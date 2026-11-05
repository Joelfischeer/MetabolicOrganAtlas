"""
LLM-based connection-type classification for organ-organ pairs.

For each pair the LLM reads up to MAX_PAPERS paper abstracts and picks
between 1 and 3 connection types (ordered by relevance) from a user-defined
list. To make the classification robust against single-sample noise, each
pair is classified 3 times independently and only the types that appear in
at least 2 of the 3 runs (majority vote) are kept; if no type reaches a
majority, the single most-voted type is kept as a fallback so every pair
always ends up with at least one type.

Results are saved incrementally to a JSON file so the run can be resumed.

If the model name starts with "claude-" (for example claude-haiku-5-5, see
CONNECTION_TYPE_LLM_MODEL in reference_network_only_metabolic/config.py) the Claude API is
used instead of Ollama: same vote scheme, the answer is a JSON list restricted to the exact
connection-type keys (structured output), and the abstracts + type list are prompt-cached so
the repeated runs of a pair cost little. Claude models reject temperature, so the runs differ
through the default sampling.
"""

import json
import re
import shutil
import textwrap
import threading
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests as _requests

OLLAMA_URL       = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT   = 180
OLLAMA_RETRIES   = 2
MAX_ABSTRACT_CHARS = 400
VOTE_RUNS        = 5     # independent classification passes per pair
VOTE_TEMPERATURE = 0.5   # higher than a single-shot call, so the runs can
                          # actually disagree — a majority vote over near-
                          # identical low-temperature replies is meaningless
MAX_TYPES        = 3


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def _check_ollama(model: str) -> bool:
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        available = [m["name"] for m in resp.json().get("models", [])]
        return any(model in m for m in available)
    except Exception:
        return False


def _call_ollama(prompt: str, model: str, temperature: float = 0.1) -> str:
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "options":  {"temperature": temperature, "num_predict": 512},
    }
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            resp = _requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            msg = resp.json()["message"]
            # Thinking models (e.g. north-mini-code) put output in 'thinking';
            # standard models use 'content'.
            content = msg.get("content", "").strip()
            if not content:
                content = msg.get("thinking", "").strip()
            return content
        except _requests.exceptions.Timeout:
            if attempt < OLLAMA_RETRIES:
                print(f"\n    [!] Timeout (attempt {attempt}/{OLLAMA_RETRIES}), retrying…",
                      end=" ", flush=True)
            else:
                raise


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    organ1: str,
    organ2: str,
    papers: list[dict],
    connection_types: dict[str, dict],
    max_papers: int,
) -> str:
    # Build numbered paper block
    if not papers:
        paper_block = "No papers available."
    else:
        lines = []
        for i, p in enumerate(papers[:max_papers], 1):
            title    = p.get("title", "").strip()
            abstract = p.get("abstract", "").strip()
            if len(abstract) > MAX_ABSTRACT_CHARS:
                abstract = abstract[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(f"[{i}] {title}\n    {abstract}")
        paper_block = "\n\n".join(lines)

    # Build numbered type list
    type_lines = "\n".join(
        f"  {i+1}. {key} — {info['label']}: {info['description']}"
        for i, (key, info) in enumerate(connection_types.items())
    )
    keys = list(connection_types.keys())
    valid_keys = ", ".join(keys)

    return textwrap.dedent(f"""
        You are an expert in metabolic physiology and inter-organ signalling.

        Task: Based on the papers below about the {organ1}–{organ2} connection,
        choose which connection type(s) from the list below best fit the
        evidence. Pick the SINGLE most-feasible type as TYPE_1. Only add
        TYPE_2 and/or TYPE_3 if they are ALSO clearly and independently
        supported by the papers — do not pad the list to reach 3. It is
        fine and common to list only 1 type.

        Connection types:
        {type_lines}

        Papers:
        {paper_block}

        Instructions:
        - Reply with ONLY the required lines below — no explanation, no
          preamble, no extra text of any kind.
        - Line 1 (required): TYPE_1: <key>
        - Line 2 (optional):  TYPE_2: <key>
        - Line 3 (optional):  TYPE_3: <key>
        - <key> must be reply only with one of these exact tokens, copied
          character-for-character — do not paraphrase, translate, combine,
          or invent a new label under any circumstances:
          {valid_keys}
        - If no type fits perfectly, you MUST still pick the closest one
          from the exact token list above. Inventing a new category name
          (e.g. "gut-heart axis") or leaving <key> blank is not allowed.
        - Order matters: TYPE_1 must be the best-matching type.
        - Omit TYPE_2/TYPE_3 entirely (do not write the line) if no further
          type clearly applies.

        Remember: <key> must be exactly one of: {valid_keys}

        Reply:
    """).strip()


# ── Parser ─────────────────────────────────────────────────────────────────────

def _parse_response(text: str, valid_keys: list[str]) -> list[str]:
    """Extract an ordered, deduplicated list of up to MAX_TYPES keys."""
    types: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("TYPE_"):
            continue
        if ":" not in line:
            continue
        candidate = line.split(":", 1)[1].strip().lower().replace(" ", "_")
        if candidate in valid_keys and candidate not in types:
            types.append(candidate)

    # Fallback: scan for any valid key mentioned anywhere in the reply,
    # in the order they appear, in case the model didn't follow the format.
    if not types:
        lowered = text.lower()
        found = [(lowered.index(key), key) for key in valid_keys if key in lowered]
        types = [key for _, key in sorted(found)]

    return types[:MAX_TYPES]


# ── Majority vote across runs ────────────────────────────────────────────────

def _majority_vote(runs: list[list[str]], max_types: int = MAX_TYPES) -> list[str]:
    """
    Combine N independent ordered type-lists into one final list.

    A type is kept if it appears in a majority of the runs (more than half).
    If no type reaches a majority (e.g. all runs disagree completely), the
    single most-voted type is kept instead, so every pair with at least one
    valid run always ends up with >=1 type. Ties are broken by earlier
    average rank (types listed first / more consistently are more relevant).
    Result is capped at `max_types`.
    """
    votes: dict[str, int] = defaultdict(int)
    rank_sum: dict[str, int] = defaultdict(int)
    order: list[str] = []
    for run in runs:
        for rank, key in enumerate(run):
            votes[key] += 1
            rank_sum[key] += rank
            if key not in order:
                order.append(key)

    if not votes:
        return []

    def sort_key(k: str):
        return (-votes[k], rank_sum[k] / votes[k])

    majority_needed = len(runs) // 2 + 1
    majority = sorted((k for k in order if votes[k] >= majority_needed), key=sort_key)

    if not majority:
        best = min(order, key=sort_key)
        return [best]

    return majority[:max_types]


def _classify_once(
    organ1: str, organ2: str, papers: list[dict],
    connection_types: dict[str, dict], max_papers: int,
    model: str, valid_keys: list[str],
) -> tuple[list[str], str]:
    """Run one independent classification pass. Returns (types, raw_reply)."""
    prompt = _build_prompt(organ1, organ2, papers, connection_types, max_papers)
    raw = _call_ollama(prompt, model, temperature=VOTE_TEMPERATURE)
    return _parse_response(raw, valid_keys), raw


# ── Claude API backend ─────────────────────────────────────────────────────────

CLAUDE_MAX_TOKENS     = 200
CLAUDE_EFFORT         = "low"
CLAUDE_WORKERS        = 8          # pairs classified in parallel
MAX_FAILED_PAIRS_IN_A_ROW = 5      # a run stops after this many pairs whose runs all failed
_CLAUDE_SYSTEM = "You are an expert in metabolic physiology and inter-organ signalling."


def _is_claude(model: str) -> bool:
    return str(model).startswith("claude-")


def _claude_prompt_blocks(organ1: str, organ2: str, papers: list[dict],
                          connection_types: dict[str, dict], max_papers: int) -> list[dict]:
    """The paper list and the type list form one cached block (identical for every vote run
    of a pair); the short instruction follows it."""
    lines = []
    for i, p in enumerate(papers[:max_papers], 1):
        abstract = (p.get("abstract") or "").strip()
        if len(abstract) > MAX_ABSTRACT_CHARS:
            abstract = abstract[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + "…"
        lines.append(f"[{i}] {(p.get('title') or '').strip()}\n    {abstract}")
    type_lines = "\n".join(f"  {k} — {info['label']}: {info['description']}"
                           for k, info in connection_types.items())
    data = (f"Papers about the {organ1}–{organ2} connection:\n\n" + "\n\n".join(lines) +
            f"\n\nConnection types (key — label: description):\n{type_lines}")
    task = (f"Based on the papers above, choose which connection type(s) from the list best fit the evidence "
            f"for the {organ1}–{organ2} connection. Return them in the JSON field \"types\", most feasible first. "
            f"Give 1 type; add a 2nd or 3rd only if it is ALSO clearly and independently supported by the "
            f"papers (do not pad the list; 1 type is common). Use the exact keys from the list.")
    return [{"type": "text", "text": data, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": task}]


def _claude_schema(valid_keys: list[str]) -> dict:
    return {"type": "object",
            "properties": {"types": {"type": "array", "items": {"type": "string", "enum": valid_keys}}},
            "required": ["types"], "additionalProperties": False}


def _parse_claude_types(raw: str, valid_keys: list[str]) -> list[str]:
    """Ordered, de-duplicated, valid keys (at most MAX_TYPES) from the JSON reply."""
    try:
        value = json.loads(raw).get("types", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    canon = {k.lower(): k for k in valid_keys}
    out: list[str] = []
    for v in value if isinstance(value, list) else []:
        k = canon.get(str(v).strip().lower())
        if k and k not in out:
            out.append(k)
    return out[:MAX_TYPES]


def _claude_call(content: list[dict], valid_keys: list[str], model: str):
    from Literature_Search import llm_condition_filter as cf
    resp = cf._get_anthropic_client().messages.create(
        model=model, max_tokens=CLAUDE_MAX_TOKENS, system=_CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": content}],
        thinking={"type": "disabled"},
        output_config={"effort": CLAUDE_EFFORT,
                       "format": {"type": "json_schema", "schema": _claude_schema(valid_keys)}})
    if resp.stop_reason == "refusal":
        raise RuntimeError("the model declined to answer (stop_reason=refusal)")
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    u = resp.usage
    usage = {"in": u.input_tokens, "out": u.output_tokens,
             "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}
    return raw, usage


def _claude_cost(model: str, usage: dict) -> "float | None":
    from Literature_Search import llm_condition_filter as cf
    price = cf.API_PRICES.get(model)
    if price is None:
        return None
    billed_in = usage["in"] + 1.25 * usage["cache_write"] + 0.1 * usage["cache_read"]
    return (billed_in * price[0] + usage["out"] * price[1]) / 1e6


def check_claude_types(model: str, valid_keys: list[str]) -> bool:
    """Tiny request in the exact format of a real classification (about $0.0001)."""
    try:
        raw, _ = _claude_call(
            [{"type": "text", "text": "Papers about the Liver–Muscle connection:\n\n[1] Insulin and glucose "
                                      "uptake\n    Insulin stimulates glucose uptake in muscle.\n\n"
                                      "Connection types:\n" + "\n".join(f"  {k}" for k in valid_keys),
              "cache_control": {"type": "ephemeral"}},
             {"type": "text", "text": "Return the single best type in \"types\"."}], valid_keys, model)
        if not _parse_claude_types(raw, valid_keys):
            print(f"[!] Claude API reachable but the reply has no valid type: {raw[:200]!r}")
            return False
        return True
    except Exception as exc:
        print(f"[!] Claude API check failed for '{model}': {type(exc).__name__}: {exc}\n"
              "    Needs ANTHROPIC_API_KEY (new terminal after `setx`) and credit in the "
              "Anthropic Console (Billing).")
        return False


def _classify_pair_claude(o1: str, o2: str, papers: list[dict], connection_types: dict[str, dict],
                          max_papers: int, model: str) -> dict:
    """VOTE_RUNS independent runs on the same cached prompt -> majority vote. Never raises."""
    valid_keys = list(connection_types)
    content = _claude_prompt_blocks(o1, o2, papers, connection_types, max_papers)
    runs, raws, errors = [], [], []
    total = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0}
    for _ in range(VOTE_RUNS):           # sequential: the first run writes the prompt cache, the others read it
        try:
            raw, usage = _claude_call(content, valid_keys, model)
            for k in total:
                total[k] += usage[k]
            runs.append(_parse_claude_types(raw, valid_keys))
            raws.append(raw)
        except Exception as exc:
            runs.append([])
            raws.append("")
            errors.append(f"{type(exc).__name__}: {exc}"[:200])
    entry = {"types": _majority_vote(runs), "runs": runs, "raw_responses": raws,
             "generated_at": datetime.now().isoformat(), "model": model, "tokens": total}
    if errors:
        entry["errors"] = errors
    return entry


def _generate_claude(organ_pairs, literature_results, connection_types, output_path: Path,
                     model: str, max_papers: int, resume: bool, reset: bool) -> dict:
    cache: dict = {}
    if reset and output_path.exists():
        output_path.unlink()
        print(f"[i] LLM type cache deleted: {output_path.name}")
    elif resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            cache = json.load(f)
        old = {v.get("model") for v in cache.values() if v.get("types")} - {model}
        if old:          # keep the earlier model's classification once, before it is replaced
            tag = re.sub(r"[^A-Za-z0-9._-]+", "_", sorted(old, key=str)[0] or "unknown")
            backup = output_path.with_name(f"{output_path.stem}.{tag}.bak{output_path.suffix}")
            if not backup.exists():
                shutil.copy2(output_path, backup)
                print(f"[i] Classification of {sorted(old, key=str)} backed up to {backup.name}")

    valid_keys = list(connection_types)
    if not check_claude_types(model, valid_keys):
        sys.exit(2)                 # stop the whole run (run_haiku_all.ps1 halts on a non-zero exit code)

    def lit_of(o1, o2):
        return literature_results.get(f"{o1}|{o2}") or literature_results.get(f"{o2}|{o1}") or {}

    def is_cached(o1, o2) -> bool:
        e = cache.get(f"{o1}|{o2}") or cache.get(f"{o2}|{o1}")
        # same rule as the Ollama path (a pair with papers but no type is unfinished), plus: the
        # entry must come from this model -- a classification by another model is replaced
        return bool(e and e.get("model") == model and (e.get("types") or not lit_of(o1, o2).get("papers")))

    todo = [p for p in organ_pairs if not is_cached(*p)]
    print(f"[i] LLM type classification using model: {model} | {len(organ_pairs)} pairs | "
          f"{len(organ_pairs) - len(todo)} cached | {len(todo)} to do | {VOTE_RUNS} votes each | "
          f"up to {max_papers} abstracts | workers: {CLAUDE_WORKERS}")

    totals = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0}
    failed_streak = n_done = 0
    t0 = time.time()
    lock = threading.Lock()

    def _work(pair):
        o1, o2 = pair
        papers = lit_of(o1, o2).get("papers", [])
        if not papers:
            return pair, {"types": [], "generated_at": datetime.now().isoformat(), "model": model}
        return pair, _classify_pair_claude(o1, o2, papers, connection_types, max_papers, model)

    chunk = CLAUDE_WORKERS * 3
    try:
        with ThreadPoolExecutor(max_workers=CLAUDE_WORKERS) as pool:
            for start in range(0, len(todo), chunk):
                if failed_streak >= MAX_FAILED_PAIRS_IN_A_ROW:
                    break
                for (o1, o2), entry in pool.map(_work, todo[start:start + chunk]):
                    n_done += 1
                    for k in totals:
                        totals[k] += entry.get("tokens", {}).get(k, 0)
                    has_papers = bool(lit_of(o1, o2).get("papers"))
                    failed_streak = failed_streak + 1 if (has_papers and not entry["types"]
                                                          and len(entry.get("errors", [])) == VOTE_RUNS) else 0
                    key = f"{o1}|{o2}"
                    with lock:
                        cache[key] = entry
                        if f"{o2}|{o1}" != key:
                            cache.pop(f"{o2}|{o1}", None)        # one entry per pair
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(cache, f, indent=2, ensure_ascii=False)
                    if not has_papers:
                        msg = "skipped (no papers)"
                    elif entry["types"]:
                        msg = f"types={'+'.join(entry['types'])}  (votes: {entry['runs']})"
                    else:
                        msg = ("ERROR: all runs failed: " + entry["errors"][0]) if len(entry.get("errors", [])) == VOTE_RUNS \
                            else "no type matched"
                    print(f"  [{n_done}/{len(todo)}] {o1} <-> {o2} ({len(lit_of(o1, o2).get('papers', []))} papers) ... {msg}",
                          flush=True)
    except KeyboardInterrupt:
        print("\n[!] Interrupted -- progress is saved; re-run the same command to resume.")
    if failed_streak >= MAX_FAILED_PAIRS_IN_A_ROW:
        print(f"[!] Stopped: {failed_streak} pairs in a row failed completely. "
              "Fix the cause (key, credit, network) and re-run.")
    cost = _claude_cost(model, totals)
    classified = sum(1 for v in cache.values() if v.get("types"))
    print(f"\n[ok] {classified}/{len(cache)} pairs classified in {(time.time() - t0) / 60:.1f} min. Saved: {output_path}"
          + (f" | {totals['in']:,} in + {totals['cache_write']:,} cache-write + {totals['cache_read']:,} cache-read"
             f" + {totals['out']:,} out tokens ~ ${cost:.3f}" if cost is not None else ""))
    return cache


# ── Main function ──────────────────────────────────────────────────────────────

def generate_connection_type_classifications(
    organ_pairs: list[tuple[str, str]],
    literature_results: dict,
    connection_types: dict[str, dict],
    output_path: "str | Path",
    model: str = "north-mini-code-1.0",
    max_papers: int = 5,
    resume: bool = True,
    reset: bool = False,
) -> dict:
    """
    Classify each organ pair into 1-3 connection types using an Ollama LLM.

    Each pair is classified VOTE_RUNS times independently; only types
    appearing in a majority of the runs are kept (falls back to the single
    most-voted type if no type reaches a majority), guaranteeing every
    classified pair gets at least one type and never more than MAX_TYPES.

    Parameters
    ----------
    organ_pairs         : sorted list of (organ1, organ2) tuples
    literature_results  : raw search JSON (keyed "organ1|organ2")
    connection_types    : dict of {key: {label, description}} from config
    output_path         : where to save/load the classification JSON
    model               : Ollama model name
    max_papers          : how many abstracts to include in the prompt
    resume              : skip pairs already in the cache
    reset               : delete cache and reclassify everything
    """
    output_path = Path(output_path)
    if _is_claude(model):
        return _generate_claude(organ_pairs, literature_results, connection_types, output_path,
                                model, max_papers, resume, reset)
    cache: dict = {}

    if reset and output_path.exists():
        output_path.unlink()
        print(f"[i] LLM type cache deleted: {output_path.name}")
    elif resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[i] Resuming LLM type classification: {len(cache)} pairs cached.")

    if not _check_ollama(model):
        print(f"[!] Ollama not reachable or model '{model}' not found.")
        print(f"    Start Ollama and run:  ollama pull {model}")
        return cache

    print(f"[i] LLM type classification using model: {model}")
    valid_keys = list(connection_types.keys())
    total = len(organ_pairs)

    for idx, (o1, o2) in enumerate(organ_pairs):
        key = f"{o1}|{o2}"
        sym = f"{o2}|{o1}"

        cached_entry = cache.get(key) or cache.get(sym)
        lit_entry = literature_results.get(key) or literature_results.get(sym) or {}
        # Only skip if the cached run actually produced a type (or there were
        # genuinely no papers to classify) -- a pair with papers but an empty
        # "types" list (e.g. the LLM didn't follow the required token format)
        # is treated as unfinished so a resumed run retries it automatically.
        if cached_entry and resume and not reset and (
            cached_entry.get("types") or not lit_entry.get("papers")
        ):
            shown = "+".join(cached_entry.get("types", [])) or cached_entry.get("primary", "?")
            print(f"  [{idx+1}/{total}] Cached ({shown}): {o1} <-> {o2}")
            continue

        lit    = literature_results.get(key) or literature_results.get(sym) or {}
        papers = lit.get("papers", [])
        n      = lit.get("n_papers_found", len(papers))

        print(f"  [{idx+1}/{total}] {o1} <-> {o2}  ({n} papers) … ",
              end="", flush=True)

        if not papers:
            print("skipped (no papers)")
            cache[key] = {
                "types": [],
                "generated_at": datetime.now().isoformat(),
                "model": model,
            }
        else:
            runs: list[list[str]] = []
            raw_responses: list[str] = []
            errors: list[str] = []
            for _ in range(VOTE_RUNS):
                try:
                    types, raw = _classify_once(
                        o1, o2, papers, connection_types, max_papers, model, valid_keys
                    )
                    runs.append(types)
                    raw_responses.append(raw)
                except Exception as exc:
                    runs.append([])
                    raw_responses.append("")
                    errors.append(str(exc))

            final_types = _majority_vote(runs)
            cache[key] = {
                "types":         final_types,
                "runs":          runs,
                "raw_responses": raw_responses,
                "generated_at":  datetime.now().isoformat(),
                "model":         model,
            }
            if errors:
                cache[key]["errors"] = errors
            if final_types:
                print(f"types={'+'.join(final_types)}  (votes: {[r for r in runs]})")
            else:
                print(f"ERROR: all {VOTE_RUNS} runs failed" if len(errors) == VOTE_RUNS
                      else "no type matched")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    classified = sum(1 for v in cache.values() if v.get("types"))
    print(f"\n[ok] {classified}/{len(cache)} pairs classified. Saved: {output_path}")
    return cache


# ── Loader helper ──────────────────────────────────────────────────────────────

def load_connection_type_classifications(path: "str | Path") -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_connection_types(
    classifications: dict,
    organ1: str,
    organ2: str,
    connection_types: dict[str, dict],
) -> list[str]:
    """
    Return an ordered list of 0-3 human-readable type labels for an organ
    pair, most-feasible first. Empty list if not classified.

    Also reads old-format caches (single "primary"/"secondary" keys, from
    before the multi-type majority-vote scheme) so a stale cache still
    renders sensibly until it's reclassified with --reset-llm-type.
    """
    entry = (
        classifications.get(f"{organ1}|{organ2}")
        or classifications.get(f"{organ2}|{organ1}")
        or {}
    )
    keys = entry.get("types")
    if keys is None:
        keys = [k for k in (entry.get("primary"), entry.get("secondary")) if k]

    return [connection_types.get(k, {}).get("label", k) for k in keys if k]
