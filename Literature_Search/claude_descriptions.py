"""
Literature summaries (organ-organ axes and single organs) written by a Claude model.

Used instead of the Ollama code path of llm_descriptions.py when the description model
name starts with "claude-" (for example "claude-haiku-5-5", see DESCRIPTION_LLM_MODEL in
reference_network_only_metabolic/config.py). Same output format as the Ollama path:
every entry holds the summary text with inline [n] citations and the list of source
papers (`papers`, so [n] = papers[n-1]), which the dashboard shows as the bibliography.

How the sources are protected against hallucination
---------------------------------------------------
* The sources are chosen BY INDEX from the papers already fetched from PubMed
  (select step: structured JSON answer, indices outside the pool are dropped). The model
  never types a title, PMID or year: the bibliography is built from our own records.
* The summary is written with the Anthropic *citations* feature: every source is passed
  as a document block and the API itself attaches each cited passage (verbatim excerpt of
  the document) to the sentence it supports. The [n] markers in the text are generated
  from those citations by this code, not typed by the model.
* A summary is only accepted if (1) every citation points to one of the sources, (2) every
  cited passage is found, word for word, in the title/abstract of that source (own regex
  check), (3) the model did not type reference numbers itself, and (4) every sentence
  carries at least one citation. Otherwise it is regenerated (3 attempts); sentences that
  are still uncited after the last attempt are removed, and if nothing citable is left the
  description stays empty. The cited passages are stored in the entry (`citations`).

Sources per summary: up to CLAUDE_TOP_N (10) papers, chosen from a random pool of up to
CLAUDE_POOL (40) papers (papers with an abstract are preferred).
"""

import html
import json
import random
import re
import shutil
import threading
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from Literature_Search import llm_condition_filter as _cf

CLAUDE_POOL           = 40      # papers sampled per pair/organ before the selection step
CLAUDE_TOP_N          = 10      # sources per summary (all papers if fewer are available)
SELECT_ABSTRACT_CHARS = 700     # abstract length shown in the selection step
SOURCE_ABSTRACT_CHARS = 3000    # abstract length of a source document
MIN_ABSTRACT_CHARS    = 200     # papers with shorter abstracts are only used if needed
MAX_ATTEMPTS          = 3
SUMMARY_MAX_TOKENS    = 6000    # citation markup is counted as output: a 150-word summary can need 1,500+ tokens
SELECT_MAX_TOKENS     = 300
EFFORT                = "medium"
WORKERS               = 8
MAX_CONSECUTIVE_ERRORS = 5
RETRY_PASSES          = 2      # extra passes over items that ended without a description

_SELECT_SCHEMA = {
    "type": "object",
    "properties": {"selected": {"type": "array", "items": {"type": "integer"}}},
    "required": ["selected"],
    "additionalProperties": False,
}
_OWN_REFERENCE_RE = re.compile(r"\[\s*\d+(?:\s*[,;\u2013-]\s*\d+)*\s*\]")   # [3] [1,2] [4-6]; not [18F]
_MARKER_RE        = re.compile(r"\[\d+(?:,\d+)*\]")
_FIRST_PERSON_REFUSAL_RE = re.compile(
    r"\b(I cannot|I can't|I can not|I am unable|I'm unable|as an AI|I apologi[sz]e)\b", re.IGNORECASE)

SYSTEM_PROMPT = "You are an expert in metabolic physiology and inter-organ communication."


def is_claude_model(model: str) -> bool:
    return _cf.is_anthropic_model(model)


# ── Text helpers ─────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", html.unescape(s or "")).lower()
    return re.sub(r"\s+", " ", s).strip()


def _doc_text(paper: dict) -> str:
    """Text of one source document: title + abstract (the only text the model may cite)."""
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) > SOURCE_ABSTRACT_CHARS:
        abstract = abstract[:SOURCE_ABSTRACT_CHARS].rsplit(" ", 1)[0]
    title = (paper.get("title") or "").strip()
    return f"{title}\n\n{abstract}" if abstract else title


def _cost(model: str, tin: int, tout: int):
    return _cf.estimate_cost(model, tin, tout)


# ── API calls ────────────────────────────────────────────────────────────────

def _create(model: str, max_tokens: int, messages: list, system: str = SYSTEM_PROMPT,
            effort: str = EFFORT, schema: "dict | None" = None):
    cfg: dict = {"effort": effort}
    if schema is not None:                       # structured output (cannot be combined with citations)
        cfg["format"] = {"type": "json_schema", "schema": schema}
    return _cf._get_anthropic_client().messages.create(
        model=model, max_tokens=max_tokens, system=system, messages=messages,
        thinking={"type": "disabled"}, output_config=cfg)


def check_claude(model: str) -> bool:
    """One tiny request that uses the citations feature (about $0.0001): proves the key,
    the credit balance, the model name and the request format before a run starts."""
    try:
        resp = _create(model, 80, [{"role": "user", "content": [
            {"type": "document", "title": "Test", "citations": {"enabled": True},
             "source": {"type": "text", "media_type": "text/plain",
                        "data": "Insulin lowers blood glucose by promoting glucose uptake in muscle."}},
            {"type": "text", "text": "In one short sentence: what does insulin do? Cite the document."}]}])
        if not any(b.type == "text" for b in resp.content):
            print("[!] Claude API reachable but returned no text.")
            return False
        return True
    except Exception as exc:
        print(f"[!] Claude API check failed for '{model}': {type(exc).__name__}: {exc}\n"
              "    Needs ANTHROPIC_API_KEY (new terminal after `setx`) and credit in the "
              "Anthropic Console (Billing).")
        return False


# ── Step 1: choose the sources ───────────────────────────────────────────────

def _select_prompt(kind: str, subject: tuple, pool: list[dict], top: int) -> str:
    listing = "\n\n".join(
        f"[{i}] ({p.get('year', '')}) {(p.get('title') or '').strip()}\n"
        f"{(p.get('abstract') or '').strip()[:SELECT_ABSTRACT_CHARS]}"
        for i, p in enumerate(pool, 1))
    if kind == "pair":
        o1, o2 = subject
        task = (f"Below are {len(pool)} papers retrieved for the {o1}-{o2} connection. Select the {top} papers "
                f"that most directly report a metabolic or hormonal interaction between {o1} and {o2}.")
    else:
        organ = subject[0]
        task = (f"Below are {len(pool)} papers retrieved for the role of the {organ} in metabolism. Select the "
                f"{top} papers most directly relevant to the {organ}'s own metabolic or hormonal function "
                f"(not its interaction with one specific other organ).")
    return (f"{task}\nAnswer with the paper numbers in the JSON field \"selected\", best first.\n\n"
            f"Papers:\n{listing}")


def _parse_selection(raw: str, n_pool: int, top: int) -> list[int]:
    """0-based indices: valid, unique, at most `top`, padded with the next unused pool papers."""
    try:
        sel = json.loads(raw).get("selected", [])
    except (json.JSONDecodeError, AttributeError):
        sel = []
    idx: list[int] = []
    for v in sel if isinstance(sel, list) else []:
        if isinstance(v, int) and 1 <= v <= n_pool and (v - 1) not in idx:
            idx.append(v - 1)
    idx = idx[:top]
    for i in range(n_pool):
        if len(idx) >= top:
            break
        if i not in idx:
            idx.append(i)
    return idx


# ── Step 2: summary with verified citations ──────────────────────────────────

def _summary_instruction(kind: str, subject: tuple) -> str:
    if kind == "pair":
        o1, o2 = subject
        task = (f"Write a scientific summary of 5-7 sentences (about 150 words) about the metabolic and hormonal "
                f"basis of the {o1}-{o2} axis. Focus on the most important signalling molecules and mechanisms.")
    else:
        organ = subject[0]
        task = (f"Write a scientific summary of 5-7 sentences (about 150 words) about the metabolic function of "
                f"the {organ}: which metabolic substrates it consumes or produces, which pathways supply its energy "
                f"and which hormones regulate its metabolism, as far as the documents show.")
    return (f"{task}\nRules:\n"
            "- Use ONLY information stated in the documents above. Add nothing from your own knowledge.\n"
            "- Write in your own words: paraphrase and combine findings across the documents. Do not copy sentences "
            "or long phrases from the documents; one short sentence may report a finding of several documents.\n"
            "- Every sentence must be supported by at least one document; cite it with the citation feature.\n"
            "- Do not write reference numbers, PMIDs, authors or titles yourself; references are added automatically.\n"
            "- Draw on every document that contributes relevant evidence.\n"
            "- Flowing prose: no bullet points, no headings, no meta-commentary. Output only the summary.")


def _documents(sources: list[dict]) -> list[dict]:
    return [{"type": "document", "title": ((p.get("title") or "").strip()[:200] or f"Paper {i}"),
             "context": f"PubMed record, {p.get('year', '')}", "citations": {"enabled": True},
             "source": {"type": "text", "media_type": "text/plain", "data": _doc_text(p)}}
            for i, p in enumerate(sources, 1)]


def assemble_cited_text(blocks, sources: list[dict]) -> tuple["str | None", list[dict], "str | None"]:
    """Turn the response blocks into text with [n] markers built from the API citations.
    Returns (text, citations, problem); problem is None if every citation is valid."""
    docs_norm = [_norm(_doc_text(p)) for p in sources]
    parts: list[tuple[str, list[int]]] = []
    cites: list[dict] = []
    raw = ""
    for b in blocks:
        if getattr(b, "type", "") != "text":
            continue
        raw += b.text
        nums: list[int] = []
        for c in (getattr(b, "citations", None) or []):
            if getattr(c, "type", "") != "char_location":
                return None, [], f"unexpected citation type {getattr(c, 'type', '?')!r}"
            i = c.document_index
            if not 0 <= i < len(sources):
                return None, [], f"citation of document {i + 1} outside the {len(sources)} sources"
            quote = _norm(c.cited_text)
            if not quote or quote not in docs_norm[i]:
                return None, [], f"cited passage not found in source {i + 1}"
            nums.append(i + 1)
            cites.append({"n": i + 1, "pmid": str(sources[i].get("pmid", "")), "quote": c.cited_text.strip()})
        parts.append((b.text, sorted(set(nums))))
    if _OWN_REFERENCE_RE.search(raw):
        return None, [], "the model wrote reference numbers itself"
    out = []
    for txt, nums in parts:
        m = re.match(r"^(.*?)([\s.!?;:,]*)$", txt, re.S)
        core, tail = m.group(1), m.group(2)
        out.append(f"{core} [{','.join(map(str, nums))}]{tail}" if nums and core.strip() else txt)
    text = re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
    return text, cites, None


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip()) if s.strip()]


MIN_UNCITED_TAIL_WORDS = 5     # an uncited clause after the last citation of a sentence this long is cut off


def trim_uncited_tails(text: str) -> tuple[str, int]:
    """Cut the clause that follows the last [n] marker of a sentence when it has at least
    MIN_UNCITED_TAIL_WORDS words: it is the model's own addition ("..., suggesting that ..."),
    not backed by a citation. Returns (text, number of clauses removed)."""
    out, removed = [], 0
    for sent in _split_sentences(text):
        marks = list(_MARKER_RE.finditer(sent))
        if marks:
            tail = sent[marks[-1].end():]
            if len(re.findall(r"[A-Za-z]{2,}", tail)) >= MIN_UNCITED_TAIL_WORDS:
                sent = sent[:marks[-1].end()] + "."
                removed += 1
        out.append(sent)
    return " ".join(out), removed


def describe(kind: str, subject: tuple, papers: list[dict], model: str, key: str, variant: int = 0) -> dict:
    """Generate one entry (pair or organ). Always returns a cache entry; never raises.
    variant > 0 (retry pass) samples a different pool of papers than the first pass."""
    base = {"generated_at": datetime.now().isoformat(), "model": model}
    tin = tout = 0
    try:
        if not papers:
            return {**base, "description": "", "n_papers_used": 0}
        rng = random.Random(f"{kind}|{key}" + (f"|retry{variant}" if variant else ""))   # reproducible sampling
        rich = [p for p in papers if len((p.get("abstract") or "").strip()) >= MIN_ABSTRACT_CHARS]
        source_pool = rich if len(rich) >= CLAUDE_TOP_N else list(papers)
        pool = source_pool if len(source_pool) <= CLAUDE_POOL else rng.sample(source_pool, CLAUDE_POOL)

        if len(pool) > CLAUDE_TOP_N:
            resp = _create(model, SELECT_MAX_TOKENS,
                           [{"role": "user", "content": _select_prompt(kind, subject, pool, CLAUDE_TOP_N)}],
                           effort="low", schema=_SELECT_SCHEMA)
            tin += resp.usage.input_tokens
            tout += resp.usage.output_tokens
            raw = "".join(b.text for b in resp.content if b.type == "text")
            sources = [pool[i] for i in _parse_selection(raw, len(pool), CLAUDE_TOP_N)]
        else:
            sources = pool

        content = _documents(sources) + [{"type": "text", "text": _summary_instruction(kind, subject)}]
        text, cites, problem, dropped, trimmed = "", [], "no attempt", 0, 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = _create(model, SUMMARY_MAX_TOKENS, [{"role": "user", "content": content}])
            tin += resp.usage.input_tokens
            tout += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                raise RuntimeError("the model declined to answer (stop_reason=refusal)")
            text, cites, problem = assemble_cited_text(resp.content, sources)
            trimmed = 0
            if problem is None and text:
                text, trimmed = trim_uncited_tails(text)
            if problem is None and resp.stop_reason == "max_tokens":
                problem = "reply cut off at max_tokens"
            if problem is None and (not text or _FIRST_PERSON_REFUSAL_RE.search(text)):
                problem = "empty reply / refusal"
            if problem is None and not _MARKER_RE.search(text):
                problem = "no citations"
            if problem is None:
                sentences = _split_sentences(text)
                uncited = [s for s in sentences if not _MARKER_RE.search(s)]
                if uncited and attempt < MAX_ATTEMPTS:
                    problem = f"{len(uncited)} sentence(s) without citation"
                elif uncited:                                   # last attempt: keep only what is cited
                    kept = [s for s in sentences if _MARKER_RE.search(s)]
                    if len(kept) < (2 if len(sources) >= 3 else 1) or len(kept) < len(uncited):
                        problem = f"only {len(kept)} of {len(sentences)} sentences carry a citation"
                    else:
                        text, dropped = " ".join(kept), len(uncited)
                        shown = {int(n) for m in _MARKER_RE.findall(text) for n in re.findall(r"\d+", m)}
                        cites = [c for c in cites if c["n"] in shown]
            if problem is None:
                break
        if problem is not None:
            return {**base, "description": "", "n_papers_used": 0, "tokens_in": tin, "tokens_out": tout,
                    "error": f"rejected after {MAX_ATTEMPTS} attempts: {problem}", "rejected_text": text}
        entry = {**base, "description": text, "n_papers_used": len(sources), "papers": sources,
                 "citations": cites, "n_cited_papers": len({c["n"] for c in cites}),
                 "pool_size": len(pool), "attempts": attempt, "tokens_in": tin, "tokens_out": tout}
        if dropped:
            entry["dropped_uncited_sentences"] = dropped
        if trimmed:
            entry["trimmed_uncited_clauses"] = trimmed
        return entry
    except Exception as exc:
        return {**base, "description": "", "n_papers_used": 0, "tokens_in": tin, "tokens_out": tout,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


# ── Runner (cache, parallel requests, incremental saving) ────────────────────

def run_claude_descriptions(items: list[dict], kind: str, output_path: "str | Path", model: str,
                            resume: bool = True, reset: bool = False,
                            workers: int = WORKERS) -> dict:
    """items: [{"key": "A|B", "alt": "B|A" or None, "subject": (A, B), "papers": [...]}, ...]
    (organs: key = organ name, alt None, subject (organ,)). Same cache semantics as the
    Ollama path: an entry is reused only if it has a description made by this model.
    Items that end without a description (rejected summary, request error) are retried in up
    to RETRY_PASSES extra passes, each with a different random pool of papers."""
    output_path = Path(output_path)
    descriptions: dict = {}
    if reset and output_path.exists():
        output_path.unlink()
        print(f"[i] Cache deleted: {output_path.name}")
    elif resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            descriptions = json.load(f)
        old_models = {v.get("model") for v in descriptions.values() if v.get("description")} - {model}
        if old_models:       # keep the earlier model's descriptions once, before they are replaced
            tag = re.sub(r"[^A-Za-z0-9._-]+", "_", sorted(old_models)[0] or "unknown")
            backup = output_path.with_name(f"{output_path.stem}.{tag}.bak{output_path.suffix}")
            if not backup.exists():
                shutil.copy2(output_path, backup)
                print(f"[i] Descriptions of {sorted(old_models)} backed up to {backup.name}")
        usable = sum(1 for v in descriptions.values() if v.get("description") and v.get("model") == model)
        print(f"[i] Cache: {usable} usable entries for model '{model}'"
              f"{f', {len(descriptions) - usable} to (re)generate' if len(descriptions) - usable else ''}.")

    if not check_claude(model):
        sys.exit(2)                 # stop the whole run (run_haiku_all.ps1 halts on a non-zero exit code)

    def has_description(it: dict) -> bool:
        e = descriptions.get(it["key"]) or (descriptions.get(it["alt"]) if it.get("alt") else None)
        return bool(e and e.get("description") and e.get("model") == model)

    todo = [it for it in items if not has_description(it)]
    print(f"[i] {len(items)} {kind} items | {len(items) - len(todo)} cached | {len(todo)} to do | "
          f"model {model} | up to {CLAUDE_TOP_N} sources each | workers: {workers}")
    tin = tout = n_ok = 0
    streak = 0
    stopped = False
    t0 = time.time()
    lock = threading.Lock()
    workers = max(1, workers)
    chunk = workers * 3

    def run_batch(batch: list[dict], variant: int, pool: ThreadPoolExecutor) -> None:
        nonlocal tin, tout, n_ok, streak, stopped
        tag = f"retry {variant}: " if variant else ""

        def _work(it: dict) -> tuple[dict, dict]:
            return it, describe(kind, it["subject"], it["papers"], model, it["key"], variant)

        done_n = 0
        for start in range(0, len(batch), chunk):
            if streak >= MAX_CONSECUTIVE_ERRORS:
                stopped = True
                return
            for it, entry in pool.map(_work, batch[start:start + chunk]):
                done_n += 1
                tin += entry.get("tokens_in", 0)
                tout += entry.get("tokens_out", 0)
                if entry.get("description"):
                    n_ok += 1
                    streak = 0
                    extra = (f", {entry.get('dropped_uncited_sentences')} uncited sentence(s) dropped"
                             if entry.get("dropped_uncited_sentences") else "")
                    msg = (f"ok ({len(entry['description'].split())} words, {entry['n_papers_used']} sources, "
                           f"{entry['n_cited_papers']} cited{extra})")
                elif entry.get("error"):
                    streak = streak + 1 if not entry["error"].startswith("rejected") else 0
                    msg = f"no description: {entry['error'][:120]}"
                else:
                    msg = "skipped (no papers)"
                with lock:
                    old = descriptions.get(it["key"])
                    # a failed retry must not erase an earlier description made by this model
                    if entry.get("description") or not (old and old.get("description") and old.get("model") == model):
                        descriptions[it["key"]] = entry
                    if it.get("alt") and it["alt"] in descriptions and it["alt"] != it["key"]:
                        del descriptions[it["alt"]]         # one entry per pair, under the canonical key
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(descriptions, f, indent=2, ensure_ascii=False)
                print(f"  [{tag}{done_n}/{len(batch)}] {it['key']} ({len(it['papers'])} papers) ... {msg}", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            run_batch(todo, 0, pool)
            for variant in range(1, RETRY_PASSES + 1):
                if stopped:
                    break
                failed = [it for it in todo if it["papers"] and not has_description(it)]
                if not failed:
                    break
                print(f"\n[i] Retry pass {variant}/{RETRY_PASSES}: {len(failed)} item(s) without a description "
                      f"-- a different random pool of papers is used", flush=True)
                run_batch(failed, variant, pool)
    except KeyboardInterrupt:
        print("\n[!] Interrupted -- progress is saved; re-run the same command to resume.")
    if stopped:
        print(f"[!] Stopped: {streak} requests failed in a row. Fix the cause (key, credit, network) and re-run.")
    still = [it["key"] for it in todo if it["papers"] and not has_description(it)]
    cost = _cost(model, tin, tout)
    print(f"\n[ok] {sum(1 for v in descriptions.values() if v.get('description'))}/{len(descriptions)} "
          f"{kind} entries have descriptions ({n_ok} new) in {(time.time() - t0) / 60:.1f} min -> {output_path}"
          + (f" | {tin:,} input + {tout:,} output tokens ~ ${cost:.3f}" if cost is not None else ""))
    if still:
        print(f"[!] {len(still)} item(s) still without a description after {RETRY_PASSES} retry passes: "
              f"{', '.join(still)}\n    Re-run the same command to try again.")
    return descriptions
