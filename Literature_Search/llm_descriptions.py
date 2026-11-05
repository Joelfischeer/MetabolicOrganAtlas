"""
Generate LLM-based literature summaries for each organ-organ connection pair.

Uses a locally running Ollama model to write a cited, evidence-based summary
(up to 10 sentences) for each edge, grounded in the PubMed papers already
fetched for that pair. If the model name starts with "claude-" (for example
claude-haiku-5-5) the Claude API is used instead: up to 10 source papers and
citations that are verified against the papers (see claude_descriptions.py).

Requirements
------------
  1. Install Ollama: https://ollama.com
  2. Pull the model:  ollama pull llama3.2
  3. Ollama must be running (it starts automatically on most systems)

Results are saved incrementally to metabolic_data/llm_descriptions.json.

Usage (standalone):
    uv run python Literature_Search/llm_descriptions.py
    uv run python Literature_Search/llm_descriptions.py --reset
    uv run python Literature_Search/llm_descriptions.py --model llama3.2
"""

import json
import re
import sys
import random
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

import requests as _requests   # already a project dependency

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

# Windows console/redirected-output encoding defaults to cp1252, which can't
# encode the arrows/ellipses used in progress prints below — force UTF-8 so
# the run doesn't crash mid-way through (e.g. when stdout is piped to a file).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUTPUT     = HERE / "metabolic_data" / "llm_descriptions.json"
DEFAULT_MODEL      = "llama3.2"
OLLAMA_URL         = "http://localhost:11434/api/chat"
SAMPLE_POOL        = 25    # papers randomly sampled per pair before selection
TOP_N_PAPERS       = 5     # papers the LLM picks as most relevant
MAX_ABSTRACT_CHARS = 400   # truncate long abstracts to keep prompts lean
OLLAMA_TIMEOUT     = 300   # seconds — local CPU inference can be slow
OLLAMA_RETRIES     = 2


def _is_claude(model: str) -> bool:
    """A model name starting with "claude-" is sent to the Claude API (claude_descriptions.py)."""
    return str(model).startswith("claude-")


def description_settings(model: str) -> dict:
    """Sampling/selection numbers of the description step for `model` (used in the dashboard text)."""
    if _is_claude(model):
        from Literature_Search import claude_descriptions as cd
        return {"pool": cd.CLAUDE_POOL, "top": cd.CLAUDE_TOP_N, "sentences": "5–7", "claude": True}
    return {"pool": SAMPLE_POOL, "top": TOP_N_PAPERS, "sentences": "3–4", "claude": False}


def configured_description_model() -> str:
    """DESCRIPTION_LLM_MODEL (else LLM_MODEL) from reference_network_only_metabolic/config.py."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_network_config", HERE / "reference_network_only_metabolic" / "config.py")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        return getattr(cfg, "DESCRIPTION_LLM_MODEL", getattr(cfg, "LLM_MODEL", DEFAULT_MODEL))
    except Exception:
        return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _paper_block(papers: list[dict], max_chars: int = MAX_ABSTRACT_CHARS) -> str:
    """Format a numbered list of papers for inclusion in a prompt."""
    lines = []
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract", "").strip()
        if len(abstract) > max_chars:
            abstract = abstract[:max_chars].rsplit(" ", 1)[0] + "…"
        lines.append(
            f"[{i}] PMID {p.get('pmid','?')} ({p.get('year','')})\n"
            f"    Title: {p.get('title','').strip()}\n"
            f"    Abstract: {abstract}"
        )
    return "\n\n".join(lines)


def _build_selection_prompt(organ1: str, organ2: str, papers: list[dict]) -> str:
    """Step 1 — ask the LLM to pick the most relevant paper indices."""
    block = _paper_block(papers)
    n = len(papers)
    top = min(TOP_N_PAPERS, n)
    return textwrap.dedent(f"""
        You are an expert in metabolic physiology and inter-organ communication.

        Below are {n} papers retrieved for the {organ1}–{organ2} connection.
        Select the {top} papers most directly relevant to the metabolic or hormonal
        interaction between {organ1} and {organ2}.

        Reply with ONLY a comma-separated list of the paper numbers you selected,
        e.g.: 3, 7, 12, 18, 22
        Do not write anything else.

        Papers:
        {block}

        Selected paper numbers:
    """).strip()


def _parse_selection(text: str, n_papers: int) -> list[int]:
    """Parse the LLM's comma-separated index reply into a list of valid 0-based indices."""
    indices = []
    for token in re.split(r"[,\s]+", text.strip()):
        token = token.strip().rstrip(".")
        if token.isdigit():
            idx = int(token) - 1   # convert 1-based to 0-based
            if 0 <= idx < n_papers:
                indices.append(idx)
    seen = set()
    return [i for i in indices if not (i in seen or seen.add(i))]


def _pad_selection(indices: list[int], pool_size: int, top_n: int) -> list[int]:
    """
    Ensure a paper-selection index list has exactly min(top_n, pool_size)
    entries, padding with the next unused pool papers (in original order)
    if the LLM's reply parsed to fewer valid indices than that. Without
    this, a partial/malformed selection reply silently means the summary
    ends up grounded in fewer sources than intended, even when more papers
    were available in the pool.
    """
    target = min(top_n, pool_size)
    indices = indices[:target]
    if len(indices) < target:
        used = set(indices)
        for i in range(pool_size):
            if len(indices) >= target:
                break
            if i not in used:
                indices.append(i)
                used.add(i)
    return indices


def _build_summary_prompt(organ1: str, organ2: str, papers: list[dict]) -> str:
    """Step 2 — ask the LLM to write a cited summary from the selected papers."""
    block = _paper_block(papers)
    n = len(papers)
    return textwrap.dedent(f"""
        You are an expert in metabolic physiology and inter-organ communication.

        Write a concise scientific summary (3–4 sentences) explaining the metabolic
        and hormonal basis of the {organ1}–{organ2} axis based solely on the papers below.
        Cite papers inline using their number in square brackets, e.g. [1] or [2,3].
        You MUST cite every one of the {n} papers below at least once — do not
        leave any of them out of your citations.
        Focus on the most important signalling molecules and mechanisms.
        Write flowing prose — no bullet points, no headings, no meta-commentary.
        Output only the scientific summary itself.

        Papers:
        {block}

        Summary:
    """).strip()


def _build_organ_selection_prompt(organ: str, papers: list[dict]) -> str:
    """Step 1 (single-organ variant) — ask the LLM to pick the most relevant
    paper indices for one organ's own metabolic literature."""
    block = _paper_block(papers)
    n = len(papers)
    top = min(TOP_N_PAPERS, n)
    return textwrap.dedent(f"""
        You are an expert in metabolic physiology.

        Below are {n} papers retrieved for {organ}'s role in metabolism.
        Select the {top} papers most directly relevant to {organ}'s own metabolic
        or hormonal function (not its interaction with any other specific organ).

        Reply with ONLY a comma-separated list of the paper numbers you selected,
        e.g.: 3, 7, 12, 18, 22
        Do not write anything else.

        Papers:
        {block}

        Selected paper numbers:
    """).strip()


def _build_organ_summary_prompt(organ: str, papers: list[dict]) -> str:
    """Step 2 (single-organ variant) — ask the LLM to write a cited summary
    of one organ's metabolic role from the selected papers."""
    block = _paper_block(papers)
    n = len(papers)
    return textwrap.dedent(f"""
        You are an expert in metabolic physiology.

        Write a concise scientific summary (3–4 sentences) explaining {organ}'s
        role in metabolism based solely on the papers below.
        Cite papers inline using their number in square brackets, e.g. [1] or [2,3].
        You MUST cite every one of the {n} papers below at least once — do not
        leave any of them out of your citations.
        Focus on the most important signalling molecules and mechanisms.
        Write flowing prose — no bullet points, no headings, no meta-commentary.
        Output only the scientific summary itself.

        Papers:
        {block}

        Summary:
    """).strip()


# ---------------------------------------------------------------------------
# Quality gate for generated summaries
# ---------------------------------------------------------------------------

MAX_SUMMARY_ATTEMPTS = 3
_REFUSAL_RE = re.compile(
    r"\b(I cannot|I can't|I can not|I am unable|I'm unable|unable to (provide|write|summari[sz]e)|"
    r"as an AI|I apologi[sz]e|please provide|if you provide|can I help you|I do not have|"
    r"I don't have|do(es)? not appear to be related|not enough information)\b", re.IGNORECASE)


def _description_problem(text: str, n_papers: int) -> "str | None":
    """Why a generated summary must be rejected, or None if it is fine."""
    if not text.strip():
        return "empty reply"
    if _REFUSAL_RE.search(text):
        return "refusal / meta text"
    cited = [int(n) for grp in re.findall(r"\[([\d,\s\-–]+)\]", text) for n in re.findall(r"\d+", grp)]
    if not cited:
        return "no citations"
    if min(cited) < 1 or max(cited) > n_papers:
        return f"citation [{max(cited)}] outside the {n_papers} sources"
    return None


# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def _check_ollama(model: str) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        available = [m["name"] for m in resp.json().get("models", [])]
        # Allow partial match (e.g. "llama3.2" matches "llama3.2:latest")
        return any(model in m for m in available)
    except Exception:
        return False


def _call_ollama(prompt: str, model: str) -> str:
    """Send a chat request to Ollama and return the response text."""
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "options":  {"temperature": 0.3, "num_predict": 512},
    }
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            resp = _requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            msg = resp.json()["message"]
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


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_llm_descriptions(
    organ_pairs: list[tuple[str, str]],
    literature_results: dict,
    output_path: "str | Path" = DEFAULT_OUTPUT,
    model: str = DEFAULT_MODEL,
    resume: bool = True,
    reset: bool = False,
) -> dict:
    """
    Generate literature-grounded LLM summaries for all organ pairs.

    Parameters
    ----------
    organ_pairs        : sorted list of (organ1, organ2) tuples
    literature_results : raw literature JSON (keyed "organ1|organ2")
    output_path        : where to save/load llm_descriptions.json
    model              : Ollama model name
    resume             : skip pairs already in the cache file
    reset              : delete cache and regenerate everything
    """
    output_path = Path(output_path)

    if _is_claude(model):       # Claude API: 10 sources, citations verified against the papers
        from Literature_Search.claude_descriptions import run_claude_descriptions
        items = []
        for o1, o2 in organ_pairs:
            key, sym = f"{o1}|{o2}", f"{o2}|{o1}"
            lit = literature_results.get(key) or literature_results.get(sym) or {}
            items.append({"key": key, "alt": sym, "subject": (o1, o2), "papers": lit.get("papers", [])})
        return run_claude_descriptions(items, "pair", output_path, model, resume=resume, reset=reset)

    # Load existing cache
    descriptions: dict = {}
    if reset and output_path.exists():
        output_path.unlink()
        print(f"[i] Cache deleted: {output_path.name}")
    elif resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            descriptions = json.load(f)
        # Count only entries that were generated by this model and have content
        usable = sum(
            1 for v in descriptions.values()
            if v.get("description") and v.get("model") == model
        )
        stale = len(descriptions) - usable
        print(f"[i] Cache: {usable} usable entries for model '{model}'"
              + (f", {stale} stale/wrong-model (will regenerate)" if stale else "") + ".")

    # Verify Ollama is running and model is available
    if not _check_ollama(model):
        print(f"[!] Ollama not reachable or model '{model}' not found.")
        print(f"    Start Ollama and run:  ollama pull {model}")
        print(f"    Then re-run this same command.")
        return descriptions

    print(f"[i] Using model: {model}")

    total = len(organ_pairs)
    for idx, (o1, o2) in enumerate(organ_pairs):
        key = f"{o1}|{o2}"
        sym = f"{o2}|{o1}"

        existing = descriptions.get(key) or descriptions.get(sym)
        if existing and existing.get("description") and existing.get("model") == model:
            print(f"  [{idx+1}/{total}] Cached: {o1} <-> {o2}")
            continue

        # Retrieve papers for this pair
        lit = (
            literature_results.get(key)
            or literature_results.get(sym)
            or {}
        )
        papers = lit.get("papers", [])
        n_papers = lit.get("n_papers_found", len(papers))

        print(f"  [{idx+1}/{total}] {o1} <-> {o2}  ({n_papers} papers) … ",
              end="", flush=True)

        if n_papers == 0:
            print("skipped (no papers)")
            descriptions[key] = {
                "description":   "",
                "generated_at":  datetime.now().isoformat(),
                "model":         model,
                "n_papers_used": 0,
            }
        else:
            try:
                # Step 1 — sample up to SAMPLE_POOL papers
                pool = papers[:SAMPLE_POOL] if len(papers) > SAMPLE_POOL else papers[:]
                if len(papers) > SAMPLE_POOL:
                    pool = random.sample(papers, SAMPLE_POOL)

                # Step 2 — ask LLM to pick TOP_N_PAPERS most relevant ones
                selected_papers = pool
                if len(pool) > TOP_N_PAPERS:
                    sel_prompt = _build_selection_prompt(o1, o2, pool)
                    sel_text   = _call_ollama(sel_prompt, model)
                    indices    = _parse_selection(sel_text, len(pool))
                    indices    = _pad_selection(indices, len(pool), TOP_N_PAPERS)
                    selected_papers = [pool[i] for i in indices]

                # Step 3 — write cited summary from selected papers. Quality gate:
                # a reply that is a refusal / meta-talk, has no citations or cites a
                # paper number outside the source list is regenerated (up to
                # MAX_SUMMARY_ATTEMPTS times); if it never passes, the description
                # is left empty instead of showing broken text.
                sum_prompt = _build_summary_prompt(o1, o2, selected_papers)
                text, problem = "", None
                for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
                    text = _call_ollama(sum_prompt, model)
                    problem = _description_problem(text, len(selected_papers))
                    if problem is None:
                        break
                    print(f"[{problem}; attempt {attempt}/{MAX_SUMMARY_ATTEMPTS}] ", end="", flush=True)
                if problem is not None:
                    descriptions[key] = {
                        "description":   "",
                        "generated_at":  datetime.now().isoformat(),
                        "model":         model,
                        "n_papers_used": 0,
                        "error":         f"rejected after {MAX_SUMMARY_ATTEMPTS} attempts: {problem}",
                        "rejected_text": text,
                    }
                    print("rejected -> no description")
                else:
                    descriptions[key] = {
                        "description":   text,
                        "generated_at":  datetime.now().isoformat(),
                        "model":         model,
                        "n_papers_used": len(selected_papers),
                        "papers":        selected_papers,
                    }
                    print(f"ok ({len(text.split())} words, {len(selected_papers)} papers)")
            except Exception as exc:
                print(f"ERROR: {exc}")
                descriptions[key] = {
                    "description":   "",
                    "generated_at":  datetime.now().isoformat(),
                    "model":         model,
                    "n_papers_used": 0,
                    "error":         str(exc),
                }

        # Save after every pair so progress is never lost
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

    done = sum(1 for v in descriptions.values() if v.get("description"))
    print(f"\n[ok] {done}/{len(descriptions)} pairs have summaries. Saved to: {output_path}")
    return descriptions


def generate_llm_organ_descriptions(
    organs: list[str],
    search_results: dict,
    output_path: "str | Path",
    model: str = DEFAULT_MODEL,
    resume: bool = True,
    reset: bool = False,
) -> dict:
    """
    Generate literature-grounded LLM summaries for individual organs (not
    pairs) — one paragraph per organ describing its own metabolic role,
    grounded in that organ's own paper pool.

    Parameters
    ----------
    organs         : organ names to summarise
    search_results : {organ: {papers, n_found, ...}} — an organ-keyed search
                      cache, e.g. loaded straight from a
                      search_results_{condition}.json produced by a
                      per-organ query (MeSH_ORGAN OR aliases) AND
                      METABOLIC_FILTER AND CONDITION_FILTER. No new PubMed
                      query is issued — this only summarises papers already
                      fetched for edge definition.
    output_path    : where to save/load the organ-descriptions JSON
    model          : Ollama model name
    resume         : skip organs already in the cache file
    reset          : delete cache and regenerate everything
    """
    output_path = Path(output_path)

    if _is_claude(model):       # Claude API: 10 sources, citations verified against the papers
        from Literature_Search.claude_descriptions import run_claude_descriptions
        items = [{"key": organ, "alt": None, "subject": (organ,),
                  "papers": search_results.get(organ, {}).get("papers", [])} for organ in organs]
        return run_claude_descriptions(items, "organ", output_path, model, resume=resume, reset=reset)

    descriptions: dict = {}
    if reset and output_path.exists():
        output_path.unlink()
        print(f"[i] Cache deleted: {output_path.name}")
    elif resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            descriptions = json.load(f)
        usable = sum(
            1 for v in descriptions.values()
            if v.get("description") and v.get("model") == model
        )
        stale = len(descriptions) - usable
        print(f"[i] Cache: {usable} usable entries for model '{model}'"
              + (f", {stale} stale/wrong-model (will regenerate)" if stale else "") + ".")

    if not _check_ollama(model):
        print(f"[!] Ollama not reachable or model '{model}' not found.")
        print(f"    Start Ollama and run:  ollama pull {model}")
        return descriptions

    print(f"[i] Using model: {model}")

    total = len(organs)
    for idx, organ in enumerate(organs):
        existing = descriptions.get(organ)
        if existing and existing.get("description") and existing.get("model") == model:
            print(f"  [{idx+1}/{total}] Cached: {organ}")
            continue

        data     = search_results.get(organ, {})
        papers   = data.get("papers", [])
        n_papers = data.get("n_found", len(papers))

        print(f"  [{idx+1}/{total}] {organ}  ({n_papers} papers) … ", end="", flush=True)

        if n_papers == 0:
            print("skipped (no papers)")
            descriptions[organ] = {
                "description":   "",
                "generated_at":  datetime.now().isoformat(),
                "model":         model,
                "n_papers_used": 0,
            }
        else:
            try:
                pool = papers[:SAMPLE_POOL] if len(papers) > SAMPLE_POOL else papers[:]
                if len(papers) > SAMPLE_POOL:
                    pool = random.sample(papers, SAMPLE_POOL)

                selected_papers = pool
                if len(pool) > TOP_N_PAPERS:
                    sel_prompt = _build_organ_selection_prompt(organ, pool)
                    sel_text   = _call_ollama(sel_prompt, model)
                    indices    = _parse_selection(sel_text, len(pool))
                    indices    = _pad_selection(indices, len(pool), TOP_N_PAPERS)
                    selected_papers = [pool[i] for i in indices]

                sum_prompt = _build_organ_summary_prompt(organ, selected_papers)
                text = _call_ollama(sum_prompt, model)
                descriptions[organ] = {
                    "description":   text,
                    "generated_at":  datetime.now().isoformat(),
                    "model":         model,
                    "n_papers_used": len(selected_papers),
                    "papers":        selected_papers,
                }
                print(f"ok ({len(text.split())} words, {len(selected_papers)} papers)")
            except Exception as exc:
                print(f"ERROR: {exc}")
                descriptions[organ] = {
                    "description":   "",
                    "generated_at":  datetime.now().isoformat(),
                    "model":         model,
                    "n_papers_used": 0,
                    "error":         str(exc),
                }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

    done = sum(1 for v in descriptions.values() if v.get("description"))
    print(f"\n[ok] {done}/{len(descriptions)} organs have summaries. Saved to: {output_path}")
    return descriptions


# ---------------------------------------------------------------------------
# Helpers used by other modules
# ---------------------------------------------------------------------------

def load_llm_descriptions(path: "str | Path") -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_description(descriptions: dict, organ1: str, organ2: str) -> str:
    """Return the LLM summary text for an organ pair, or empty string."""
    entry = (
        descriptions.get(f"{organ1}|{organ2}")
        or descriptions.get(f"{organ2}|{organ1}")
        or {}
    )
    return entry.get("description", "")


def get_organ_description(descriptions: dict, organ: str) -> str:
    """Return the LLM summary text for a single organ, or empty string."""
    return descriptions.get(organ, {}).get("description", "")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _load_edge_metadata_from_csv(csv_path) -> dict:
    """Edge texts of a symmetric organ x organ CSV (upper triangle): {(organ1, organ2): text}, both directions.
    Fallback edge source of main() when neither --edge-filter nor healthy_cohort_connections.csv exists
    (this function replaces the import from the Data_Loader module of the predecessor project, so the atlas is self-contained)."""
    import pandas as pd
    csv_path = Path(csv_path)
    metadata: dict = {}
    if not csv_path.exists():
        print("[i] No edge metadata CSV found.")
        return metadata
    df = pd.read_csv(csv_path, index_col=0, encoding="utf-8").fillna("")
    df = df.astype(str).apply(lambda col: col.str.strip('"'))
    nodes = list(df.index)
    for i, n1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            text = df.loc[n1, nodes[j]]
            if str(text).strip() == "":
                continue
            metadata[(n1, nodes[j])] = text
            metadata[(nodes[j], n1)] = text
    print(f"[ok] Loaded metadata for {len(metadata) // 2} edges")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate literature-grounded LLM summaries for organ-organ connections."
    )
    parser.add_argument("--reset", action="store_true",
                        help="Delete cache and regenerate all descriptions.")
    parser.add_argument("--model", default=None,
                        help="Ollama model, or a Claude API model such as claude-haiku-5-5 "
                             "(default: DESCRIPTION_LLM_MODEL from config.py).")
    parser.add_argument("--edge-filter", default=None,
                        help="Path to a 0/1 adjacency matrix CSV (e.g. healthy_cohort_connections.csv). "
                             "When provided, pairs are read from this file instead of connection_data.csv.")
    parser.add_argument("--literature", default=None,
                        help="Path to the literature results JSON to summarise. "
                             "Defaults to metabolic_data/literature_results.json.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Where to save the descriptions JSON (default: {DEFAULT_OUTPUT}).")
    args = parser.parse_args()

    from Literature_Search.pubmed_search import load_literature_results

    # Resolve edge source — explicit flag > healthy_cohort_connections.csv > connection_data.csv
    _cohort_csv     = HERE / "reference_network_only_metabolic" / "healthy_cohort_connections.csv"
    _connection_csv = HERE / "metabolic_data" / "connection_data.csv"

    edge_filter = Path(args.edge_filter) if args.edge_filter else (
        _cohort_csv if _cohort_csv.exists() else None
    )

    if edge_filter:
        import csv as _csv
        pairs_set: set[tuple[str, str]] = set()
        with open(edge_filter, encoding="utf-8") as f:
            reader = _csv.reader(f)
            header = next(reader)
            col_organs = header[1:]
            for row in reader:
                if not row:
                    continue
                row_organ = row[0].strip()
                for col_idx, val in enumerate(row[1:]):
                    val = val.strip()
                    if val == "1" and col_idx < len(col_organs):
                        col_organ = col_organs[col_idx].strip()
                        if row_organ and col_organ and row_organ != col_organ:
                            pairs_set.add((min(row_organ, col_organ),
                                           max(row_organ, col_organ)))
        pairs = sorted(pairs_set)
        print(f"[i] Edge source: {edge_filter.name} → {len(pairs)} pairs")
    else:
        edge_metadata = _load_edge_metadata_from_csv(_connection_csv)
        pairs = sorted({
            (o1, o2) if o1 < o2 else (o2, o1)
            for (o1, o2) in edge_metadata
        })
        print(f"[i] Edge source: {_connection_csv.name} → {len(pairs)} pairs")

    # Resolve literature results — explicit flag > network-specific > global
    _network_lit = HERE / "reference_network_only_metabolic" / "metabolic_literature_results.json"
    _global_lit  = HERE / "metabolic_data" / "literature_results.json"
    literature_path = Path(args.literature) if args.literature else (
        _network_lit if _network_lit.exists() else _global_lit
    )

    # Resolve output — explicit flag > network-specific file
    _network_out = HERE / "reference_network_only_metabolic" / "metabolic_llm_descriptions.json"
    output_path = Path(args.output) if args.output != str(DEFAULT_OUTPUT) else (
        _network_out if edge_filter == _cohort_csv else DEFAULT_OUTPUT
    )

    lit_results = load_literature_results(literature_path)
    print(f"[i] Literature: {literature_path.name}")
    print(f"[i] Output:     {output_path}")

    print(f"[i] {len(pairs)} organ-organ pairs to summarise.")

    generate_llm_descriptions(
        organ_pairs=pairs,
        literature_results=lit_results,
        output_path=output_path,
        model=args.model or configured_description_model(),
        resume=not args.reset,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
