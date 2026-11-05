"""
PubMed search + LLM summary for individual organs.

For each organ, fetches recent papers on its metabolic function then uses a
local Ollama model to write a 5-sentence description focused on consumed
substrates and hormonal regulation.

Results are saved incrementally so the process can be resumed.

Usage (standalone):
    uv run python Literature_Search/organ_descriptions.py
    uv run python Literature_Search/organ_descriptions.py --reset
    uv run python Literature_Search/organ_descriptions.py --model llama3.2
"""

import json
import sys
import time
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests as _requests

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

# Windows console/redirected-output encoding defaults to cp1252, which can't
# encode the arrows/ellipses used in progress prints below — force UTF-8 so
# the run doesn't crash mid-way through (e.g. when stdout is piped to a file).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SEARCH_OUTPUT = HERE / "metabolic_data" / "organ_search_results.json"
DEFAULT_LLM_OUTPUT    = HERE / "metabolic_data" / "organ_descriptions.json"
DEFAULT_MODEL         = "llama3.2"
MAX_PAPERS_PROMPT     = 6
MAX_ABSTRACT_CHARS    = 400
YEARS_BACK            = 5
MAX_RESULTS           = 50
DELAY                 = 0.4

ORGAN_METABOLIC_QUERY = (
    "(metabolism[Title/Abstract] OR metabolic[Title/Abstract] "
    "OR catabolism[Title/Abstract] OR anabolism[Title/Abstract] "
    "OR biosynthesis[Title/Abstract] OR metabolite[Title/Abstract] "
    "OR flux[Title/Abstract] OR substrate[Title/Abstract] "
    "OR nutrient[Title/Abstract] OR fuel[Title/Abstract] "
    "OR glucose[Title/Abstract] OR \"glucose uptake\"[Title/Abstract] "
    "OR glycolysis[Title/Abstract] OR gluconeogenesis[Title/Abstract] "
    "OR energy[Title/Abstract] OR bioenergetics[Title/Abstract] "
    "OR mitochondria[Title/Abstract] OR mitochondrial[Title/Abstract] "
    "OR thermogenesis[Title/Abstract] OR \"Krebs cycle\"[Title/Abstract] "
    "OR \"electron transport\"[Title/Abstract] OR oxidation[Title/Abstract])"
)


# ---------------------------------------------------------------------------
# PubMed search for a single organ
# ---------------------------------------------------------------------------

def _search_organ(organ: str, years_back: int, max_results: int, delay: float) -> list[dict]:
    """Search PubMed for metabolic function papers for a single organ."""
    from Literature_Search.pubmed_search import (
        ORGAN_ALIASES, ORGAN_MESH, _ncbi_get, fetch_abstracts,
    )

    aliases  = ORGAN_ALIASES.get(organ, [organ])
    mesh     = ORGAN_MESH.get(organ, f'"{organ}"[MeSH Terms]')
    alias_q  = "(" + " OR ".join(f'"{a}"[Title/Abstract]' for a in aliases) + ")"
    query    = f"({mesh} OR {alias_q}) AND {ORGAN_METABOLIC_QUERY}"

    min_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y/%m/%d")
    params = {
        "db":       "pubmed",
        "term":     query,
        "retmax":   max_results,
        "retmode":  "json",
        "mindate":  min_date,
        "datetype": "pdat",
    }
    resp = _ncbi_get("esearch.fcgi", params)
    if resp is None:
        return []
    pmids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []
    time.sleep(delay)
    return fetch_abstracts(pmids, delay=delay)


def run_organ_search(
    organs: list[str],
    output_path: "str | Path" = DEFAULT_SEARCH_OUTPUT,
    years_back: int = YEARS_BACK,
    max_results: int = MAX_RESULTS,
    delay: float = DELAY,
    resume: bool = True,
    reset: bool = False,
) -> dict:
    """Search PubMed for each organ and save results incrementally."""
    output_path = Path(output_path)

    results: dict = {}
    if reset and output_path.exists():
        output_path.unlink()
        print(f"[i] Cache deleted: {output_path.name}")
    elif resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[i] Resuming organ search: {len(results)} already done.")

    total = len(organs)
    for idx, organ in enumerate(organs):
        if organ in results:
            n = results[organ].get("n_papers", 0)
            print(f"  [{idx+1}/{total}] Cached ({n} papers): {organ}")
            continue

        print(f"  [{idx+1}/{total}] Searching: {organ} … ", end="", flush=True)
        papers = _search_organ(organ, years_back, max_results, delay)
        print(f"{len(papers)} papers")

        results[organ] = {
            "organ":       organ,
            "n_papers":    len(papers),
            "papers":      papers,
            "search_date": datetime.now().isoformat(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] Organ search done. Saved to: {output_path}")
    return results


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

def _build_organ_prompt(organ: str, papers: list[dict]) -> str:
    if not papers:
        paper_block = "No papers available."
    else:
        lines = []
        for i, p in enumerate(papers[:MAX_PAPERS_PROMPT], 1):
            abstract = p.get("abstract", "").strip()
            if len(abstract) > MAX_ABSTRACT_CHARS:
                abstract = abstract[:MAX_ABSTRACT_CHARS].rsplit(" ", 1)[0] + "…"
            lines.append(
                f"[{i}] PMID {p.get('pmid','?')} ({p.get('year','')})\n"
                f"    Title: {p.get('title','').strip()}\n"
                f"    Abstract: {abstract}"
            )
        paper_block = "\n\n".join(lines)

    return (
        f"You are an expert in metabolic physiology.\n\n"
        f"Task: Write exactly 5 sentences describing the metabolic function of the "
        f"{organ}. Focus on: (1) which metabolic substrates it primarily consumes "
        f"or produces, and (2) which key hormones regulate its metabolism. "
        f"Use the papers below as references where relevant and cite them with their "
        f"number in square brackets, e.g. [1] or [2,3]. You may also draw on your "
        f"general knowledge of physiology. Write flowing prose — no bullet points, "
        f"no headings, no refusals.\n\n"
        f"Papers:\n{paper_block}\n\nDescription:"
    )


_REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i won't", "i will not", "i don't", "i do not",
    "cannot fulfill", "cannot provide", "can't fulfill",
    "not able to", "unable to fulfill", "unable to provide",
    "i apologize", "i'm sorry", "i must decline",
    "as an ai", "as a language model",
]

def _is_refusal(text: str) -> bool:
    """Return True if the model's response looks like a refusal."""
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def _fallback_prompt(organ: str) -> str:
    """Minimal prompt used when the main prompt triggers a refusal."""
    return (
        f"You are an expert in metabolic physiology. "
        f"Write exactly 5 sentences about the metabolic function of the {organ}. "
        f"Describe what metabolic substrates it consumes or produces, "
        f"and which hormones regulate its metabolism. "
        f"Write in flowing scientific prose."
    )


def generate_organ_llm_descriptions(
    organs: list[str],
    search_results: dict,
    output_path: "str | Path" = DEFAULT_LLM_OUTPUT,
    model: str = DEFAULT_MODEL,
    resume: bool = True,
    reset: bool = False,
) -> dict:
    """Generate LLM descriptions for each organ."""
    from Literature_Search.llm_descriptions import _check_ollama, _call_ollama, _is_claude

    output_path = Path(output_path)

    if _is_claude(model):       # Claude API: grounded in up to 10 papers, citations verified
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
        usable = sum(1 for v in descriptions.values()
                     if v.get("description") and v.get("model") == model)
        stale  = len(descriptions) - usable
        print(f"[i] Cache: {usable} usable entries for model '{model}'"
              + (f", {stale} stale (will regenerate)" if stale else "") + ".")

    if not _check_ollama(model):
        print(f"[!] Ollama not reachable or model '{model}' not found.")
        print(f"    Start Ollama and run:  ollama pull {model}")
        return descriptions

    print(f"[i] Using model: {model}")
    total = len(organs)

    for idx, organ in enumerate(organs):
        existing = descriptions.get(organ, {})
        if existing.get("description") and existing.get("model") == model:
            print(f"  [{idx+1}/{total}] Cached: {organ}")
            continue

        organ_data = search_results.get(organ, {})
        papers     = organ_data.get("papers", [])
        n_papers   = organ_data.get("n_papers", 0)

        print(f"  [{idx+1}/{total}] {organ}  ({n_papers} papers) … ", end="", flush=True)

        prompt = _build_organ_prompt(organ, papers)
        try:
            text = _call_ollama(prompt, model)
            if _is_refusal(text):
                print(f"refusal detected, retrying with fallback … ", end="", flush=True)
                text = _call_ollama(_fallback_prompt(organ), model)
            descriptions[organ] = {
                "description":   text,
                "papers":        papers[:MAX_PAPERS_PROMPT],
                "generated_at":  datetime.now().isoformat(),
                "model":         model,
                "n_papers_used": min(len(papers), MAX_PAPERS_PROMPT),
            }
            print(f"ok ({len(text.split())} words)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            descriptions[organ] = {
                "description":   "",
                "papers":        [],
                "generated_at":  datetime.now().isoformat(),
                "model":         model,
                "n_papers_used": 0,
                "error":         str(exc),
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

    done = sum(1 for v in descriptions.values() if v.get("description"))
    print(f"\n[ok] {done}/{len(descriptions)} organs have descriptions. Saved to: {output_path}")
    return descriptions


# ---------------------------------------------------------------------------
# Helpers used by other modules
# ---------------------------------------------------------------------------

def load_organ_descriptions(path: "str | Path") -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search PubMed and generate LLM descriptions for each organ."
    )
    parser.add_argument("--reset",      action="store_true",
                        help="Delete all caches and start fresh.")
    parser.add_argument("--reset-search", action="store_true",
                        help="Delete only the search cache.")
    parser.add_argument("--reset-llm",  action="store_true",
                        help="Delete only the LLM description cache.")
    parser.add_argument("--model",      default=None,
                        help="Ollama model, or a Claude API model such as claude-haiku-5-5 "
                             "(default: DESCRIPTION_LLM_MODEL from config.py).")
    args = parser.parse_args()
    from Literature_Search.llm_descriptions import configured_description_model
    model = args.model or configured_description_model()

    import csv as _csv
    _cohort_csv = HERE / "reference_network_only_metabolic" / "all_organ_connections.csv"
    if not _cohort_csv.exists():
        print(f"[!] Edge filter not found: {_cohort_csv}")
        sys.exit(1)
    _organ_set: set[str] = set()
    with open(_cohort_csv, encoding="utf-8") as _f:
        _reader = _csv.reader(_f)
        _header = next(_reader)
        _col_organs = [c.strip() for c in _header[1:]]
        _organ_set.update(_col_organs)
        for _row in _reader:
            if _row:
                _organ_set.add(_row[0].strip())
    _organ_set.discard("")
    organs = sorted(_organ_set)
    print(f"[i] Edge filter: {_cohort_csv.name} → {len(organs)} organs to process.")

    search_results = run_organ_search(
        organs=organs,
        output_path=DEFAULT_SEARCH_OUTPUT,
        resume=not (args.reset or args.reset_search),
        reset=(args.reset or args.reset_search),
    )

    generate_organ_llm_descriptions(
        organs=organs,
        search_results=search_results,
        output_path=DEFAULT_LLM_OUTPUT,
        model=model,
        resume=not (args.reset or args.reset_llm),
        reset=(args.reset or args.reset_llm),
    )


if __name__ == "__main__":
    main()
