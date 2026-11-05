"""
Complete the organ pairs whose PubMed search hit the 10,000-record limit of esearch (found by
pubmed_final_query_check.py), under the FINAL query and the PubMed state of the original search date.

Phase 1 (default, changes nothing in the results):
  * complete PMID list of every capped pair under the final query (split by publication year so that no slice reaches the
    10,000 limit; the sum of the slices must equal the total number of hits),
  * download title + abstract of all of them (efetch, same function as the pipeline),
  * apply the pipeline's same-sentence filter with the final rules (same call as run_search),
  * check: every paper that is already kept for the pair is kept again (anything else stops the script),
  * write the staging file reference_network_only_metabolic/capped_pairs_completion_cache.json.

Phase 2 (--apply, reads the staging file, no PubMed calls):
  * snapshot of the paper lists of all views before the change (needed to find stale descriptions/types afterwards),
  * backup of the results file (metabolic_literature_results.json.before_capped_pair_completion.<time>.bak),
  * capped pairs: the new kept papers are APPENDED to the existing list (existing order is kept), key players recomputed,
  * all pairs: pubmed_query = final query, n_papers_found_raw = number of hits of the final query (the values of the
    original fetch are kept in pubmed_query_fetched / n_papers_found_raw_fetched),
  * pubmed_step1_final.json with the step-1 numbers (hits, unique papers) for the funnel figure.

Usage:   python complete_capped_pairs.py            # phase 1
         python complete_capped_pairs.py --apply    # phase 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_metabolic_lit_search as rl  # noqa: E402
import pubmed_final_query_check as chk  # noqa: E402
from Literature_Search import llm_condition_filter as cf  # noqa: E402
from Literature_Search.pubmed_search import ORGAN_ALIASES, extract_key_players  # noqa: E402

D = HERE / "reference_network_only_metabolic"
CACHE = D / "capped_pairs_completion_cache.json"
SNAPSHOT = D / "view_snapshot_before_capped_pair_completion.json"
STEP1 = D / "pubmed_step1_final.json"


def year_slices() -> list[tuple[int, int]]:
    return [(1000, 1999)] + [(y, y) for y in range(2000, 2028)] + [(2028, 3000)]


def complete_pmids(query: str, hits: int, known: list[str]) -> list[str]:
    """All PMIDs of the query (as of the search date). Uses the list of the check run when it was not truncated."""
    if hits <= chk.RETMAX and len(known) == hits:
        return list(known)
    ids, total = [], 0
    for a, b in year_slices():
        slice_q = f'{query} AND ("{a}/01/01"[PDAT] : "{b}/12/31"[PDAT])'
        n, got = chk.esearch(slice_q)
        if n > chk.RETMAX:
            raise SystemExit(f"[!] slice {a}-{b} has {n} hits (> {chk.RETMAX}); split it further")
        total += n
        ids.extend(got)
    uniq = list(dict.fromkeys(ids))
    # the slice counts may add up to more than `hits`: a record with an electronic and a print date matches two years
    if len(uniq) != hits:
        raise SystemExit(f"[!] slices give {len(uniq)} unique PMIDs ({total} slice hits), the query has {hits}")
    print(f"      year slices: {total} slice hits, {len(uniq)} unique PMIDs = all {hits} hits")
    return uniq


def fetch_all(pmids: list[str]) -> tuple[list[dict], list[str]]:
    papers: dict[str, dict] = {}
    for i in range(0, len(pmids), 1000):
        for p in rl.fetch_abstracts(pmids[i:i + 1000], delay=chk.DELAY):
            papers[p["pmid"]] = p
        print(f"      fetched {min(i + 1000, len(pmids))}/{len(pmids)}", flush=True)
    missing = [p for p in pmids if p not in papers]
    if missing:
        print(f"      {len(missing)} PMIDs not returned, retrying once", flush=True)
        for p in rl.fetch_abstracts(missing, delay=chk.DELAY):
            papers[p["pmid"]] = p
        missing = [p for p in pmids if p not in papers]
    if missing:
        print(f"      [!] still missing after the retry: {len(missing)} (first: {missing[:5]})")
    return [papers[p] for p in pmids if p in papers], missing


def phase1() -> None:
    results = json.loads(rl.OUTPUT_JSON.read_text(encoding="utf-8"))
    check = json.loads(chk.OUT_JSON.read_text(encoding="utf-8"))
    rows = {r["pair"]: r for r in check["pairs"]}
    capped = [k for k, r in rows.items() if r["capped at 10,000"]]
    print(f"[i] capped pairs: {capped}")
    crosstalk_patterns = rl._compile_patterns(rl.CROSSTALK_KEYWORDS)
    out = {"search date": chk.SEARCH_DATE, "created": datetime.now().isoformat(timespec="seconds"), "pairs": {}}
    for key in capped:
        e = results[key]
        o1, o2 = e["organ1"], e["organ2"]
        query = rl.build_query(o1, o2)
        hits = rows[key]["final query hits"]
        print(f"\n[{key}] final query: {hits} hits")
        pmids = complete_pmids(query, hits, check["final_query_pmids"][key])
        print(f"      complete PMID list: {len(pmids)}")
        papers, missing = fetch_all(pmids)
        kept = rl._filter_same_sentence_crosstalk(
            papers, rl._compile_patterns(ORGAN_ALIASES.get(o1, [o1])), rl._compile_patterns(ORGAN_ALIASES.get(o2, [o2])),
            crosstalk_patterns, organ1_name=o1, organ2_name=o2, all_organ_patterns=rl.ALL_ORGAN_PATTERNS)
        old_ids = [str(p["pmid"]) for p in e["papers"]]
        new_ids = {str(p["pmid"]) for p in kept}
        lost = [p for p in old_ids if p not in new_ids]
        added = [p for p in kept if str(p["pmid"]) not in set(old_ids)]
        print(f"      kept now {len(kept)} | kept before {len(old_ids)} | already kept but not kept again: {len(lost)} | new: {len(added)}")
        if lost:
            raise SystemExit(f"[!] {len(lost)} papers of the existing list are not kept again (first: {lost[:5]}); stopping")
        out["pairs"][key] = {"query": query, "hits": hits, "pmids": pmids, "fetched": len(papers), "missing": missing,
                             "kept_total": len(kept), "new_papers": added}
    CACHE.write_text(json.dumps(out), encoding="utf-8")
    print(f"\n[ok] staging file written: {CACHE.name} (results file not changed)")


def view_lists(results: dict, labels: dict) -> dict:
    filters = {c["slug"]: cf.LLMConditionFilter(c["slug"], labels) for c in cf.CONDITIONS}
    out = {"all": {}, **{s: {} for s in filters}}
    for e in results.values():
        o1, o2 = e.get("organ1", ""), e.get("organ2", "")
        canon = f"{min(o1, o2)}|{max(o1, o2)}"
        papers = e.get("papers", [])
        out["all"][canon] = [str(p["pmid"]) for p in papers]
        for s, f in filters.items():
            out[s][canon] = [str(p["pmid"]) for p in papers if f.accepts(p)]
    return out


def phase2() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    check = json.loads(chk.OUT_JSON.read_text(encoding="utf-8"))
    rows = {r["pair"]: r for r in check["pairs"]}
    results = json.loads(rl.OUTPUT_JSON.read_text(encoding="utf-8"))
    labels = cf.load_condition_labels(rl.CONDITION_LABELS, rl.CONDITION_LLM_MODEL)

    SNAPSHOT.write_text(json.dumps(view_lists(results, labels)), encoding="utf-8")
    print(f"[i] snapshot of all views before the change: {SNAPSHOT.name}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = rl.OUTPUT_JSON.with_suffix(f".json.before_capped_pair_completion.{stamp}.bak")
    shutil.copy2(rl.OUTPUT_JSON, backup)
    print(f"[i] backup of the results: {backup.name}")

    new_pmids = set()
    for key, c in cache["pairs"].items():
        e = results[key]
        have = {str(p["pmid"]) for p in e["papers"]}
        added = [p for p in c["new_papers"] if str(p["pmid"]) not in have]
        e["papers"] = e["papers"] + added
        e["n_papers_found"] = len(e["papers"])
        e["key_players"] = extract_key_players(e["papers"])
        e["completed_with_final_query"] = cache["created"]
        new_pmids |= {str(p["pmid"]) for p in added}
        print(f"  {key}: +{len(added)} papers -> {e['n_papers_found']}")
    for key, e in results.items():
        r = rows[key]
        e.setdefault("pubmed_query_fetched", e.get("pubmed_query", ""))
        e.setdefault("n_papers_found_raw_fetched", e.get("n_papers_found_raw", 0))
        e["pubmed_query"] = rl.build_query(e["organ1"], e["organ2"])
        e["n_papers_found_raw"] = r["final query hits"]
    rl.OUTPUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    uniq = set()
    for key, ids in check["final_query_pmids"].items():
        uniq |= set(cache["pairs"][key]["pmids"]) if key in cache["pairs"] else set(ids)
    step1 = {"search date": chk.SEARCH_DATE, "pairs": len(rows),
             "hits (assignments)": sum(r["final query hits"] for r in rows.values()),
             "unique papers": len(uniq), "completed pairs": sorted(cache["pairs"]),
             "kept assignments": sum(e["n_papers_found"] for e in results.values())}
    STEP1.write_text(json.dumps(step1, indent=2), encoding="utf-8")
    unl = sorted(p for p in new_pmids if p not in labels)
    print(f"\n[ok] results updated | step 1: {step1}")
    print(f"new papers: {len(new_pmids)} | of them without an LLM label yet: {len(unl)} (already labelled: {len(new_pmids) - len(unl)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="phase 2: merge the staging file into the results")
    phase2() if ap.parse_args().apply else phase1()
