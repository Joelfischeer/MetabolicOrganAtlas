"""
Step 1 of the funnel restricted to the 20-year window: count-only PubMed run (esearch, no abstracts, no LLM).

For every organ pair the FINAL query is run for the records that were in PubMed on the search date (Entrez date <= 2026/09/17)
and whose publication date ([PDAT]: print or electronic date) lies in 2006/09/22 - 2026/09/17.

Validation before anything is written (a malformed date query would show up here):
  * the papers kept by the pipeline (already restricted to the window by apply_date_window.py) must be in these lists,
  * the papers that apply_date_window.py removed must not be in them.
Only if both hold (<= 1% exceptions, the rest are papers with partial dates at the window edge) the results are updated:
  n_papers_found_raw        = hits of the final query inside the window  (what the funnels show as step 1)
  n_papers_found_raw_all_years = hits of the final query, all years (before this script)
and pubmed_step1_final.json gets 'window hits (assignments)' / 'window unique papers'.

Backup: metabolic_literature_results.json.before_step1_window.<time>.bak.   Usage: python recount_step1_window.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_metabolic_lit_search as rl  # noqa: E402
import pubmed_final_query_check as chk  # noqa: E402

D = HERE / "reference_network_only_metabolic"
STEP1 = D / "pubmed_step1_final.json"
OUT = D / "pubmed_step1_window.json"
REMOVED_CSV = D / "removed_outside_date_window.csv"
WINDOW = ' AND ("2006/09/22"[PDAT] : "2026/09/17"[PDAT])'


def main() -> None:
    results = json.loads(rl.OUTPUT_JSON.read_text(encoding="utf-8"))
    removed: dict[str, set[str]] = {}
    with open(REMOVED_CSV, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            for pair in r[5].split("; "):
                removed.setdefault(pair, set()).add(r[0])
    rows, lists = [], {}
    for i, (key, e) in enumerate(sorted(results.items()), 1):
        n, ids = chk.esearch(rl.build_query(e["organ1"], e["organ2"]) + WINDOW)
        if n > chk.RETMAX or len(ids) != n:
            raise SystemExit(f"[!] {key}: {n} hits in the window, {len(ids)} returned (limit {chk.RETMAX}); slice the query first")
        got = set(ids)
        kept = {str(p["pmid"]) for p in e["papers"]}
        rem = removed.get(key, set())
        rows.append({"pair": key, "hits all years (final query)": e["n_papers_found_raw"], "hits in window": n,
                     "kept": len(kept), "kept not in window list": len(kept - got),
                     "removed by date check": len(rem), "removed but in window list": len(rem & got)})
        lists[key] = ids
        print(f"  [{i}/{len(results)}] {key}: all years {e['n_papers_found_raw']} | window {n} | kept {len(kept)} (missing {len(kept - got)}) | removed {len(rem)} (in list {len(rem & got)})", flush=True)

    uniq = {p for ids in lists.values() for p in ids}
    s = {"window": "2006-09-22..2026-09-17", "search date": chk.SEARCH_DATE,
         "hits all years (assignments)": sum(r["hits all years (final query)"] for r in rows),
         "window hits (assignments)": sum(r["hits in window"] for r in rows),
         "window unique papers": len(uniq),
         "pairs with window hits": sum(1 for r in rows if r["hits in window"]),
         "kept (assignments)": sum(r["kept"] for r in rows), "kept not in window list": sum(r["kept not in window list"] for r in rows),
         "removed by date check (assignments)": sum(r["removed by date check"] for r in rows),
         "removed but in window list": sum(r["removed but in window list"] for r in rows)}
    OUT.write_text(json.dumps({"summary": s, "pairs": rows}, indent=2), encoding="utf-8")
    print("\n" + json.dumps(s, indent=2))
    bad_kept = s["kept not in window list"] / max(1, s["kept (assignments)"])
    bad_rem = s["removed but in window list"] / max(1, s["removed by date check (assignments)"])
    print(f"\nvalidation: kept papers missing from the window lists {100 * bad_kept:.2f}% | removed papers present {100 * bad_rem:.2f}%")
    if bad_kept > 0.01 or bad_rem > 0.01:
        raise SystemExit("[!] the window counts disagree with the local date decisions by more than 1%; results NOT updated "
                         f"(details in {OUT.name})")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = rl.OUTPUT_JSON.with_suffix(f".json.before_step1_window.{stamp}.bak")
    shutil.copy2(rl.OUTPUT_JSON, backup)
    for r in rows:
        e = results[r["pair"]]
        e.setdefault("n_papers_found_raw_all_years", e["n_papers_found_raw"])
        e["n_papers_found_raw"] = r["hits in window"]
    rl.OUTPUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    step1 = json.loads(STEP1.read_text(encoding="utf-8")) if STEP1.exists() else {}
    step1.update({"window": s["window"], "window hits (assignments)": s["window hits (assignments)"],
                  "window unique papers": s["window unique papers"], "pairs with window hits": s["pairs with window hits"]})
    STEP1.write_text(json.dumps(step1, indent=2), encoding="utf-8")
    print(f"[ok] results updated (backup {backup.name}); step 1 = {s['window hits (assignments)']} hits, {s['window unique papers']} unique papers")


if __name__ == "__main__":
    main()
