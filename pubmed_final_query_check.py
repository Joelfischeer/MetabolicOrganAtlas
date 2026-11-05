"""
Count-only PubMed check of step 1 of the funnel (no abstracts downloaded, no LLM).

The stored literature results were fetched on 2026-09-17 with the queries stored in the results file; the
current config has two terms less in the query (`pathway`, and `microbiome` / `gut microbiota` for the Colon pairs).
This script runs, for all organ pairs,
  * the OLD query (as stored)   -> must reproduce the stored n_papers_found_raw (validates the method), and
  * the FINAL query (current config, `build_query`)  -> the step-1 numbers under the final rules,
both restricted to records that were in PubMed on the search date (Entrez date <= 2026/09/17), and compares the
PMID lists with the papers that the pipeline kept.

Notes on how the search behaves (found while writing this check):
  * esearch returns at most 10,000 PMIDs per query, so pairs with more hits were cut off at ~10,000 (`MAX_PAPERS` = 20,000
    in the config has no effect above that); `count` is the total number of hits, `retrieved` what the pipeline got.
  * `mindate` without `maxdate` is ignored by PubMed, so `YEARS_BACK` = 20 was not applied: all years were searched.
    This script uses the same parameters (no date window), only the Entrez-date cap is added.

Writes reference_network_only_metabolic/pubmed_final_query_check.json (+ .csv per pair).
Usage:  python pubmed_final_query_check.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_metabolic_lit_search as rl  # noqa: E402

RESULTS = HERE / "reference_network_only_metabolic" / "metabolic_literature_results.json"
OUT_JSON = HERE / "reference_network_only_metabolic" / "pubmed_final_query_check.json"
OUT_CSV = HERE / "reference_network_only_metabolic" / "pubmed_final_query_check.csv"
SEARCH_DATE = "2026/09/17"
DATE_CAP = f' AND ("1900/01/01"[EDAT] : "{SEARCH_DATE}"[EDAT])'
RETMAX = 10000          # esearch hard limit
DELAY = 0.4


def esearch(term: str) -> tuple[int, list[str]]:
    params = {"db": "pubmed", "term": term + DATE_CAP, "retmax": RETMAX, "retmode": "json"}
    resp = rl._ncbi_get("esearch.fcgi", params, method="POST")
    time.sleep(DELAY)
    if resp is None:
        raise SystemExit(f"[!] PubMed request failed for: {term[:100]}...")
    res = resp.json()["esearchresult"]
    return int(res["count"]), list(res["idlist"])


def main() -> None:
    stored = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows, final_ids = [], {}
    for i, (key, v) in enumerate(sorted(stored.items()), 1):
        o1, o2 = v["organ1"], v["organ2"]
        old_count, old_ids = esearch(v["pubmed_query"])
        new_count, new_ids = esearch(rl.build_query(o1, o2))
        kept = {p["pmid"] for p in v["papers"]}
        old_set, new_set = set(old_ids), set(new_ids)
        capped = new_count > RETMAX or old_count > RETMAX
        rows.append({
            "pair": key, "stored raw": v["n_papers_found_raw"], "old query hits": old_count, "old query retrieved": len(old_ids),
            "final query hits": new_count, "final query retrieved": len(new_ids), "capped at 10,000": capped,
            "kept by pipeline": len(kept), "kept found in final list": len(kept & new_set),
            "final list not in old list": len(new_set - old_set),
        })
        final_ids[key] = new_ids
        print(f"  [{i}/{len(stored)}] {key}: stored {v['n_papers_found_raw']} | old {old_count}/{len(old_ids)} | "
              f"final {new_count}/{len(new_ids)} | kept {len(kept)} found {len(kept & new_set)}", flush=True)

    uniq = {p for ids in final_ids.values() for p in ids}
    summary = {
        "search date": SEARCH_DATE,
        "pairs": len(rows),
        "stored raw (sum)": sum(r["stored raw"] for r in rows),
        "old query hits (sum)": sum(r["old query hits"] for r in rows),
        "old query retrieved (sum)": sum(r["old query retrieved"] for r in rows),
        "final query hits (sum)": sum(r["final query hits"] for r in rows),
        "final query retrieved (sum)": sum(r["final query retrieved"] for r in rows),
        "final query retrieved, unique papers": len(uniq),
        "pairs capped at 10,000": sum(r["capped at 10,000"] for r in rows),
        "kept by pipeline (sum)": sum(r["kept by pipeline"] for r in rows),
        "kept found in final list (sum)": sum(r["kept found in final list"] for r in rows),
        "final list not in old list (sum)": sum(r["final list not in old list"] for r in rows),
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "pairs": rows, "final_query_pmids": final_ids}), encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n" + json.dumps(summary, indent=2))
    print(f"[ok] {OUT_JSON.name}, {OUT_CSV.name}")


if __name__ == "__main__":
    main()
