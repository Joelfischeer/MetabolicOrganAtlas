"""
reprocess_same_sentence_filter.py
==================================
Re-apply the same-sentence organ+organ+crosstalk filter to every cached
pair in reference_network_only_metabolic/metabolic_literature_results.json,
using the already-cached papers — no PubMed calls. Use this after adding or
changing _filter_same_sentence_crosstalk() in run_metabolic_lit_search.py to
bring an already-fully-cached results file up to date without a full
--reset re-search (which would needlessly re-hit PubMed for every pair).

The cached papers already satisfy the PubMed query's document-level
crosstalk requirement (both organs + a crosstalk keyword somewhere in the
document) — this script only narrows that down further to the stricter
same-sentence co-location, then re-derives key_players/connection_type from
the narrowed paper set.

Run from the Metabolic_Reference_Atlas/ directory:
    uv run python reprocess_same_sentence_filter.py
Then rebuild the dashboard:
    uv run python run_metabolic_lit_search.py --viz-only
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from Literature_Search.pubmed_search import (
    ORGAN_ALIASES, extract_key_players,
)
from run_metabolic_lit_search import (
    CROSSTALK_KEYWORDS, _compile_patterns, _filter_same_sentence_crosstalk,
    ALL_ORGAN_PATTERNS,
)

HERE = Path(__file__).parent
RESULTS_JSON = HERE / "reference_network_only_metabolic" / "metabolic_literature_results.json"


def main():
    if not RESULTS_JSON.exists():
        print(f"[!] Not found: {RESULTS_JSON}")
        return

    # Timestamped: a fixed name would be overwritten by the next run, losing
    # the file as it was before the earlier reprocessing.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = RESULTS_JSON.with_suffix(f".json.presamesentence.{stamp}.bak")
    shutil.copy2(RESULTS_JSON, backup)
    print(f"[i] Backup saved: {backup}")

    with open(RESULTS_JSON, encoding="utf-8") as f:
        results = json.load(f)

    crosstalk_patterns = _compile_patterns(CROSSTALK_KEYWORDS)

    total = len(results)
    for idx, (key, entry) in enumerate(results.items()):
        o1, o2 = entry.get("organ1", ""), entry.get("organ2", "")
        papers_raw = entry.get("papers", [])
        if not papers_raw or not o1 or not o2:
            continue

        organ1_patterns = _compile_patterns(ORGAN_ALIASES.get(o1, [o1]))
        organ2_patterns = _compile_patterns(ORGAN_ALIASES.get(o2, [o2]))
        papers = _filter_same_sentence_crosstalk(papers_raw, organ1_patterns,
                                                 organ2_patterns, crosstalk_patterns,
                                                 organ1_name=o1, organ2_name=o2,
                                                 all_organ_patterns=ALL_ORGAN_PATTERNS)

        entry["n_papers_found_pre_same_sentence"] = len(papers_raw)
        entry["papers"]         = papers
        entry["n_papers_found"] = len(papers)
        key_players = extract_key_players(papers)
        entry["key_players"] = key_players

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"  [{idx+1}/{total}] reprocessed")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    dropped_to_zero = sum(1 for v in results.values() if v.get("n_papers_found", 0) == 0)
    print(f"\n[ok] Reprocessed {total} pairs ({dropped_to_zero} now have 0 papers). "
          f"Saved: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
