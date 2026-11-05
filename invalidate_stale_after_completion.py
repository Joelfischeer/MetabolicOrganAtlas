"""
After complete_capped_pairs.py --apply and the classification of the new papers (run_haiku_all.ps1 -Stage classify):
remove (with backups) the connection types and descriptions of every view whose input papers changed, so that the next
run (-Stage views) regenerates exactly those and nothing else.

  * a connection type entry is stale when the first LLM_MAX_PAPERS (100) papers of the pair in that view are not the same
    as before (the LLM only reads those),
  * a description is stale when one of the papers it cites is no longer among the papers of the pair in that view.

Needs a snapshot of the paper lists of all views as they were when the descriptions/types were last (re)generated:
  view_snapshot_before_capped_pair_completion.json  (written by complete_capped_pairs.py --apply; before that completion)
  view_snapshot_before_date_window.json             (lists before apply_date_window.py --apply; after the completion)
Changes nothing else, no API calls. Backups: <file>.<label>_<date>.json.
Usage:  python invalidate_stale_after_completion.py [--snapshot FILE] [--label NAME]
        (default: the date-window snapshot if it exists, label before_date_window)
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_metabolic_lit_search as rl  # noqa: E402
from Literature_Search import llm_condition_filter as cf  # noqa: E402

D = HERE / "reference_network_only_metabolic"
SNAPSHOT = D / "view_snapshot_before_date_window.json"


def canon_of(key: str) -> str:
    a, b = key.split("|")
    return f"{min(a, b)}|{max(a, b)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--label", default="before_date_window")
    args = ap.parse_args()
    before = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%Y%m%d")
    res = json.loads(rl.OUTPUT_JSON.read_text(encoding="utf-8"))
    labels = cf.load_condition_labels(rl.CONDITION_LABELS, rl.CONDITION_LLM_MODEL)
    filters = {c["slug"]: cf.LLMConditionFilter(c["slug"], labels) for c in cf.CONDITIONS}
    after = {"all": {}, **{s: {} for s in filters}}
    for e in res.values():
        o1, o2 = e.get("organ1", ""), e.get("organ2", "")
        canon = f"{min(o1, o2)}|{max(o1, o2)}"
        papers = e.get("papers", [])
        after["all"][canon] = [str(p["pmid"]) for p in papers]
        for s, f in filters.items():
            after[s][canon] = [str(p["pmid"]) for p in papers if f.accepts(p)]
    print("papers per view (assignments) before -> after:")
    for s in after:
        print(f"  {s:<15}{sum(len(v) for v in before[s].values()):>8} -> {sum(len(v) for v in after[s].values()):>8}")

    total_d = total_t = 0
    print(f"\n{'view':<15}{'descriptions removed':>22}{'types removed':>16}")
    for slug in after:
        suffix = "" if slug == "all" else f"_{slug}"
        dp, tp = D / f"metabolic_llm_descriptions{suffix}.json", D / f"metabolic_connection_types{suffix}.json"
        if not dp.exists() or not tp.exists():
            print(f"{slug:<15} files missing, skipped")
            continue
        d, t = json.loads(dp.read_text(encoding="utf-8")), json.loads(tp.read_text(encoding="utf-8"))
        new_sets = {k: set(v) for k, v in after[slug].items()}
        stale_d = [k for k, v in d.items() if v.get("description") and
                   (not after[slug].get(canon_of(k)) or any(str(p["pmid"]) not in new_sets[canon_of(k)] for p in v.get("papers", [])))]
        stale_t = [k for k, v in t.items() if v.get("runs") and
                   before[slug].get(canon_of(k), [])[:rl.LLM_MAX_PAPERS] != after[slug].get(canon_of(k), [])[:rl.LLM_MAX_PAPERS]]
        for path, data, stale in ((dp, d, stale_d), (tp, t, stale_t)):
            bak = path.with_name(f"{path.stem}.{args.label}_{stamp}{path.suffix}")
            if stale and not bak.exists():
                shutil.copy2(path, bak)
            for k in stale:
                del data[k]
            if stale:
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        total_d += len(stale_d)
        total_t += len(stale_t)
        print(f"{slug:<15}{len(stale_d):>22}{len(stale_t):>16}")
    print(f"\ntotal removed for regeneration: {total_d} descriptions, {total_t} type entries")


if __name__ == "__main__":
    main()
