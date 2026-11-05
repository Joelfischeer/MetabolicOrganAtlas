"""
Apply the 20-year publication window to the stored papers.

The PubMed search was meant to cover the 20 years before the search date, but PubMed ignored the date (mindate without
maxdate), so all years were searched. This script applies the window afterwards: a paper stays when one of its publication
dates -- the print date of the journal issue or the electronic date, as in PubMed's [PDAT] -- lies in
[WINDOW_START, WINDOW_END] = [2006-09-22, 2026-09-17]. A partial date (year only, year+month) counts as its whole period.

Phase 1 (default): look up the dates of every stored paper (efetch, 200 PMIDs per request, cached in
  reference_network_only_metabolic/date_window_dates_cache.json) and report; the results are NOT changed.
Phase 2 (--apply): backup of the results file (.before_date_window.<time>.bak), papers outside the window removed from every pair,
  n_papers_found / key_players recomputed, list of the removed papers in removed_outside_date_window.csv.

Labels (per PMID) are not touched. Afterwards: invalidate_stale_after_completion.py, run_haiku_all.ps1 -Stage views, etc.
Usage:  python apply_date_window.py            # look up + report
        python apply_date_window.py --apply    # remove the papers outside the window
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
import shutil
import sys
import time
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_metabolic_lit_search as rl  # noqa: E402
from Literature_Search.pubmed_search import extract_key_players  # noqa: E402

D = HERE / "reference_network_only_metabolic"
CACHE = D / "date_window_dates_cache.json"
REMOVED_CSV = D / "removed_outside_date_window.csv"
WINDOW_START = date(2006, 9, 22)
WINDOW_END = date(2026, 9, 17)
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}


def _month(txt: str | None) -> int | None:
    if not txt:
        return None
    txt = txt.strip()
    if txt.isdigit():
        return int(txt)
    return MONTHS.get(txt[:3].lower())


def _interval(y: int, m: int | None, d: int | None) -> tuple[date, date]:
    """First and last day covered by a possibly partial date."""
    if m is None:
        return date(y, 1, 1), date(y, 12, 31)
    last = calendar.monthrange(y, m)[1]
    return (date(y, m, d), date(y, m, d)) if d else (date(y, m, 1), date(y, m, last))


def parse_dates(block: str) -> list[dict]:
    """Print date of the issue and electronic dates of one <PubmedArticle> block."""
    out = []
    m = re.search(r"<JournalIssue[^>]*>.*?<PubDate>(.*?)</PubDate>", block, re.DOTALL)
    if m:
        pd_ = m.group(1)
        y = re.search(r"<Year>(\d{4})</Year>", pd_)
        if y:
            mo = re.search(r"<Month>([^<]+)</Month>", pd_)
            dd = re.search(r"<Day>(\d+)</Day>", pd_)
            out.append({"kind": "print", "y": int(y.group(1)), "m": _month(mo.group(1)) if mo else None, "d": int(dd.group(1)) if dd else None})
        else:
            ml = re.search(r"<MedlineDate>(\d{4})\s*([A-Za-z]{3})?", pd_)
            if ml:
                out.append({"kind": "print", "y": int(ml.group(1)), "m": _month(ml.group(2)), "d": None})
    for a in re.findall(r"<ArticleDate[^>]*DateType=\"Electronic\"[^>]*>(.*?)</ArticleDate>", block, re.DOTALL):
        y, mo, dd = (re.search(rf"<{t}>(\d+)</{t}>", a) for t in ("Year", "Month", "Day"))
        if y:
            out.append({"kind": "electronic", "y": int(y.group(1)), "m": int(mo.group(1)) if mo else None, "d": int(dd.group(1)) if dd else None})
    return out


def lookup(pmids: list[str]) -> dict[str, list[dict]]:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [p for p in pmids if p not in cache]
    print(f"[i] dates cached: {len(pmids) - len(todo)} | to look up: {len(todo)}")
    for i in range(0, len(todo), 200):
        batch = todo[i:i + 200]
        resp = rl._ncbi_get("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
        if resp is None:
            raise SystemExit("[!] PubMed request failed; run the script again to continue (the dates found so far are cached)")
        for block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", resp.text, re.DOTALL):
            pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
            if pm:
                cache[pm.group(1)] = parse_dates(block)
        for p in batch:
            cache.setdefault(p, [])          # not returned as an article: no dates
        if (i // 200) % 10 == 9 or i + 200 >= len(todo):
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"    {min(i + 200, len(todo))}/{len(todo)}", flush=True)
        time.sleep(0.4)
    return cache


def verdict(dates: list[dict]) -> tuple[str, bool]:
    """('in' | 'before' | 'after' | 'nodate', boundary) ; boundary = decision depends on how a partial date is read."""
    if not dates:
        return "nodate", False
    inside, before, after, full_inside = False, False, False, False
    for dt in dates:
        a, b = _interval(dt["y"], dt["m"], dt["d"])
        if b < WINDOW_START:
            before = True
        elif a > WINDOW_END:
            after = True
        else:
            inside = True
            if a >= WINDOW_START and b <= WINDOW_END:
                full_inside = True
    if inside:
        return "in", not full_inside      # boundary: only a partial date overlapping the edge puts the paper inside
    return ("before" if before and not after else "after"), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    apply = ap.parse_args().apply
    results = json.loads(rl.OUTPUT_JSON.read_text(encoding="utf-8"))
    info = {}
    for e in results.values():
        for p in e["papers"]:
            info[str(p["pmid"])] = p
    dates = lookup(sorted(info))
    verd = {pm: verdict(dates.get(pm, [])) for pm in info}
    cnt = {k: sum(1 for v, _ in verd.values() if v == k) for k in ("in", "before", "after", "nodate")}
    print(f"\nunique papers {len(info)}: in window {cnt['in']} | before {WINDOW_START} {cnt['before']} | after {WINDOW_END} {cnt['after']} | no date {cnt['nodate']}")
    print(f"boundary cases (partial date overlapping a window edge, kept): {sum(1 for v, b in verd.values() if v == 'in' and b)}")
    # compare with the stored year
    y_out = {pm for pm, p in info.items() if str(p.get('year', '')).isdigit() and int(p['year']) <= 2005}
    print(f"stored year <= 2005: {len(y_out)} | of them judged inside the window by their dates: {sum(1 for pm in y_out if verd[pm][0] == 'in')}")
    if not apply:
        print("\n[i] nothing changed (run with --apply to remove the papers outside the window)")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = rl.OUTPUT_JSON.with_suffix(f".json.before_date_window.{stamp}.bak")
    shutil.copy2(rl.OUTPUT_JSON, backup)
    print(f"[i] backup: {backup.name}")
    removed = {}
    for key, e in results.items():
        keep = [p for p in e["papers"] if verd[str(p["pmid"])][0] in ("in", "nodate")]
        gone = [p for p in e["papers"] if verd[str(p["pmid"])][0] not in ("in", "nodate")]
        e["n_papers_found_all_years"] = len(e["papers"])
        e["papers"] = keep
        e["n_papers_found"] = len(keep)
        e["key_players"] = extract_key_players(keep)
        e["date_window"] = f"{WINDOW_START}..{WINDOW_END}"
        for p in gone:
            removed.setdefault(str(p["pmid"]), {"pmid": str(p["pmid"]), "year": p.get("year"), "reason": verd[str(p["pmid"])][0],
                                                "dates": json.dumps(dates.get(str(p["pmid"]), [])), "title": p.get("title", ""), "pairs": []})["pairs"].append(key)
    rl.OUTPUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(REMOVED_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "year (stored)", "reason", "dates", "n pairs", "pairs", "title"])
        for r in removed.values():
            w.writerow([r["pmid"], r["year"], f"published {r['reason']} the window", r["dates"], len(r["pairs"]), "; ".join(r["pairs"]), r["title"]])
    print(f"[ok] results updated: {sum(e['n_papers_found'] for e in results.values())} assignments, "
          f"{len({str(p['pmid']) for e in results.values() for p in e['papers']})} unique papers; removed {len(removed)} unique papers -> {REMOVED_CSV.name}")


if __name__ == "__main__":
    main()
