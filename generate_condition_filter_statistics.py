"""
generate_condition_filter_statistics.py
=======================================
Statistics of the LLM condition filter (Literature_Search/llm_condition_filter.py).

Reads the base literature results and the per-paper LLM labels
(reference_network_only_metabolic/metabolic_condition_labels.jsonl) and writes,
into condition_filter_statistics/:

  condition_filter_statistics.xlsx   Summary | Label x status | Cases |
                                     Per pair | Keyword vs LLM | Hallucinations | Spot-check
  condition_labels_per_paper.csv     one row per classified paper
  per_pair_label_counts.csv          label counts for each organ pair

Works on partial results (a classification run that is still going on): every
number refers to the papers classified so far.

Run (from the Metabolic_Reference_Atlas/ folder):
    python generate_condition_filter_statistics.py
"""

import csv
import html
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import run_metabolic_lit_search as rl                                  # noqa: E402
from Literature_Search import llm_condition_filter as cf               # noqa: E402

OUT_DIR = HERE / "condition_filter_statistics"


def _pct(a: int, b: int) -> float:
    return round(100 * a / b, 2) if b else 0.0


def main() -> None:
    with open(rl.OUTPUT_JSON, encoding="utf-8") as f:
        results = json.load(f)
    labels = cf.load_condition_labels(rl.CONDITION_LABELS, rl.CONDITION_LLM_MODEL)
    if not labels:
        sys.exit(f"No LLM condition labels for model {rl.CONDITION_LLM_MODEL} / prompt "
                 f"{cf.PROMPT_VERSION} in {rl.CONDITION_LABELS}. Run --classify-conditions first.")

    # papers of the base search -------------------------------------------------
    paper_info: dict[str, dict] = {}
    assignments: list[tuple[str, str]] = []                 # (pair key, pmid)
    for key, entry in results.items():
        for p in entry.get("papers", []):
            pid = str(p.get("pmid", ""))
            if not pid:
                continue
            assignments.append((key, pid))
            paper_info.setdefault(pid, p)
    n_unique, n_assign = len(paper_info), len(assignments)
    classified = {pid: r for pid, r in labels.items() if pid in paper_info}
    n_cls = len(classified)

    status = Counter(r["status"] for r in classified.values())
    parsed = status[cf.OK] + status[cf.HALLUCINATED] + status[cf.TOO_SHORT]
    halluc = status[cf.HALLUCINATED] + status[cf.TOO_SHORT]
    ok_labels = Counter(r["label"] for r in classified.values() if r["status"] == cf.OK)
    parsed_labels = Counter(r["label"] for r in classified.values() if r["label"])
    cls_assign = [(k, pid) for k, pid in assignments if pid in classified]

    # ---- Summary ---------------------------------------------------------------
    times = [r.get("seconds", 0) for r in classified.values() if r.get("seconds")]
    dates = sorted(r["classified_at"] for r in classified.values() if r.get("classified_at"))
    summary = [
        ("Model", rl.CONDITION_LLM_MODEL), ("Prompt version", cf.PROMPT_VERSION),
        ("Classification period", f"{dates[0]}  ->  {dates[-1]}" if dates else ""),
        ("Mean seconds per paper", round(sum(times) / len(times), 1) if times else ""),
        ("", ""),
        ("Unique papers in the base search", n_unique),
        ("Paper-pair assignments in the base search", n_assign),
        ("Unique papers classified so far", n_cls),
        ("Coverage of unique papers (%)", _pct(n_cls, n_unique)),
        ("Coverage of assignments (%)", _pct(len(cls_assign), n_assign)),
        ("Papers classified from the title only (no abstract)",
         sum(1 for r in classified.values() if r.get("title_only"))),
        ("", ""),
    ]
    for s in cf.STATUSES:
        summary.append((f"Status: {s}", f"{status[s]}  ({_pct(status[s], n_cls)}%)"))
    summary += [
        ("", ""),
        ("Hallucination rate (hallucinated + too-short evidence, of parsed answers) %",
         _pct(halluc, parsed)),
        ("   of which: sentence not found in title/abstract", status[cf.HALLUCINATED]),
        ("   of which: evidence empty or shorter than "
         f"{cf.MIN_EVIDENCE_WORDS} words", status[cf.TOO_SHORT]),
        ("Usable papers (valid label + verified evidence)", f"{status[cf.OK]}  ({_pct(status[cf.OK], n_cls)}%)"),
        ("", ""),
        ("Label distribution among usable papers", ""),
    ]
    for lab in cf.LABELS:
        summary.append((f"   {lab}", f"{ok_labels[lab]}  ({_pct(ok_labels[lab], status[cf.OK])}%)"))
    df_summary = pd.DataFrame(summary, columns=["Metric", "Value"])

    # ---- Label x status --------------------------------------------------------
    lab_status = pd.DataFrame(
        [[Counter(r["status"] for r in classified.values() if r["label"] == lab)[s]
          for s in cf.STATUSES] for lab in cf.LABELS],
        index=cf.LABELS, columns=cf.STATUSES)
    lab_status["total with this label"] = lab_status.sum(axis=1)
    lab_status["hallucination rate %"] = [
        _pct(row[cf.HALLUCINATED] + row[cf.TOO_SHORT], row[cf.OK] + row[cf.HALLUCINATED] + row[cf.TOO_SHORT])
        for _, row in lab_status.iterrows()]
    lab_status.index.name = "LLM label"
    df_lab_status = lab_status.reset_index()

    # ---- Cases (No condition / Healthy / Obese) ---------------------------------
    case_rows = []
    pairs_total = len(results)
    pairs_with_papers = sum(1 for e in results.values() if e.get("papers"))
    case_rows.append(["No condition (all papers)", "all papers, any label", n_assign, n_unique, pairs_with_papers,
                      100.0, 100.0])
    for case, lab in cf.CONDITION_TO_LABEL.items():
        filt = cf.LLMConditionFilter(case, labels)
        a = [(k, pid) for k, pid in assignments if filt.accepts(paper_info[pid])]
        pids = {pid for _, pid in a}
        case_rows.append([f"{case.capitalize()} case", f"label '{lab}', evidence verified", len(a), len(pids),
                          len({k for k, _ in a}), _pct(len(a), n_assign), _pct(len(pids), n_unique)])
        lost = sum(1 for r in classified.values()
                   if r.get("label") == lab and r["status"] in (cf.HALLUCINATED, cf.TOO_SHORT))
        case_rows.append([f"   {case.capitalize()}-labelled but rejected (hallucinated evidence)", "unique papers",
                          "", lost, "", "", _pct(lost, n_unique)])
    df_cases = pd.DataFrame(case_rows, columns=["Case", "Definition", "Paper-pair assignments", "Unique papers",
                                                "Pairs with >=1 paper", "% of all assignments",
                                                "% of unique papers"])

    # ---- Per pair ---------------------------------------------------------------
    pair_rows = []
    for key, entry in sorted(results.items()):
        pids = [str(p.get("pmid", "")) for p in entry.get("papers", []) if p.get("pmid")]
        recs = [classified.get(pid) for pid in pids]
        done = [r for r in recs if r]
        c = Counter(r["label"] for r in done if r["status"] == cf.OK)
        row = {"pair": key.replace("|", " - "), "papers": len(pids), "classified": len(done),
               "usable": sum(1 for r in done if r["status"] == cf.OK),
               "hallucinated": sum(1 for r in done if r["status"] in (cf.HALLUCINATED, cf.TOO_SHORT)),
               "unusable": sum(1 for r in done if r["status"] in (cf.INVALID_JSON, cf.INVALID_LABEL, cf.ERROR))}
        row.update({lab: c[lab] for lab in cf.LABELS})
        row["Healthy %"] = _pct(c[cf.LABEL_HEALTHY], row["usable"])
        row["Obesity/metabolic/diabetes %"] = _pct(c[cf.LABEL_METABOLIC], row["usable"])
        pair_rows.append(row)
    df_pair = pd.DataFrame(pair_rows)

    # ---- Legacy keyword filter vs LLM ---------------------------------------------
    kw_rows = []
    pats = {c: rl._compile_patterns(rl.LEGACY_CONDITION_KEYWORDS[c]) for c in ("healthy", "obese")}
    legacy_view = {"healthy": "healthy", "obesity": "obese"}     # view slug -> legacy keyword list
    usable = {pid: r for pid, r in classified.items() if r["status"] == cf.OK}
    for case, lab in cf.CONDITION_TO_LABEL.items():
        if case not in legacy_view:
            continue            # the old keyword filter only knew healthy / obese
        kw = {pid for pid in usable
              if rl._any_match(pats[legacy_view[case]], (paper_info[pid].get("title", "") + ". "
                                            + paper_info[pid].get("abstract", "")).lower())}
        llm = {pid for pid, r in usable.items() if r["label"] == lab}
        both = kw & llm
        kw_rows.append([case, len(kw), len(llm), len(both), len(kw - llm), len(llm - kw),
                        _pct(len(both), len(kw)), _pct(len(both), len(llm)),
                        _pct(len(both), len(kw | llm))])
    df_kw = pd.DataFrame(kw_rows, columns=[
        "case", "legacy keyword filter: papers", "LLM filter: papers", "in both", "keyword only", "LLM only",
        "% of keyword papers the LLM also accepts", "% of LLM papers the keywords also match", "Jaccard %"])
    # what the LLM says about the papers the old keyword filter would have kept
    kw_dist_rows = []
    for case in ("healthy", "obese"):
        kw = [pid for pid in usable
              if rl._any_match(pats[case], (paper_info[pid].get("title", "") + ". "
                                            + paper_info[pid].get("abstract", "")).lower())]
        c = Counter(usable[pid]["label"] for pid in kw)
        for lab in cf.LABELS:
            kw_dist_rows.append([f"legacy {case} keywords", len(kw), lab, c[lab], _pct(c[lab], len(kw))])
    df_kw_dist = pd.DataFrame(kw_dist_rows, columns=["papers selected by", "n papers", "LLM label",
                                                     "n", "% of those papers"])

    # ---- Hallucination examples + spot-check sample ---------------------------------
    def _row(r: dict) -> dict:
        p = paper_info.get(r["pmid"], {})
        return {"pmid": r["pmid"], "label": r.get("label"), "status": r["status"], "reason": r.get("reason", ""),
                "evidence": r.get("evidence", ""),
                "title": html.unescape(p.get("title", "")), "year": p.get("year", ""), "doi": p.get("doi", "")}

    rng = random.Random(0)
    bad = [r for r in classified.values() if r["status"] in (cf.HALLUCINATED, cf.TOO_SHORT)]
    df_hall = pd.DataFrame([_row(r) for r in rng.sample(bad, min(200, len(bad)))])
    spot = []
    for lab in (cf.LABEL_HEALTHY, cf.LABEL_METABOLIC):
        pool = [r for r in usable.values() if r["label"] == lab]
        spot += [_row(r) for r in rng.sample(pool, min(40, len(pool)))]
    df_spot = pd.DataFrame(spot)
    if not df_spot.empty:
        df_spot["your judgement (correct / wrong)"] = ""

    # ---- write ------------------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)
    xlsx = OUT_DIR / "condition_filter_statistics.xlsx"
    sheets = {"Summary": df_summary, "Label x status": df_lab_status, "Cases": df_cases,
              "Per pair": df_pair, "Keyword vs LLM": df_kw, "Keyword papers by label": df_kw_dist,
              "Hallucinations": df_hall, "Spot-check": df_spot}
    def _xl_clean(df):      # PubMed text sometimes holds control characters, which .xlsx cannot store
        # (no dtype test: newer pandas stores text columns as "str", not "object")
        return df.apply(lambda col: col.map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v))

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in sheets.items():
            _xl_clean(df).to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col_cells in ws.columns:
                width = max((len(str(c.value)) for c in col_cells[:60] if c.value is not None), default=8)
                ws.column_dimensions[col_cells[0].column_letter].width = min(80, max(10, width + 2))
            ws.freeze_panes = "A2"

    rows = [_row(r) for r in classified.values()]
    pd.DataFrame(rows).to_csv(OUT_DIR / "condition_labels_per_paper.csv", index=False, encoding="utf-8-sig")
    df_pair.to_csv(OUT_DIR / "per_pair_label_counts.csv", index=False, encoding="utf-8-sig")

    # ---- console ------------------------------------------------------------------------
    print(f"Condition filter statistics  ({datetime.now():%Y-%m-%d %H:%M})")
    print(f"  model {rl.CONDITION_LLM_MODEL}, prompt {cf.PROMPT_VERSION}")
    print(f"  classified {n_cls} of {n_unique} unique papers ({_pct(n_cls, n_unique)}%)")
    print("  status: " + ", ".join(f"{s}={status[s]}" for s in cf.STATUSES))
    print(f"  hallucination rate {_pct(halluc, parsed)}% of {parsed} parsed answers")
    print("  usable papers by label:")
    for lab in cf.LABELS:
        print(f"    {lab:<38} {ok_labels[lab]:>6}  ({_pct(ok_labels[lab], status[cf.OK])}%)")
    print(df_cases.to_string(index=False))
    print(f"\n  written: {xlsx}")


if __name__ == "__main__":
    main()
