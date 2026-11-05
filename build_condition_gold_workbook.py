"""
build_condition_gold_workbook.py
================================
Builds the labelling workbook for the LLM condition filter: ~150 papers that
YOU label (dropdown), so the filter's accuracy can be measured per class
(see evaluate_condition_filter.py).

The sample is enriched for the rare classes (Healthy, Obesity/metabolic/diabetes,
...), which a plain random sample would hardly contain. Enrichment uses cheap
cues only to PICK papers -- the cues are never shown and never count as labels.
The stratum of each paper is stored on a hidden "Key" sheet (blind labelling).
The LLM's predictions are not in the workbook either.

Run (from the Metabolic_Reference_Atlas/ folder):
    python build_condition_gold_workbook.py            # 150 papers, seed 2026
    python build_condition_gold_workbook.py --seed 7
"""

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import run_metabolic_lit_search as rl                         # noqa: E402
from Literature_Search import llm_condition_filter as cf      # noqa: E402

OUT_DIR = HERE / "condition_gold_set"

# (stratum, number of papers, matcher) -- drawn in this order; a paper is used once
CUES = {
    "healthy_title": r"\bhealthy (volunteers?|subjects?|men|women|adults?|humans?|individuals?|participants?|"
                     r"young|elderly|people|persons)\b|\bin healthy\b|\bnormal (physiology|mice|rats)\b",
    "obese_title":   r"\bobes(e|ity)\b|overweight|diabet|metabolic syndrome|insulin resistance|high[- ]fat diet",
    "cancer":        r"\b(cancers?|tumou?rs?|carcinoma|leuk(a)?emia|lymphoma|melanoma|neoplas\w*|malignan\w*)\b",
    "cardio":        r"heart failure|myocardial infarction|cardiomyopathy|atheroscleros\w*|hypertension|"
                     r"cardiovascular|stroke|arrhythmia",
    "infectious":    r"\binfection|sepsis|\bvirus|\bviral\b|\bCOVID|\bHIV\b|malaria|tubercul\w*|pathogen",
    "mental":        r"depress\w*|anxiety|schizophren\w*|bipolar|autism|\bADHD\b|addiction|psychiatric",
    "genetic":       r"cystic fibrosis|dystrophy|inherited|congenital|lysosomal|phenylketonuria|Down syndrome|hereditary",
    "review":        r"\breview\b|perspective|overview|\bcommentary\b",
}
PLAN = [("healthy_title", 20, "title"), ("legacy_healthy_kw", 25, "text"), ("obese_title", 15, "title"),
        ("legacy_obese_kw", 15, "text"), ("cancer", 12, "title"), ("cardio", 10, "title"),
        ("infectious", 10, "title"), ("mental", 8, "title"), ("genetic", 6, "title"),
        ("review", 12, "title"), ("random", 17, "any")]

HDR_FILL = PatternFill("solid", fgColor="1F3864")
REV_FILL = PatternFill("solid", fgColor="FFF2CC")
REV_HDR_FILL = PatternFill("solid", fgColor="C55A11")
WRAP = Alignment(wrap_text=True, vertical="top")

RULES = [
    "Choose ONE label per paper: the health condition of the people/animals the paper studies "
    "(for a review: the condition the review is about). Judge from title + abstract only.",
    "Healthy = the studied subjects are healthy people or normal animals. Healthy subjects that only serve as the "
    "control group of a disease do not count: choose the disease.",
    "No condition = the paper does not study subjects with a defined health condition.",
    "If several conditions occur, pick the one the paper focuses on. Use 'Unsure' only if you really cannot decide "
    "(unsure papers are left out of the accuracy numbers).",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "condition_gold_set.xlsx")
    args = ap.parse_args()

    if args.out.exists():
        from openpyxl import load_workbook
        wb0 = load_workbook(args.out, read_only=True)
        if any(r[5] not in (None, "") for r in wb0["Papers"].iter_rows(min_row=2, values_only=True)):
            sys.exit(f"{args.out} already contains labels -- refusing to overwrite (use --out for a new file).")
        wb0.close()

    with open(rl.OUTPUT_JSON, encoding="utf-8") as f:
        results = json.load(f)
    papers: dict[str, dict] = {}
    for entry in results.values():
        for p in entry.get("papers", []):
            pid = str(p.get("pmid", ""))
            if pid and p.get("abstract", "").strip() and pid not in papers:
                papers[pid] = p
    pids = sorted(papers)
    rng = random.Random(args.seed)
    rng.shuffle(pids)

    kw = {c: rl._compile_patterns(rl.LEGACY_CONDITION_KEYWORDS[c]) for c in ("healthy", "obese")}
    cues = {k: re.compile(v, re.I) for k, v in CUES.items()}

    def matches(stratum: str, p: dict) -> bool:
        title = html.unescape(p.get("title", ""))
        text = (title + ". " + html.unescape(p.get("abstract", ""))).lower()
        if stratum == "legacy_healthy_kw":
            return rl._any_match(kw["healthy"], text)
        if stratum == "legacy_obese_kw":
            return rl._any_match(kw["obese"], text)
        if stratum == "random":
            return True
        return bool(cues[stratum].search(title))

    chosen: list[tuple[str, str]] = []                # (pid, stratum)
    used: set[str] = set()
    for stratum, n, _ in PLAN:
        got = 0
        for pid in pids:
            if pid in used or not matches(stratum, papers[pid]):
                continue
            chosen.append((pid, stratum))
            used.add(pid)
            got += 1
            if got == n:
                break
        if got < n:
            print(f"[!] stratum {stratum}: only {got} of {n} papers found")
    rng.shuffle(chosen)

    wb = Workbook()
    ws = wb.active
    ws.title = "Papers"
    heads = [("ID", 7), ("PMID", 11), ("Year", 6), ("Title", 45), ("Abstract", 100),
             ("YOUR LABEL", 30), ("Paper type (optional)", 16), ("Notes", 28)]
    for c, (h, w) in enumerate(heads, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = REV_HDR_FILL if c >= 6 else HDR_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"

    for r, (pid, stratum) in enumerate(chosen, start=2):
        p = papers[pid]
        abstract = ILLEGAL_CHARACTERS_RE.sub("", html.unescape(p["abstract"])).strip()
        title = ILLEGAL_CHARACTERS_RE.sub("", html.unescape(p.get("title", ""))).strip()
        for c, v in enumerate([f"G{r - 1:03d}", pid, p.get("year", ""), title, abstract, None, None, None], start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.alignment = WRAP
            if c >= 6:
                cell.fill = REV_FILL
            if c == 5:
                cell.font = Font(size=9)
        link = ws.cell(row=r, column=2)
        link.hyperlink = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
        link.font = Font(color="0563C1", underline="single")
        ws.row_dimensions[r].height = max(60, min(400, 13 * (len(abstract) // 110 + 2)))
    n = len(chosen)

    opts = ",".join(cf.LABELS + ["Unsure"])
    dv = DataValidation(type="list", formula1=f'"{opts}"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Invalid label", error="Pick a label from the list.")
    ws.add_data_validation(dv)
    dv.add(f"F2:F{n + 1}")
    dv2 = DataValidation(type="list", formula1='"review,method,in_vitro,study"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"G2:G{n + 1}")

    # README sheet
    wr = wb.create_sheet("README", 0)
    wr.column_dimensions["A"].width = 38
    wr.column_dimensions["B"].width = 120
    rows = [("Condition labelling set", None),
            ("What to do", f"Label the {n} papers on the 'Papers' sheet with the dropdown in the orange column "
                           "(YOUR LABEL). ~30-40 s per paper, 1.5-2 h in total. The LLM's predictions are not "
                           "shown, on purpose."),
            ("Rules", None)]
    rows += [(f"  {i}", t) for i, t in enumerate(RULES, 1)]
    rows += [("", None), ("The seven labels", None)]
    rows += [(lab, cf.LABEL_DESCRIPTIONS[lab]) for lab in cf.LABELS]
    rows += [("Unsure", "cannot decide from title + abstract; excluded from the accuracy numbers")]
    rows += [("", None), ("Afterwards", "python evaluate_condition_filter.py  -> per-class precision/recall, "
                                         "confusion matrix, Healthy/Obese case metrics.")]
    for i, (a, b) in enumerate(rows, start=1):
        ca, cb = wr.cell(row=i, column=1, value=a), wr.cell(row=i, column=2, value=b)
        ca.alignment = cb.alignment = WRAP
        ca.font = Font(bold=True, size=14 if i == 1 else 11)
        if b is None and a:
            ca.font = Font(bold=True, color="FFFFFF")
            ca.fill = cb.fill = HDR_FILL

    # hidden key sheet (blind labelling)
    wk = wb.create_sheet("Key")
    wk.append(["ID", "PMID", "stratum (sampling cue, NOT a label)"])
    for r, (pid, stratum) in enumerate(chosen, start=1):
        wk.append([f"G{r:03d}", pid, stratum])
    wk.sheet_state = "hidden"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    from collections import Counter
    print(f"Saved {args.out}: {n} papers")
    print("  strata:", dict(Counter(s for _, s in chosen)))


if __name__ == "__main__":
    main()
