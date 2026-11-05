"""
build_llm_review_workbook.py
============================
Manual-review workbook for the LLM-filtered reference network, built the same way as the
earlier review workbook (Assignments / Molecules / Summary) but with the seven LLM
condition labels as the conditions:

  Assignments  random paper-pair assignments, N per LLM label. The reviewer sees the LLM's
               label and the evidence sentence it quoted (highlighted in the abstract) and rates
               (1) organ relationship, (2) LLM label Correct / Wrong / Unsure, (3) the right
               label if wrong, (4) study system.
  Molecules    every molecule the pipeline extracted from each sampled paper (as before).
  Summary      live formulas: % and Wilson 95% CI per metric, overall and per LLM label.

Two stages, one workbook:
  python build_llm_review_workbook.py                       # stage 1: --per-label 5 (default)
  python build_llm_review_workbook.py --top-up --per-label 15
      adds up to (15 - current) assignments per label, drawn ONLY from papers classified after
      the stage-1 snapshot (the "yet to be classified" ones), and keeps every existing row and
      every rating (the workbook is regenerated, ratings carried over by ID).

Sampling: random, one assignment per paper (a paper's label is the same under every pair), only
papers whose LLM answer is usable (valid label + verified evidence sentence).
State (seed, snapshot, drawn rows) lives in validation/llm_review_state.json.
"""

import argparse
import html
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import run_metabolic_lit_search as rl                         # noqa: E402
from Literature_Search import pubmed_search as ps             # noqa: E402
from Literature_Search import llm_condition_filter as cf      # noqa: E402

OUT_DIR = HERE / "validation"
WORKBOOK = OUT_DIR / "reference_network_review_workbook_LLM.xlsx"
STATE = OUT_DIR / "llm_review_state.json"

RELATION_OPTS = ["Confirmed", "Co-regulation", "Co-mention only", "Wrong", "Cannot tell"]
LABEL_CHECK_OPTS = ["Correct", "Wrong", "Unsure"]
SYSTEM_OPTS = ["Human", "Animal", "In vitro", "Review", "Other"]
MOLECULE_OPTS = ["Correct", "Wrong category", "False positive", "Cannot tell"]

# header names of the reviewer columns (also used to carry ratings over on a top-up)
H_REL, H_LAB, H_FIX, H_SYS, H_NOTE = ("REVIEW: organ relationship", "REVIEW: LLM label",
                                      "REVIEW: right label (if wrong)", "REVIEW: study system", "REVIEW: notes")
H_MOL, H_MNOTE = "REVIEW: molecule check", "REVIEW: notes"

HDR_FILL = PatternFill("solid", fgColor="1F3864")
REV_HDR_FILL = PatternFill("solid", fgColor="C55A11")
REV_FILL = PatternFill("solid", fgColor="FFF2CC")
HDR_FONT = Font(bold=True, color="FFFFFF")
WRAP_TOP = Alignment(wrap_text=True, vertical="top")
LINK_FONT = Font(color="0563C1", underline="single")


# ───────────────────────────── helpers ─────────────────────────────────────────

def clean(text: str) -> str:
    # control characters in PubMed text would make openpyxl raise IllegalCharacterError
    return ILLEGAL_CHARACTERS_RE.sub("", html.unescape(text or "")).strip()


def est_height(items: list[tuple[str, float]], line_pt: float = 12.5, min_pt: float = 30.0) -> float:
    lines = 1
    for text, width in items:
        n = sum(max(1, math.ceil(len(p) / max(1.0, width * 1.05))) for p in (text or "").split("\n"))
        lines = max(lines, n)
    return max(min_pt, min(409.0, lines * line_pt))


def evidence_sentences(paper: dict, o1: str, o2: str, organ_pat: dict, crosstalk_pat: list) -> list[str]:
    """Sentences accepted by the pipeline's same-sentence filter (reproduced one-for-one)."""
    text = paper.get("title", "") + ". " + paper.get("abstract", "")
    out = []
    for sent in re.split(r"(?<=[.!?;])\s+", text):
        low = sent.lower()
        if not (rl._any_match(organ_pat[o1], low) and rl._any_match(organ_pat[o2], low)
                and rl._any_match(crosstalk_pat, low)):
            continue
        if not rl._passes_multi_organ_check(low, o1, o2, rl.ALL_ORGAN_PATTERNS):
            continue
        out.append(clean(sent))
    return out


def find_hits(matcher, text_lower: str) -> dict:
    automaton, term_to_canon = matcher
    hits: dict = {}
    for end_idx, term in automaton.iter(text_lower):
        start = end_idx - len(term) + 1
        if start > 0 and ps._is_word_char(text_lower[start - 1]):
            continue
        if end_idx + 1 < len(text_lower) and ps._is_word_char(text_lower[end_idx + 1]):
            continue
        excl = ps._EXCLUDE_FOLLOWING.get(term)
        if excl and text_lower.startswith(excl, end_idx + 1):
            continue
        hits.setdefault(term_to_canon.get(term, term), []).append((start, end_idx + 1, term))
    return hits


def molecule_rows(paper: dict) -> list[dict]:
    """All molecules the pipeline extracts from one paper, with matched context."""
    title, abstract = paper.get("title", ""), paper.get("abstract", "")
    text = title + " " + abstract + " " + " ".join(paper.get("keywords", []))
    low = text.lower()
    same_len = len(low) == len(text)
    abstract_end = len(title) + 1 + len(abstract)
    rows = []
    for category, matcher in (("metabolite", ps._METABOLITE_MATCHER), ("hormone", ps._HORMONE_MATCHER),
                              ("protein", ps._PROTEIN_MATCHER)):
        hits = find_hits(matcher, low)
        assert set(hits) == ps._find_canonical_terms(matcher[0], matcher[1], low), (paper.get("pmid"), category)
        for canon in sorted(hits):
            start, end, _term = hits[canon][0]
            src = text if same_len else low
            ctx = src[max(0, start - 70):start] + "[[" + src[start:end] + "]]" + src[end:end + 70]
            rows.append({"category": category, "molecule": canon, "matched": src[start:end],
                         "context": clean(ctx).replace("\n", " "),
                         "where": "title/abstract" if start < abstract_end else "MeSH keywords only",
                         "mentions": len(hits[canon])})
    return rows


def evidence_spans(evidence: str, text: str) -> list[tuple[int, int]]:
    """Character spans of the LLM's evidence sentence inside the (cleaned) abstract."""
    ev = cf._clean_text(evidence).strip()
    out, pos = [], 0
    for frag in [f for f in cf._ELLIPSIS_RE.split(ev) if re.search(r"\w", f)]:
        toks = re.findall(r"\w+", frag)
        m = re.compile(r"\W+".join(re.escape(t) for t in toks), re.IGNORECASE).search(text, pos) if toks else None
        if not m:
            return []
        out.append(m.span())
        pos = m.end()
    return out


def highlighted(text: str, spans: list) -> "CellRichText | str":
    if not spans:
        return text
    flags = [False] * len(text)
    for a, b in spans:
        for i in range(a, b):
            flags[i] = True
    runs, start = [], 0
    for i in range(1, len(text) + 1):
        if i == len(text) or flags[i] != flags[start]:
            font = InlineFont(sz=9, b=flags[start] or None, color="1F4E9E" if flags[start] else None)
            runs.append(TextBlock(font, text[start:i]))
            start = i
    return CellRichText(*runs)


def write_header(ws, cols: list[dict]) -> None:
    for i, c in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=i, value=c["name"])
        cell.fill = REV_HDR_FILL if c.get("review") else HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = c["width"]
        if c.get("help"):
            cell.comment = Comment(c["help"], "build_llm_review_workbook")
    ws.row_dimensions[1].height = 36
    ws.freeze_panes = "D2"


def add_dropdown(ws, col_idx: int, options: list[str], n_rows: int) -> None:
    letter = ws.cell(row=1, column=col_idx).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Invalid entry", error="Pick a value from the list.")
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{n_rows + 1}")


# ───────────────────────────── sampling / state ─────────────────────────────────

def load_data():
    with open(rl.OUTPUT_JSON, encoding="utf-8") as f:
        results = json.load(f)
    labels = cf.load_condition_labels(rl.CONDITION_LABELS, rl.CONDITION_LLM_MODEL)
    return results, labels


def draw(results: dict, labels: dict, want: dict, exclude_pmids: set, only_pmids: "set | None",
         rng: random.Random) -> list[dict]:
    """Random assignments: `want[label]` per label, one per paper, usable answers only."""
    by_label: dict[str, list[tuple[str, str]]] = {lab: [] for lab in cf.LABELS}
    for key in sorted(results):
        for p in results[key].get("papers", []):
            pid = str(p.get("pmid", ""))
            rec = labels.get(pid)
            if not rec or rec["status"] != cf.OK or pid in exclude_pmids:
                continue
            if only_pmids is not None and pid not in only_pmids:
                continue
            by_label[rec["label"]].append((key, pid))
    rows, taken = [], set()
    for lab in cf.LABELS:
        pool = by_label[lab]
        rng.shuffle(pool)
        n = 0
        for key, pid in pool:
            if n >= want.get(lab, 0):
                break
            if pid in taken:
                continue
            taken.add(pid)
            rows.append({"label": lab, "pair": key, "pmid": pid})
            n += 1
        if n < want.get(lab, 0):
            print(f"[!] {lab}: only {n} of {want[lab]} wanted assignments available")
    rng.shuffle(rows)
    return rows


# ───────────────────────────── workbook ───────────────────────────────────────────

def read_old_ratings(path: Path) -> tuple[dict, dict]:
    """Ratings of an existing workbook: Assignments by ID, Molecules by (ID, molecule, category)."""
    ra, rm = {}, {}
    if not path.exists():
        return ra, rm
    wb = load_workbook(path, read_only=True)
    ws = wb["Assignments"]
    rows = list(ws.iter_rows(values_only=True))
    head = {h: i for i, h in enumerate(rows[0])}
    for r in rows[1:]:
        if r[0]:
            ra[r[0]] = {h: r[head[h]] for h in (H_REL, H_LAB, H_FIX, H_SYS, H_NOTE) if h in head}
    if "Molecules" in wb.sheetnames:
        rows = list(wb["Molecules"].iter_rows(values_only=True))
        head = {h: i for i, h in enumerate(rows[0])}
        for r in rows[1:]:
            if r[0]:
                rm[(r[0], r[head["Molecule (pipeline name)"]], r[head["Category (pipeline)"]])] = {
                    h: r[head[h]] for h in (H_MOL, H_MNOTE) if h in head}
    wb.close()
    return ra, rm


def build_categories(wb: Workbook, n_assign: int, n_mol: int) -> None:
    """Counts and shares of every rating category (live formulas): organ relationship and LLM label per LLM label,
    molecule annotation per molecule category."""
    ws = wb.create_sheet("Categories", 1)
    Ar = lambda c: f"Assignments!${c}$2:${c}${n_assign + 1}"      # noqa: E731
    Mr = lambda c: f"Molecules!${c}$2:${c}${max(n_mol, 1) + 1}"    # noqa: E731
    label_groups = [("All papers", None)] + [(lab, lab) for lab in cf.LABELS]
    mol_groups = [("All molecules", None)] + [(c, c) for c in ("metabolite", "hormone", "protein")]
    ncol = max(len(label_groups), len(mol_groups))

    def count_a(col, value, label):
        if label is None:
            return f'COUNTIF({Ar(col)},"{value}")'
        return f'COUNTIFS({Ar("C")},"{label}",{Ar(col)},"{value}")'

    def count_m(col, value, category):
        if category is None:
            return f'COUNTIF({Mr(col)},"{value}")'
        return f'COUNTIFS({Mr("E")},"{category}",{Mr(col)},"{value}")'

    ws["A1"], ws["A1"].font = "Validation: share of every rating category", Font(bold=True, size=14)
    ws["A2"] = ("Live formulas over the yellow reviewer columns of the Assignments and Molecules sheets. Direct relationship = the "
                "rating 'Confirmed'. The sample has equal numbers per LLM label, so the 'All' columns are not the accuracy over the corpus.")
    ws["A2"].alignment = WRAP_TOP
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol + 1)
    ws.row_dimensions[2].height = 34

    def block(top, title, groups, count, col, cats, unit, note, excluded=None):
        n = len(groups)
        ws.cell(row=top, column=1, value=title).font = HDR_FONT
        for c in range(1, ncol + 2):
            ws.cell(row=top, column=c).fill = HDR_FILL
        ws.cell(row=top + 1, column=1, value="Count").font = Font(bold=True)
        for j, (name, _) in enumerate(groups, 2):
            h = ws.cell(row=top + 1, column=j, value=name)
            h.font = Font(bold=True)
            h.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.row_dimensions[top + 1].height = 32
        r0 = top + 2
        for i, (disp, value) in enumerate(cats):
            ws.cell(row=r0 + i, column=1, value=disp)
            for j, (_, g) in enumerate(groups, 2):
                ws.cell(row=r0 + i, column=j, value="=" + count(col, value, g))
        rt = r0 + len(cats)
        ws.cell(row=rt, column=1, value=f"Rated {unit} (n)").font = Font(bold=True)
        for j in range(2, n + 2):
            L = ws.cell(row=rt, column=j).column_letter
            ws.cell(row=rt, column=j, value=f"=SUM({L}{r0}:{L}{rt - 1})").font = Font(bold=True)
        nxt = rt + 1
        if excluded:
            ws.cell(row=nxt, column=1, value=excluded[0])
            for j, (_, g) in enumerate(groups, 2):
                ws.cell(row=nxt, column=j, value="=" + count(col, excluded[1], g))
            nxt += 1
        ws.cell(row=nxt + 1, column=1, value=f"Share of rated {unit}").font = Font(bold=True)
        rp = nxt + 2
        for i, (disp, _) in enumerate(cats):
            ws.cell(row=rp + i, column=1, value=disp)
            for j in range(2, n + 2):
                L = ws.cell(row=rp + i, column=j).column_letter
                ws.cell(row=rp + i, column=j, value=f'=IF({L}${rt}>0,{L}{r0 + i}/{L}${rt},"")').number_format = "0.0%"
        end = rp + len(cats)
        ws.cell(row=end, column=1, value=note).alignment = WRAP_TOP
        ws.merge_cells(start_row=end, start_column=1, end_row=end, end_column=ncol + 1)
        ws.row_dimensions[end].height = 30
        return r0, rt, end

    rel_cats = [("Direct relationship (Confirmed)", "Confirmed"), ("Co-regulation", "Co-regulation"),
                ("Co-mention only", "Co-mention only"), ("Wrong", "Wrong")]
    r0, rt, end = block(4, "ORGAN RELATIONSHIP", label_groups, count_a, "L", rel_cats, "papers",
                        "Shares of the papers rated with one of the four categories; 'Cannot tell' is not counted.",
                        excluded=("Cannot tell (not counted)", "Cannot tell"))
    lab_cats = [("Correct", "Correct"), ("Unsure", "Unsure"), ("Wrong", "Wrong")]
    l0, lt, lend = block(end + 2, "LLM CONDITION LABEL", label_groups, count_a, "M", lab_cats, "papers",
                         "Shares of all rated papers; Unsure is counted (it is not excluded here).")
    mol_cats = [("Correct", "Correct"), ("Wrong category", "Wrong category"), ("False positive", "False positive")]
    m0, mt, mend = block(lend + 2, "MOLECULE ANNOTATION (by molecule category)", mol_groups, count_m, "J", mol_cats, "molecules",
                         "Shares of the molecules rated Correct / Wrong category / False positive; 'Cannot tell' is not counted.",
                         excluded=("Cannot tell (not counted)", "Cannot tell"))

    # 95% CI of the overall shares (Wilson, z from the Summary sheet)
    top = mend + 2
    ws.cell(row=top, column=1, value="OVERALL SHARES WITH 95% CONFIDENCE INTERVAL (Wilson)").font = HDR_FONT
    for c in range(1, 7):
        ws.cell(row=top, column=c).fill = HDR_FILL
    for c, h in enumerate(["Category", "k", "n", "Share", "95% CI low", "95% CI high"], 1):
        ws.cell(row=top + 1, column=c, value=h).font = Font(bold=True)
    r = top + 2
    for title, first, last, ntot in (("organ relationship", r0, rt - 1, rt), ("LLM label", l0, lt - 1, lt),
                                     ("molecule", m0, mt - 1, mt)):
        for k_row in range(first, last + 1):
            ws.cell(row=r, column=1, value=f'="{title}: "&A{k_row}')
            ws.cell(row=r, column=2, value=f"=B{k_row}")
            ws.cell(row=r, column=3, value=f"=B{ntot}")
            ws.cell(row=r, column=4, value=f'=IF(C{r}>0,B{r}/C{r},"")').number_format = "0.0%"
            core = f"D{r}*(1-D{r})/C{r}+Summary!$B$3^2/(4*C{r}^2)"
            ws.cell(row=r, column=5, value=f'=IF(C{r}>0,(D{r}+Summary!$B$3^2/(2*C{r})-Summary!$B$3*SQRT({core}))/(1+Summary!$B$3^2/C{r}),"")').number_format = "0.0%"
            ws.cell(row=r, column=6, value=f'=IF(C{r}>0,(D{r}+Summary!$B$3^2/(2*C{r})+Summary!$B$3*SQRT({core}))/(1+Summary!$B$3^2/C{r}),"")').number_format = "0.0%"
            r += 1
    ws.column_dimensions["A"].width = 44
    for j in range(2, ncol + 2):
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = 16
    ws.freeze_panes = "B4"


def build_summary(wb: Workbook, n_assign: int, n_mol: int) -> None:
    ws = wb.create_sheet("Summary", 0)
    Ar = lambda c: f"Assignments!${c}$2:${c}${n_assign + 1}"      # noqa: E731
    Mr = lambda c: f"Molecules!${c}$2:${c}${max(n_mol, 1) + 1}"    # noqa: E731

    def cnt(conds):
        return "COUNTIFS(" + ",".join(f'{r},"{c}"' for r, c in conds) + ")"

    def total(col, base, values):
        return "+".join(cnt(base + [(col, v)]) for v in values)

    ws["A1"], ws["A1"].font = "Review summary: LLM condition filter", Font(bold=True, size=14)
    ws["A2"] = "Live formulas over the yellow reviewer columns. Intervals are Wilson 95% score intervals."
    ws["A3"], ws["B3"] = "z (95% CI)", 1.96
    ws["A5"] = "PROGRESS"
    for c, h in enumerate(["Sheet / rating", "Rated", "Total", "% rated"], 1):
        ws.cell(row=6, column=c, value=h)
    prog = [("Assignments: organ relationship", f"=COUNTA({Ar('L')})", n_assign),
            ("Assignments: LLM label check", f"=COUNTA({Ar('M')})", n_assign),
            ("Molecules", f"=COUNTA({Mr('J')})", n_mol)]
    for i, (name, f, tot) in enumerate(prog, 7):
        ws.cell(row=i, column=1, value=name); ws.cell(row=i, column=2, value=f); ws.cell(row=i, column=3, value=tot)
        ws.cell(row=i, column=4, value=f'=IF(C{i}>0,B{i}/C{i},"")').number_format = "0%"

    hdr = 12
    ws.cell(row=hdr - 1, column=1, value="RESULTS")
    for c, h in enumerate(["Metric", "Group", "k (successes)", "n (denominator)", "Proportion", "95% CI low",
                           "95% CI high", "Excluded from n", "Report as"], 1):
        ws.cell(row=hdr, column=c, value=h)

    groups = [("All", "*")] + [(lab, lab) for lab in cf.LABELS]
    mgroups = [("All", [(Mr("B"), "*")])] + [(c, [(Mr("E"), c)]) for c in ("metabolite", "hormone", "protein")] \
        + [(lab, [(Mr("B"), lab)]) for lab in cf.LABELS]
    REL = ["Confirmed", "Co-regulation", "Co-mention only", "Wrong"]
    MOL = ["Correct", "Wrong category", "False positive"]
    rows: list[dict] = []
    rows.append({"section": "LLM condition label (as rated by the reviewer)"})
    for g, crit in groups:
        b = [(Ar("C"), crit)]
        rows.append({"metric": "LLM label correct (Correct of Correct+Wrong)", "group": g, "key": ("lab", g),
                     "k": cnt(b + [(Ar("M"), "Correct")]), "n": total(Ar("M"), b, ["Correct", "Wrong"]),
                     "x": cnt(b + [(Ar("M"), "Unsure")])})
    rows.append({"section": "Organ relationship"})
    for g, crit in groups:
        b = [(Ar("C"), crit)]
        rows.append({"metric": "Organ relationship: direct relationship (Confirmed)", "group": g, "key": ("rel", g),
                     "k": cnt(b + [(Ar("L"), "Confirmed")]), "n": total(Ar("L"), b, REL),
                     "x": cnt(b + [(Ar("L"), "Cannot tell")])})
    for g, crit in groups:
        b = [(Ar("C"), crit)]
        rows.append({"metric": "Organ relationship: co-regulation (same effector)", "group": g, "key": ("coreg", g),
                     "k": cnt(b + [(Ar("L"), "Co-regulation")]), "n": total(Ar("L"), b, REL),
                     "x": cnt(b + [(Ar("L"), "Cannot tell")])})
    for g, crit in groups:
        b = [(Ar("C"), crit)]
        rows.append({"metric": "Organ relationship: co-mention only", "group": g, "key": ("comention", g),
                     "k": cnt(b + [(Ar("L"), "Co-mention only")]), "n": total(Ar("L"), b, REL),
                     "x": cnt(b + [(Ar("L"), "Cannot tell")])})
    for g, crit in groups:
        b = [(Ar("C"), crit)]
        rows.append({"metric": "Organ relationship: wrong", "group": g, "key": ("wrong", g),
                     "k": cnt(b + [(Ar("L"), "Wrong")]), "n": total(Ar("L"), b, REL),
                     "x": cnt(b + [(Ar("L"), "Cannot tell")])})
    rows.append({"section": "Molecules (group = category or LLM label of the paper)"})
    for g, base in mgroups:
        rows.append({"metric": "Molecule annotation correct (strict: Correct)", "group": g, "key": ("mol", g),
                     "k": cnt(base + [(Mr("J"), "Correct")]), "n": total(Mr("J"), base, MOL),
                     "x": cnt(base + [(Mr("J"), "Cannot tell")])})
    for g, base in mgroups:
        rows.append({"metric": "Real molecule (Correct or Wrong category)", "group": g,
                     "k": total(Mr("J"), base, ["Correct", "Wrong category"]), "n": total(Mr("J"), base, MOL),
                     "x": cnt(base + [(Mr("J"), "Cannot tell")])})

    key_row, sections, r = {}, [], hdr + 1
    for item in rows:
        if "section" in item:
            ws.cell(row=r, column=1, value=item["section"]); sections.append(r); r += 1
            continue
        ws.cell(row=r, column=1, value=item["metric"]); ws.cell(row=r, column=2, value=item["group"])
        ws.cell(row=r, column=3, value="=" + item["k"]); ws.cell(row=r, column=4, value="=" + item["n"])
        ws.cell(row=r, column=5, value=f'=IF(D{r}>0,C{r}/D{r},"")').number_format = "0.0%"
        core = f"E{r}*(1-E{r})/D{r}+$B$3^2/(4*D{r}^2)"
        ws.cell(row=r, column=6, value=f'=IF(D{r}>0,(E{r}+$B$3^2/(2*D{r})-$B$3*SQRT({core}))/(1+$B$3^2/D{r}),"")').number_format = "0.0%"
        ws.cell(row=r, column=7, value=f'=IF(D{r}>0,(E{r}+$B$3^2/(2*D{r})+$B$3*SQRT({core}))/(1+$B$3^2/D{r}),"")').number_format = "0.0%"
        ws.cell(row=r, column=8, value="=" + item["x"])
        ws.cell(row=r, column=9, value=(f'=IF(D{r}=0,"",C{r}&"/"&D{r}&" ("&TEXT(E{r},"0%")&"; 95% CI "'
                                        f'&TEXT(F{r},"0%")&"–"&TEXT(G{r},"0%")&")")'))
        if "key" in item:
            key_row[item["key"]] = r
        r += 1

    r += 1
    ws.cell(row=r, column=1, value="DRAFT RESULTS SENTENCE (fills in from the cells above)")
    sent_hdr = r
    rel, lab, mol = key_row[("rel", "All")], key_row[("lab", "All")], key_row[("mol", "All")]
    coreg = key_row[("coreg", "All")]
    ci = lambda x: f'"95% CI "&TEXT(F{x},"0%")&"–"&TEXT(G{x},"0%")'    # noqa: E731
    sentence = (f'=IF(OR(D{rel}=0,D{lab}=0,D{mol}=0),"(Rate all three columns to generate this sentence.)",'
                f'"Independent review of "&D{rel}&" retrieved paper-pair assignments, sampled across the seven LLM '
                f'condition labels, confirmed a relevant organ relationship in "&TEXT(E{rel},"0%")&" ("&{ci(rel)}&") and showed '
                f'co-regulation of both organs by the same effector in a further "&TEXT(E{coreg},"0%")&". '
                f'The LLM condition label was judged correct in "&TEXT(E{lab},"0%")&" of "&D{lab}&" rated papers ("&{ci(lab)}'
                f'&"), and molecular annotations were correct in "&TEXT(E{mol},"0%")&" of "&D{mol}&" reviewed instances ("&{ci(mol)}&").")')
    ws.cell(row=r + 1, column=1, value=sentence)
    ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=9)
    ws.cell(row=r + 1, column=1).alignment = WRAP_TOP
    ws.row_dimensions[r + 1].height = 95
    notes = ["NOTES",
             "n excludes 'Unsure' / 'Cannot tell' (see 'Excluded from n'). 'Correct of Correct+Wrong' is the precision of the "
             "LLM's label as judged by you; per-label rows show where the LLM is weak. Rows are not independent "
             "(molecules per paper, papers per pair); the CI is a simple binomial interval.",
             "'Co-regulation' (the two organs are influenced by the same effector) counts in n like the other relationship "
             "ratings but is not counted as 'confirmed'; it has its own rows.",
             "The sample is stratified by LLM label (equal numbers per label), so 'All' is NOT the accuracy over the corpus: "
             "the corpus has far more 'Other diseases' / 'Obesity' papers than 'Healthy' ones. Report per-label numbers."]
    nr = sent_hdr + 3
    for i, t in enumerate(notes):
        c = ws.cell(row=nr + i, column=1, value=t); c.alignment = WRAP_TOP
        if i == 0:
            c.font = HDR_FONT
            for cc in range(1, 10):
                ws.cell(row=nr, column=cc).fill = HDR_FILL
        else:
            ws.merge_cells(start_row=nr + i, start_column=1, end_row=nr + i, end_column=9)
            ws.row_dimensions[nr + i].height = 52
    for col, w in zip("ABCDEFGHI", [52, 38, 13, 15, 11, 11, 11, 15, 34]):
        ws.column_dimensions[col].width = w
    for rr in (5, hdr - 1, sent_hdr):
        for cc in range(1, 10):
            ws.cell(row=rr, column=cc).fill = HDR_FILL
        ws.cell(row=rr, column=1).font = HDR_FONT
    for rr, upto in ((6, 5), (hdr, 10)):
        for cc in range(1, upto):
            cell = ws.cell(row=rr, column=cc)
            cell.font = Font(bold=True); cell.fill = PatternFill("solid", fgColor="D9E1F2")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
    for rr in sections:
        for cc in range(1, 10):
            ws.cell(row=rr, column=cc).fill = PatternFill("solid", fgColor="EDEDED")
            ws.cell(row=rr, column=cc).font = Font(bold=True)
    ws.freeze_panes = ws.cell(row=hdr + 1, column=1)


def write_workbook(path: Path, state: dict, results: dict, labels: dict) -> dict:
    ratings_a, ratings_m = read_old_ratings(path)
    organ_pat = {o: rl._compile_patterns(a) for o, a in rl.ORGAN_ALIASES.items()}
    crosstalk_pat = rl._compile_patterns(rl.CROSSTALK_KEYWORDS)
    papers = {}
    for key, entry in results.items():
        for p in entry.get("papers", []):
            papers.setdefault((key, str(p.get("pmid", ""))), p)

    wb = Workbook()
    wa = wb.active
    wa.title = "Assignments"
    acols = [
        {"name": "ID", "width": 7}, {"name": "Stage", "width": 6},
        {"name": "LLM condition label", "width": 22}, {"name": "Organ 1", "width": 14}, {"name": "Organ 2", "width": 14},
        {"name": "PMID", "width": 10}, {"name": "Year", "width": 6}, {"name": "Title", "width": 38},
        {"name": "Evidence sentence quoted by the LLM", "width": 44},
        {"name": "Sentence(s) the same-sentence filter accepted", "width": 44},
        {"name": "Abstract (LLM evidence in blue bold)", "width": 95},
        {"name": H_REL, "width": 16, "review": True,
         "help": "Confirmed = the paper describes a metabolic / crosstalk relationship between the two organs.\n"
                 "Co-regulation = the two organs are influenced by the same effector in this paper (the same hormone, "
                 "drug, diet, gene, ... acts on both), without an interaction between the organs themselves.\n"
                 "Co-mention only = both organs named, no relationship between them.\nWrong = an organ term means "
                 "something else, or the relationship is not between these organs.\nCannot tell."},
        {"name": H_LAB, "width": 14, "review": True,
         "help": "Is the LLM's condition label (column C) right for the studied subjects of this paper "
                 "(for a review: the condition it is about)?\nCorrect / Wrong / Unsure. Healthy = healthy people or normal "
                 "animals as the studied subjects, not merely controls."},
        {"name": H_FIX, "width": 24, "review": True, "help": "Only if 'Wrong': the label you would give."},
        {"name": H_SYS, "width": 11, "review": True}, {"name": H_NOTE, "width": 26, "review": True},
    ]
    write_header(wa, acols)
    rows = [(st["stage"], r) for st in state["stages"] for r in st["rows"]]
    for i, (stage, r) in enumerate(rows, start=2):
        p = papers[(r["pair"], r["pmid"])]
        o1, o2 = results[r["pair"]]["organ1"], results[r["pair"]]["organ2"]
        rec = labels.get(r["pmid"], {})
        abstract = cf._clean_text(p.get("abstract", "")).strip()
        ev = rec.get("evidence", "")
        filt = evidence_sentences(p, o1, o2, organ_pat, crosstalk_pat)
        filt_text = "\n".join(f"• {s}" for s in filt[:3]) if filt else "(none reproduced)"
        old = ratings_a.get(r["id"], {})
        vals = [r["id"], stage, r["label"], o1, o2, r["pmid"], p.get("year", ""), clean(p.get("title", "")), ev,
                filt_text, highlighted(abstract, evidence_spans(ev, abstract)),
                old.get(H_REL), old.get(H_LAB), old.get(H_FIX), old.get(H_SYS), old.get(H_NOTE)]
        for c, v in enumerate(vals, 1):
            cell = wa.cell(row=i, column=c, value=v)
            cell.alignment = WRAP_TOP
            if acols[c - 1].get("review"):
                cell.fill = REV_FILL
            if c == 11 and not isinstance(v, CellRichText):
                cell.font = Font(size=9)
        link = wa.cell(row=i, column=6)
        link.hyperlink = f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/"
        link.font = LINK_FONT
        wa.row_dimensions[i].height = est_height([(abstract, 95 * 1.2), (ev, 44), (filt_text, 44),
                                                  (clean(p.get("title", "")), 38)], min_pt=60)
    n_assign = len(rows)
    add_dropdown(wa, 12, RELATION_OPTS, n_assign)
    add_dropdown(wa, 13, LABEL_CHECK_OPTS, n_assign)
    add_dropdown(wa, 14, cf.LABELS, n_assign)
    add_dropdown(wa, 15, SYSTEM_OPTS, n_assign)
    wa.auto_filter.ref = f"A1:P{n_assign + 1}"

    wm = wb.create_sheet("Molecules")
    mcols = [{"name": "Assignment ID", "width": 11}, {"name": "LLM condition label", "width": 22},
             {"name": "PMID", "width": 10}, {"name": "Molecule (pipeline name)", "width": 28},
             {"name": "Category (pipeline)", "width": 12}, {"name": "Matched text", "width": 18},
             {"name": "Found in", "width": 18}, {"name": "Mentions", "width": 9},
             {"name": "Context (match in [[ ]])", "width": 80},
             {"name": H_MOL, "width": 16, "review": True,
              "help": "Correct = the text really names this molecule and the category fits.\nWrong category = real "
                      "molecule, filed under the wrong category.\nFalse positive = not this molecule (e.g. 'AMP' inside "
                      "'AMPK', a gene-symbol collision, a generic word).\nCannot tell."},
             {"name": H_MNOTE, "width": 28, "review": True}]
    write_header(wm, mcols)
    mr = 2
    for stage, r in rows:
        p = papers[(r["pair"], r["pmid"])]
        for m in molecule_rows(p):
            old = ratings_m.get((r["id"], m["molecule"], m["category"]), {})
            vals = [r["id"], r["label"], r["pmid"], m["molecule"], m["category"], m["matched"], m["where"],
                    m["mentions"], m["context"], old.get(H_MOL), old.get(H_MNOTE)]
            for c, v in enumerate(vals, 1):
                cell = wm.cell(row=mr, column=c, value=v)
                cell.alignment = WRAP_TOP
                if mcols[c - 1].get("review"):
                    cell.fill = REV_FILL
                if c == 9:
                    cell.font = Font(size=9)
            wm.row_dimensions[mr].height = est_height([(m["context"], 80 * 1.2)], min_pt=28)
            mr += 1
    n_mol = mr - 2
    if n_mol:
        add_dropdown(wm, 10, MOLECULE_OPTS, n_mol)
        wm.auto_filter.ref = f"A1:K{n_mol + 1}"

    build_summary(wb, n_assign, n_mol)
    build_categories(wb, n_assign, n_mol)

    wr = wb.create_sheet("README", 0)
    wr.column_dimensions["A"].width = 30
    wr.column_dimensions["B"].width = 120
    per_label = {lab: sum(1 for _, r in rows if r["label"] == lab) for lab in cf.LABELS}
    lines = [("Reference network review: LLM condition filter", None, "title"),
             ("Built", datetime.now().strftime("%Y-%m-%d %H:%M"), None),
             ("Model / prompt", f"{rl.CONDITION_LLM_MODEL} / prompt {cf.PROMPT_VERSION}", None),
             ("Sampling", f"Random paper-pair assignments, one per paper, equal numbers per LLM label (seed "
                          f"{state['seed']}); only papers with a usable LLM answer (valid label + verified evidence). "
                          "Stage 1 = papers classified when this workbook was first built; later stages only add papers "
                          "classified afterwards. Existing rows and ratings are never changed.", None),
             ("Rows now", " | ".join(f"{lab}: {per_label[lab]}" for lab in cf.LABELS) + f" | total {n_assign}", None),
             ("", "", None), ("SHEETS", None, "section"),
             ("Assignments", "For each paper rate (1) organ relationship (Confirmed / Co-regulation = both organs influenced by the same effector / Co-mention only / Wrong / Cannot tell), (2) whether the LLM's condition label is Correct / "
                             "Wrong / Unsure, (3) the right label if wrong, (4) study system. The LLM's label and the sentence "
                             "it quoted are shown (evidence highlighted in the abstract); judge the paper by its abstract.", None),
             ("Molecules", "Every molecule the pipeline extracted from the sampled papers; rate Correct / Wrong category / "
                           "False positive.", None),
             ("Summary", "Live formulas: % and Wilson 95% CI per rating, overall and per LLM label, progress, draft sentence.", None),
             ("Categories", "Counts and shares of every category: organ relationship (direct relationship / co-regulation / co-mention only / "
                            "wrong) and LLM label (correct / unsure / wrong) per LLM label, molecule annotation (correct / wrong category / "
                            "false positive) per molecule category, with 95% intervals.", None),
             ("", "", None), ("THE SEVEN LABELS", None, "section")]
    lines += [(lab, cf.LABEL_DESCRIPTIONS[lab], None) for lab in cf.LABELS]
    lines += [("", "", None), ("ADDING MORE ROWS", None, "section"),
              ("Top-up", "python build_llm_review_workbook.py --top-up --per-label 15   (after more papers are classified). "
                         "Close the workbook in Excel first; ratings are carried over by ID.", None),
              ("Rewrite", "python build_llm_review_workbook.py --rewrite   (rebuilds the workbook from the state file after a change "
                          "of the layout or of the rating options; all rows and ratings are kept).", None)]
    for i, (a, b, kind) in enumerate(lines, 1):
        ca, cb = wr.cell(row=i, column=1, value=a), wr.cell(row=i, column=2, value=b)
        ca.alignment = cb.alignment = WRAP_TOP
        if kind == "title":
            ca.font = Font(bold=True, size=14)
        elif kind == "section":
            ca.font = HDR_FONT; ca.fill = cb.fill = HDR_FILL
        else:
            ca.font = Font(bold=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return {"assignments": n_assign, "molecules": n_mol, "per_label": per_label}


# ───────────────────────────── CLI ───────────────────────────────────────────────

def refill(args, results: dict, labels: dict) -> dict:
    """Keep the sample valid after the filtered paper set changed: rows whose (pair, paper) assignment no
    longer exists are removed (and written, with their ratings, to a CSV); replacements of the same label are
    drawn at random from the current set. The kept rows are still a uniform random sample of the new set."""
    import csv
    import shutil
    state = json.loads(args.state.read_text(encoding="utf-8"))
    valid = {k: {str(p.get("pmid", "")) for p in e.get("papers", [])} for k, e in results.items()}
    kept_rows, removed = 0, []
    for st in state["stages"]:
        keep = []
        for r in st["rows"]:
            (keep if str(r["pmid"]) in valid.get(r["pair"], set()) else removed).append(r)
        kept_rows += len(keep)
        st["rows"] = keep
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.workbook.parent / f"before_refill_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in (args.workbook, args.state):
        if f.exists():
            shutil.copy2(f, backup_dir / f.name)
    # the removed rows with their ratings and titles
    ratings_a, _ = read_old_ratings(args.workbook)
    titles = {}
    if args.workbook.exists():
        ws = load_workbook(args.workbook, read_only=True)["Assignments"]
        rows_ = list(ws.iter_rows(values_only=True))
        head = {h: i for i, h in enumerate(rows_[0])}
        titles = {r[0]: r[head["Title"]] for r in rows_[1:] if r[0]}
    csv_path = args.workbook.parent / f"removed_rows_after_filter_change_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "LLM condition label", "pair", "PMID", "title", H_REL, H_LAB, H_FIX, H_SYS, H_NOTE])
        for r in removed:
            old = ratings_a.get(r["id"], {})
            w.writerow([r["id"], r["label"], r["pair"], r["pmid"], titles.get(r["id"], ""),
                        *[old.get(h, "") or "" for h in (H_REL, H_LAB, H_FIX, H_SYS, H_NOTE)]])
    target = state["stages"][0].get("per_label_target", 20)
    have = {lab: sum(1 for st in state["stages"] for r in st["rows"] if r["label"] == lab) for lab in cf.LABELS}
    want = {lab: max(0, target - have[lab]) for lab in cf.LABELS}
    used = {str(r["pmid"]) for st in state["stages"] for r in st["rows"]} | {str(r["pmid"]) for r in removed}
    stage_no = len(state["stages"]) + 1
    rng = random.Random(args.seed + 1000 + stage_no)
    rows = draw(results, labels, want, used, None, rng)
    all_ids = [int(r["id"][1:]) for st in state["stages"] for r in st["rows"]] + [int(r["id"][1:]) for r in removed]
    for i, r in enumerate(rows, max(all_ids, default=0) + 1):
        r["id"] = f"R{i:03d}"
    state["stages"].append({"stage": stage_no, "built": datetime.now().isoformat(timespec="seconds"),
                            "per_label_target": target, "refill": True, "rows": rows})
    state.setdefault("removed", []).extend({**r, "removed": datetime.now().isoformat(timespec="seconds")} for r in removed)
    rated = sum(1 for r in removed if any((ratings_a.get(r["id"], {}) or {}).values()))
    print(f"refill: kept {kept_rows} rows, removed {len(removed)} ({rated} of them rated -> {csv_path.name}), "
          f"drew {len(rows)} replacements; backup in {backup_dir.name}/")
    return state


def _assert_closed(path: Path) -> None:
    for lock in (path.with_name(f"~${path.name}"), path.with_name(f".~lock.{path.name}#")):
        if lock.exists():
            sys.exit(f"{path.name} is open in Excel / LibreOffice ({lock.name} exists): save and close it first.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-label", type=int, default=None, help="target assignments per label (default 5; 15 with --top-up)")
    ap.add_argument("--top-up", action="store_true", help="add rows from papers classified after the stage-1 snapshot")
    ap.add_argument("--rewrite", action="store_true",
                    help="rebuild the workbook from the state file (new layout / rating options) keeping all rows and ratings")
    ap.add_argument("--refill", action="store_true",
                    help="after the paper set changed (e.g. a crosstalk keyword was removed): remove the rows whose paper is "
                         "no longer assigned to its organ pair (saved with their ratings to a CSV) and draw random "
                         "replacements of the same label; ratings of all other rows are kept")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workbook", type=Path, default=WORKBOOK)
    ap.add_argument("--state", type=Path, default=STATE)
    args = ap.parse_args()

    results, labels = load_data()
    _assert_closed(args.workbook)
    if args.rewrite:
        import shutil
        if not (args.state.exists() and args.workbook.exists()):
            sys.exit("No workbook/state found; nothing to rewrite.")
        backup_dir = args.workbook.parent / f"before_rewrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in (args.workbook, args.state):
            shutil.copy2(f, backup_dir / f.name)
        state = json.loads(args.state.read_text(encoding="utf-8"))
        print(f"rewrite: {sum(len(st['rows']) for st in state['stages'])} rows kept, ratings carried over; backup in {backup_dir.name}/")
    elif args.refill:
        if not args.state.exists():
            sys.exit("No state file found; nothing to refill.")
        state = refill(args, results, labels)
    elif not args.top_up:
        if args.state.exists() or args.workbook.exists():
            sys.exit(f"{args.workbook.name} / state already exist. Use --top-up to add rows (ratings are kept), "
                     f"or delete both files to start over.")
        n = args.per_label or 5
        rng = random.Random(args.seed)
        rows = draw(results, labels, {lab: n for lab in cf.LABELS}, set(), None, rng)
        for i, r in enumerate(rows, 1):
            r["id"] = f"R{i:03d}"
        state = {"seed": args.seed, "created": datetime.now().isoformat(timespec="seconds"),
                 "first_snapshot": sorted(labels),            # every paper that had ANY record at stage 1
                 "stages": [{"stage": 1, "built": datetime.now().isoformat(timespec="seconds"),
                             "per_label_target": n, "rows": rows}]}
    else:
        if not args.state.exists():
            sys.exit("No stage-1 state found; run without --top-up first.")
        state = json.loads(args.state.read_text(encoding="utf-8"))
        target = args.per_label or 15
        have = {lab: sum(1 for st in state["stages"] for r in st["rows"] if r["label"] == lab) for lab in cf.LABELS}
        want = {lab: max(0, target - have[lab]) for lab in cf.LABELS}
        snapshot = set(state["first_snapshot"])
        newer = {pid for pid in labels if pid not in snapshot}            # classified after stage 1
        used = {r["pmid"] for st in state["stages"] for r in st["rows"]}
        stage_no = len(state["stages"]) + 1
        rng = random.Random(args.seed + stage_no)
        rows = draw(results, labels, want, used, newer, rng)
        start = sum(len(st["rows"]) for st in state["stages"])
        for i, r in enumerate(rows, start + 1):
            r["id"] = f"R{i:03d}"
        print(f"papers classified since stage 1: {len(newer)}; wanted {sum(want.values())}, drew {len(rows)}")
        state["stages"].append({"stage": stage_no, "built": datetime.now().isoformat(timespec="seconds"),
                                "per_label_target": target, "rows": rows})
    info = write_workbook(args.workbook, state, results, labels)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state), encoding="utf-8")
    print(f"saved {args.workbook}")
    print(f"  assignments {info['assignments']}, molecule rows {info['molecules']}")
    print("  per label:", info["per_label"])


if __name__ == "__main__":
    main()
