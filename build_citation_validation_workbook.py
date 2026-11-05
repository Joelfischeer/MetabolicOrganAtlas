"""
build_citation_validation_workbook.py
=====================================
Manual check of the LLM-generated organ-organ descriptions: are the sentences really supported by the papers they cite?

The descriptions (Claude Haiku with the citations feature) are lists of sentences, every sentence followed by reference numbers
[n] that point to the source papers of that description. This script draws N random sentences (default 30) from all valid cited
descriptions of the dashboard (the "All papers" view and the seven condition views) and writes the workbook
validation/citation_validation_workbook.xlsx:

  Sentences   one row per sampled sentence (as shown in the dashboard) with its cited papers; the reviewer rates whether the
              sentence is supported by the cited papers (Supported / Partly supported / Not supported / Cannot tell) and the main
              problem.
  Citations   one row per sentence-paper pair: the sentence next to the title + abstract of the cited paper, the passages the model
              cited highlighted in blue bold; optional rating per paper.
  Summary     live formulas: shares with Wilson 95% intervals, problem types, progress, draft results sentence.
  README      how the sample was drawn and how to rate.

Sampling: uniform over all sentences of the frame (sentences with at least one valid reference number), without replacement.
The sampled sentences and the papers they cite are stored in validation/citation_validation_state.json, so the workbook can be
rebuilt (--rewrite, ratings carried over by ID) and stays what it was when drawn even if descriptions are regenerated later;
--check tells whether the sampled sentences are still in the dashboard's descriptions.

  python build_citation_validation_workbook.py                 # draw 30 sentences (seed 2026) and write the workbook
  python build_citation_validation_workbook.py --n 40 --seed 7
  python build_citation_validation_workbook.py --rewrite       # rebuild from the state file, keep ratings
  python build_citation_validation_workbook.py --check         # are the sampled sentences still in the descriptions?
"""
import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_llm_review_workbook as bw                       # noqa: E402  (styles, helpers, lock check)
import run_metabolic_lit_search as rl                        # noqa: E402
from Literature_Search import claude_descriptions as cd      # noqa: E402
from Literature_Search import llm_condition_filter as cf     # noqa: E402

D = HERE / "reference_network_only_metabolic"
OUT_DIR = HERE / "validation"
WORKBOOK = OUT_DIR / "citation_validation_workbook.xlsx"
STATE = OUT_DIR / "citation_validation_state.json"

SUPPORT_OPTS = ["Supported", "Partly supported", "Not supported", "Cannot tell"]
PROBLEM_OPTS = ["Claim not in the cited papers", "Overstated or generalised", "Wrong organ / molecule / species",
                "Wrong direction or mechanism", "Combines findings that no single paper states", "Other"]
PAPER_OPTS = ["Yes", "Partly", "No", "Cannot tell"]
H_SUP, H_PROB, H_NOTE = "REVIEW: sentence supported by the cited papers?", "REVIEW: main problem (if not fully supported)", "REVIEW: notes"
H_PSUP, H_PNOTE = "REVIEW: does this paper support the sentence?", "REVIEW: notes"
VIEWS = [("all", "All papers")] + [(c["slug"], c["name"]) for c in cf.CONDITIONS]


# ───────────────────────────── frame and sampling ─────────────────────────────────

def description_file(slug: str) -> Path:
    return D / f"metabolic_llm_descriptions{'' if slug == 'all' else '_' + slug}.json"


def build_frame() -> tuple[list[dict], dict]:
    """Every sentence with at least one valid reference number of every valid cited pair description."""
    frame, info = [], {"descriptions": 0, "sentences": 0, "skipped sentences": 0, "per view": {}}
    for slug, name in VIEWS:
        path = description_file(slug)
        if not path.exists():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        n_desc = n_sent = 0
        for pair, e in entries.items():
            papers, text = e.get("papers") or [], (e.get("description") or "").strip()
            if not text or not papers or not e.get("citations"):
                continue
            n_desc += 1
            for i, sent in enumerate(cd._split_sentences(text)):
                nums = sorted({int(n) for m in cd._MARKER_RE.findall(sent) for n in re.findall(r"\d+", m)})
                if not nums or any(not 1 <= n <= len(papers) for n in nums):
                    info["skipped sentences"] += 1
                    continue
                n_sent += 1
                frame.append({"slug": slug, "view": name, "pair": pair, "index": i, "sentence": sent, "nums": nums,
                              "generated_at": e.get("generated_at", ""), "model": e.get("model", "")})
        info["per view"][name] = {"descriptions": n_desc, "sentences": n_sent}
        info["descriptions"] += n_desc
        info["sentences"] += n_sent
    return frame, info


def cited_papers(slug: str, pair: str, nums: list[int]) -> list[dict]:
    e = json.loads(description_file(slug).read_text(encoding="utf-8"))[pair]
    out = []
    for n in nums:
        p = e["papers"][n - 1]
        quotes = [c["quote"] for c in e["citations"] if c["n"] == n]
        out.append({"n": n, "pmid": str(p.get("pmid", "")), "year": str(p.get("year", "")), "doi": p.get("doi", ""),
                    "title": p.get("title", ""), "abstract": p.get("abstract", ""), "quotes": quotes})
    return out


def draw_sample(n: int, seed: int) -> dict:
    frame, info = build_frame()
    rows = random.Random(seed).sample(frame, n)
    for i, r in enumerate(rows, 1):
        r["id"] = f"C{i:03d}"
        r["papers"] = cited_papers(r["slug"], r["pair"], r["nums"])
        r["sha"] = hashlib.sha1(r["sentence"].encode("utf-8")).hexdigest()[:12]
    return {"seed": seed, "n": n, "created": datetime.now().isoformat(timespec="seconds"), "frame": info, "rows": rows}


# ───────────────────────────── workbook ─────────────────────────────────────────

def read_old_ratings(path: Path) -> tuple[dict, dict]:
    sent, cit = {}, {}
    if not path.exists():
        return sent, cit
    wb = load_workbook(path, read_only=True)
    ws = wb["Sentences"]
    rows = list(ws.iter_rows(values_only=True))
    head = {h: i for i, h in enumerate(rows[0])}
    for r in rows[1:]:
        if r[0]:
            sent[r[0]] = {h: r[head[h]] for h in (H_SUP, H_PROB, H_NOTE)}
    ws = wb["Citations"]
    rows = list(ws.iter_rows(values_only=True))
    head = {h: i for i, h in enumerate(rows[0])}
    for r in rows[1:]:
        if r[0]:
            cit[(r[0], r[head["Ref no."]])] = {h: r[head[h]] for h in (H_PSUP, H_PNOTE)}
    return sent, cit


def abstract_cell(paper: dict):
    title, abstract = bw.clean(paper["title"]), bw.clean(paper["abstract"])
    text = f"{title}\n\n{abstract}" if abstract else title
    spans = []
    for q in paper["quotes"]:
        spans += bw.evidence_spans(q, text)
    return bw.highlighted(text, sorted(spans)), text, sum(1 for q in paper["quotes"] if bw.evidence_spans(q, text))


def write_workbook(path: Path, state: dict, old_sent: dict, old_cit: dict) -> dict:
    rows = state["rows"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Sentences"
    cols = [
        {"name": "ID", "width": 7},
        {"name": "View", "width": 16},
        {"name": "Organ 1", "width": 14},
        {"name": "Organ 2", "width": 14},
        {"name": "Sentence (as shown in the dashboard)", "width": 70,
         "help": "One sentence of an LLM-generated organ-organ description. [n] = the cited paper(s) of that description."},
        {"name": "Cited papers", "width": 62, "help": "Reference number, PMID, year and title of every paper the sentence cites. "
                                                      "The abstracts are on the sheet Citations."},
        {"name": H_SUP, "width": 20, "review": True,
         "help": "Supported = everything the sentence states is stated in (or follows directly from) the cited papers. "
                 "Partly supported = part of it is, another part (a detail, a generalisation, a mechanism, an organ) is not. "
                 "Not supported = the cited papers do not state it or contradict it. Cannot tell = the abstracts are not enough."},
        {"name": H_PROB, "width": 30, "review": True, "help": "Only if the sentence is not fully supported: the main reason."},
        {"name": H_NOTE, "width": 40, "review": True},
        {"name": "Description generated", "width": 18},
    ]
    bw.write_header(ws, cols)
    ws.freeze_panes = "F2"
    for i, r in enumerate(rows, start=2):
        old = old_sent.get(r["id"], {})
        refs = "\n".join(f"[{p['n']}] PMID {p['pmid']}, {p['year']}: {bw.clean(p['title'])}" for p in r["papers"])
        o1, o2 = r["pair"].split("|")
        vals = [r["id"], r["view"], o1, o2, bw.clean(r["sentence"]), refs,
                old.get(H_SUP), old.get(H_PROB), old.get(H_NOTE), r["generated_at"][:16].replace("T", " ")]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(row=i, column=c, value=v)
            cell.alignment = bw.WRAP_TOP
            if c in (7, 8, 9):
                cell.fill = bw.REV_FILL
        ws.row_dimensions[i].height = bw.est_height([(vals[4], 70), (refs, 62)])
    n_s = len(rows)
    bw.add_dropdown(ws, 7, SUPPORT_OPTS, n_s)
    bw.add_dropdown(ws, 8, PROBLEM_OPTS, n_s)
    ws.auto_filter.ref = f"A1:J{n_s + 1}"

    wc = wb.create_sheet("Citations")
    ccols = [
        {"name": "Sentence ID", "width": 9},
        {"name": "Ref no.", "width": 7, "help": "The reference number [n] of the sentence."},
        {"name": "View", "width": 14},
        {"name": "Organ pair", "width": 20},
        {"name": "Sentence", "width": 52},
        {"name": "PMID", "width": 11},
        {"name": "Year", "width": 6},
        {"name": "Title + abstract (passages the model cited in blue bold)", "width": 92,
         "help": "The blue bold passages are ALL passages the model cited from this paper anywhere in the description; the pipeline "
                 "does not record which passage belongs to which sentence. Judge the sentence against the whole abstract."},
        {"name": "Passages cited from this paper", "width": 50},
        {"name": H_PSUP, "width": 17, "review": True,
         "help": "Optional: Yes = this paper alone supports the sentence; Partly = it supports a part; No = it does not support it."},
        {"name": H_PNOTE, "width": 32, "review": True},
    ]
    bw.write_header(wc, ccols)
    wc.freeze_panes = "F2"
    r_i, unfound, total_q = 2, 0, 0
    for r in rows:
        for p in r["papers"]:
            old = old_cit.get((r["id"], p["n"]), {})
            cell_abs, plain, found = abstract_cell(p)
            unfound += len(p["quotes"]) - found
            total_q += len(p["quotes"])
            quotes = "\n".join(f"“{bw.clean(q)}”" for q in p["quotes"])
            vals = [r["id"], p["n"], r["view"], r["pair"].replace("|", " | "), bw.clean(r["sentence"]), p["pmid"], p["year"],
                    cell_abs, quotes, old.get(H_PSUP), old.get(H_PNOTE)]
            for c, v in enumerate(vals, 1):
                cell = wc.cell(row=r_i, column=c, value=v)
                cell.alignment = bw.WRAP_TOP
                if c in (10, 11):
                    cell.fill = bw.REV_FILL
            pm = wc.cell(row=r_i, column=6)
            pm.hyperlink = f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"
            pm.font = bw.LINK_FONT
            wc.row_dimensions[r_i].height = bw.est_height([(vals[4], 52), (plain, 92), (quotes, 50)], min_pt=60)
            r_i += 1
    n_c = r_i - 2
    bw.add_dropdown(wc, 10, PAPER_OPTS, n_c)
    wc.auto_filter.ref = f"A1:K{n_c + 1}"

    build_summary(wb, n_s, n_c)
    build_readme(wb, state, n_s, n_c, unfound, total_q)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return {"sentences": n_s, "citations": n_c, "quotes": total_q, "quotes not located in the abstract": unfound}


def build_summary(wb: Workbook, n_s: int, n_c: int) -> None:
    ws = wb.create_sheet("Summary", 0)
    S = lambda c: f"Sentences!${c}$2:${c}${n_s + 1}"      # noqa: E731
    C = lambda c: f"Citations!${c}$2:${c}${max(n_c, 1) + 1}"   # noqa: E731
    ws["A1"], ws["A1"].font = "Summary: are the sentences supported by their citations?", Font(bold=True, size=14)
    ws["A2"] = "Live formulas over the yellow reviewer columns. Intervals are Wilson 95% score intervals; 'Cannot tell' is not counted."
    ws["A3"], ws["B3"] = "z (95% CI)", 1.96
    ws["A5"] = "PROGRESS"
    for c, h in enumerate(["Sheet / rating", "Rated", "Total", "% rated"], 1):
        ws.cell(row=6, column=c, value=h)
    for i, (name, f, tot) in enumerate([("Sentences: supported by the cited papers?", f"=COUNTA({S('G')})", n_s),
                                        ("Citations: does this paper support the sentence? (optional)", f"=COUNTA({C('J')})", n_c)], 7):
        ws.cell(row=i, column=1, value=name)
        ws.cell(row=i, column=2, value=f)
        ws.cell(row=i, column=3, value=tot)
        ws.cell(row=i, column=4, value=f'=IF(C{i}>0,B{i}/C{i},"")').number_format = "0%"
    hdr = 11
    ws.cell(row=hdr - 1, column=1, value="RESULTS")
    for c, h in enumerate(["Metric", "k (successes)", "n (denominator)", "Proportion", "95% CI low", "95% CI high",
                           "Excluded from n (Cannot tell)", "Report as"], 1):
        ws.cell(row=hdr, column=c, value=h)
    cnt = lambda rng, v: f'COUNTIF({rng},"{v}")'          # noqa: E731
    sn = "+".join(cnt(S("G"), v) for v in SUPPORT_OPTS[:3])
    pn = "+".join(cnt(C("J"), v) for v in PAPER_OPTS[:3])
    metrics = [("Sentence fully supported (Supported)", cnt(S("G"), "Supported"), sn, cnt(S("G"), "Cannot tell")),
               ("Sentence supported or partly supported", f"{cnt(S('G'), 'Supported')}+{cnt(S('G'), 'Partly supported')}", sn, cnt(S("G"), "Cannot tell")),
               ("Sentence partly supported", cnt(S("G"), "Partly supported"), sn, cnt(S("G"), "Cannot tell")),
               ("Sentence not supported", cnt(S("G"), "Not supported"), sn, cnt(S("G"), "Cannot tell")),
               ("Cited paper supports the sentence (Yes)", cnt(C("J"), "Yes"), pn, cnt(C("J"), "Cannot tell"))]
    key = {}
    for i, (name, k, n, x) in enumerate(metrics, hdr + 1):
        ws.cell(row=i, column=1, value=name)
        ws.cell(row=i, column=2, value="=" + k)
        ws.cell(row=i, column=3, value="=" + n)
        ws.cell(row=i, column=4, value=f'=IF(C{i}>0,B{i}/C{i},"")').number_format = "0.0%"
        core = f"D{i}*(1-D{i})/C{i}+$B$3^2/(4*C{i}^2)"
        ws.cell(row=i, column=5, value=f'=IF(C{i}>0,(D{i}+$B$3^2/(2*C{i})-$B$3*SQRT({core}))/(1+$B$3^2/C{i}),"")').number_format = "0.0%"
        ws.cell(row=i, column=6, value=f'=IF(C{i}>0,(D{i}+$B$3^2/(2*C{i})+$B$3*SQRT({core}))/(1+$B$3^2/C{i}),"")').number_format = "0.0%"
        ws.cell(row=i, column=7, value="=" + x)
        ws.cell(row=i, column=8, value=f'=IF(C{i}=0,"",B{i}&"/"&C{i}&" ("&TEXT(D{i},"0%")&"; 95% CI "&TEXT(E{i},"0%")&"–"&TEXT(F{i},"0%")&")")')
        key[name] = i
    r = hdr + len(metrics) + 3
    ws.cell(row=r - 1, column=1, value="MAIN PROBLEM OF THE SENTENCES THAT ARE NOT FULLY SUPPORTED")
    for c, h in enumerate(["Problem", "Count"], 1):
        ws.cell(row=r, column=c, value=h)
    for j, opt in enumerate(PROBLEM_OPTS, r + 1):
        ws.cell(row=j, column=1, value=opt)
        ws.cell(row=j, column=2, value="=" + cnt(S("H"), opt))
    r = r + len(PROBLEM_OPTS) + 3
    ws.cell(row=r - 1, column=1, value="DRAFT RESULTS SENTENCE (fills in from the cells above)")
    a, b, c_, d_ = (key["Sentence fully supported (Supported)"], key["Sentence supported or partly supported"],
                    key["Sentence partly supported"], key["Sentence not supported"])
    ci = lambda x: f'"95% CI "&TEXT(E{x},"0%")&"–"&TEXT(F{x},"0%")'      # noqa: E731
    ws.cell(row=r, column=1, value=(
        f'=IF(C{a}=0,"(Rate the column on the sheet Sentences to generate this sentence.)",'
        f'"Of "&C{a}&" randomly sampled sentences of the LLM-generated organ-organ descriptions, "&TEXT(D{a},"0%")&" ("&{ci(a)}&") were fully supported by '
        f'the papers they cite, "&TEXT(D{c_},"0%")&" were partly supported and "&TEXT(D{d_},"0%")&" were not supported.")'))
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
    ws.cell(row=r, column=1).alignment = bw.WRAP_TOP
    ws.row_dimensions[r].height = 48
    ws.column_dimensions["A"].width = 58
    for col in "BCDEFG":
        ws.column_dimensions[col].width = 16
    ws.column_dimensions["H"].width = 34
    for row in (5, hdr - 1, hdr + len(metrics) + 2):
        ws.cell(row=row, column=1).font = Font(bold=True)
    for cc in range(1, 9):
        ws.cell(row=hdr, column=cc).font = Font(bold=True)


def build_readme(wb: Workbook, state: dict, n_s: int, n_c: int, unfound: int, total_q: int) -> None:
    wr = wb.create_sheet("README", 0)
    wr.column_dimensions["A"].width = 30
    wr.column_dimensions["B"].width = 120
    fr = state["frame"]
    per_view = " | ".join(f"{v}: {d['sentences']} sentences in {d['descriptions']} descriptions" for v, d in fr["per view"].items())
    lines = [("Check: are the LLM-generated sentences supported by their citations?", None, "title"),
             ("Built", datetime.now().strftime("%Y-%m-%d %H:%M"), None),
             ("What is checked", "The organ-organ descriptions of the dashboard are written by Claude (Haiku 5.5) from up to 10 papers per pair and view, "
                                 "with the citations feature: every sentence carries reference numbers [n] of the papers it rests on. The passages "
                                 "the model cites were verified to occur verbatim in the papers, but whether a sentence is really supported by "
                                 "those papers is a judgement. This workbook asks for that judgement on a random sample.", None),
             ("Sample", f"{n_s} sentences, drawn uniformly at random without replacement (seed {state['seed']}) from all {fr['sentences']} cited sentences of "
                        f"{fr['descriptions']} valid pair descriptions (the 'All papers' view and the seven condition views). {per_view}.", None),
             ("", "", None), ("SHEETS", None, "section"),
             ("Sentences", "One row per sampled sentence. Read the sentence, open the cited papers on the sheet Citations, and rate (yellow columns) whether the "
                           "sentence is supported by the cited papers, and, if not fully, the main problem.", None),
             ("Citations", f"One row per sentence-paper pair ({n_c} rows): the sentence beside the title + abstract of the cited paper. The passages the model cited "
                           f"are in blue bold ({total_q - unfound} of {total_q} located in the abstract); they are all passages cited from that paper anywhere "
                           "in the description (the pipeline does not store which passage belongs to which sentence), so judge against the whole abstract. "
                           "The per-paper rating is optional.", None),
             ("Summary", "Live formulas: shares with Wilson 95% intervals, problem types, progress, draft results sentence.", None),
             ("", "", None), ("RATING", None, "section"),
             ("Supported", "Everything the sentence states is stated in the cited papers (or follows directly from them).", None),
             ("Partly supported", "Part of the sentence is supported, another part is not: an added detail, a generalisation, a mechanism, a direction of "
                                  "effect, a different organ / molecule / species.", None),
             ("Not supported", "The cited papers do not state what the sentence claims, or they contradict it.", None),
             ("Cannot tell", "The title + abstract are not enough to decide (the full text might support it). Not counted in the statistics.", None),
             ("Judge by", "Only the cited papers (the numbers in the sentence), not by what you know to be true. A sentence with several numbers may "
                          "rest on the papers together; if it needs a finding that none of them states, rate it Partly / Not supported and pick "
                          "'Combines findings that no single paper states'.", None),
             ("", "", None), ("REBUILD", None, "section"),
             ("Rewrite", "python build_citation_validation_workbook.py --rewrite   (rebuilds the workbook from validation/citation_validation_state.json, "
                         "ratings are kept by ID; close the workbook first).", None),
             ("Check", "python build_citation_validation_workbook.py --check   (are the sampled sentences still in the dashboard's descriptions? They stay "
                       "in this workbook as drawn even if a description is regenerated).", None)]
    for i, (a, b, kind) in enumerate(lines, 1):
        ca, cb = wr.cell(row=i, column=1, value=a), wr.cell(row=i, column=2, value=b)
        ca.alignment = cb.alignment = bw.WRAP_TOP
        if kind == "title":
            ca.font = Font(bold=True, size=14)
        elif kind == "section":
            ca.font = bw.HDR_FONT
            ca.fill = cb.fill = bw.HDR_FILL
        else:
            ca.font = Font(bold=True)


# ───────────────────────────── CLI ──────────────────────────────────────────────

def check(state: dict) -> None:
    cache = {}
    same = gone = 0
    for r in state["rows"]:
        e = cache.setdefault(r["slug"], json.loads(description_file(r["slug"]).read_text(encoding="utf-8")))
        text = (e.get(r["pair"]) or {}).get("description", "")
        ok = r["sentence"] in text
        same += ok
        gone += not ok
        if not ok:
            print(f"  {r['id']} ({r['view']}, {r['pair']}): the description was regenerated or removed; the sentence is no longer shown")
    print(f"check: {same} of {same + gone} sampled sentences are still in the dashboard's descriptions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="number of sentences to draw (default 30)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rewrite", action="store_true", help="rebuild the workbook from the state file, keep the ratings")
    ap.add_argument("--check", action="store_true", help="report whether the sampled sentences are still in the descriptions")
    ap.add_argument("--workbook", type=Path, default=WORKBOOK)
    ap.add_argument("--state", type=Path, default=STATE)
    args = ap.parse_args()

    if args.check:
        check(json.loads(args.state.read_text(encoding="utf-8")))
        return
    bw._assert_closed(args.workbook)
    if args.rewrite:
        if not (args.state.exists() and args.workbook.exists()):
            sys.exit("No workbook/state found; nothing to rewrite.")
        import shutil
        backup = args.workbook.parent / f"before_citation_rewrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup.mkdir(parents=True, exist_ok=True)
        for f in (args.workbook, args.state):
            shutil.copy2(f, backup / f.name)
        state = json.loads(args.state.read_text(encoding="utf-8"))
        old_s, old_c = read_old_ratings(args.workbook)
        print(f"rewrite: {len(state['rows'])} sentences kept, ratings carried over; backup in {backup.name}/")
    else:
        if args.state.exists() or args.workbook.exists():
            sys.exit(f"{args.workbook.name} / state already exist. Use --rewrite to rebuild keeping the ratings, or move both files away "
                     "to draw a new sample.")
        state = draw_sample(args.n, args.seed)
        old_s, old_c = {}, {}
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = write_workbook(args.workbook, state, old_s, old_c)
    print(f"saved {args.workbook}\n  {stats}")
    fr = state["frame"]
    print(f"  frame: {fr['sentences']} sentences in {fr['descriptions']} descriptions ({fr['skipped sentences']} sentences without a valid reference skipped)")


if __name__ == "__main__":
    main()
