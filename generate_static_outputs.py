"""
generate_static_outputs.py
==========================
Static statistics and figures of the LLM-condition-filter network (current results).

Writes to static_outputs/ :

  statistics/paper_funnel.xlsx (+ csv)    papers fetched, kept by the same-sentence filter and per condition view,
                                          LLM answer status, per-pair table, the rules in force
  paper_funnel.png                        the same as a figure
  overview_statistics.png / overview_table.png / statistics/overview_table.csv
                                          edge counts, papers per pair, OO coefficients, overlap with the PET cohorts
  <view>/sankey.png, papers_per_organ_barplot.png,
  <view>/connection_type_frequency_barplot.png, connection_type_per_organ_stacked_barplot.png
                                          one folder per view: all, nocondition, healthy, obesity, cardiovascular,
                                          cancers, infectious, other
  papers_per_organ_overview_grid.png, connection_types_grid.png
                                          all views in one figure

A view = the papers of one LLM condition label with a verified evidence sentence ("All papers" = no condition
filter). Everything is computed from the cached results, labels and connection types; nothing is searched or
sent to an LLM.

    python generate_static_outputs.py                (from this folder, with the project's .venv)
"""

import collections
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_metabolic_lit_search as rl                                   # noqa: E402
from Literature_Search import llm_condition_filter as cf                # noqa: E402
from Literature_Search.pubmed_search import ORGAN_ALIASES                # noqa: E402
from threshold_utils import kneedle_elbow                                # noqa: E402
from Visualisation import static_plots as sp                             # noqa: E402

OUT = HERE / "static_outputs"
STAT = OUT / "statistics"
STAT.mkdir(parents=True, exist_ok=True)
sp.GRID_DPI = 130
INK, MUTED, GRID = "#1e293b", "#64748b", "#e2e8f0"

VIEWS = [("all", "All papers")] + [(c["slug"], c["name"]) for c in cf.CONDITIONS]
VIEW_NAME = dict(VIEWS)
VIEW_COLOR = dict(cf.CONDITION_COLOR)
COHORT_COLOR = {"Healthy PET cohort": "#22d3ee", "Overweight PET cohort": "#fbbf24", "Lung cancer PET cohort": "#a78bfa"}
COHORT_N_ORGANS: dict = {}                                                # PET cohort -> organs in its network (filled by overview())
canon =lambda a, b: (min(a, b), max(a, b))                              # noqa: E731


# ── data ─────────────────────────────────────────────────────────────────────

def load_data():
    results = json.loads(Path(rl.OUTPUT_JSON).read_text(encoding="utf-8"))
    labels = cf.load_condition_labels(rl.CONDITION_LABELS, rl.CONDITION_LLM_MODEL)
    by_pair = {canon(e["organ1"], e["organ2"]): e for e in results.values() if e.get("organ1") and e.get("organ2")}
    pairs = rl.load_edge_filter(rl.EDGE_FILTER_CSV)
    view_papers = {"all": {p: by_pair.get(p, {}).get("papers", []) for p in pairs}}
    for c in cf.CONDITIONS:
        f = cf.LLMConditionFilter(c["slug"], labels)
        view_papers[c["slug"]] = {p: [x for x in view_papers["all"][p] if f.accepts(x)] for p in pairs}
    return results, labels, pairs, view_papers


def organ_links(pair_papers: dict):
    active = {p: ps for p, ps in pair_papers.items() if ps}
    organs = sorted({o for p in active for o in p})
    organ_pmids = {o: set() for o in organs}
    for (a, b), ps in active.items():
        pm = {str(x.get("pmid")) for x in ps if x.get("pmid")}
        organ_pmids[a] |= pm
        organ_pmids[b] |= pm
    organ_papers = {o: len(s) for o, s in organ_pmids.items()}
    links = []
    for (a, b), ps in active.items():
        s, t = (a, b) if organ_papers[a] >= organ_papers[b] else (b, a)
        links.append({"source": s, "target": t, "value": len(ps)})
    return organ_papers, links


def load_types(slug: str) -> dict:
    d = HERE / "reference_network_only_metabolic"
    path = d / "metabolic_connection_types.json" if slug == "all" else rl.condition_output_paths(slug)["llm_types"]
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {canon(*k.split("|", 1)): v.get("types", []) for k, v in raw.items()}


def oo_map(pair_papers: dict) -> dict:
    organ_pmids = {}
    for (a, b), ps in pair_papers.items():
        pm = {str(x.get("pmid")) for x in ps if x.get("pmid")}
        organ_pmids.setdefault(a, set()).update(pm)
        organ_pmids.setdefault(b, set()).update(pm)
    n = {o: len(s) for o, s in organ_pmids.items()}
    out = {}
    for (a, b), ps in pair_papers.items():
        co = len([x for x in ps if x.get("pmid")])
        den = (n.get(a, 0) * n.get(b, 0)) ** 0.5
        out[(a, b)] = co / den if den > 0 else 0.0
    return out


def unique_pmids(pair_papers: dict) -> set:
    return {str(x.get("pmid")) for ps in pair_papers.values() for x in ps if x.get("pmid")}


# ── 1. paper funnel ──────────────────────────────────────────────────────────
# PubMed fetch (the paper lists were not kept, only the count of every pair: n_papers_found_raw)
#   -> same-sentence filter (both organs by their aliases + a crosstalk keyword in one sentence, plus the sentence rules)
#   -> LLM condition views.

def build_funnel(results, labels, pairs, view_papers):
    fetched = {canon(e["organ1"], e["organ2"]): int(e.get("n_papers_found_raw", 0) or 0) for e in results.values() if e.get("organ1")}
    n_fetched = sum(fetched.get(p, 0) for p in pairs)
    final_pp = view_papers["all"]
    ref_a, ref_u = sum(len(v) for v in final_pp.values()), len(unique_pmids(final_pp))
    try:       # count-only run of recount_step1_window.py (the paper lists of the PubMed step were not kept)
        n_fetched_unique = json.loads((Path(rl.OUTPUT_JSON).parent / "pubmed_step1_final.json").read_text(encoding="utf-8")).get("window unique papers")
    except (OSError, ValueError):
        n_fetched_unique = None
    steps = [
        {"step": "1  Fetched from PubMed",
         "description": "papers returned by the pair queries within the 20-year window (both organs + metabolic term + crosstalk term "
                        "anywhere in title/abstract; PubMed state of 2026-09-17); counts from a count-only PubMed run, the paper lists were not kept",
         "paper-pair assignments": n_fetched, "unique papers": n_fetched_unique,
         "pairs with >=1 paper": sum(1 for p in pairs if fetched.get(p, 0)),
         "% of fetched assignments": 100.0, "% of All-papers unique papers": None},
        {"step": "2  Same-sentence filter  (= All papers)",
         "description": "both organs (by their aliases) and a crosstalk keyword in one sentence; rejected: co-listed organs, a third organ "
                        "named first, parallel locations ('in the brain and in the lung')",
         "paper-pair assignments": ref_a, "unique papers": ref_u,
         "pairs with >=1 paper": sum(1 for v in final_pp.values() if v),
         "% of fetched assignments": round(100 * ref_a / n_fetched, 1) if n_fetched else None,
         "% of All-papers unique papers": 100.0},
    ]
    cond_rows = []
    for slug, name in VIEWS[1:]:
        pp = view_papers[slug]
        a_, u_ = sum(len(v) for v in pp.values()), len(unique_pmids(pp))
        cond_rows.append({"view": name, "slug": slug, "paper-pair assignments": a_, "unique papers": u_,
                          "pairs with >=1 paper": sum(1 for v in pp.values() if v),
                          "% of All-papers assignments": round(100 * a_ / ref_a, 1), "% of All-papers unique papers": round(100 * u_ / ref_u, 1)})
        steps.append({"step": f"3  LLM condition: {name}", "description": "label of the paper + verified evidence sentence",
                      "paper-pair assignments": a_, "unique papers": u_,
                      "pairs with >=1 paper": sum(1 for v in pp.values() if v),
                      "% of fetched assignments": round(100 * a_ / n_fetched, 1) if n_fetched else None,
                      "% of All-papers unique papers": round(100 * u_ / ref_u, 1)})
    uniq = unique_pmids(final_pp)
    st = collections.Counter(labels[p]["status"] if p in labels else "not classified" for p in uniq)
    status_rows = [{"LLM answer of the papers in All papers": k, "unique papers": v, "%": round(100 * v / len(uniq), 2)}
                   for k, v in st.most_common()]
    by_label = collections.Counter(labels[p]["label"] for p in uniq if p in labels and labels[p]["status"] == "ok")
    label_rows = [{"label": lab, "unique papers (usable)": by_label.get(lab, 0),
                   "%": round(100 * by_label.get(lab, 0) / max(1, sum(by_label.values())), 1)} for lab in cf.LABELS]
    per_pair = []
    for p in pairs:
        row = {"pair": f"{p[0]} | {p[1]}", "fetched from PubMed": fetched.get(p, 0), "same-sentence filter (All papers)": len(final_pp[p])}
        for slug, name in VIEWS[1:]:
            row[name] = len(view_papers[slug][p])
        per_pair.append(row)
    rules = [{"rule": "same-sentence filter", "value": "both organ names (any alias) and at least one crosstalk keyword in the same sentence; "
                                                       "rejected: organs that are only co-listed, a third organ named first, parallel locations "
                                                       "('in the brain and in the lung')"},
             {"rule": "crosstalk keywords (%d)" % len(rl.CROSSTALK_KEYWORDS), "value": ", ".join(rl.CROSSTALK_KEYWORDS)}]
    rules += [{"rule": f"aliases: {o}", "value": ", ".join(a_)} for o, a_ in ORGAN_ALIASES.items()]
    rules += [{"rule": "gut terms", "value": "'gut' (also in gut microbiota, gut microbiome, microbiome in the gut) counts as Small Intestine only; "
                                              "Colon needs a colon word; a bare 'microbiome'/'microbiota' counts for no organ"},
              {"rule": "condition model / prompt", "value": f"{rl.CONDITION_LLM_MODEL} / {cf.PROMPT_VERSION}"},
              {"rule": "generated", "value": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}]
    return steps, cond_rows, status_rows, label_rows, per_pair, rules


def panel(letter, text):
    """Panel title with a bold panel letter (mathtext only for the letter)."""
    return rf"$\mathbf{{{letter}}}$  {text}"


PANEL_SIZES = {"A": (11.0, 3.6), "B": (6.8, 4.5), "C": (6.3, 4.3), "D": (7.8, 5.4), "E": (7.8, 5.4), "F": (8.6, 3.8),
               "G": (7.6, 3.0), "H": (6.6, 6.0), "I": (7.2, 7.2)}


def build_panels(steps, cond_rows, status_rows, validation, mol_validation, cit_validation, organ_overview):
    """The panels of the funnel figure: [{letter, slug, rows, cols, draw(ax, standalone)}] and the number of grid rows.
    standalone=True is the single-panel export: no panel letter and no title (a key that only the title carried is added as a legend)."""
    rows_n = 2 + bool(validation) + bool(validation and (mol_validation or cit_validation))
    panels = []

    def add(letter, slug, rows, cols, draw):
        panels.append({"letter": letter, "slug": slug, "rows": rows, "cols": cols, "draw": draw})

    main = [s for s in steps if "LLM condition" not in s["step"]]
    h = 0.36

    def draw_a(ax, standalone):
        y = np.arange(len(main))[::-1]
        a_vals = [s["paper-pair assignments"] for s in main]
        ax.barh(y + h / 2, a_vals, h, color="#6366f1", label="paper-pair assignments")
        for yy, v, s in zip(y, a_vals, main):
            pct = s["% of fetched assignments"]
            ax.text(v, yy + h / 2, f"  {v:,}" + (f"  ({pct:.1f}% of fetched)" if pct is not None and pct < 100 else ""), va="center", fontsize=10, color=INK)
        u_pairs = [(yy, s["unique papers"]) for yy, s in zip(y, main) if s["unique papers"] is not None]
        ax.barh([yy - h / 2 for yy, _ in u_pairs], [v for _, v in u_pairs], h, color="#22d3ee", label="unique papers")
        for yy, v in u_pairs:
            ax.text(v, yy - h / 2, f"  {v:,}", va="center", fontsize=10, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([s["step"] for s in main], fontsize=11, color=INK)
        ax.set_xlim(0, max(a_vals) * 1.25)
        if not standalone:
            ax.set_title(panel("A", "Papers fetched and kept by the same-sentence filter"
                               + ("" if main[0]["unique papers"] is not None else " (unique papers were not recorded for the PubMed step)")),
                         loc="left", fontsize=13, color=INK)
        ax.legend(frameon=False, loc="center right")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)

    def draw_b(ax, standalone):
        names = [r["view"] for r in cond_rows]
        y = np.arange(len(names))[::-1]
        ax.barh(y + h / 2, [r["paper-pair assignments"] for r in cond_rows], h, color=[VIEW_COLOR[r["slug"]] for r in cond_rows])
        ax.barh(y - h / 2, [r["unique papers"] for r in cond_rows], h, color=[VIEW_COLOR[r["slug"]] for r in cond_rows], alpha=0.55)
        for yy, r in zip(y, cond_rows):
            ax.text(r["paper-pair assignments"], yy + h / 2, f"  {r['paper-pair assignments']:,}", va="center", fontsize=9, color=INK)
            ax.text(r["unique papers"], yy - h / 2, f"  {r['unique papers']:,}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10, color=INK)
        ax.set_xlim(0, max(r["paper-pair assignments"] for r in cond_rows) * 1.2)
        if standalone:        # the title of the combined figure carried this key
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor="#475569", label="paper-pair assignments (dark)"),
                               Patch(facecolor="#475569", alpha=0.55, label="unique papers (light)")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
        else:
            ax.set_title(panel("B", "Condition views (LLM label + verified evidence)") + "\ndark = paper-pair assignments, light = unique papers",
                         loc="left", fontsize=12, color=INK)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)

    def draw_c(ax, standalone):
        ks = [r["LLM answer of the papers in All papers"] for r in status_rows]
        vs = [r["unique papers"] for r in status_rows]
        pal = {"ok": "#34d399", "hallucinated": "#f97316", "too_short": "#facc15", "invalid_json": "#a78bfa",
               "invalid_label": "#a78bfa", "error": "#ef4444", "not classified": "#94a3b8"}
        y = np.arange(len(ks))[::-1]
        ax.barh(y, vs, color=[pal.get(k, "#94a3b8") for k in ks])
        for yy, v in zip(y, vs):
            ax.text(v, yy, f"  {v:,} ({100 * v / sum(vs):.1f}%)", va="center", fontsize=9, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(ks, fontsize=10, color=INK)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, max(vs) * 8)
        if not standalone:
            ax.set_title(panel("C", "LLM answers of the unique papers in All papers") + "\n(only 'ok' = valid label + evidence found in the paper is used)",
                         loc="left", fontsize=12, color=INK)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)

    add("A", "papers_fetched_and_kept", 0, slice(0, 2), draw_a)
    add("B", "condition_views", 1, 0, draw_b)
    add("C", "llm_answers", 1, 1, draw_c)

    if validation:
        n_papers = len(validation)
        rel = validation_counts(validation, 1, REL_CATS)
        lab = validation_counts(validation, 2, LAB_CATS)
        cannot = sum(1 for r in validation if r[1] == "Cannot tell")
        add("D", "validation_organ_relationship", 2, 0, lambda ax, st: draw_stacked(
            ax, rel, REL_CATS, None if st else panel("D", f"Validation: organ relationship ({n_papers} reviewed papers"
                                                     + (f", {cannot} 'cannot tell' not counted" if cannot else "") + ")")))
        add("E", "validation_llm_condition_label", 2, 1, lambda ax, st: draw_stacked(
            ax, lab, LAB_CATS, None if st else panel("E", "Validation: LLM condition label (same rows as D)"), names=st))
    letters = iter("GHIJKL")                    # panel letters after F are handed out in order, so there is never a gap
    both = bool(mol_validation and cit_validation)          # bottom row: F left, G right
    if validation and mol_validation:
        mol = molecule_counts(mol_validation)
        cannot_m = sum(1 for r in mol_validation if r[1] == "Cannot tell")
        title_f = (f"Validation: key player annotation ({len(mol_validation)} reviewed key players" if both else
                   f"Validation: key player annotation by key player category ({len(mol_validation)} reviewed key players")
        add("F", "validation_key_players", 3, 0 if both else slice(0, 2), lambda ax, st: draw_stacked(
            ax, mol, MOL_CATS, None if st else panel("F", title_f + (f", {cannot_m} 'cannot tell' not counted" if cannot_m else "") + ")"),
            unit="key players", annotate_small=True if (st or not both) else "label", legend_y=-0.24))
    if validation and cit_validation:
        cit = citation_counts(cit_validation)
        cannot_c = sum(1 for r in cit_validation[0] if r[4] == "Cannot tell")
        n_sent = cit["Sentences"][0]
        letter_g = next(letters)
        add(letter_g, "validation_citations", 3, 1 if mol_validation else slice(0, 2), lambda ax, st: draw_stacked(
            ax, cit, CIT_CATS, None if st else panel(letter_g, "Validation: description sentences\nsupported by their citations "
                                                    f"({n_sent} sentences" + (f", {cannot_c} 'cannot tell' not counted" if cannot_c else "") + ")"),
            unit="sentences / citations", annotate_small=True, legend_y=-0.24, min_slots=3))
    if organ_overview:
        organ_papers, links = organ_overview
        top_rows = 0 if rows_n < 3 else slice(0, 2)
        bottom_rows = 1 if rows_n < 3 else slice(2, rows_n)
        letter_bars, letter_chord = next(letters), next(letters)      # third column, top to bottom: bars, chord diagram

        def draw_bars(ax, standalone):
            sp._draw_barplot_ax(ax, organ_papers, label_fontsize=9)
            if not standalone:
                ax.set_title(panel(letter_bars, "All papers: papers per organ\n(a paper counts for each organ of its pairs)"),
                             loc="left", fontsize=12, color=INK)

        def draw_chord(ax, standalone):
            if sp._draw_chord_diagram_ax(ax, organ_papers, links) and not standalone:
                ax.set_ylim(-1.75, 2.05)          # headroom for the title
                ax.set_anchor("N")
            if not standalone:
                ax.set_title(panel(letter_chord, "All papers: organ cross-talk\nribbon width = papers of the organ pair"),
                             loc="left", fontsize=12, color=INK)

        add(letter_bars, "papers_per_organ", top_rows, 2, draw_bars)
        add(letter_chord, "organ_crosstalk_chord", bottom_rows, 2, draw_chord)
    return panels, rows_n


def export_panels(panels, out_dir, dpi=600):
    """Every panel of the funnel figure on its own, high resolution, without panel letter and title."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for pn in panels:
        fig, ax = plt.subplots(figsize=PANEL_SIZES[pn["letter"]] if pn["letter"] in PANEL_SIZES else (7, 5), dpi=dpi)
        fig.patch.set_facecolor("white")
        pn["draw"](ax, True)
        out = out_dir / f"{pn['letter']}_{pn['slug']}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [ok] {sp._rel(out)}")


def draw_funnel(steps, cond_rows, status_rows, out_path, validation=None, mol_validation=None, cit_validation=None,
                organ_overview=None, panel_dir=None):
    """organ_overview: (organ_papers, links) of the All-papers view -> third column with the papers-per-organ bars and the chord diagram.
    panel_dir: also write every panel on its own (high resolution, no letter, no title) into this folder."""
    panels, rows_n = build_panels(steps, cond_rows, status_rows, validation, mol_validation, cit_validation, organ_overview)
    ncols = 3 if organ_overview else 2
    fig = plt.figure(figsize=({2: 17, 3: 24.5}[ncols], {2: 9.5, 3: 14.5, 4: 17.6}[rows_n]), dpi=200)
    gs = fig.add_gridspec(rows_n, ncols, height_ratios=[0.8, 1, 1.15, 0.85][:rows_n], width_ratios=[1.25, 1, 1.45][:ncols],
                          hspace=0.62, wspace=0.42, top=0.93)
    fig.patch.set_facecolor("white")
    for pn in panels:
        pn["draw"](fig.add_subplot(gs[pn["rows"], pn["cols"]]), False)
    fig.suptitle("Literature extraction and manual literature validation overview" if validation else "Literature extraction overview",
                 fontsize=17, y=0.965, color=INK)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {sp._rel(out_path)}")
    if panel_dir:
        export_panels(panels, Path(panel_dir))


# ── validation (manual review of the sampled papers, filled in the review workbook) ──────────────

VALIDATION_XLSX = HERE / "validation" / "reference_network_review_workbook_LLM.xlsx"
LABEL_NAME = {c["label"]: c["name"] for c in cf.CONDITIONS}
REL_CATS = [("Direct relationship", "Confirmed", "#34d399"), ("Co-regulation", "Co-regulation", "#38bdf8"),
            ("Co-mention only", "Co-mention only", "#fbbf24"), ("False", "Wrong", "#ef4444")]
# (name shown in the figure and the statistics sheets, rating value in the review workbook, colour)
LAB_CATS = [("Correct", "Correct", "#34d399"), ("Unclear", "Unsure", "#fbbf24"), ("False", "Wrong", "#ef4444")]
MOL_CATS = [("Correct", "Correct", "#34d399"), ("False category", "Wrong category", "#fbbf24"),
            ("False positive", "False positive", "#ef4444")]
MOL_GROUPS = ("metabolite", "hormone", "protein")


def load_validation():
    """[(LLM label of the paper, organ-relationship rating, LLM-label rating)] from the review workbook, or None."""
    import openpyxl
    if not VALIDATION_XLSX.exists():
        return None
    ws = openpyxl.load_workbook(VALIDATION_XLSX, read_only=True, data_only=True)["Assignments"]
    rows = list(ws.iter_rows(values_only=True))
    head = {h: i for i, h in enumerate(rows[0])}
    out = [(r[head["LLM condition label"]], r[head["REVIEW: organ relationship"]], r[head["REVIEW: LLM label"]])
           for r in rows[1:] if r[0]]
    return out if any(rel or lab for _, rel, lab in out) else None


def load_molecule_validation():
    """[(pipeline molecule category, molecule rating, molecule, matched text, PMID, note)] from the Molecules sheet, or None."""
    import openpyxl
    if not VALIDATION_XLSX.exists():
        return None
    wb = openpyxl.load_workbook(VALIDATION_XLSX, read_only=True, data_only=True)
    if "Molecules" not in wb.sheetnames:
        return None
    rows = list(wb["Molecules"].iter_rows(values_only=True))
    head = {h: i for i, h in enumerate(rows[0])}
    cols = ("Category (pipeline)", "REVIEW: molecule check", "Molecule (pipeline name)", "Matched text", "PMID", "REVIEW: notes")
    out = [tuple(r[head[c]] for c in cols) for r in rows[1:] if r[0]]
    return out if any(r[1] for r in out) else None


def molecule_counts(rows):
    """{group: (n, {rating: count})}; group 'All rated key players' + one per key player category."""
    out = {}
    valid = {v for _, v, _ in MOL_CATS}
    for group in ("All rated key players",) + MOL_GROUPS:
        sel = [r for r in rows if group == "All rated key players" or r[0] == group]
        c = collections.Counter(r[1] for r in sel if r[1] in valid)
        out[group] = (sum(c.values()), c)
    return out


def molecule_table(rows):
    recs = []
    for group, (n, c) in molecule_counts(rows).items():
        for disp, value, _ in MOL_CATS:
            recs.append({"question": "key player annotation", "group": group, "category": disp, "count": c.get(value, 0),
                         "rated n": n, "share %": round(100 * c.get(value, 0) / n, 1) if n else None})
    return pd.DataFrame(recs)


def molecule_flags(rows):
    """Reviewed molecule matches that were not rated Correct (with the matched text and the reviewer's note)."""
    return pd.DataFrame([{"PMID": r[4], "molecule (pipeline name)": r[2], "category (pipeline)": r[0], "matched text": r[3],
                          "rating": r[1], "reviewer note": r[5]} for r in rows if r[1] and r[1] != "Correct"])


CIT_XLSX = HERE / "validation" / "citation_validation_workbook.xlsx"
CIT_CATS = [("Supported", "Supported", "#34d399"), ("Partly supported", "Partly supported", "#fbbf24"),
            ("Not supported", "Not supported", "#ef4444")]
PAPER_TO_SENTENCE = {"Yes": "Supported", "Partly": "Partly supported", "No": "Not supported"}   # per-paper rating -> same categories


def load_citation_validation():
    """(sentence rows, per-paper ratings) from the citation workbook, or None while no sentence is rated.
    sentence row: (ID, view, organ pair, sentence, rating, main problem, note)."""
    import openpyxl
    if not CIT_XLSX.exists():
        return None
    wb = openpyxl.load_workbook(CIT_XLSX, read_only=True, data_only=True)
    rows = list(wb["Sentences"].iter_rows(values_only=True))
    h = {k: i for i, k in enumerate(rows[0])}
    sent = [(r[0], r[h["View"]], f"{r[h['Organ 1']]} | {r[h['Organ 2']]}", r[h["Sentence (as shown in the dashboard)"]],
             r[h["REVIEW: sentence supported by the cited papers?"]], r[h["REVIEW: main problem (if not fully supported)"]],
             r[h["REVIEW: notes"]]) for r in rows[1:] if r[0]]
    crow = list(wb["Citations"].iter_rows(values_only=True))
    ch = {k: i for i, k in enumerate(crow[0])}
    papers = [r[ch["REVIEW: does this paper support the sentence?"]] for r in crow[1:] if r[0]]
    return (sent, papers) if any(r[4] for r in sent) else None


def citation_counts(cit):
    """{'Sentences': (n, Counter), 'Citations': (n, Counter)} on the same three categories ('Cannot tell' not counted).
    A citation = one sentence-paper pair; its rating says whether that cited paper on its own supports the sentence."""
    sent, papers = cit
    valid = {v for _, v, _ in CIT_CATS}
    c1 = collections.Counter(r[4] for r in sent if r[4] in valid)
    out = {"Sentences": (sum(c1.values()), c1)}
    c2 = collections.Counter(PAPER_TO_SENTENCE[p] for p in papers if p in PAPER_TO_SENTENCE)
    if c2 and all(papers):                  # the optional per-paper rating is only plotted when every cited paper is rated
        out["Citations"] = (sum(c2.values()), c2)
    return out


def citation_table(cit):
    recs = []
    for group, (n, c) in citation_counts(cit).items():
        for disp, value, _ in CIT_CATS:
            recs.append({"question": "sentence supported by the cited papers" if group == "Sentences" else "citation supports the sentence (the cited paper alone)",
                         "group": group, "category": disp, "count": c.get(value, 0), "rated n": n,
                         "share %": round(100 * c.get(value, 0) / n, 1) if n else None})
    problems = collections.Counter(r[5] for r in cit[0] if r[4] in ("Partly supported", "Not supported") and r[5])
    for prob, k in problems.most_common():
        recs.append({"question": "main problem of the sentences not fully supported", "group": "Sentences", "category": prob,
                     "count": k, "rated n": None, "share %": None})
    return pd.DataFrame(recs)


def citation_flags(cit):
    """Rated sentences that are not fully supported, with the reviewer's problem type and note."""
    return pd.DataFrame([{"ID": r[0], "view": r[1], "organ pair": r[2], "sentence": r[3], "rating": r[4], "main problem": r[5],
                          "reviewer note": r[6]} for r in cit[0] if r[4] in ("Partly supported", "Not supported")])


def validation_counts(rows, col, cats):
    """{group: (n, {category value: count})}; group 'All rated papers' + one per LLM label (name)."""
    out = {}
    for group in ["All rated papers"] + [LABEL_NAME[l] for l in cf.LABELS]:
        sel = [r for r in rows if group == "All rated papers" or LABEL_NAME.get(r[0]) == group]
        c = collections.Counter(r[col] for r in sel if r[col] in {v for _, v, _ in cats})
        out[group] = (sum(c.values()), c)
    return out


def validation_table(rows):
    recs = []
    for question, col, cats in (("organ relationship", 1, REL_CATS), ("LLM condition label", 2, LAB_CATS)):
        for group, (n, c) in validation_counts(rows, col, cats).items():
            for disp, value, _ in cats:
                recs.append({"question": question, "group": group, "category": disp, "count": c.get(value, 0), "rated n": n,
                             "share %": round(100 * c.get(value, 0) / n, 1) if n else None})
    return pd.DataFrame(recs)


def draw_stacked(ax, counts, cats, title, names=True, unit="papers", annotate_small=False, legend_y=-0.16, min_slots=0):
    groups = list(counts)
    y = np.arange(len(groups))[::-1]
    left = np.zeros(len(groups))
    for disp, value, color in cats:
        share = np.array([counts[g][1].get(value, 0) / counts[g][0] if counts[g][0] else 0 for g in groups])
        ax.barh(y, share * 100, left=left, color=color, edgecolor="white", linewidth=0.8, label=disp, height=0.72)
        for yy, sh, lf in zip(y, share, left):
            if sh >= 0.06:
                ax.text(lf + sh * 50, yy, f"{sh * 100:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left += share * 100
    small_txt = {}
    for g in groups:      # categories too thin for a percentage label
        n, c = counts[g]
        small_txt[g] = ", ".join(f"{c.get(v, 0)} {d.lower()}" for d, v, _ in cats if n and 0 < c.get(v, 0) / n < 0.06)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} (n={counts[g][0]}" + (f"; {small_txt[g]}" if annotate_small == "label" and small_txt[g] else "") + ")"
                        for g in groups] if names else [], fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    if min_slots > len(groups):            # same bar thickness as a panel with min_slots bars
        pad = (min_slots - len(groups)) / 2
        ax.set_ylim(-0.5 - pad, len(groups) - 0.5 + pad)
    if annotate_small is True:             # write the counts right of the bar
        for yy, g in zip(y, groups):
            if small_txt[g]:
                ax.text(101, yy, small_txt[g], va="center", fontsize=9, color=INK, clip_on=False)
    ax.set_xlabel(f"share of rated {unit} (%)", color=INK)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=len(cats), frameon=False, fontsize=9)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)


# ── 2. overview statistics ──────────────────────────────────────────────────

def overview(view_papers, pairs):
    cohorts = {}
    for name, fn in (("Healthy PET cohort", rl.HEALTHY_FILTER_CSV), ("Overweight PET cohort", rl.OBESE_FILTER_CSV),
                     ("Lung cancer PET cohort", rl.LUNG_FILTER_CSV)):
        if Path(fn).exists():
            cohorts[name] = set(rl.load_edge_filter(fn))
            COHORT_N_ORGANS[name] = len(pd.read_csv(fn, index_col=0).columns)   # lung cancer: 11 organs (55 pairs), the others 13 (78 pairs)
    rows, per_view = [], {}
    for slug, name in VIEWS:
        pp = view_papers[slug]
        counts = [len(v) for v in pp.values()]
        oo = oo_map(pp)
        nonzero = sorted(v for v in oo.values() if v > 0)
        thr = kneedle_elbow(nonzero) if len(nonzero) >= 3 else 0.0
        sel = {p for p, v in oo.items() if v > 0 and v >= thr}
        per_view[slug] = {"counts": counts, "oo": nonzero, "thr": thr, "sel": sel}
        row = {"view": name, "pairs with >=1 paper": sum(1 for c in counts if c), "paper-pair assignments": sum(counts),
               "unique papers": len(unique_pmids(pp)), "median papers per pair (pairs with papers)":
               float(np.median([c for c in counts if c])) if any(counts) else 0.0,
               "OO elbow threshold": round(thr, 4), "OO-elbow edges": len(sel)}
        for cname, cset in cohorts.items():
            row[f"edges also in {cname}"] = len(sel & cset)
        rows.append(row)
    return pd.DataFrame(rows), per_view, cohorts


def draw_overview(df, per_view, cohorts, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=200)
    fig.patch.set_facecolor("white")
    names = [VIEW_NAME[s] for s, _ in VIEWS]
    cols = [VIEW_COLOR[s] for s, _ in VIEWS]
    x = np.arange(len(names))

    ax = axes[0, 0]
    w = 0.38
    ax.bar(x - w / 2, df["pairs with >=1 paper"], w, color=cols, label="pairs with at least one paper")
    ax.bar(x + w / 2, df["OO-elbow edges"], w, color=cols, alpha=0.5, label="edges above the OO elbow")
    for xi, a, b in zip(x, df["pairs with >=1 paper"], df["OO-elbow edges"]):
        ax.text(xi - w / 2, a, str(a), ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, b, str(b), ha="center", va="bottom", fontsize=9)
    for cname, cset in cohorts.items():
        ax.axhline(len(cset), ls="--", lw=1, color=MUTED)
        ax.text(len(names) - 0.45, len(cset), f"  {cname}: {len(cset)} edges", ha="left", va="center", fontsize=8, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(names) + 1.5)
    ax.set_title("A  Edges per view (of 91 organ pairs)", loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[0, 1]
    data = [[c for c in per_view[s]["counts"] if c] or [0] for s, _ in VIEWS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(rng.normal(i, 0.07, len(d)), d, s=9, color=cols[i - 1], alpha=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("papers per organ pair (log)")
    ax.set_title("B  Papers per organ pair (pairs with at least one paper)", loc="left", fontsize=13, color=INK)

    ax = axes[1, 0]
    if cohorts:
        k = len(cohorts)
        w = 0.8 / k
        for j, (cname, cset) in enumerate(cohorts.items()):
            vals = [len(per_view[s]["sel"] & cset) for s, _ in VIEWS]
            tot = [len(per_view[s]["sel"]) for s, _ in VIEWS]
            ax.bar(x + (j - (k - 1) / 2) * w, vals, w, color=[COHORT_COLOR.get(cname, "#fbbf24")] * len(x),
                   alpha=0.9, label=f"{cname} ({len(cset)} edges, {COHORT_N_ORGANS.get(cname, '?')} organs)")
            for xi, v, t in zip(x + (j - (k - 1) / 2) * w, vals, tot):
                ax.text(xi, v, f"{v}/{t}", ha="center", va="bottom", fontsize=8, rotation=90)    # rotated: three bars per view
        ax.margins(y=0.14)
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("OO-elbow edges also in the cohort")
    ax.set_title("C  Overlap of literature edges with the PET cohort networks (shared / literature edges)", loc="left", fontsize=13, color=INK)

    ax = axes[1, 1]
    data = [per_view[s]["oo"] or [0.0] for s, _ in VIEWS]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], cols):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    for i, s in enumerate([s for s, _ in VIEWS], 1):
        ax.hlines(per_view[s]["thr"], i - 0.4, i + 0.4, colors=INK, linestyles="--", lw=1)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Otsuka-Ochiai coefficient")
    ax.set_title("D  Edge strength (non-zero OO); dashed = elbow threshold", loc="left", fontsize=13, color=INK)

    for a in axes.flat:
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        a.grid(axis="y", color=GRID)
        a.set_axisbelow(True)
    fig.suptitle("Overview of the condition views", fontsize=17, y=0.995, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {sp._rel(out_path)}")


def draw_table(df, out_path):
    show = df.copy()
    for c in show.columns:
        if c == "OO elbow threshold":
            show[c] = show[c].map(lambda v: f"{v:.4f}")
        elif show[c].dtype.kind in "fi":
            show[c] = show[c].map(lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}")
    import textwrap
    n = len(show.columns)
    fig, ax = plt.subplots(figsize=(3.0 + 1.75 * (n - 1), 0.6 * (len(show) + 3)), dpi=200)
    ax.axis("off")
    heads = ["\n".join(textwrap.wrap(c, 12)) for c in show.columns]
    widths = [0.2] + [0.8 / (n - 1)] * (n - 1)
    tb = ax.table(cellText=show.values, colLabels=heads, colWidths=widths, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 2.3)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.9)
        elif c == 0:
            cell.set_text_props(weight="bold", color=INK)
            slug = VIEWS[r - 1][0]
            cell.set_facecolor(VIEW_COLOR[slug] + "33")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {sp._rel(out_path)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Remove the outputs of the previous run first: on this PC an existing PNG of a synced folder could not be
    # opened for overwriting ("OSError: [Errno 22]"), a fresh file is always fine.
    for old in list(OUT.rglob("*.png")) + list(STAT.glob("*.xlsx")) + list(STAT.glob("*.csv")):
        try:
            old.unlink()
        except OSError as exc:
            print(f"  [!] could not remove {old.name}: {exc}")
    results, labels, pairs, view_papers = load_data()
    print(f"[i] {len(pairs)} organ pairs | {sum(len(v) for v in view_papers['all'].values())} paper-pair assignments | "
          f"{len(unique_pmids(view_papers['all']))} unique papers | labels from {rl.CONDITION_LLM_MODEL}")

    # 1. funnel -------------------------------------------------------------------
    print("[i] paper funnel")
    steps, cond_rows, status_rows, label_rows, per_pair, rules = build_funnel(results, labels, pairs, view_papers)
    validation = load_validation()
    mol_validation = load_molecule_validation()
    cit_validation = load_citation_validation()
    organ_overview = organ_links(view_papers["all"])
    if cit_validation:
        print(f"[i] citation validation: {sum(1 for r in cit_validation[0] if r[4])}/{len(cit_validation[0])} sentences rated, "
              f"{sum(1 for r in cit_validation[1] if r)}/{len(cit_validation[1])} citations rated (bar only when all are)")
    else:
        print("[i] citation validation: no sentence rated yet in validation/citation_validation_workbook.xlsx -> panel G not drawn")
    sheets = {"Funnel": pd.DataFrame(steps), "Conditions": pd.DataFrame(cond_rows), "LLM answers": pd.DataFrame(status_rows),
              "Usable labels": pd.DataFrame(label_rows), "Per pair": pd.DataFrame(per_pair)}
    if validation:
        sheets["Validation"] = validation_table(validation)
    if mol_validation:
        sheets["Validation molecules"] = molecule_table(mol_validation)
        sheets["Molecule flags"] = molecule_flags(mol_validation)
    if cit_validation:
        sheets["Validation citations"] = citation_table(cit_validation)
        sheets["Citation flags"] = citation_flags(cit_validation)
    sheets["Papers per organ"] = pd.DataFrame(
        [{"organ": o, "unique papers (All papers)": n} for o, n in sorted(organ_overview[0].items(), key=lambda kv: -kv[1])])
    sheets["Rules in force"] = pd.DataFrame(rules)
    with pd.ExcelWriter(STAT / "paper_funnel.xlsx", engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:60] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(90, max(10, w + 2))
            ws.freeze_panes = "A2"
    for name, df in sheets.items():
        df.to_csv(STAT / f"paper_funnel_{name.lower().replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
    print(f"  [ok] {sp._rel(STAT / 'paper_funnel.xlsx')}")
    print(pd.DataFrame(steps)[["step", "paper-pair assignments", "unique papers", "pairs with >=1 paper"]].to_string(index=False))
    draw_funnel(steps, cond_rows, status_rows, OUT / "paper_funnel.png", validation, mol_validation, cit_validation, organ_overview,
                panel_dir=OUT / "paper_funnel_panels")

    # 2. overview -------------------------------------------------------------------
    print("[i] overview statistics")
    df, per_view, cohorts = overview(view_papers, pairs)
    df.to_csv(STAT / "overview_table.csv", index=False, encoding="utf-8-sig")
    draw_overview(df, per_view, cohorts, OUT / "overview_statistics.png")
    draw_table(df, OUT / "overview_table.png")

    # 3. per view figures --------------------------------------------------------------
    types = {slug: load_types(slug) for slug, _ in VIEWS}
    ref_pairs = [p for p in pairs if view_papers["all"][p]]
    all_count, _ = sp._connection_type_counts(types["all"], ref_pairs)
    order = sorted((k for k in sp.CONNECTION_TYPE_LABELS if k != sp.UNCLASSIFIED_KEY), key=lambda k: -all_count.get(k, 0))
    order.append(sp.UNCLASSIFIED_KEY)
    colors = sp._type_colors(order[:-1])
    overview_rows, type_rows = [], []
    for slug, name in VIEWS:
        print(f"[i] view {slug}")
        organ_papers, links = organ_links(view_papers[slug])
        title = f"Metabolic Reference Network - {name}"
        subtitle = "Total papers found per organ (union of the papers of all its organ-organ axes in this view)"
        d = OUT / slug
        sp.draw_chord_diagram(organ_papers, links, title, d / "sankey.png")
        sp.draw_barplot(organ_papers, title, subtitle, d / "papers_per_organ_barplot.png")
        overview_rows.append((name, organ_papers, links))
        vp = [p for p in pairs if view_papers[slug][p]]
        if types[slug]:
            tc, otc = sp._connection_type_counts(types[slug], vp)
            sp.draw_connection_type_barplot(tc, order, colors, title,
                                            f"How often each connection type was assigned ({len(vp)} organ pairs, LLM-classified, up to 3 types per pair)",
                                            d / "connection_type_frequency_barplot.png")
            sp.draw_organ_connection_type_stacked_barplot(otc, order, colors, title,
                                                          "Connection types touching each organ (stacked; a pair with several types counts toward each)",
                                                          d / "connection_type_per_organ_stacked_barplot.png")
            type_rows.append((name, tc, otc))
        else:
            print(f"  [!] no connection types for {slug}")
    print("[i] grids")
    sp.draw_reference_overview_grid(overview_rows, OUT / "papers_per_organ_overview_grid.png")
    if type_rows:
        sp.draw_connection_type_grid(type_rows, order, colors, OUT / "connection_types_grid.png")
    print(f"\n[ok] everything written to {sp._rel(OUT)}")


if __name__ == "__main__":
    main()
