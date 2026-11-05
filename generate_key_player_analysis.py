"""
Key player figures and enrichment for the LLM-based condition views (port of generate_key_player_pieplots.py of the predecessor project and the condition
enrichment of generate_key_player_analysis.py of the predecessor project, which were written for the keyword-based healthy / obese networks).

Views: All papers + the seven condition views (No condition, Healthy, Obesity/metabolic/diabetes, Cardiovascular, Cancers, Infectious,
Other diseases), each over the 91 organ pairs with the papers of that view (same-sentence filter, 20-year window).

1. Pie plots  (hormones | metabolites | proteins/transporters per view)
   Share of the mentions of every key player, summed over all organ pairs of the view (key_players counts of the view's result file;
   a paper that matches several pairs counts once per pair, as in the original pies). Top 15 players named, the rest = "Other".
   -> key_player_analysis/key_player_pies/<view>_key_players_pie.png   (600 dpi), key_player_analysis/key_players_grid.png
   -> key_player_analysis/tables/key_player_pie_counts.xlsx
2. Condition enrichment  (counting unit: unique papers, a paper counts once per player)
   For every key player: Fisher's exact test (two-sided), Benjamini-Hochberg FDR within each category, log2 odds ratio with a 0.5
   continuity correction, terms mentioned in >= 5 papers of the two groups. Comparisons: each condition vs the papers of the six
   other conditions ("rest"; papers without a usable LLM label are in neither group), and Healthy vs Obesity/metabolic/diabetes.
   Every paper has exactly one label, so the groups never overlap.
   -> key_player_analysis/enrichment/key_player_enrichment_<condition>_vs_rest.png, ..._healthy_vs_obesity.png
   -> key_player_analysis/tables/key_player_condition_enrichment.xlsx, key_player_enrichment_healthy_vs_obesity.csv

Usage (from this folder):  python generate_key_player_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_static_outputs as g                                     # noqa: E402  (data, views, colours)
import run_metabolic_lit_search as rl                                   # noqa: E402
from Literature_Search import llm_condition_filter as cf                # noqa: E402
from Literature_Search.pubmed_search import extract_key_players         # noqa: E402

D = HERE / "reference_network_only_metabolic"
OUT = HERE / "key_player_analysis"
PIE_DIR, ENR_DIR, TAB_DIR = OUT / "key_player_pies", OUT / "enrichment", OUT / "tables"

CATS = ("hormones", "metabolites", "proteins")
CAT_LABEL = {"hormones": "Hormones", "metabolites": "Metabolites", "proteins": "Proteins / Transporters"}
VIEWS = [("all", "All papers")] + [(c["slug"], c["name"]) for c in cf.CONDITIONS]
TOP_N = 15                  # named slices per pie
MIN_PAPERS = 5              # min papers (condition + rest) for a term to be tested
TOP_BARS = 15               # bars per direction in the enrichment figures
DPI = 600                   # single pies
GRID_MAX_MPIX = 150         # the grid is physically large: choose its dpi so that it stays below ~150 megapixels (PIL's limit is ~180)
OTHER_COLOR = "#64748b"
_TAB20 = matplotlib.colormaps["tab20"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# ── 1. pie plots ─────────────────────────────────────────────────────────────

def view_results(slug: str) -> dict:
    path = rl.OUTPUT_JSON if slug == "all" else D / f"metabolic_literature_results_{slug}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(results: dict) -> dict[str, dict[str, float]]:
    totals = {c: defaultdict(float) for c in CATS}
    for e in results.values():
        kp = e.get("key_players") or {}
        for cat in CATS:
            for term, n in (kp.get(f"{cat}_counts") or {}).items():
                if term.strip():
                    totals[cat][term.strip()] += float(n)
    return {c: dict(v) for c, v in totals.items()}


def top_n_plus_other(counts: dict, n: int = TOP_N):
    items = sorted(counts.items(), key=lambda x: -x[1])
    labels, values = [t for t, _ in items[:n]], [v for _, v in items[:n]]
    if items[n:]:
        labels.append("Other")
        values.append(sum(v for _, v in items[n:]))
    return labels, values


def draw_pie_ax(ax, counts: dict, category: str) -> None:
    ax.set_facecolor("white")
    n_players = sum(1 for v in counts.values() if v > 0)
    n_mentions = sum(counts.values())
    title = f"{CAT_LABEL[category]}\n{n_players:,} distinct players recovered ({n_mentions:,.0f} mentions)"
    if not counts or n_mentions == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="#94a3b8", fontsize=13)
        ax.set_title(title, fontsize=14, pad=12, color="#1e293b", fontweight="bold")
        ax.axis("off")
        return
    labels, values = top_n_plus_other(counts)
    total = sum(values)
    has_other = labels[-1] == "Other"
    n_named = len(labels) - (1 if has_other else 0)
    colors = [_TAB20(i / max(n_named - 1, 1)) for i in range(n_named)] + ([OTHER_COLOR] if has_other else [])
    wedges, _ = ax.pie(values, colors=colors, startangle=90, wedgeprops={"linewidth": 0.8, "edgecolor": "white"}, counterclock=False)
    ax.legend(wedges, [f"{lbl}  ({v / total * 100:.1f}%)" for lbl, v in zip(labels, values)], loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=9, frameon=True, framealpha=0.95, edgecolor="#e2e8f0",
              handlelength=1.2, handletextpad=0.5, columnspacing=1.0)
    ax.set_title(title, fontsize=14, pad=12, color="#1e293b", fontweight="bold")


def save_fig(fig, out: Path, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):          # a PNG that was just viewed can be locked for a moment on Windows
        try:
            fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)
    print("  [ok]", out.relative_to(HERE))


def draw_single_pie(kp: dict, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.patch.set_facecolor("white")
    for ax, cat in zip(axes, CATS):
        draw_pie_ax(ax, kp.get(cat, {}), cat)
    fig.suptitle(f"{title}\nKey Player Distribution Across All Organ–Organ Connections", fontsize=15, y=1.01, color="#1e293b")
    save_fig(fig, out, DPI)


def draw_grid(rows: list, out: Path) -> None:
    n = len(rows)
    row_h = 10.0
    dpi = int(min(400, math.sqrt(GRID_MAX_MPIX * 1e6 / (22 * row_h * n))))
    fig, axes = plt.subplots(n, 3, figsize=(22, row_h * n), dpi=dpi, gridspec_kw={"hspace": 0.42, "wspace": 0.18, "top": 0.972, "bottom": 0.012})
    fig.patch.set_facecolor("white")
    for i, (label, kp) in enumerate(rows):
        for j, cat in enumerate(CATS):
            draw_pie_ax(axes[i, j], kp.get(cat, {}), cat)
        axes[i, 0].text(-0.18, 0.5, label, transform=axes[i, 0].transAxes, rotation=90, ha="center", va="center",
                        fontsize=15, fontweight="bold", color="#1e293b")
    fig.suptitle("Key Player Distribution by Condition", fontsize=20, y=0.995, color="#1e293b")
    print(f"  grid: {n} rows at {dpi} dpi")
    save_fig(fig, out, dpi)


def pies(results_by_view: dict) -> pd.DataFrame:
    rows, long = [], []
    for slug, name in VIEWS:
        res = results_by_view[slug]
        kp = aggregate(res)
        n_pairs = sum(1 for e in res.values() if e.get("papers"))
        n_papers = len({str(p["pmid"]) for e in res.values() for p in e.get("papers", [])})
        draw_single_pie(kp, f"Metabolic Reference Network — {name} ({n_papers:,} papers, {n_pairs} pairs)",
                        PIE_DIR / f"{slug}_key_players_pie.png")
        rows.append((f"{name}\n({n_papers:,} papers, {n_pairs} pairs)", kp))
        for cat in CATS:
            tot = sum(kp[cat].values())
            for rank, (term, n) in enumerate(sorted(kp[cat].items(), key=lambda x: -x[1]), 1):
                long.append({"view": name, "category": cat, "rank": rank, "key player": term, "mentions": n,
                             "share of the category's mentions (%)": round(100 * n / tot, 3) if tot else np.nan})
    draw_grid(rows, OUT / "key_players_grid.png")
    return pd.DataFrame(long)


# ── 2. enrichment (unique papers) ────────────────────────────────────────────

def _logfact(n: int) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])


def fisher_two_sided(a: int, n1: int, c: int, n2: int, lf: np.ndarray) -> float:
    """2x2 [[a, n1-a], [c, n2-c]]; two-sided exact p (sum of tables no likelier than observed). Copied from generate_key_player_analysis.py of the predecessor project."""
    k, n = a + c, n1 + n2
    lo, hi = max(0, k - n2), min(k, n1)
    xs = np.arange(lo, hi + 1)
    lp = (lf[n1] - lf[xs] - lf[n1 - xs]) + (lf[n2] - lf[k - xs] - lf[n2 - k + xs]) - (lf[n] - lf[k] - lf[n - k])
    obs = lp[a - lo]
    return float(min(1.0, np.exp(lp[lp <= obs + 1e-9]).sum()))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q values. Copied from generate_key_player_analysis.py of the predecessor project."""
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1]):
        run = min(run, p[idx] * n / rank)
        q[idx] = run
    return q


def paper_players(papers: dict) -> dict:
    out = {}
    for i, (pm, p) in enumerate(papers.items()):
        kp = extract_key_players([p])
        out[pm] = {c: set(kp.get(f"{c}_counts", {})) for c in CATS}
        if (i + 1) % 4000 == 0:
            print(f"   players extracted for {i + 1:,}/{len(papers):,} papers", flush=True)
    return out


def enrichment(pp: dict, set_a: set, set_b: set, min_papers: int = MIN_PAPERS) -> pd.DataFrame:
    """Terms enriched in set_a (the condition) vs set_b. Positive log2 odds ratio = more in set_a."""
    na, nb = len(set_a), len(set_b)
    lf = _logfact(na + nb)
    rows = []
    for cat in CATS:
        ca, cb = Counter(), Counter()
        for pm in set_a:
            ca.update(pp[pm][cat])
        for pm in set_b:
            cb.update(pp[pm][cat])
        res = []
        for t in sorted(t for t in set(ca) | set(cb) if ca[t] + cb[t] >= min_papers):    # sorted: a set has no fixed order
            a, c = ca[t], cb[t]
            res.append((t, a, c, fisher_two_sided(a, na, c, nb, lf),
                        math.log2(((a + .5) / (na - a + .5)) / ((c + .5) / (nb - c + .5)))))
        if not res:
            continue
        q = bh(np.array([r[3] for r in res]))
        for (t, a, c, p, lor), qq in zip(res, q):
            rows.append({"category": cat, "term": t, "n_papers_A": a, "N_A": na, "pct_A": round(100 * a / na, 2),
                         "n_papers_B": c, "N_B": nb, "pct_B": round(100 * c / nb, 2), "log2_odds_ratio": round(lor, 3),
                         "p_value": p, "q_value_BH": qq, "significant_q<0.05": bool(qq < 0.05)})
    return pd.DataFrame(rows)


def plot_enrichment(df: pd.DataFrame, name_a: str, name_b: str, color_a: str, color_b: str, na: int, nb: int, out: Path,
                    short_b: str | None = None, short_a: str | None = None) -> None:
    """One figure per comparison: hormones | metabolites | proteins. The percentages (% of papers mentioning the term, A vs B) are written
    right at the end of each bar (bars extending to the left: next to the middle line), the same as in the combined figure."""
    sb = short_b or name_b                       # short names of the groups for the panel titles and axis labels
    sa = short_a or name_a
    fig, axes = plt.subplots(1, 3, figsize=(30, 12.2), layout="constrained")          # same size and fonts as one row of the combined figure
    for ax, cat in zip(axes, CATS):
        draw_enrichment_panel(ax, df, cat, None, sa, sb, color_a, color_b)
    fig.suptitle(f"Condition-specific key players: {name_a} (n={na:,}) vs {name_b} (n={nb:,})\n"
                 f"Fisher's exact test, Benjamini–Hochberg q < 0.05, top {TOP_BARS} per side shown; numbers next to the bars = % of papers "
                 f"mentioning the term ({name_a} vs {sb})", fontsize=FONTS["suptitle"])      # placed by the constrained layout
    fig.get_layout_engine().set(w_pad=0.3, h_pad=0.25, wspace=0.03, rect=(0, 0, 0.972, 1))
    for _ in range(3):                                      # the panel widths depend on the limits and vice versa: iterate until stable
        fig.canvas.draw()
        for ax in axes:
            fit_enrichment_xlim(fig, ax)
    fig.canvas.draw()
    save_fig(fig, out, 300)


FONTS = {"title": 15, "tick": 13, "pct": 12.5, "xlabel": 13, "suptitle": 18, "row": 19}     # fonts of the combined figure
SHORT_A = {"obesity": "Obesity"}                                                                 # short condition name in the panel titles (the long one stays in the figure title)
PCT_GAP_PT = 5                                                                                 # gap between a bar end and its percentages


def draw_enrichment_panel(ax, df: pd.DataFrame, cat: str, letter: str | None, name_a: str, sb: str, color_a: str, color_b: str,
                          fonts: dict = FONTS) -> None:
    """One panel (one category) of an enrichment figure: top bars per side among the significant terms (q < 0.05).
    The key player names are the y tick labels; '% of papers (condition vs rest)' is written right at the end of each bar
    (bars extending to the left: next to the middle line)."""
    d = df[(df.category == cat) & (df["significant_q<0.05"])]
    top_a = d[d.log2_odds_ratio > 0].nlargest(TOP_BARS, "log2_odds_ratio")
    top_b = d[d.log2_odds_ratio < 0].nsmallest(TOP_BARS, "log2_odds_ratio")
    sel = pd.concat([top_b.sort_values("log2_odds_ratio", ascending=False), top_a.sort_values("log2_odds_ratio")]).reset_index(drop=True)
    head = rf"$\mathbf{{{letter}}}$  " if letter else ""
    ax.set_title(head + CAT_LABEL[cat] + f"\n{int((d.log2_odds_ratio > 0).sum())} enriched in {name_a} · "
                 f"{int((d.log2_odds_ratio < 0).sum())} enriched in {sb}  ({int((df.category == cat).sum())} tested)",
                 fontsize=fonts["title"], fontweight="bold", pad=10)
    if sel.empty:
        ax.text(0.5, 0.5, "no significant terms (q < 0.05)", ha="center", va="center", transform=ax.transAxes, fontsize=fonts["tick"])
        ax.axis("off")
        return
    y = np.arange(len(sel))
    ax.barh(y, sel.log2_odds_ratio, color=[color_a if v > 0 else color_b for v in sel.log2_odds_ratio], height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels([r.term for r in sel.itertuples()], fontsize=fonts["tick"])
    anns = []
    for yi, r in zip(y, sel.itertuples()):               # percentages right at the end of the bar (bars to the left: next to the middle line)
        anns.append(ax.annotate(f"{r.pct_A:.1f}% vs {r.pct_B:.1f}%", xy=(max(r.log2_odds_ratio, 0), yi), xytext=(PCT_GAP_PT, 0),
                                textcoords="offset points", ha="left", va="center", fontsize=fonts["pct"], color="#334155",
                                annotation_clip=False))
    ax.tick_params(axis="x", labelsize=fonts["tick"] - 1)
    ax.axvline(0, color="#334155", lw=0.9)
    lim = max(abs(sel.log2_odds_ratio).max() * 1.1, 1)
    ax.set_xlim(-lim, lim * 1.7)                         # provisional; fit_enrichment_xlim sets the final limits once the panel width is known
    ax._kp_fit = (sel.log2_odds_ratio.to_numpy(), anns)
    ax.set_xlabel(f"log2 odds ratio  (← enriched in {sb} | enriched in {name_a} →)", fontsize=fonts["xlabel"])


def fit_enrichment_xlim(fig, ax) -> None:
    """x limits of a panel so that the percentage text at the end of every bar stays inside the axes (needs the current panel width):
    left = longest bar to the left; right = smallest R with  bar_end_i + text_width_i * (L + R) / axes_width <= R  for every bar i."""
    if not hasattr(ax, "_kp_fit"):
        return
    vals, anns = ax._kp_fit
    rend = fig.canvas.get_renderer()
    width_px = ax.get_window_extent(rend).width
    left = max(-min(vals.min(), 0) * 1.08, 0.08 * abs(vals).max(), 0.3)
    need = 0.0
    for v, a in zip(vals, anns):
        c = (a.get_window_extent(rend).width + 6 * fig.dpi / 72) / width_px         # text + small safety gap as a share of the axes width
        need = max(need, (max(v, 0) + c * left) / (1 - c))
    ax.set_xlim(-left, need)


def plot_enrichment_combined(parts: list, out: Path) -> None:
    """parts: [(df, name_a, short_a, color_a), ...] condition-vs-rest tables, one row each; columns = hormones | metabolites | proteins."""
    w, h = 30.0, 12.2 * len(parts)
    fig, axes = plt.subplots(len(parts), 3, figsize=(w, h), layout="constrained")
    letters = iter("ABCDEFGHIJKL")
    for r, (df, name_a, short_a, color_a) in enumerate(parts):
        for c, cat in enumerate(CATS):
            draw_enrichment_panel(axes[r, c], df, cat, next(letters), short_a, "rest", color_a, OTHER_COLOR)
    names = " and ".join(n for _, n, _, _ in parts)
    fig.suptitle(f"Condition-specific key players: {names} vs the papers of the other six conditions\n"
                 f"Fisher's exact test, Benjamini–Hochberg q < 0.05, top {TOP_BARS} per side shown; numbers next to the bars = % of papers "
                 f"mentioning the term (condition vs rest)", fontsize=FONTS["suptitle"])      # placed by the constrained layout
    # constrained layout: every column gets the room its own labels need; the left margin is kept free for the row labels
    fig.get_layout_engine().set(w_pad=0.3, h_pad=0.25, wspace=0.03, hspace=0.05, rect=(0.026, 0, 0.966, 1))
    for _ in range(3):                                      # the panel widths depend on the limits and vice versa: iterate until stable
        fig.canvas.draw()
        for ax in axes.ravel():
            fit_enrichment_xlim(fig, ax)
    fig.canvas.draw()
    for r, (df, name_a, _, _) in enumerate(parts):         # row labels (after the layout, so the positions are final)
        pos = axes[r, 0].get_position()
        na, nb = int(df["N_A"].iloc[0]), int(df["N_B"].iloc[0])
        fig.text(0.012, (pos.y0 + pos.y1) / 2, f"{name_a} vs rest\n(n = {na:,} vs {nb:,})", rotation=90, ha="center", va="center",
                 fontsize=FONTS["row"], fontweight="bold", color="#1e293b")
    # very good quality: a vector PDF (text stays text, sharp at any size) and a PNG with the highest dpi that keeps it below
    # ~150 megapixels (image libraries refuse much larger files)
    dpi = int(min(500, math.sqrt(145e6 / (w * 1.06 * h * 1.06))))
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("  [ok]", out.with_suffix(".pdf").relative_to(HERE))
    print(f"  png at {dpi} dpi")
    save_fig(fig, out, dpi)


def combined_enrichment_figure() -> None:
    """Cancers vs rest and Obesity/metabolic/diabetes vs rest in one 2 x 3 figure, from the saved enrichment table (no recomputation)."""
    x = pd.ExcelFile(TAB_DIR / "key_player_condition_enrichment.xlsx")
    parts = [(x.parse("cancers_vs_rest"), "Cancers", "Cancers", g.VIEW_COLOR["cancers"]),
             (x.parse("obesity_vs_rest"), "Obesity/metabolic/diabetes", "Obesity", g.VIEW_COLOR["obesity"])]
    plot_enrichment_combined(parts, ENR_DIR / "key_player_enrichment_cancers_and_obesity_vs_rest.png")


def single_enrichment_figures() -> None:
    """Redraw the single comparison figures (7 x condition vs rest, healthy vs obesity) from the saved enrichment table (no recomputation)."""
    x = pd.ExcelFile(TAB_DIR / "key_player_condition_enrichment.xlsx")
    for c in cf.CONDITIONS:
        df = x.parse(f"{c['slug']}_vs_rest")
        plot_enrichment(df, c["name"], "the papers of the other six conditions", g.VIEW_COLOR[c["slug"]], OTHER_COLOR,
                        int(df["N_A"].iloc[0]), int(df["N_B"].iloc[0]), ENR_DIR / f"key_player_enrichment_{c['slug']}_vs_rest.png", short_b="rest",
                        short_a=SHORT_A.get(c["slug"]))
    df = x.parse("healthy_vs_obesity")
    plot_enrichment(df, "healthy", "obesity/metabolic/diabetes", g.VIEW_COLOR["healthy"], g.VIEW_COLOR["obesity"],
                    int(df["N_A"].iloc[0]), int(df["N_B"].iloc[0]), ENR_DIR / "key_player_enrichment_healthy_vs_obesity.png", short_b="obesity")


def condition_enrichment(results: dict, labels: dict) -> tuple[dict, dict]:
    papers = {}
    for e in results.values():
        for p in e.get("papers", []):
            papers[str(p["pmid"])] = p
    print(f"[i] {len(papers):,} unique papers; extracting key players per paper")
    pp = paper_players(papers)
    members = {}
    for c in cf.CONDITIONS:
        flt = cf.LLMConditionFilter(c["slug"], labels)
        members[c["slug"]] = {pm for pm, p in papers.items() if flt.accepts(p)}
    labelled = set().union(*members.values())
    print(f"    papers with a usable label: {len(labelled):,} | " + ", ".join(f"{c['name']} {len(members[c['slug']]):,}" for c in cf.CONDITIONS))
    tables = {}
    for c in cf.CONDITIONS:
        slug, name = c["slug"], c["name"]
        a, b = members[slug], labelled - members[slug]
        df = enrichment(pp, a, b)
        df.insert(0, "comparison", f"{name} vs rest")
        tables[f"{slug}_vs_rest"] = (df, name, "rest", len(a), len(b), g.VIEW_COLOR[slug])
        plot_enrichment(df, name, "the papers of the other six conditions", g.VIEW_COLOR[slug], OTHER_COLOR, len(a), len(b),
                        ENR_DIR / f"key_player_enrichment_{slug}_vs_rest.png", short_b="rest", short_a=SHORT_A.get(slug))
    a, b = members["healthy"], members["obesity"]
    df = enrichment(pp, a, b)
    df.insert(0, "comparison", "Healthy vs Obesity/metabolic/diabetes")
    tables["healthy_vs_obesity"] = (df, "healthy", "obesity/metabolic/diabetes", len(a), len(b), None)
    plot_enrichment(df, "healthy", "obesity/metabolic/diabetes", g.VIEW_COLOR["healthy"], g.VIEW_COLOR["obesity"], len(a), len(b),
                    ENR_DIR / "key_player_enrichment_healthy_vs_obesity.png", short_b="obesity")
    return tables, members


def write_enrichment_tables(tables: dict) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for key, (df, na_, nb_, na, nb, _) in tables.items():
        for cat in CATS:
            d = df[df.category == cat]
            summary.append({"comparison": df["comparison"].iloc[0], "category": cat, "n papers (A = condition)": na, "n papers (B = rest)": nb,
                            "terms tested": len(d), "significant q<0.05": int(d["significant_q<0.05"].sum()),
                            "enriched in A": int(((d["significant_q<0.05"]) & (d.log2_odds_ratio > 0)).sum()),
                            "enriched in B": int(((d["significant_q<0.05"]) & (d.log2_odds_ratio < 0)).sum())})
    with pd.ExcelWriter(TAB_DIR / "key_player_condition_enrichment.xlsx") as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name="summary", index=False)
        for key, (df, *_rest) in tables.items():
            df.sort_values(["category", "q_value_BH", "p_value", "term"], kind="stable").to_excel(xw, sheet_name=key[:31], index=False)
            xw.sheets[key[:31]].freeze_panes = "A2"
    tables["healthy_vs_obesity"][0].sort_values(["category", "q_value_BH", "p_value", "term"], kind="stable").to_csv(
        TAB_DIR / "key_player_enrichment_healthy_vs_obesity.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))


def main() -> None:
    if "--combined-only" in sys.argv:                # only the combined enrichment figure, from the saved table
        combined_enrichment_figure()
        return
    if "--singles-only" in sys.argv:                 # only the 8 single enrichment figures, from the saved table
        single_enrichment_figures()
        return
    only_pies = "--pies-only" in sys.argv          # redraw just the pies and the grid (the enrichment takes a few minutes)
    for d in ((PIE_DIR,) if only_pies else (PIE_DIR, ENR_DIR, TAB_DIR)):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            try:
                old.unlink()
            except OSError as exc:
                print(f"  [!] could not remove {old.name}: {exc}")
    results_by_view = {slug: view_results(slug) for slug, _ in VIEWS}
    print("[i] pie plots")
    long = pies(results_by_view)
    with pd.ExcelWriter(TAB_DIR / "key_player_pie_counts.xlsx") as xw:
        long[long["rank"] <= TOP_N].to_excel(xw, sheet_name=f"top {TOP_N}", index=False)
        long.to_excel(xw, sheet_name="all players", index=False)
        for sh in xw.sheets.values():
            sh.freeze_panes = "A2"
    long[long["rank"] <= TOP_N].to_csv(TAB_DIR / "key_player_pie_counts_top15.csv", index=False, encoding="utf-8-sig")
    if only_pies:
        print("\n[ok] pies and grid redrawn (--pies-only)")
        return
    print("[i] condition enrichment")
    labels = cf.load_condition_labels(rl.CONDITION_LABELS, rl.CONDITION_LLM_MODEL)
    tables, _ = condition_enrichment(results_by_view["all"], labels)
    write_enrichment_tables(tables)
    combined_enrichment_figure()
    print("\n[ok] key player analysis written to", OUT.relative_to(HERE))


if __name__ == "__main__":
    main()
