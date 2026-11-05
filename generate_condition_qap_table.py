"""
QAP correlation between the condition networks of the LLM-filtered reference network (without the All-papers network).

Networks: the seven condition views (No condition, Healthy, Obesity/metabolic/diabetes, Cardiovascular, Cancers, Infectious, Other diseases),
each over the same 14 organs / 91 organ pairs. Edge weight = OO (Otsuka-Ochiai) coefficient of the organ pair in that view:
    OO(a, b) = papers of the pair / sqrt(unique papers of organ a x unique papers of organ b)   (papers = papers of the view),
0 for pairs without papers. Weighted density = mean OO over all 91 pairs ("mean normalized literature support").

For every pair of networks: Pearson r between the two 91-pair weight vectors (upper triangles of the organ weight matrices) and a two-sided
QAP (Mantel-type) permutation p: the organ labels of network B are permuted (rows and columns together), r is recomputed, 10,000 permutations,
seed 42, p = (1 + #{|r_perm| >= |r_obs|}) / (10,000 + 1). `corr` and `qap_pvalue` are copied unchanged from
generate_matrix_pearson_table.py of the predecessor project (the earlier healthy/obese analysis of the keyword-based project) so that the method is identical;
only numpy is needed here.

Bonferroni correction: the 21 pairwise comparisons form one family; adjusted p = min(1, 21 x p). The smallest attainable adjusted p is
21 / 10,001 = 0.0021 (no permutation reached the observed |r|).

Writes statistical_tests/ (next to this script):
  condition_networks_qap_correlation.csv / .xlsx   long table (network A, network B, r, p QAP) + matrices + densities (xlsx sheets)
  condition_networks_qap_table.png                 table for the manuscript (lower triangle r, upper triangle QAP p, weighted density)
  condition_networks_qap_table_bonferroni.png      the same with the Bonferroni-adjusted QAP p in the upper triangle
Usage:  python generate_condition_qap_table.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_static_outputs as g                                # noqa: E402  (data loading, views, OO)

OUT = HERE / "statistical_tests"
N_PERM = 10_000
SEED = 42
DPI = 300
ORDER = [("nocondition", "No condition"), ("healthy", "Healthy"), ("obesity", "Obesity/metabolic/diabetes"),
         ("cardiovascular", "Cardiovascular"), ("cancers", "Cancers"), ("infectious", "Infectious"), ("other", "Other diseases")]


# ── copied unchanged from generate_matrix_pearson_table.py of the predecessor project (numpy part) ─────────────────────────

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def qap_pvalue(x, y, pair_list, n_perm=N_PERM, seed=SEED):
    """Two-sided QAP (Mantel-type) permutation p-value of the Pearson r between two matrices.

    x, y are the upper-triangle vectors over `pair_list` (organ pairs). The organ labels of network B are
    permuted (rows and columns together), which keeps the structure of each network but breaks the organ
    correspondence; r is recomputed for every permutation. p = (1 + #{|r_perm| >= |r_obs|}) / (n_perm + 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    organs = sorted({o for pair in pair_list for o in pair})
    n, m = len(organs), len(pair_list)
    pos = {o: i for i, o in enumerate(organs)}
    I = np.array([pos[a] for a, _ in pair_list])
    J = np.array([pos[b] for _, b in pair_list])
    assert m == n * (n - 1) // 2 and len({(min(i, j), max(i, j)) for i, j in zip(I, J)}) == m
    idx = np.full((n, n), -1)                                   # position of each organ pair in the vector
    idx[I, J] = idx[J, I] = np.arange(m)
    perms = np.random.default_rng(seed).permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)
    y_perm = y[idx[perms[:, I], perms[:, J]]]                   # (n_perm, m): B'[i, j] = B[perm[i], perm[j]]
    xc = x - x.mean()
    yc = y_perm - y_perm.mean(axis=1, keepdims=True)
    r_perm = yc @ xc / (np.linalg.norm(xc) * np.linalg.norm(yc, axis=1))
    r_obs = corr(x, y)
    return float((1 + np.sum(np.abs(r_perm) >= abs(r_obs) - 1e-12)) / (n_perm + 1))


def fmt_p(p):
    floor = 1 / (N_PERM + 1)
    if p != p:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── data ───────────────────────────────────────────────────────────────────────────────────────

def condition_vectors():
    results, labels, pairs, view_papers = g.load_data()
    vec, info = {}, []
    for slug, name in ORDER:
        pp = view_papers[slug]
        oo = g.oo_map(pp)
        vec[name] = np.array([oo.get(p, 0.0) for p in pairs], dtype=float)
        info.append({"network": name, "unique papers": len(g.unique_pmids(pp)), "paper-pair assignments": sum(len(v) for v in pp.values()),
                     "pairs with papers": sum(1 for v in pp.values() if v),
                     "weighted density (OO, %)": 100 * float(vec[name].mean())})
    return pairs, vec, pd.DataFrame(info)


def main():
    pairs, vec, dens = condition_vectors()
    names = [n for _, n in ORDER]
    rows = []
    R = pd.DataFrame(np.nan, index=names, columns=names)
    P = pd.DataFrame(np.nan, index=names, columns=names)
    PA = pd.DataFrame(np.nan, index=names, columns=names)           # Bonferroni-adjusted
    n_tests = len(names) * (len(names) - 1) // 2                      # 21 comparisons = one family
    for a, b in combinations(names, 2):
        r, p = corr(vec[a], vec[b]), qap_pvalue(vec[a], vec[b], pairs)
        p_adj = min(1.0, p * n_tests)
        R.loc[a, b] = R.loc[b, a] = r
        P.loc[a, b] = P.loc[b, a] = p
        PA.loc[a, b] = PA.loc[b, a] = p_adj
        rows.append({"network A": a, "network B": b, "pearson r (OO weights)": round(r, 4), "p (QAP)": p,
                     "p (QAP, Bonferroni)": p_adj, "significant after Bonferroni (p < 0.05)": bool(p_adj < 0.05),
                     "n comparisons": n_tests, "n organ pairs": len(pairs), "n permutations": N_PERM, "seed": SEED})
    long = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    long.to_csv(OUT / "condition_networks_qap_correlation.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "condition_networks_qap_correlation.xlsx", engine="openpyxl") as xw:
        long.to_excel(xw, sheet_name="Pairs", index=False)
        R.round(4).to_excel(xw, sheet_name="r matrix")
        P.to_excel(xw, sheet_name="QAP p matrix")
        PA.to_excel(xw, sheet_name="QAP p Bonferroni matrix")
        dens.round(4).to_excel(xw, sheet_name="Weighted density", index=False)
    print(dens.round(2).to_string(index=False))
    print(long.sort_values("pearson r (OO weights)", ascending=False).to_string(index=False))
    draw_table(names, R, P, dens)
    draw_table(names, R, PA, dens, adjusted=True, n_tests=n_tests)
    print(f"Bonferroni ({n_tests} comparisons): significant {int(long['significant after Bonferroni (p < 0.05)'].sum())} of {n_tests}; "
          f"not significant: {long.loc[~long['significant after Bonferroni (p < 0.05)'], ['network A', 'network B']].agg(' vs '.join, axis=1).tolist()}")
    return long, dens


def draw_table(names, R, P, dens, adjusted=False, n_tests=21):
    n = len(names)
    fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=DPI)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.1, 5.4)
    ax.axis("off")
    short = {"Obesity/metabolic/diabetes": "Obesity/metab./diabetes"}
    x0, w_name, w_dens, w_cell = 0.2, 2.5, 1.35, 1.34
    y_head, row_h = 4.15, 0.46
    caption = ("OO-weighted condition networks of the LLM-labelled literature (14 organs, 91 organ pairs): weighted density (mean OO over all pairs) and QAP correlation "
               "between the organ weight matrices. Lower triangle: Pearson r of the 91 pair weights; upper triangle: "
               + (f"Bonferroni-adjusted two-sided QAP p (raw p x {n_tests} comparisons, capped at 1; " if adjusted else "two-sided QAP p (")
               + f"{N_PERM:,} permutations of the organ labels, seed {SEED}).")
    import textwrap
    ax.text(x0, 5.25, "\n".join(textwrap.wrap(caption, 150)), ha="left", va="top", fontsize=8.5, fontweight="bold", color="#0f172a", linespacing=1.3)
    ax.plot([x0, x0 + w_name + w_dens + n * w_cell], [y_head + 0.42, y_head + 0.42], color="#0f172a", lw=1.8)
    ax.text(x0 + 0.05, y_head, "Network", fontsize=9, fontweight="bold", va="center")
    ax.text(x0 + w_name + w_dens / 2, y_head, "Weighted\ndensity (%)", fontsize=8.5, fontweight="bold", va="center", ha="center")
    for j, nm in enumerate(names):
        ax.text(x0 + w_name + w_dens + (j + 0.5) * w_cell, y_head, {"Obesity/metabolic/diabetes": "Obesity/metab.\n/diabetes", "Other diseases": "Other\ndiseases"}.get(nm, nm), fontsize=8, fontweight="bold",
                va="center", ha="center")
    ax.plot([x0, x0 + w_name + w_dens + n * w_cell], [y_head - 0.34, y_head - 0.34], color="#0f172a", lw=0.9)
    for i, nm in enumerate(names):
        y = y_head - 0.34 - (i + 0.5) * row_h
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((x0, y - row_h / 2), w_name + w_dens + n * w_cell, row_h, fc="#f1f5f9", ec="none", zorder=0))
        ax.text(x0 + 0.05, y, nm, fontsize=9, fontweight="bold", va="center")
        ax.text(x0 + w_name + w_dens / 2, y, f"{dens.loc[dens['network'] == nm, 'weighted density (OO, %)'].iloc[0]:.2f}", fontsize=9, va="center", ha="center")
        for j, nm2 in enumerate(names):
            cx = x0 + w_name + w_dens + (j + 0.5) * w_cell
            if i == j:
                ax.text(cx, y, "–", fontsize=9, va="center", ha="center", color="#94a3b8")
            elif i > j:
                ax.text(cx, y, f"{R.loc[nm, nm2]:.3f}", fontsize=9, va="center", ha="center", color="#0f172a", fontweight="bold")
            else:
                pv = P.loc[nm, nm2]
                txt = (f"{pv:.3f}" if adjusted else fmt_p(pv))
                ax.text(cx, y, txt, fontsize=9, va="center", ha="center",
                        color="#0f172a" if (adjusted and pv < 0.05) else "#475569", fontweight="bold" if (adjusted and pv < 0.05) else "normal")
    y_end = y_head - 0.34 - n * row_h
    ax.plot([x0, x0 + w_name + w_dens + n * w_cell], [y_end, y_end], color="#0f172a", lw=1.8)
    note = (f"Bold = significant after Bonferroni correction (adjusted p < 0.05). The smallest attainable adjusted p is {n_tests}/10,001 = {n_tests / 10001:.4f} "
            "(no permutation reached the observed |r|)." if adjusted else
            "p < 0.001 = no permutation reached the observed |r| (smallest attainable p = 1/10,001).")
    ax.text(x0, y_end - 0.2, note, fontsize=7.6, color="#475569", va="top")
    out = OUT / ("condition_networks_qap_table_bonferroni.png" if adjusted else "condition_networks_qap_table.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", out.name)


if __name__ == "__main__":
    main()
