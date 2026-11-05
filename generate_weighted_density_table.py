"""
Weighted density of all networks (PET partial-correlation networks and literature networks) on the SAME organs.

Weighted density D = (sum of w_ij over all unordered organ pairs i < j) / (N (N - 1) / 2), absent edges w_ij = 0.  All eligible pairs are
included (also zero-weight pairs), the diagonal is excluded.  Literature networks: descriptive, no hypothesis test.

Organs     the minimum set of organs shared by all networks = the 10 organs present in all three PET networks (the lung cancer network has
           11 organs, healthy / overweight 13, the literature networks 14): Brain, Heart, Kidney, Liver, Muscle, Pancreas, Small Intestine,
           Spleen, Thyroid, WAT  ->  N = 10, 45 pairs.
Networks   PET (3)        healthy (BMI <= 25), overweight (BMI > 25), lung cancer;  w_ij = |partial correlation| of the significant edges
                          (p < 0.05, uncorrected), every other pair 0  (same weight as "|r|-weighted" in generate_density_resampling_tests.py of the predecessor project)
           literature (8) All papers, No condition, Healthy, Obesity/metabolic/diabetes, Cardiovascular, Cancers, Infectious, Other diseases;
                          w_ij = Otsuka-Ochiai coefficient of the pair in the view (0 without papers), computed on the 14-organ network
                          and then restricted to the 10 organs (as in generate_condition_qap_table.py)
Supplementary weights (sheet "Supplementary weights"): PET full |r| (all pairs, no significance threshold) and literature OO >= elbow
           (Kneedle elbow of the view, as in the overview figure).
PET and literature weights are on different scales (|r| vs OO): compare the networks within a group, not between PET and literature.

Writes statistical_tests/weighted_density.csv / .xlsx and weighted_density_table.png.
Usage:  python generate_weighted_density_table.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_pet_network_comparison as pnc                          # noqa: E402  (PET matrices, literature OO vectors)
import generate_static_outputs as g                                    # noqa: E402  (view names)

OUT = HERE / "statistical_tests"
DPI = 300
DARK, MID, LIGHT = "#1e293b", "#475569", "#94a3b8"


def density_row(name, group, weight, w, n_org, n_pairs):
    w = np.asarray(w, float)
    assert len(w) == n_pairs == n_org * (n_org - 1) // 2 and (w >= 0).all()
    return {"network": name, "group": group, "edge weight w_ij": weight, "N organs": n_org, "pairs N(N-1)/2": n_pairs,
            "non-zero pairs": int((w > 0).sum()), "sum of w_ij": float(w.sum()), "weighted density D": float(w.sum() / n_pairs)}


def main():
    pet = pnc.load_pet()
    organs = sorted(set.intersection(*[set(pc.columns) for pc, _ in pet.values()]))
    pairs = pnc.pairs_of(organs)
    n, m = len(organs), len(pairs)
    lit_pairs, lit, thr = pnc.literature_networks()
    lit_idx = {p: i for i, p in enumerate(lit_pairs)}
    li = [lit_idx[p] for p in pairs]

    main_rows, supp_rows = [], []
    for key, _ in pnc.COHORTS:
        pc, sig = pet[key]
        name = pnc.COHORT_LONG[key]
        main_rows.append(density_row(name, "PET partial correlation", "|r| of significant edges (p < 0.05), other pairs 0", np.abs(pnc.pet_vector(pc, sig, pairs, "sig")), n, m))
        supp_rows.append(density_row(name, "PET partial correlation", "|r| of all pairs (no significance threshold)", np.abs(pnc.pet_vector(pc, sig, pairs, "full")), n, m))
    for slug, vname in g.VIEWS:
        main_rows.append(density_row(vname, "literature", "OO (papers of the view), 0 without papers", lit[slug]["OO"][li], n, m))
        supp_rows.append(density_row(vname, "literature", f"OO of the edges at or above the elbow ({thr[slug]:.4f}), other pairs 0", lit[slug]["elbow"][li], n, m))

    df, sup = pd.DataFrame(main_rows), pd.DataFrame(supp_rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "weighted_density.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "weighted_density.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Weighted density", index=False)
        sup.to_excel(xw, sheet_name="Supplementary weights", index=False)
        pd.DataFrame({"pair": [f"{a} - {b}" for a, b in pairs]}).assign(
            **{r["network"] + " (" + r["group"] + ")": v for r, v in zip(
                main_rows, [np.abs(pnc.pet_vector(*pet[k], pairs, "sig")) for k, _ in pnc.COHORTS]
                + [lit[s]["OO"][li] for s, _ in g.VIEWS])}).to_excel(xw, sheet_name="Edge weights (45 pairs)", index=False)
    pd.set_option("display.width", 220)
    print(f"organs (N = {n}): {', '.join(organs)}; {m} pairs")
    print(df[["network", "group", "non-zero pairs", "sum of w_ij", "weighted density D"]].round(4).to_string(index=False))
    print(sup[["network", "non-zero pairs", "weighted density D"]].round(4).to_string(index=False))
    draw_table(df, organs)


def draw_table(df, organs):
    n_rows = len(df)
    fig_h = 2.1 + 0.34 * n_rows + 0.6 + 1.35          # last term: room for the footnote under the table
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=DPI)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.patch.set_facecolor("white")
    Y = lambda inches: 1 - inches / fig_h
    ax.text(0.0, Y(0.12), "Weighted density of the partial-correlation and literature networks", fontsize=11.5, fontweight="bold", color=DARK, va="top")
    ax.text(0.0, Y(0.45), f"same {len(organs)} organs in every network (N = {len(organs)}, {len(organs) * (len(organs) - 1) // 2} organ pairs)",
            fontsize=9, color=MID, va="top")
    xs = [0.0, 0.215, 0.545, 0.64, 0.755, 0.9]
    heads = ["Network", "Edge weight w$_{ij}$", "Non-zero\npairs", "Sum of w$_{ij}$", "Weighted\ndensity D"]
    xh = [xs[0], xs[1], xs[2] + 0.04, xs[3] + 0.06, xs[4] + 0.12]
    y_in = 1.05
    for x, h in zip(xh, heads):
        ax.text(x, Y(y_in), h, fontsize=9, fontweight="bold", color=DARK, va="center",
                ha="left" if x < 0.4 else "center", multialignment="center")
    ax.plot([0, 1], [Y(y_in + 0.22), Y(y_in + 0.22)], color=DARK, lw=0.9)
    y_in += 0.5
    prev, band = None, 0
    for _, r in df.iterrows():
        if r["group"] != prev:
            if prev is not None:
                y_in += 0.14
            band = 0
            ax.text(0.0, Y(y_in), {"PET partial correlation": "PET partial-correlation networks (FDG-PET)", "literature": "Literature networks (LLM condition views)"}[r["group"]],
                    fontsize=9, fontweight="bold", color="#059669" if r["group"].startswith("PET") else "#0284c7", va="center")
            y_in += 0.3
            prev = r["group"]
        if band % 2 == 0:
            ax.add_patch(plt.Rectangle((0, Y(y_in) - 0.5 * 0.31 / fig_h), 1, 0.31 / fig_h, fc="#f1f5f9", ec="none", zorder=0))
        wdef = "|r| of significant edges" if r["group"].startswith("PET") else "Otsuka–Ochiai (OO)"
        cells = [(r["network"], xs[0], "left", False), (wdef, xs[1], "left", False), (f"{r['non-zero pairs']} / {r['pairs N(N-1)/2']}", xh[2], "center", False),
                 (f"{r['sum of w_ij']:.3f}", xh[3], "center", False), (f"{r['weighted density D']:.4f}", xh[4], "center", True)]
        for txt, x, ha, bold in cells:
            ax.text(x + (0.01 if x == 0.0 else 0), Y(y_in), txt, fontsize=8.8, color=DARK, va="center", ha=ha, fontweight="bold" if bold else "normal")
        y_in += 0.31
        band += 1
    note = (r"Weighted density D = (Σ w_ij) / (N (N − 1) / 2): the sum of all edge weights w_ij over the unordered organ pairs (i < j), divided by the number of "
            "possible pairs; absent edges have w_ij = 0, all eligible pairs are included (also zero-weight pairs) and the diagonal is excluded. "
            f"N = {len(organs)}: the organs present in all networks ({', '.join(organs)}); the lung cancer network has no Adrenal Glands, Bone Marrow or Lung, "
            "so the 13-organ healthy / overweight networks and the 14-organ literature networks are restricted to these organs. "
            "PET: w_ij = |partial correlation| of the edges with p < 0.05 (uncorrected), all other pairs 0 (healthy BMI ≤ 25 and overweight BMI > 25 groups; lung cancer: SUV mean / aorta SUV mean). "
            "Literature: w_ij = Otsuka–Ochiai coefficient of the organ pair in the view (papers of the pair / sqrt(unique papers of the two organs)), 0 without papers; "
            "no hypothesis test (descriptive). The two groups have different weight scales (|r| vs OO), so densities are comparable within a group, not between PET and literature.")
    ax.text(0.0, 0.0, textwrap.fill(note, 170), fontsize=6.8, color=LIGHT, va="bottom")
    out = OUT / "weighted_density_table.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", out.relative_to(HERE))


if __name__ == "__main__":
    main()
