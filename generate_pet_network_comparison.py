"""
Network comparisons of the FDG-PET partial-correlation networks: healthy vs overweight vs lung cancer, and each of them against the
literature networks (All papers + the seven LLM condition views).  Port of generate_matrix_pearson_table.py of the predecessor project (keyword-based project,
healthy vs obese) to the LLM project with the third cohort.

PET networks   healthy (BMI <= 25, 13 organs), overweight (BMI > 25, 13 organs), lung cancer (11 organs: no Adrenal Glands / Bone
               Marrow / Lung, but Colon).  Matrices: *_cohort_PC_connections.csv (signed partial correlation r) and
               *_cohort_PC_significance.csv (p < 0.05, uncorrected).  Two weight definitions:
                 "full"  every pair carries its partial correlation (signed r)
                 "sig"   only significant edges keep their r, all other pairs 0
Literature     the 8 views (All papers, No condition, Healthy, Obesity/metabolic/diabetes, Cardiovascular, Cancers, Infectious, Other),
               OO (Otsuka-Ochiai) weight of each organ pair within the view (as in generate_condition_qap_table.py), and
               "OO >= elbow" = OO kept only for the edges at or above the Kneedle elbow of that view (as in the overview figure).
Comparison     Pearson r between the upper triangles (organ pairs) of two matrices over the SAME organs, naive p (treats the pairs as
               independent: descriptive) and a two-sided QAP p (organ labels of network B permuted, 10,000 permutations, seed 42;
               `corr` / `qap_pvalue` as in generate_condition_qap_table.py).  No multiple-testing correction.
               PET vs literature: the literature weights are unsigned, so the PET partial correlations enter as |r|.
Organ sets     "common"  the 10 organs present in all three PET networks (Brain, Heart, Kidney, Liver, Muscle, Pancreas, Small
                         Intestine, Spleen, Thyroid, WAT; 45 pairs) -> every cohort and view is compared on the same pairs
               "own"     the organs of the PET network itself (healthy / overweight 13, lung cancer 11), literature restricted to them.
Edge overlap   significant PET edges vs the literature OO-elbow edges: shared / only PET / only literature, Jaccard index.

Writes statistical_tests/:
  pet_network_comparison.xlsx / pet_vs_pet.csv / pet_vs_literature.csv / pet_vs_literature_edge_overlap.csv / pet_weights_common_organs.csv
  pet_vs_pet_table.png, pet_vs_literature_grid.png
Usage:  python generate_pet_network_comparison.py
"""
from __future__ import annotations

import sys
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_static_outputs as g                                    # noqa: E402  (views, OO, paper sets)
from threshold_utils import kneedle_elbow                              # noqa: E402

REF = HERE / "reference_network_only_metabolic"
OUT = HERE / "statistical_tests"
N_PERM, SEED, DPI = 10_000, 42, 300
DARK, MID, LIGHT = "#1e293b", "#475569", "#94a3b8"
COHORTS = [("healthy", "Healthy"), ("obese", "Overweight"), ("lung_cancer", "Lung cancer")]
COHORT_LONG = {"healthy": "Healthy (BMI ≤ 25)", "obese": "Overweight (BMI > 25)", "lung_cancer": "Lung cancer"}
MATCHED = {"healthy": "healthy", "obese": "obesity", "lung_cancer": "cancers"}     # PET cohort -> literature view of the same condition
PET_DEF = {"sig": "significant edges only", "full": "full partial correlation"}
LIT_W = {"OO": "OO", "elbow": "OO ≥ elbow"}


# ── method (same as generate_condition_qap_table.py / generate_matrix_pearson_table.py of the predecessor project) ─────────

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def qap_pvalue(x, y, pair_list, n_perm=N_PERM, seed=SEED):
    """Two-sided QAP (Mantel-type) permutation p of the Pearson r between two organ-pair vectors: the organ labels of network B
    are permuted (rows and columns together), r is recomputed; p = (1 + #{|r_perm| >= |r_obs|}) / (n_perm + 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    organs = sorted({o for pair in pair_list for o in pair})
    n, m = len(organs), len(pair_list)
    pos = {o: i for i, o in enumerate(organs)}
    I = np.array([pos[a] for a, _ in pair_list])
    J = np.array([pos[b] for _, b in pair_list])
    assert m == n * (n - 1) // 2 and len({(min(i, j), max(i, j)) for i, j in zip(I, J)}) == m
    idx = np.full((n, n), -1)
    idx[I, J] = idx[J, I] = np.arange(m)
    perms = np.random.default_rng(seed).permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)
    y_perm = y[idx[perms[:, I], perms[:, J]]]
    xc = x - x.mean()
    yc = y_perm - y_perm.mean(axis=1, keepdims=True)
    r_perm = yc @ xc / (np.linalg.norm(xc) * np.linalg.norm(yc, axis=1))
    r_obs = corr(x, y)[0]
    return float((1 + np.sum(np.abs(r_perm) >= abs(r_obs) - 1e-12)) / (n_perm + 1))


def fmt_p(p, qap=False):
    """p as text; for QAP p-values the smallest attainable value 1 / (n_perm + 1) is shown as '≤ 0.0001'."""
    if p != p:
        return "—"
    if qap and p <= 1 / (N_PERM + 1) * (1 + 1e-9):
        return f"≤{1 / (N_PERM + 1):.4f}"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ── data ────────────────────────────────────────────────────────────────────────────────────

def load_pet():
    """cohort -> (signed partial-correlation DataFrame, 0/1 significance DataFrame), organs as index / columns."""
    out = {}
    for key, _ in COHORTS:
        pc = pd.read_csv(REF / f"{key}_cohort_PC_connections.csv", index_col=0)
        sig = pd.read_csv(REF / f"{key}_cohort_PC_significance.csv", index_col=0).reindex(index=pc.index, columns=pc.columns)
        out[key] = (pc, sig)
    return out


def pairs_of(organs):
    organs = sorted(organs)
    return [(a, b) for a, b in combinations(organs, 2)]


def pet_vector(pc, sig, pairs, kind):
    r = np.array([pc.loc[a, b] for a, b in pairs], float)
    s = np.array([sig.loc[a, b] for a, b in pairs]) == 1
    return r if kind == "full" else np.where(s, r, 0.0)


def pet_edges(pc, sig, pairs):
    """signed r of the significant edges only (0 elsewhere) -> sign vector"""
    return pet_vector(pc, sig, pairs, "sig")


def literature_networks():
    results, labels, lit_pairs, view_papers = g.load_data()
    vec, thr = {}, {}
    for slug, name in g.VIEWS:
        oo = g.oo_map(view_papers[slug])
        w = np.array([oo.get(p, 0.0) for p in lit_pairs], float)
        nonzero = sorted(v for v in w if v > 0)
        t = kneedle_elbow(nonzero) if len(nonzero) >= 3 else 0.0
        keep = (w > 0) & (w >= t)
        vec[slug] = {"OO": w, "elbow": np.where(keep, w, 0.0)}
        thr[slug] = float(t)
    return lit_pairs, vec, thr


# ── analysis ────────────────────────────────────────────────────────────────────────────────

def pet_vs_pet(pet):
    organs = {k: set(pet[k][0].columns) for k in pet}
    common = set.intersection(*organs.values())
    rows = []
    for (ka, na), (kb, nb) in combinations(COHORTS, 2):
        bases = [("common organs of the 3 cohorts", common)]
        if ka == "healthy" and kb == "obese":
            bases.append(("own organs of both networks", organs[ka] & organs[kb]))
        for basis, org in bases:
            pairs = pairs_of(org)
            (pca, sga), (pcb, sgb) = pet[ka], pet[kb]
            ea, eb = pet_edges(pca, sga, pairs), pet_edges(pcb, sgb, pairs)
            both = (ea != 0) & (eb != 0)
            for kind in ("sig", "full"):
                x, y = pet_vector(pca, sga, pairs, kind), pet_vector(pcb, sgb, pairs, kind)
                r, p = corr(x, y)
                rows.append({"comparison": f"{COHORT_LONG[ka]} vs {COHORT_LONG[kb]}", "network A": na, "network B": nb,
                             "organs basis": basis, "n organs": len(org), "n pairs": len(pairs), "matrix": PET_DEF[kind],
                             "significant edges A": int((ea != 0).sum()), "significant edges B": int((eb != 0).sum()),
                             "shared edges": int(both.sum()), "shared with the same sign": int((both & (np.sign(ea) == np.sign(eb))).sum()),
                             "Jaccard (edges)": float(both.sum() / max(1, ((ea != 0) | (eb != 0)).sum())),
                             "pearson r": r, "p (naive)": p, "p (QAP)": qap_pvalue(x, y, pairs),
                             "n permutations": N_PERM, "seed": SEED})
    return sorted(common), pd.DataFrame(rows)


def pet_vs_literature(pet, common):
    lit_pairs, lit, thr = literature_networks()
    lit_idx = {p: i for i, p in enumerate(lit_pairs)}
    rows, overlap = [], []
    for key, name in COHORTS:
        pc, sig = pet[key]
        for basis, org in (("common organs of the 3 cohorts", set(common)), ("own organs of the PET network", set(pc.columns))):
            pairs = pairs_of(org)
            li = [lit_idx[p] for p in pairs]
            for slug, vname in g.VIEWS:
                e_pet = pet_edges(pc, sig, pairs) != 0
                e_lit = lit[slug]["elbow"][li] > 0
                overlap.append({"PET network": COHORT_LONG[key], "literature network": vname, "organs basis": basis, "n organs": len(org),
                                "n pairs": len(pairs), "PET significant edges": int(e_pet.sum()), "literature OO-elbow edges": int(e_lit.sum()),
                                "shared": int((e_pet & e_lit).sum()), "only PET": int((e_pet & ~e_lit).sum()),
                                "only literature": int((~e_pet & e_lit).sum()),
                                "Jaccard": float((e_pet & e_lit).sum() / max(1, (e_pet | e_lit).sum())),
                                "same condition": MATCHED[key] == slug, "elbow threshold": thr[slug]})
                for kind in ("sig", "full"):
                    x = np.abs(pet_vector(pc, sig, pairs, kind))
                    for wk, wl in LIT_W.items():
                        y = lit[slug][wk][li]
                        r, p = corr(x, y)
                        rows.append({"PET network": COHORT_LONG[key], "literature network": vname, "same condition": MATCHED[key] == slug,
                                     "organs basis": basis, "n organs": len(org), "n pairs": len(pairs),
                                     "PET matrix (|r|)": PET_DEF[kind], "literature weight": wl,
                                     "PET non-zero pairs": int((x > 0).sum()), "literature non-zero pairs": int((y > 0).sum()),
                                     "pearson r": r, "p (naive)": p, "p (QAP)": qap_pvalue(x, y, pairs),
                                     "n permutations": N_PERM, "seed": SEED})
    return pd.DataFrame(rows), pd.DataFrame(overlap), thr


def weights_long(pet, common):
    pairs = pairs_of(common)
    rows = []
    for a, b in pairs:
        row = {"organ 1": a, "organ 2": b}
        for key, _ in COHORTS:
            pc, sig = pet[key]
            row[f"{key} r"] = pc.loc[a, b]
            row[f"{key} significant"] = int(sig.loc[a, b])
        rows.append(row)
    return pd.DataFrame(rows)


# ── figures ─────────────────────────────────────────────────────────────────────────────────

def draw_pet_table(df, common):
    n = len(df)
    n_blocks = df[["comparison", "organs basis"]].drop_duplicates().shape[0]
    fig_h = 2.6 + 0.33 * n + 0.45 * n_blocks
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=DPI)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.patch.set_facecolor("white")
    Y = lambda inches: 1 - inches / fig_h
    ax.text(0.0, Y(0.12), "PET partial-correlation networks: healthy vs overweight vs lung cancer", fontsize=11.5, fontweight="bold",
            color=DARK, va="top")
    xs = [0.0, 0.25, 0.42, 0.49, 0.585, 0.69, 0.78, 0.865, 0.945]
    heads = ["Networks compared", "Matrix", "Organs", "Pairs", "Sig. edges\n(A / B)", "Shared\n(same sign)", "Pearson r",
             "p (naive)", "p (QAP)"]
    y_in = 0.62
    for x, h in zip(xs, heads):
        ax.text(x, Y(y_in), h, fontsize=9, fontweight="bold", color=DARK, va="center", ha="left" if x < 0.4 else "center",
                multialignment="center")
    ax.plot([0, 1], [Y(y_in + 0.2), Y(y_in + 0.2)], color=DARK, lw=0.9)
    y_in += 0.45
    prev = None
    for _, r in df.iterrows():
        blk = (r["comparison"], r["organs basis"])
        if blk != prev:
            if prev is not None:
                y_in += 0.14
            band = 0
            tag = "common organs" if r["organs basis"].startswith("common") else "own organs"
            ax.text(0.0, Y(y_in), f"{r['comparison']}  ({tag})", fontsize=9, fontweight="bold", color="#059669", va="center")
            y_in += 0.3
            prev = blk
        if band % 2 == 0:
            ax.add_patch(plt.Rectangle((0, Y(y_in) - 0.5 * 0.31 / fig_h), 1, 0.31 / fig_h, fc="#f1f5f9", ec="none", zorder=0))
        cells = [(r["matrix"], xs[1]), (f"{r['n organs']}", xs[2]), (f"{r['n pairs']}", xs[3]),
                 (f"{r['significant edges A']} / {r['significant edges B']}", xs[4]),
                 (f"{r['shared edges']} ({r['shared with the same sign']})", xs[5]),
                 (f"{r['pearson r']:+.3f}", xs[6]), (fmt_p(r["p (naive)"]), xs[7]), (fmt_p(r["p (QAP)"], qap=True), xs[8])]
        for txt, x in cells:
            ax.text(x, Y(y_in), txt, fontsize=8.8, color=DARK, va="center", ha="left" if x < 0.4 else "center",
                    fontweight="bold" if x == xs[6] else "normal")
        y_in += 0.31
        band += 1
    note = ("Pearson correlation between the upper triangles (organ pairs) of two PET partial-correlation matrices over the same organs. "
            f"Common organs = the {len(common)} organs present in all three networks ({', '.join(common)}; {len(common) * (len(common) - 1) // 2} pairs); "
            "the lung cancer file has no Adrenal Glands, Bone Marrow or Lung (it has Colon, which the healthy / overweight PET data lack). "
            "Matrix: full = every pair with its signed partial correlation; significant edges only = pairs with p < 0.05 (uncorrected) keep their r, "
            "all others 0. Shared edges = significant in both networks (in brackets: with the same sign). Healthy and overweight: BMI ≤ 25 / > 25 groups "
            "of the healthy file, group-mean imputation; lung cancer: one cohort, SUV mean / aorta SUV mean, column-mean imputation. "
            "p (naive) = ordinary Pearson test, which treats the organ pairs as independent (they are not; descriptive). "
            f"p (QAP) = two-sided permutation p from {N_PERM:,} random permutations of the organ labels of network B (seed {SEED}); "
            f"smallest attainable value {1 / (N_PERM + 1):.4f}. No correction for multiple comparisons.")
    ax.text(0.0, 0.0, textwrap.fill(note, 165), fontsize=6.8, color=LIGHT, va="bottom")
    out = OUT / "pet_vs_pet_table.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", out.relative_to(HERE))


def draw_lit_grid(df):
    views = [(s, n) for s, n in g.VIEWS]
    d = df[(df["organs basis"].str.startswith("common")) & (df["literature weight"] == "OO")]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 3.9), dpi=DPI)
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("RdBu_r")
    for ax, kind in zip(axes, ("sig", "full")):
        sub = d[d["PET matrix (|r|)"] == PET_DEF[kind]]
        R = np.full((3, len(views)), np.nan)
        P = np.full_like(R, np.nan)
        for i, (key, _) in enumerate(COHORTS):
            for j, (slug, vname) in enumerate(views):
                row = sub[(sub["PET network"] == COHORT_LONG[key]) & (sub["literature network"] == vname)].iloc[0]
                R[i, j], P[i, j] = row["pearson r"], row["p (QAP)"]
        ax.imshow(R, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
        for i in range(3):
            for j in range(len(views)):
                p = P[i, j]
                txt = "n/a" if np.isnan(R[i, j]) else f"{R[i, j]:+.2f}\np = {fmt_p(p, qap=True)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.2, color=DARK,
                        fontweight="bold" if (p == p and p < 0.05) else "normal")
        for i, (key, _) in enumerate(COHORTS):                                   # box = literature view of the same condition
            j = [s for s, _ in views].index(MATCHED[key])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=DARK, lw=1.8))
        ax.set_xticks(range(len(views)))
        ax.set_xticklabels([n for _, n in views], fontsize=8.3, color=DARK, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(3))
        ax.set_yticklabels([COHORT_LONG[k] for k, _ in COHORTS], fontsize=9, color=DARK)
        ax.tick_params(length=0)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(("A  Significant PET edges only (|r|)" if kind == "sig" else "B  Full PET partial correlation (|r|)")
                     + "  vs  literature OO", loc="left", fontsize=10.5, fontweight="bold", color=DARK)
    fig.suptitle("PET networks vs literature networks: Pearson r over the 45 organ pairs of the 10 common organs "
                 "(QAP p; bold = p < 0.05; box = literature view of the same condition)", fontsize=10, color=MID, y=1.0)
    fig.tight_layout()
    out = OUT / "pet_vs_literature_grid.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", out.relative_to(HERE))


def main():
    pet = load_pet()
    common, vs_pet = pet_vs_pet(pet)
    vs_lit, overlap, thr = pet_vs_literature(pet, common)
    OUT.mkdir(exist_ok=True)
    vs_pet.to_csv(OUT / "pet_vs_pet.csv", index=False, encoding="utf-8-sig")
    vs_lit.to_csv(OUT / "pet_vs_literature.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(OUT / "pet_vs_literature_edge_overlap.csv", index=False, encoding="utf-8-sig")
    w = weights_long(pet, common)
    w.to_csv(OUT / "pet_weights_common_organs.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "pet_network_comparison.xlsx", engine="openpyxl") as xw:
        vs_pet.to_excel(xw, sheet_name="PET vs PET", index=False)
        vs_lit.to_excel(xw, sheet_name="PET vs literature", index=False)
        overlap.to_excel(xw, sheet_name="Edge overlap", index=False)
        w.to_excel(xw, sheet_name="PET weights (common organs)", index=False)
        pd.DataFrame({"view": [n for _, n in g.VIEWS], "OO elbow threshold": [thr[s] for s, _ in g.VIEWS]}).to_excel(
            xw, sheet_name="Literature elbow", index=False)
    pd.set_option("display.width", 250)
    print(f"common organs ({len(common)}): {', '.join(common)}")
    print(vs_pet[["comparison", "organs basis", "n pairs", "matrix", "significant edges A", "significant edges B", "shared edges",
                  "shared with the same sign", "pearson r", "p (naive)", "p (QAP)"]].round(4).to_string(index=False))
    sel = vs_lit[(vs_lit["organs basis"].str.startswith("common")) & (vs_lit["literature weight"] == "OO")
                 & (vs_lit["PET matrix (|r|)"] == PET_DEF["sig"])]
    print(sel.pivot(index="PET network", columns="literature network", values="pearson r").round(3).to_string())
    draw_pet_table(vs_pet, common)
    draw_lit_grid(vs_lit)


if __name__ == "__main__":
    main()
