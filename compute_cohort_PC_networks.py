"""
Partial correlation networks of the healthy and overweight FDG-PET cohorts (13 organs).

Input   ../PET_data/Data FDG healthy 20250910 anonym.csv   (PET_data lies one folder above the atlas; columns "<organ> cB", BMI; one row per scan)
Split   BMI <= 25 = healthy (n = 112), BMI > 25 = overweight (n = 85); rows without BMI are dropped.
        Missing organ values are replaced by the within-group column mean before the fit.
        (The code and the files still call the overweight group "obese": obese_cohort_*.csv, OBESE_FILTER_CSV.)

Outputs (reference_network_only_metabolic/), for healthy_ and obese_:
  *_cohort_PC_connections.csv    full partial-correlation matrix (r)
  *_cohort_PC_significance.csv   binary significance mask (p < 0.05, uncorrected)
  *_cohort_PC_pvalues.csv        raw p-values
  *_cohort_connections.csv       binary adjacency (13 x 13), used as the "cohort" edge sets of the dashboard

The lung cancer cohort is computed by compute_lung_cancer_PC_network.py (same partial-correlation function).
The dashboard reads these files in build_viz() (Visualisation/ui_patches.py), nothing is patched into an HTML file here.
Taken over from the predecessor project (compute_cohort_PC_networks.py); the HTML patching of that script was removed.

    python compute_cohort_PC_networks.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

HERE = Path(__file__).parent
DATA_FILE = HERE.parent / "PET_data" / "Data FDG healthy 20250910 anonym.csv"     # PET_data lies one folder above the atlas
OUT_DIR = HERE / "reference_network_only_metabolic"

ORGAN_COLS = [
    "AdrGl max cB",
    "Brain cB",
    "intestine cB",
    "heart cB",
    "Kidney cB",
    "Liver cB",
    "Lungs max cB",
    "Pancreas cB",
    "Gluteus cB",
    "Spleen cB",
    "Fat sub cB",
    "BM max cB",
    "Thyroid cB",
]

# Canonical organ names of the dashboard (same order as ORGAN_COLS)
ORGAN_NAMES = [
    "Adrenal Glands",
    "Brain",
    "Small Intestine",
    "Heart",
    "Kidney",
    "Liver",
    "Lung",
    "Pancreas",
    "Muscle",
    "Spleen",
    "WAT",
    "Bone Marrow",
    "Thyroid",
]


def partial_correlation_network(
    data: pd.DataFrame,
    calc_significance: str | float = "significance",
    alpha: float = 0.05,
) -> dict:
    cov = np.cov(data.values, rowvar=False)
    precision = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    partial_corr_df = pd.DataFrame(partial_corr, index=data.columns, columns=data.columns)

    results = {"partial_corr": partial_corr_df}

    n, p = data.shape
    dof = n - p

    if isinstance(calc_significance, str):
        # Clamp to avoid divide-by-zero at |r|=1 (diagonal handled by fill_diagonal)
        r2_clamp = np.clip(partial_corr ** 2, 0, 1 - 1e-10)
        t_stats = partial_corr * np.sqrt(dof / (1 - r2_clamp))
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=dof))
        np.fill_diagonal(p_values, 1.0)
        pval_df = pd.DataFrame(p_values, index=data.columns, columns=data.columns)
        results["p_values"] = pval_df

        if calc_significance == "significance":
            sig_mask = (p_values < alpha).astype(int)
        elif calc_significance == "bonferroni":
            m = p * (p - 1) / 2
            sig_mask = (p_values < alpha / m).astype(int)
        else:
            raise ValueError("calc_significance must be 'significance' or 'bonferroni'")

        np.fill_diagonal(sig_mask, 0)
        results["edges"] = pd.DataFrame(sig_mask, index=data.columns, columns=data.columns)

    elif isinstance(calc_significance, (float, int)):
        edge_mask = np.abs(partial_corr) >= calc_significance
        np.fill_diagonal(edge_mask, 0)
        results["edges"] = pd.DataFrame(edge_mask.astype(int), index=data.columns, columns=data.columns)

    return results


def _to_binary_adjacency_csv(sig_df: pd.DataFrame) -> str:
    """Render significance DataFrame as a 0/1 full adjacency CSV string."""
    organs = list(sig_df.columns)
    lines = ["," + ",".join(organs)]
    for row in organs:
        cells = [row]
        for col in organs:
            if row == col:
                cells.append("0")
            else:
                cells.append(str(int(sig_df.loc[row, col])))
        lines.append(",".join(cells))
    return "\n".join(lines)


def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"[!] {DATA_FILE} not found: the PET_data folder must lie one folder above the atlas")
    df = pd.read_csv(DATA_FILE)
    print(f"[i] Loaded {len(df)} rows from {DATA_FILE.name}")

    df = df[df["BMI"].notna()].copy()
    print(f"[i] {len(df)} rows have BMI values")

    col_map = dict(zip(ORGAN_COLS, ORGAN_NAMES))

    healthy_raw = df[df["BMI"] <= 25][ORGAN_COLS].rename(columns=col_map).copy()
    obese_raw = df[df["BMI"] > 25][ORGAN_COLS].rename(columns=col_map).copy()
    print(f"[i] Healthy (BMI <= 25): {len(healthy_raw)} patients")
    print(f"[i] Overweight (BMI > 25): {len(obese_raw)} patients")

    healthy = healthy_raw.fillna(healthy_raw.mean())
    obese = obese_raw.fillna(obese_raw.mean())

    for group_name, data in [("healthy", healthy), ("obese", obese)]:
        print(f"\n[i] Computing partial correlation network - {group_name} (n={len(data)})")
        res = partial_correlation_network(data, calc_significance="significance", alpha=0.05)

        pc, sig, pv = res["partial_corr"], res["edges"], res["p_values"]
        n_edges = int(sig.values[np.triu_indices_from(sig.values, k=1)].sum())
        print(f"    Significant edges (p < 0.05): {n_edges}")

        pc.to_csv(OUT_DIR / f"{group_name}_cohort_PC_connections.csv")
        sig.to_csv(OUT_DIR / f"{group_name}_cohort_PC_significance.csv")
        pv.to_csv(OUT_DIR / f"{group_name}_cohort_PC_pvalues.csv")
        (OUT_DIR / f"{group_name}_cohort_connections.csv").write_text(_to_binary_adjacency_csv(sig), encoding="utf-8")
        print(f"[ok] {group_name}_cohort_PC_connections / _PC_significance / _PC_pvalues / _connections written to {OUT_DIR.name}/")

        organs = list(sig.columns)
        for i, a in enumerate(organs):
            for j, b in enumerate(organs):
                if j > i and sig.loc[a, b] == 1:
                    print(f"      {a} - {b}: r={pc.loc[a, b]:.4f}, p={pv.loc[a, b]:.4f}")


if __name__ == "__main__":
    main()
