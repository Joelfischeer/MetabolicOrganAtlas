"""
Partial correlation network of the FDG-PET lung cancer cohort (one cohort, no BMI split).

Input   ../PET_data/Lung_cancer_moose_SUVs.csv   (PET_data lies one folder above the atlas; ';'-separated, MOOSE segmentations)
Steps   1. take the SUV *mean* of the organs listed in ORGAN_COLUMNS (not max / min)
        2. divide every organ column by the patient's own aorta_mean_automatic (aorta = reference, not a node)
        3. save the normalised SUV means  -> ../PET_data/Lung_cancer_SUVmean_aorta_normalized.csv
        4. replace the few missing values by the column mean (as for the healthy / overweight cohorts)
        5. partial correlation network, edges at p < 0.05 (uncorrected, same function as the healthy cohort)

Outputs (next to healthy_cohort_* / obese_cohort_* in reference_network_only_metabolic/):
        lung_cancer_cohort_PC_connections.csv    partial correlation matrix (r)
        lung_cancer_cohort_PC_significance.csv   0/1 mask, p < 0.05
        lung_cancer_cohort_PC_pvalues.csv        p-values
        lung_cancer_cohort_connections.csv       0/1 adjacency (binary reference, like the other cohorts)

The dashboard picks these files up in build_viz() (button "Lung cancer cohort").

    python compute_lung_cancer_PC_network.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist

HERE = Path(__file__).parent
PET_DIR = HERE.parent / "PET_data"          # PET_data lies one folder above the atlas (next to Metabolic_Reference_Atlas/)
DATA_FILE = PET_DIR / "Lung_cancer_moose_SUVs.csv"
SUV_OUT = PET_DIR / "Lung_cancer_SUVmean_aorta_normalized.csv"
OUT_DIR = HERE / "reference_network_only_metabolic"
PREFIX = "lung_cancer_cohort"

REFERENCE_COLUMN = "aorta_mean_automatic"

# dashboard organ name -> SUV-mean column of the lung cancer file.
# Adrenal Glands, Bone Marrow and Lung are not in this file; Colon is (it is not in the healthy PET data).
ORGAN_COLUMNS = {
    "Brain":           "brain_mean_automatic",
    "Colon":           "colon_mean_automatic",
    "Heart":           "heart_mean_automatic",
    "Kidney":          "kidneys_mean_automatic",
    "Liver":           "liver_mean_automatic",
    "Muscle":          "muscle_mean_automatic",
    "Pancreas":        "pancreas_mean_automatic",
    "Small Intestine": "small_bowel_mean_automatic",
    "Spleen":          "spleen_mean_automatic",
    "Thyroid":         "thyroid_gland_mean_automatic",
    "WAT":             "subcutaneous_fat_mean_automatic",
}


def partial_correlation_network(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Same computation as partial_correlation_network() in compute_cohort_PC_networks.py (this folder)
    (calc_significance="significance"): partial correlation from the pseudo-inverse of the covariance
    matrix, two-sided t-test with n - p degrees of freedom, no multiple-testing correction."""
    cov = np.cov(data.values, rowvar=False)
    precision = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)

    n, p = data.shape
    dof = n - p
    r2_clamp = np.clip(partial_corr ** 2, 0, 1 - 1e-10)
    t_stats = partial_corr * np.sqrt(dof / (1 - r2_clamp))
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=dof))
    np.fill_diagonal(p_values, 1.0)
    sig_mask = (p_values < alpha).astype(int)
    np.fill_diagonal(sig_mask, 0)

    as_df = lambda a: pd.DataFrame(a, index=data.columns, columns=data.columns)
    return {"partial_corr": as_df(partial_corr), "p_values": as_df(p_values), "edges": as_df(sig_mask)}


def binary_adjacency_csv(sig: pd.DataFrame) -> str:
    organs = list(sig.columns)
    lines = ["," + ",".join(organs)]
    for row in organs:
        lines.append(",".join([row] + ["0" if row == col else str(int(sig.loc[row, col])) for col in organs]))
    return "\n".join(lines)


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"[!] {DATA_FILE} not found: the PET_data folder must lie one folder above the atlas")
    raw = pd.read_csv(DATA_FILE, sep=";")
    print(f"[i] Loaded {len(raw)} patients from {DATA_FILE.name}")
    if raw["Patient"].duplicated().any():
        raise SystemExit("[!] duplicated patient ids")

    ref = raw[REFERENCE_COLUMN]
    print(f"[i] Reference {REFERENCE_COLUMN}: {int(ref.isna().sum())} missing, "
          f"min {ref.min():.3f}, max {ref.max():.3f}")
    suv = pd.DataFrame({organ: raw[col] / ref for organ, col in ORGAN_COLUMNS.items()})
    suv.insert(0, "Patient", raw["Patient"])
    suv.to_csv(SUV_OUT, index=False)
    print(f"[ok] Aorta-normalised SUV means ({len(ORGAN_COLUMNS)} organs) -> {SUV_OUT}")

    data = suv.drop(columns="Patient")
    n_missing = data.isna().sum()
    print("[i] Missing values replaced by the column mean:",
          ", ".join(f"{o} {int(n)}" for o, n in n_missing.items() if n) or "none")
    data = data.fillna(data.mean())

    print(f"[i] Computing partial correlation network (n={len(data)}, {data.shape[1]} organs)")
    res = partial_correlation_network(data, alpha=0.05)
    pc, sig, pv = res["partial_corr"], res["edges"], res["p_values"]
    pc.to_csv(OUT_DIR / f"{PREFIX}_PC_connections.csv")
    sig.to_csv(OUT_DIR / f"{PREFIX}_PC_significance.csv")
    pv.to_csv(OUT_DIR / f"{PREFIX}_PC_pvalues.csv")
    (OUT_DIR / f"{PREFIX}_connections.csv").write_text(binary_adjacency_csv(sig), encoding="utf-8")

    organs = list(sig.columns)
    edges = [(a, b) for i, a in enumerate(organs) for b in organs[i + 1:] if sig.loc[a, b] == 1]
    print(f"[ok] {len(edges)} significant edges (p < 0.05) of {len(organs) * (len(organs) - 1) // 2} possible "
          f"-> {OUT_DIR}")
    for a, b in edges:
        print(f"      {a} - {b}: r={pc.loc[a, b]:+.3f}, p={pv.loc[a, b]:.4f}")


if __name__ == "__main__":
    main()
