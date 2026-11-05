"""
Post-processing of the generated dashboard HTML (applied by build_viz() after the
HTML and the cohort buttons have been written).

1. apply_ui_patches():  the interface refinements that had been made directly in the
   generated HTML (compact top bar, slide-in sidebar, reference-network mode with
   signed partial-correlation colouring, "r = ..." badge on connections, ...).
   They are stored as (new_block, old_block) text replacements in ui_patches.json, so a
   rebuild of the dashboard no longer loses them.
2. inject_weighted_cohort_matrices():  replaces the 0/1 cohort matrices behind the
   "Healthy cohort" / "Overweight cohort" / "Lung cancer cohort" buttons by the partial-correlation
   matrices (|r| of significant edges keeps its sign, everything else 0), read from the
   <condition>_cohort_PC_connections.csv / _PC_significance.csv files. Positive edges
   are drawn blue, negative edges red. Only the organs of the PET network are written
   (13 for healthy / overweight, no Colon; 11 for lung cancer, with Colon, no Adrenal Glands / Bone Marrow / Lung).
3. patch_cohort_reference_view():  organs without a significant edge (WAT in the overweight group)
   stay visible in the cohort reference view, and the buttons' hover texts.
"""
import json
import re
from pathlib import Path

import pandas as pd

PATCH_FILE = Path(__file__).with_name("ui_patches.json")


def apply_ui_patches(html_path: Path) -> None:
    text = Path(html_path).read_text(encoding="utf-8")
    patches = json.loads(PATCH_FILE.read_text(encoding="utf-8"))
    applied = skipped = 0
    for p in patches:
        n = text.count(p["new"])
        if n == 1:
            text = text.replace(p["new"], p["old"], 1)
            applied += 1
        elif text.count(p["old"]) == 1:
            skipped += 1                       # already in the patched state
        else:
            print(f"  [!] UI patch near line {p['where_new_line']} not applicable "
                  f"({n} matches) -- the template changed; regenerate ui_patches.json")
    Path(html_path).write_text(text, encoding="utf-8")
    print(f"  [ok] UI patches applied: {applied}/{len(patches)}"
          + (f" ({skipped} already present)" if skipped else ""))


def _weighted_csv(pc_csv: Path, sig_csv: Path, organs: list[str]) -> str:
    pc = pd.read_csv(pc_csv, index_col=0)
    sig = pd.read_csv(sig_csv, index_col=0)
    have = set(sig.columns)
    # Only organs that are part of the PET partial-correlation network (Colon was not measured) are
    # written -- including organs without any significant edge (e.g. WAT in the overweight group), which
    # the page then shows as isolated nodes (see patch_cohort_reference_view).
    organs = [o for o in organs if o in have]
    lines = ["," + ",".join(organs)]
    for row in organs:
        cells = [row]
        for col in organs:
            if row == col:
                cells.append("0")
            elif row in have and col in have and sig.loc[row, col] == 1:
                cells.append(str(float(pc.loc[row, col])))
            else:
                cells.append("0")
        lines.append(",".join(cells))
    return "\n".join(lines)


def inject_weighted_cohort_matrices(html_path: Path, ref_dir: Path) -> None:
    text = Path(html_path).read_text(encoding="utf-8")
    for var, cond in (("_HC", "healthy"), ("_OC", "obese"), ("_LC", "lung_cancer")):
        pc_csv = Path(ref_dir) / f"{cond}_cohort_PC_connections.csv"
        sig_csv = Path(ref_dir) / f"{cond}_cohort_PC_significance.csv"
        m = re.search(r"var " + var + r" = `([^`]*)`", text)
        if not (m and pc_csv.exists() and sig_csv.exists()):
            print(f"  [!] weighted {cond} cohort matrix not injected "
                  f"({'no placeholder' if not m else 'PC csv missing'})")
            continue
        organs = m.group(1).split("\n")[0].split(",")[1:]       # same organ order as the page
        csv_text = _weighted_csv(pc_csv, sig_csv, organs).replace("`", "\\`")
        text = text[:m.start()] + "var " + var + " = `" + csv_text + "`" + text[m.end():]
    Path(html_path).write_text(text, encoding="utf-8")
    print("  [ok] weighted partial-correlation cohort matrices injected")


# (anchor, replacement) edits of the page script; each anchor must occur exactly once.
_REF = 'let uploadedRefPairs = null; // Set of "OrganA__OrganB" (alphabetically sorted) or null\n'
_COHORT_VIEW_EDITS = [
    # remember the organs listed in the loaded matrix (header), not only those with an edge
    (_REF,
     _REF + "  // Organs listed in the loaded reference matrix (also those without any edge)\n"
            "  let uploadedRefOrgans = null;\n"),
    ("    return pairs;\n  }\n\n  // IDs of cohort buttons",
     "    uploadedRefOrgans = new Set(canonHeader.filter(Boolean));\n"
     "    return pairs;\n  }\n\n  // IDs of cohort buttons"),
    # cohort reference networks: keep organs without a significant edge visible (WAT in the overweight group)
    ("      uploadedRefPairs.forEach((_, k) => k.split('__').forEach(o => uploadedOrgans.add(o)));\n",
     "      uploadedRefPairs.forEach((_, k) => k.split('__').forEach(o => uploadedOrgans.add(o)));\n"
     "      // Healthy/overweight cohort reference networks: organs without any significant edge (e.g. WAT in\n"
     "      // the overweight group) stay visible as isolated nodes\n"
     "      if (uploadedRefOrgans && (comparisonSource === 'healthy_cohort' || comparisonSource === 'obese_cohort'\n"
     "                                || comparisonSource === 'lung_cancer_cohort'))\n"
     "        uploadedRefOrgans.forEach(o => uploadedOrgans.add(o));\n"),
]
_COHORT_BUTTON_TITLES = [
    ('title="Show healthy PC reference network"',
     'title="Show healthy partial correlation network, significant connections"'),
    ('title="Show overweight PC reference network"',
     'title="Show overweight partial correlation network, significant connections"'),
    ('title="Show lung cancer PC reference network"',
     'title="Show lung cancer partial correlation network, significant connections"'),
]


def patch_cohort_reference_view(html_path: Path) -> None:
    """1. In the healthy/overweight cohort reference view, organs without a significant edge (WAT in the
    overweight group) stay visible as isolated nodes.  2. Hover text of the "Healthy cohort" / "Overweight
    cohort" buttons.  Idempotent."""
    text = Path(html_path).read_text(encoding="utf-8")
    if "uploadedRefOrgans" in text:
        print("  [ok] cohort reference view: organ visibility already patched")
    else:
        ok = all(text.count(a) == 1 for a, _ in _COHORT_VIEW_EDITS)
        if ok:
            for anchor, repl in _COHORT_VIEW_EDITS:
                text = text.replace(anchor, repl, 1)
            print("  [ok] cohort reference view: organs without significant edges stay visible")
        else:
            print("  [!] cohort reference view not patched -- the page script changed "
                  f"(anchor matches: {[text.count(a) for a, _ in _COHORT_VIEW_EDITS]})")
    for old, new in _COHORT_BUTTON_TITLES:
        text = text.replace(old, new)
    Path(html_path).write_text(text, encoding="utf-8")
