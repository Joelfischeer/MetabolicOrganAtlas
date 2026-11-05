"""
evaluate_condition_filter.py
============================
Measures the accuracy of the LLM condition filter against YOUR labels in
condition_gold_set/condition_gold_set.xlsx (see build_condition_gold_workbook.py).

The papers are classified with the CURRENT prompt/model of
Literature_Search/llm_condition_filter.py (results are cached per model +
prompt version in condition_gold_set/predictions_<model>_<prompt>.jsonl, so a
re-run only classifies what is missing; ~17 s per paper on CPU).

Reports: accuracy (two ways), precision/recall/F1 per class, confusion matrix,
the metrics of the Healthy and Obese CASE filters, accuracy per sampling
stratum and every error with its evidence sentence.

    python evaluate_condition_filter.py                       # current model + prompt
    python evaluate_condition_filter.py --model qwen2.5:7b    # another Ollama model
    python evaluate_condition_filter.py --model claude-haiku-5-5   # Claude API (needs ANTHROPIC_API_KEY; ~$0.02)
    python evaluate_condition_filter.py --no-classify         # only evaluate what is cached
"""

import argparse
import html
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import run_metabolic_lit_search as rl                         # noqa: E402
from Literature_Search import llm_condition_filter as cf      # noqa: E402

GOLD_DIR = HERE / "condition_gold_set"
NO_PRED = "(no usable prediction)"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def read_gold(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, read_only=True, data_only=True)
    strata = {str(r[1]): r[2] for r in wb["Key"].iter_rows(min_row=2, values_only=True)} \
        if "Key" in wb.sheetnames else {}
    rows = []
    for r in wb["Papers"].iter_rows(min_row=2, values_only=True):
        if r[1] is None:
            continue
        rows.append({"id": r[0], "pmid": str(r[1]), "title": r[3], "gold": (r[5] or "").strip(),
                     "stratum": strata.get(str(r[1]), "")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_DIR / "condition_gold_set.xlsx")
    ap.add_argument("--model", default=rl.CONDITION_LLM_MODEL)
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--no-classify", action="store_true")
    args = ap.parse_args()

    gold = read_gold(args.gold)
    labelled = gold[gold["gold"] != ""]
    unsure = int((labelled["gold"] == "Unsure").sum())
    gold = labelled[labelled["gold"] != "Unsure"].copy()
    bad = sorted(set(gold["gold"]) - set(cf.LABELS))
    if bad:
        sys.exit(f"Unknown labels in the workbook: {bad}")
    if gold.empty:
        sys.exit("No labelled papers yet -- fill in the YOUR LABEL column of the workbook first.")
    print(f"gold labels: {len(gold)} papers ({unsure} 'Unsure' excluded, "
          f"{len(labelled) - len(gold) - unsure} other) of {len(read_gold(args.gold))}")

    pred_path = args.predictions or GOLD_DIR / f"predictions_{_safe(args.model)}_{cf.PROMPT_VERSION}.jsonl"
    preds = cf.load_condition_labels(pred_path, args.model)
    todo = [p for p in gold["pmid"] if p not in preds]
    if todo and not args.no_classify:
        if not cf.check_model(args.model):
            sys.exit(f"Model '{args.model}' is not available (see the message above).")
        with open(rl.OUTPUT_JSON, encoding="utf-8") as f:
            res = json.load(f)
        papers = cf.collect_unique_papers(res)
        missing = [p for p in todo if p not in papers]
        if missing:       # gold papers that a later (stricter) crosstalk filter removed: use the pre-filter backups
            for bak in sorted(rl.OUTPUT_JSON.parent.glob(rl.OUTPUT_JSON.name + ".presamesentence.*.bak")):
                with open(bak, encoding="utf-8") as f:
                    old = cf.collect_unique_papers(json.load(f))
                for p in list(missing):
                    if p in old:
                        papers[p] = old[p]
                        missing.remove(p)
                if not missing:
                    break
            if missing:
                sys.exit(f"No title/abstract found for gold papers: {missing}")
        workers = cf.ANTHROPIC_WORKERS if cf.is_anthropic_model(args.model) else 1
        print(f"classifying {len(todo)} papers with {args.model} / prompt {cf.PROMPT_VERSION} ...")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        tok_in = tok_out = 0
        with open(pred_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
            for i, rec in enumerate(pool.map(lambda pid: cf.classify_paper(papers[pid], args.model), todo), 1):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                preds[rec["pmid"]] = rec
                tok_in += rec.get("tokens_in", 0)
                tok_out += rec.get("tokens_out", 0)
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}]", flush=True)
        cost = cf.estimate_cost(args.model, tok_in, tok_out)
        if cost is not None:
            print(f"API usage: {tok_in:,} input + {tok_out:,} output tokens ~ ${cost:.3f}")
    gold = gold[gold["pmid"].isin(preds)].copy()
    n = len(gold)
    if n == 0:
        sys.exit("No predictions available (run without --no-classify).")

    gold["status"] = [preds[p]["status"] for p in gold["pmid"]]
    gold["pred_raw"] = [preds[p].get("label") for p in gold["pmid"]]
    gold["pred"] = [preds[p]["label"] if preds[p]["status"] == cf.OK else NO_PRED for p in gold["pmid"]]
    gold["evidence"] = [preds[p].get("evidence", "") for p in gold["pmid"]]
    gold["correct"] = gold["gold"] == gold["pred"]

    usable = gold[gold["status"] == cf.OK]
    acc_all = gold["correct"].mean() * 100
    acc_ok = usable["correct"].mean() * 100 if len(usable) else float("nan")
    halluc = int(gold["status"].isin([cf.HALLUCINATED, cf.TOO_SHORT]).sum())

    # per class -------------------------------------------------------------------
    rows = []
    for lab in cf.LABELS:
        tp = int(((gold["gold"] == lab) & (gold["pred"] == lab)).sum())
        support = int((gold["gold"] == lab).sum())
        n_pred = int((gold["pred"] == lab).sum())
        prec = 100 * tp / n_pred if n_pred else float("nan")
        rec = 100 * tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
        rows.append({"label": lab, "gold n": support, "predicted n": n_pred, "correct": tp,
                     "precision %": round(prec, 1), "recall %": round(rec, 1), "F1 %": round(f1, 1)})
    per_class = pd.DataFrame(rows)
    macro_f1 = per_class["F1 %"].mean(skipna=True)

    # confusion matrix ---------------------------------------------------------------
    cols = cf.LABELS + [NO_PRED]
    conf = pd.crosstab(gold["gold"], gold["pred"]).reindex(index=cf.LABELS, columns=cols, fill_value=0)
    conf.index.name = "gold \\ predicted"
    conf = conf.reset_index()

    # case filters -----------------------------------------------------------------------
    case_rows = []
    for case, lab in cf.CONDITION_TO_LABEL.items():
        tp = int(((gold["gold"] == lab) & (gold["pred"] == lab)).sum())
        fp = int(((gold["gold"] != lab) & (gold["pred"] == lab)).sum())
        fn = int(((gold["gold"] == lab) & (gold["pred"] != lab)).sum())
        case_rows.append({"case": case, "label": lab, "papers the filter keeps": tp + fp, "true positives": tp,
                          "false positives": fp, "missed (false negatives)": fn,
                          "precision %": round(100 * tp / (tp + fp), 1) if tp + fp else float("nan"),
                          "recall %": round(100 * tp / (tp + fn), 1) if tp + fn else float("nan")})
    cases = pd.DataFrame(case_rows)

    strat = (gold.groupby("stratum")["correct"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "accuracy"})
             .assign(accuracy=lambda d: (d["accuracy"] * 100).round(1)).reset_index())
    errors = gold[~gold["correct"]][["id", "pmid", "gold", "pred", "pred_raw", "status", "title", "evidence"]]
    errors = errors.assign(title=errors["title"].map(lambda t: html.unescape(str(t))))

    summary = pd.DataFrame([
        ("Model", args.model), ("Prompt version", cf.PROMPT_VERSION), ("Labelled papers evaluated", n),
        ("Accuracy, all papers (unusable answers count as wrong) %", round(acc_all, 1)),
        ("Accuracy, usable answers only %", round(acc_ok, 1)),
        ("Macro-F1 over classes %", round(macro_f1, 1)),
        ("Hallucinated / too-short evidence", f"{halluc} ({100 * halluc / n:.1f}%)"),
        ("Invalid or failed answers", int(gold["status"].isin([cf.INVALID_JSON, cf.INVALID_LABEL, cf.ERROR]).sum())),
        ("Note", "the sample is enriched for rare classes: precision/recall are valid within this sample; "
                 "class prevalence in the full corpus is different"),
    ], columns=["Metric", "Value"])

    out = GOLD_DIR / f"evaluation_{_safe(args.model)}_{cf.PROMPT_VERSION}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in (("Summary", summary), ("Per class", per_class), ("Confusion", conf), ("Case filters", cases),
                         ("By stratum", strat), ("Errors", errors)):
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:60] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(70, max(10, w + 2))
            ws.freeze_panes = "A2"

    print(f"\n=== {args.model} / prompt {cf.PROMPT_VERSION} on {n} labelled papers")
    print(f"accuracy: {acc_all:.1f}% (all)  {acc_ok:.1f}% (usable answers only) | macro-F1 {macro_f1:.1f}% | "
          f"hallucinated {halluc}")
    print(per_class.to_string(index=False))
    print()
    print(cases.to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
