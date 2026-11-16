"""Evaluate split_block.py (step 1: splitting only).

  python training/eval_splitter.py                       # uses the default paths below
  python training/eval_splitter.py --preds balanced_predictions.csv

Sections
  1. synthetic: split_eval_blocks.jsonl from make_pairs.py (raw, unexpanded text, as the splitter sees
     it in production), where the correct lines and hours are known.
     Caveat: make_pairs joins tasks with "; " and "(h)", the splitter's strongest cue, so this is
     closer to a regression test than a measure of real-world accuracy.
  2. gold: human block_billed labels. Y should split into 2+ lines, N should stay single.
     This is the real accuracy number.
  3. classifier (optional, needs --preds): status mix for classifier-flagged Y vs N.

Writes per-entry results to --out so misses can be inspected.
"""
import argparse
import difflib
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # phase2_model/
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from split_block import split_entry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default=os.path.join(ROOT, "data", "outputs", "pairs"))
ap.add_argument("--gold", default=os.path.join(ROOT, "data", "raw", "label_sample_labeled_gold.csv"))
ap.add_argument("--preds", default=None, help="CSV with Narrative, Hours, block_billed_pred")
ap.add_argument("--out", default=os.path.join(ROOT, "data", "outputs", "eval_splitter"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pd.set_option("display.width", 200)

key = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(t).lower())).strip()
sim = lambda x, y: difflib.SequenceMatcher(None, key(x), key(y)).ratio()
summary = {}

# ---------------------------------------------------------------- 1. synthetic
blocks_path = os.path.join(a.pairs, "split_eval_blocks.jsonl")
bb = [json.loads(l) for l in open(blocks_path, encoding="utf-8")] if os.path.exists(blocks_path) else []
if not bb:
    print(f"=== 1. synthetic: skipped (no {blocks_path}; run make_pairs.py)")
else:
    out = []
    for r in bb:
        t_text = [l["text"] for l in r["lines"]]
        t_hours = [l["hours"] for l in r["lines"]]
        truth = t_text
        got = split_entry(r["input"], r["hours"])["lines"]
        same_n = len(got) == len(truth)
        text_ok = same_n and all(sim(g["text"], t) >= 0.9 for g, t in zip(got, t_text))
        has_t = t_hours[0] is not None
        hours_ok = same_n and has_t and all(g["hours"] is not None and abs(g["hours"] - h) < 1e-6
                                            for g, h in zip(got, t_hours))
        out.append(dict(input=r["input"], n_true=len(truth), n_got=len(got), count_ok=same_n,
                        text_ok=text_ok, has_times=has_t, hours_ok=hours_ok,
                        got=" || ".join(g["text"] for g in got), want=" || ".join(t_text)))
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(a.out, "synthetic_results.csv"), index=False)
    n, wt = len(s), s.has_times.sum()
    summary.update(synth_count=s.count_ok.mean(), synth_text=s.text_ok.mean(),
                   synth_hours=s[s.has_times].hours_ok.mean() if wt else float("nan"))
    print(f"=== 1. synthetic ({n} merged entries; regression test, see caveat in docstring)")
    print(f"  right number of lines:              {s.count_ok.sum()}/{n}")
    print(f"  line text matches (>=90% similar):  {s.text_ok.sum()}/{n}")
    print(f"  hours match (entries with times):   {s[s.has_times].hours_ok.sum()}/{wt}")
    for r in s[~s.text_ok].head(5).itertuples():
        print(f"   TEXT MISS\n     got : {r.got[:150]}\n     want: {r.want[:150]}")

# ---------------------------------------------------------------- 2. gold
g = pd.read_csv(a.gold)
res = [split_entry(t, h) for t, h in zip(g.narrative, g.hours)]
g["status"] = [r["status"] for r in res]
g["n_lines"] = [len(r["lines"]) for r in res]
g["lines"] = [" || ".join(f"{l['text']} ({l['hours']})" for l in r["lines"]) for r in res]
g["flags"] = ["|".join(r["flags"]) for r in res]
g["split"] = g.n_lines >= 2
y, nn = g[g.block_billed == "Y"], g[g.block_billed == "N"]
tp, fn, fp = y.split.sum(), (~y.split).sum(), nn.split.sum()
prec = tp / (tp + fp) if tp + fp else float("nan")
rec = tp / (tp + fn) if tp + fn else float("nan")
f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
hours_ok = (y[y.split].status == "split_ok").sum()
summary.update(gold_precision=prec, gold_recall=rec, gold_f1=f1,
               gold_hours_reconciled=hours_ok / tp if tp else float("nan"))

print(f"\n=== 2. gold ({len(g)} human-labeled entries)")
print(pd.crosstab(g.block_billed, g.status, margins=True).to_string())
print(f"\n  recall    (Y split into 2+ lines):  {tp}/{tp + fn} = {rec:.1%}")
print(f"  precision (splits that were Y):     {tp}/{tp + fp} = {prec:.1%}")
print(f"  F1:                                 {f1:.3f}")
print(f"  hours reconcile on correct splits:  {hours_ok}/{tp}")
print("\n  MISSED (Y not split):")
for r in y[~y.split].itertuples():
    print("   -", str(r.narrative)[:150])
print("\n  FALSE SPLITS (N split):")
for r in nn[nn.split].itertuples():
    print("   -", str(r.narrative)[:150])
cols = ["sample_id", "block_billed", "quality_label", "status", "n_lines", "flags", "narrative", "lines"]
g[cols].to_csv(os.path.join(a.out, "gold_results.csv"), index=False)

# ---------------------------------------------------------------- 3. classifier
if a.preds:
    b = pd.read_csv(a.preds)
    b["status"] = [split_entry(t, h)["status"] for t, h in zip(b.Narrative, b.Hours)]
    print("\n=== 3. by classifier prediction")
    print(pd.crosstab(b.block_billed_pred, b.status, margins=True).to_string())
else:
    print("\n=== 3. classifier: skipped (pass --preds file with a block_billed_pred column)")

# ---------------------------------------------------------------- summary
print("\n=== SUMMARY")
for k, v in summary.items():
    print(f"  {k:24s} {v:.3f}")
with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump({k: round(float(v), 4) for k, v in summary.items()}, f, indent=2)
print(f"\n  per-entry results -> {a.out}")
