"""Fine-tune the rewriter (step 4): flan-t5-small, flagged line -> line with placeholders.

  python training/train_rewriter.py [--pairs data/outputs/pairs] [--out models] [--epochs 2] [--limit N]

Input  : review.rewriter_prompt(flags, line), the same format the service builds at run time
Target : the pair's target, with [SUGGESTED: X] turned back into the plain placeholder (the model never
         guesses a name; the service lists candidates instead)
Junk pairs are left out: junk lines are flagged, not rewritten.
CPU is fine: about 20-30 minutes for 2 epochs on ~24k pairs. Writes models/rewriter/ and models/rewriter_report.json.
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # phase2_model/
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from expand_shorthand import load_bank  # noqa: E402
from review import SEVERITY, allowed_words, rewriter_prompt, safe_rewrite, tidy  # noqa: E402
from train_classifier import pair_flags, placeholder_target  # noqa: E402


def examples(path, limit=None, seed=0):
    d = pd.read_json(path, lines=True)
    d = d[d.type != "junk"]
    rows = []
    for p in d.to_dict("records"):
        flags = [f for f in SEVERITY if f in pair_flags(p)]
        rows.append((rewriter_prompt(flags, p["input_model"]), tidy(placeholder_target(p)), p["type"]))
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", default=os.path.join(ROOT, "data", "outputs", "pairs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    ap.add_argument("--base", default="google/flan-t5-small")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=None, help="train on N pairs only (quick test)")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))

    tr = examples(os.path.join(a.pairs, "pairs_train.jsonl"), a.limit)
    dv = examples(os.path.join(a.pairs, "pairs_dev.jsonl"), 400, seed=1)
    print(f"train {len(tr)}  dev (scored) {len(dv)}  base {a.base}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    steps = int(len(tr) / a.batch * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(steps, 1)))

    model.train()
    t0, step = time.time(), 0
    while step < steps:
        order = sorted(range(len(tr)), key=lambda i: len(tr[i][0]))          # similar lengths per batch
        batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        random.Random(step).shuffle(batches)
        for b in batches:
            if step >= steps:
                break
            enc = tok([tr[i][0] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=192)
            lab = tok([tr[i][1] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == steps:
                el = time.time() - t0
                print(f"step {step}/{steps}  loss {loss.item():.3f}  {el / 60:.1f} min  "
                      f"eta {el / step * (steps - step) / 60:.1f} min", flush=True)

    # ---- dev: exact match after tidying, and how often the service's safety check would reject the output
    model.eval()
    rows, allowed = [], allowed_words(load_bank())
    for i in range(0, len(dv), 32):
        chunk = dv[i:i + 32]
        enc = tok([x[0] for x in chunk], return_tensors="pt", padding=True, truncation=True, max_length=192)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, num_beams=2)
        for (prompt, tgt, typ), pred in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
            line = prompt.split("| line: ", 1)[1]
            rows.append(dict(type=typ, exact=tidy(pred) == tgt, safe=safe_rewrite(line, pred, allowed),
                             prompt=prompt, target=tgt, pred=pred))
    r = pd.DataFrame(rows)
    by = r.groupby("type")[["exact", "safe"]].mean().round(3)
    print("\n=== dev (synthetic): exact match and passes safety check, by pair type")
    print(by.to_string())
    print(f"  overall exact {r.exact.mean():.3f}  safe {r.safe.mean():.3f}")

    dest = os.path.join(a.out, "rewriter")
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    r.to_csv(os.path.join(a.out, "rewriter_dev_predictions.csv"), index=False)
    with open(os.path.join(a.out, "rewriter_report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(exact=round(float(r.exact.mean()), 3), safe=round(float(r.safe.mean()), 3),
                       by_type=json.loads(by.to_json(orient="index"))), f, indent=2)
    meta_path = os.path.join(a.out, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    meta["rewriter"] = dict(trained=datetime.now(timezone.utc).isoformat(timespec="seconds"), base=a.base,
                            n_train=len(tr), epochs=a.epochs, dev_exact=round(float(r.exact.mean()), 3))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
