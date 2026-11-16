"""Remove evaluation entries from the training source file, so the model is never trained on its test set.

  python training/decontaminate.py                                   # default paths below
  python training/decontaminate.py balanced.csv out.csv --eval gold.csv [--eval other_eval.csv ...] [--overlap 0.9]

A balanced row is removed if its narrative, or any of its split task lines, matches an evaluation entry
(or one of the evaluation entry's split lines) by:
  exact      same text after lower-casing and dropping punctuation
  overlap    >= --overlap (default 0.9) of the shorter text's 5-word sequences also appear in the other text.
             Catches one text containing the other (a gold line inside a balanced block entry, or the
             reverse) and near-duplicates (scrape artifacts, a changed word).
Split lines under 8 words are ignored and overlap matching needs 8+ words on both sides, so boilerplate
("confer with team re same") does not remove unrelated entries. A whole entry still matches exactly at any length.

Whole documents are NOT removed: the gold sample shares 168 fee applications with 38% of the balanced
file, and the leakage risk is the entry text, not the document.

Outputs: out.csv (kept rows, same columns) and out_removed.csv (removed rows + why + which eval entry).
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # phase2_model/
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from split_block import split_entry  # noqa: E402

N = 5                 # words per shingle
MIN_LINE_WORDS = 8    # shorter split lines are not matched (boilerplate)
MIN_OVERLAP_WORDS = 8 # overlap matching needs both texts at least this long


def key(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(t).lower())).strip()


def shingles(k):
    w = k.split()
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def texts_of(narrative, hours):
    """The whole entry (matched at any length) plus its split lines (only if >= MIN_LINE_WORDS words, so
    boilerplate such as 'confer with team re same' does not knock out unrelated entries)."""
    whole = key(narrative)
    out = {whole} if whole else set()
    try:
        for ln in split_entry(narrative, hours)["lines"]:
            k = key(ln["text"])
            if len(k.split()) >= MIN_LINE_WORDS:
                out.add(k)
    except Exception:
        pass
    return out, whole


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("balanced", nargs="?", default=os.path.join(ROOT, "data", "raw", "time_entries_clean_balanced.csv"))
    ap.add_argument("out", nargs="?", default=os.path.join(ROOT, "data", "outputs", "time_entries_decontaminated.csv"))
    ap.add_argument("--eval", action="append",
                    help="evaluation CSV(s) with a narrative column (default: the gold label sample)")
    ap.add_argument("--overlap", type=float, default=0.9,
                    help="0.8 caught different entries with similar wording; 0.9 kept only real duplicates")
    a = ap.parse_args()
    evals = a.eval or [os.path.join(ROOT, "data", "raw", "label_sample_labeled_gold.csv")]

    # ---- index every evaluation text
    exact, by_shingle, sh_of, owner = {}, defaultdict(set), {}, {}
    for path in evals:
        e = pd.read_csv(path)
        tcols = [c for c in ("narrative", "narrative_raw", "Narrative") if c in e.columns]
        hcol = next((c for c in ("hours", "Hours") if c in e.columns), None)
        idcol = next((c for c in ("sample_id", "Time ID") if c in e.columns), None)
        for i, r in e.iterrows():
            eid = f"{os.path.basename(path)}:{r[idcol] if idcol else i}"
            for c in tcols:
                if pd.isna(r[c]):
                    continue
                for t in texts_of(r[c], r[hcol] if hcol else 0)[0]:
                    exact.setdefault(t, eid)
                    s = shingles(t) if len(t.split()) >= MIN_OVERLAP_WORDS else set()
                    if s and t not in sh_of:
                        sh_of[t], owner[t] = s, eid
                        for x in s:
                            by_shingle[x].add(t)
    print(f"evaluation texts indexed: {len(exact)} from {len(evals)} file(s)")

    # ---- scan the balanced file
    b = pd.read_csv(a.balanced)
    keep, why, who = [], [], []
    for r in b.itertuples(index=False):
        narrative, hours = getattr(r, "Narrative"), getattr(r, "Hours")
        hit = None
        texts, whole = texts_of(narrative, hours)
        for t in texts:
            if t in exact and (t == whole or len(t.split()) >= MIN_LINE_WORDS):
                hit = ("exact", exact[t])
                break
            s = shingles(t) if len(t.split()) >= MIN_OVERLAP_WORDS else set()
            if not s:
                continue
            cands = set().union(*(by_shingle.get(x, ()) for x in s))
            for c in cands:
                ov = len(s & sh_of[c]) / min(len(s), len(sh_of[c]))
                if ov >= a.overlap:
                    hit = (f"overlap {ov:.2f}", owner[c])
                    break
            if hit:
                break
        keep.append(hit is None)
        why.append(hit[0] if hit else "")
        who.append(hit[1] if hit else "")

    b["_keep"], b["match"], b["eval_entry"] = keep, why, who
    kept = b[b._keep].drop(columns=["_keep", "match", "eval_entry"])
    removed = b[~b._keep].drop(columns=["_keep"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    kept.to_csv(a.out, index=False)
    rem_path = os.path.splitext(a.out)[0] + "_removed.csv"
    removed.to_csv(rem_path, index=False)
    kinds = removed.match.str.split().str[0].value_counts().to_dict()
    print(f"balanced rows: {len(b)}  kept: {len(kept)}  removed: {len(removed)}  {kinds}")
    print(f"evaluation entries with at least one match: {removed.eval_entry.nunique()}")
    print(f"-> {a.out}\n-> {rem_path}")


if __name__ == "__main__":
    main()
