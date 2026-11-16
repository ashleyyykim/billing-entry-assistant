"""Train the line classifier (step 3) on the pairs from make_pairs.py.

  python training/train_classifier.py [--pairs data/outputs/pairs] [--out models]

Model: TF-IDF (words 1-2 + characters 3-5) and one logistic regression per flag, so a line can get
several flags at once. Learns MISSING_PARTICIPANT, MISSING_SUBJECT, VAGUE and JUNK.
AMBIGUOUS_SHORTHAND and FORMATTING are exact rules in review.py and are not learned.

Reports
  dev   synthetic dev pairs (same generator as training, so optimistic)
  gold  the 240 hand-labeled entries, entry level (worst line's flag vs the human label): the real number
Writes models/classifier.joblib, models/metadata.json and models/classifier_report.json.
Rerun after make_pairs.py whenever the data changes.
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # phase2_model/
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from expand_shorthand import DEFAULT_BANK  # noqa: E402
from review import MODEL_FLAGS  # noqa: E402

TYPE_FLAG = {"missing_participant": "MISSING_PARTICIPANT", "missing_subject": "MISSING_SUBJECT",
             "vague_object": "VAGUE", "junk": "JUNK", "all_caps": "FORMATTING"}
SUGGESTED = re.compile(r"\[SUGGESTED:[^\]]*\]")


def suggested_kind(target):
    """Which placeholder a context_suggested target replaced, from the words before it."""
    m = SUGGESTED.search(target)
    before = target[:m.start()].rstrip().split()[-1:] if m else []
    if before and before[0].lower() in {"with", "to", "from", "between", "and"}:
        return "PARTICIPANT"
    if before and before[0].lower() in {"regarding", "re", "re:", "concerning", "about", "connection"}:
        return "SUBJECT"
    return "DOCUMENT"


def pair_flags(p):
    """All flags for a pair, including AMBIGUOUS_SHORTHAND (from the pair's label)."""
    flags = []
    if p["type"] in TYPE_FLAG:
        flags.append(TYPE_FLAG[p["type"]])
    elif p["type"] == "context_suggested":
        flags.append({"PARTICIPANT": "MISSING_PARTICIPANT", "SUBJECT": "MISSING_SUBJECT",
                      "DOCUMENT": "VAGUE"}[suggested_kind(p["target"])])
    if "AMBIGUOUS_SHORTHAND" in p.get("label", []):
        flags.append("AMBIGUOUS_SHORTHAND")
    return flags


def placeholder_target(p):
    """The model never guesses: [SUGGESTED: X] becomes the plain placeholder. Candidates are listed separately."""
    t = p["target"]
    if p["type"] == "context_suggested":
        t = SUGGESTED.sub(f"[{suggested_kind(t)}]", t, count=1)
    return t


def load_pairs(path):
    return pd.read_json(path, lines=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", default=os.path.join(ROOT, "data", "outputs", "pairs"))
    ap.add_argument("--gold", default=os.path.join(ROOT, "data", "raw", "label_sample_labeled_gold.csv"))
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr, dv = (load_pairs(os.path.join(a.pairs, f"pairs_{s}.jsonl")) for s in ("train", "dev"))
    Y = lambda d: np.array([[f in pair_flags(p) for f in MODEL_FLAGS] for p in d.to_dict("records")], dtype=int)
    ytr, ydv = Y(tr), Y(dv)
    print(f"train {len(tr)}  dev {len(dv)}  positives per flag (train): "
          f"{dict(zip(MODEL_FLAGS, ytr.sum(0).tolist()))}")

    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 max_features=200_000)),
    ])
    Xtr = vec.fit_transform(tr.input_model)
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000), n_jobs=-1)
    clf.fit(Xtr, ytr)

    pdv = clf.predict_proba(vec.transform(dv.input_model))
    rep_dev = classification_report(ydv, (pdv >= 0.5).astype(int), target_names=MODEL_FLAGS,
                                    zero_division=0, output_dict=True)
    print("\n=== dev (synthetic, optimistic)")
    print(classification_report(ydv, (pdv >= 0.5).astype(int), target_names=MODEL_FLAGS, zero_division=0))

    joblib.dump(dict(vectorizer=vec, model=clf, flags=MODEL_FLAGS,
                     thresholds={f: 0.5 for f in MODEL_FLAGS}), os.path.join(a.out, "classifier.joblib"))
    shutil.copy(a.bank, os.path.join(a.out, "shorthand_bank.csv"))

    # ---- gold, entry level, through the same code the service runs
    report = dict(dev=rep_dev)
    if os.path.exists(a.gold):
        from review import Reviewer
        r = Reviewer(a.out, use_rewriter=False)
        g = pd.read_csv(a.gold)
        res = r.review_many([dict(narrative=t, hours=h) for t, h in zip(g.narrative, g.hours)])
        g["pred"] = [x["worst_flag"] or "OK" for x in res]
        g["pred_flagged"] = g.pred != "OK"
        g["gold_flagged"] = ~g.quality_label.isin(["GOOD", "SPECIFIC"])
        ct = pd.crosstab(g.quality_label, g.pred, margins=True)
        tp = (g.pred_flagged & g.gold_flagged).sum()
        prec = tp / max(g.pred_flagged.sum(), 1)
        rec = tp / max(g.gold_flagged.sum(), 1)
        print("=== gold (240 hand-labeled entries; rows = human label, columns = worst predicted flag)")
        print(ct.to_string())
        print(f"\n  any problem flagged: precision {prec:.2f}  recall {rec:.2f}")
        report["gold"] = dict(flag_precision=round(float(prec), 3), flag_recall=round(float(rec), 3),
                              crosstab=json.loads(ct.to_json()))
        g[["sample_id", "quality_label", "pred", "narrative"]].to_csv(
            os.path.join(a.out, "classifier_gold_predictions.csv"), index=False)

    with open(os.path.join(a.out, "classifier_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    meta_path = os.path.join(a.out, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    meta["classifier"] = dict(trained=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              pairs=os.path.relpath(a.pairs, ROOT), n_train=len(tr), flags=MODEL_FLAGS,
                              gold=report.get("gold", {}).get("flag_precision") and
                              dict(precision=report["gold"]["flag_precision"], recall=report["gold"]["flag_recall"]))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
