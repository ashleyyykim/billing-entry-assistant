"""Simple scikit-learn matter matcher.  Usage: python simple_sklearn.py time_entries.xlsx
Turns each entry into one text string (narrative + fields + typed CMID + timekeeper history tokens),
then trains two TF-IDF + logistic regression models: one for the matter (CMID), one for the action."""
import re, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score, classification_report

path = sys.argv[1]
entries = pd.read_excel(path, "Time Entries")
hist = pd.read_excel(path, "TK Matter History")

# top-3 matters each timekeeper billed to in Q2 (a hint for entries with vague narratives)
hist["CMID"] = hist["CMID"].map("{:.3f}".format)
top3 = hist.sort_values("Hours (04/01-06/30/2026)", ascending=False).groupby("TK ID")["CMID"].apply(lambda s: list(s[:3]))

def typed_cmid(x):
    """'6103.1' -> '6103.001'; unreadable/missing -> 'none'."""
    m = re.fullmatch(r"(\d{4})[./](\d{1,3})", str(x).strip())
    return f"{m[1]}.{m[2].zfill(3)}" if m else "none"

def to_text(r):
    typed = typed_cmid(r["CMID"])
    hist_tokens = " ".join("hist_" + c for c in top3.get(r["TK ID"], []))
    return (f"{r['Narrative']} || {r['Client']} || {r['Matter']} || tk_{r['TK ID']} || src_{r['Source Format']} "
            f"|| typed_{typed} typedclient_{typed[:4]} || {hist_tokens}").lower()

entries["text"] = entries.apply(to_text, axis=1)
train, val, test = (entries[entries["KEY: Split"] == s] for s in ("TRAIN", "VAL", "TEST"))

def make_model(**kw):  # word + character n-grams (characters help with OCR typos)
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
    return make_pipeline(feats, LogisticRegression(C=20, max_iter=500, **kw))

# Model 1: which matter?  (NONE = no matching matter)
matter = make_model().fit(train["text"], train["KEY: True CMID"].astype(str))
# Model 2: what should happen to the entry?
action = make_model(class_weight="balanced").fit(train["text"], train["KEY: Expected Action"])  # rare actions are ~8% each

for name, df in (("VAL", val), ("TEST", test)):
    pm, pa = matter.predict(df["text"]), action.predict(df["text"])
    truth = df["KEY: True CMID"].astype(str)
    hard = df["KEY: Trivial"] == "N"
    silent = df["KEY: Silent Error"] == "Y"
    print(f"\n== {name} ==")
    print(f"matter accuracy {accuracy_score(truth, pm):.3f} | non-trivial {accuracy_score(truth[hard], pm[hard]):.3f}"
          f" | silent errors {accuracy_score(truth[silent], pm[silent]):.3f}")
    print(f"action accuracy {accuracy_score(df['KEY: Expected Action'], pa):.3f}")
    if name == "TEST":
        print(classification_report(df["KEY: Expected Action"], pa, digits=3))
