"""Rule-based baseline: shorthand detection/expansion + simple compliance flags.

Usage: python rule_baseline.py entries.csv shorthand_bank.csv out.csv
Input CSV needs a `Narrative` column (and optionally `Hours`).
"""
import re
import sys
import pandas as pd

VAGUE_PHRASES = [
    r"\battention to\b", r"\bwork(ed)? on\b", r"\bfile (review|maintenance)\b",
    r"\breview(ed)? file\b", r"\bmisc(ellaneous)?\b", r"\bvarious\b",
    r"\bmatters?\b$", r"\bfollow(ed)? up\b$", r"\bprepar(e|ation) for\b$",
]
ADMIN_PHRASES = [r"\bscan(ned|ning)?\b", r"\bfil(e|ed|ing) documents\b",
                 r"\bschedul(e|ed|ing)\b", r"\bcopy(ing)?\b", r"\badmin\b"]
TASK_SPLIT = re.compile(r";|\band\b(?= (review|draft|prepare|attend|call|email|revise|conference))", re.I)
MIN_WORDS = 5


def load_bank(path):
    bank = pd.read_csv(path).fillna("")
    rules = []
    for _, r in bank.iterrows():
        flags = 0 if r["case_sensitive"] == "Y" else re.I
        rules.append({**r.to_dict(), "rx": re.compile(r["pattern"], flags)})
    return rules  # file order = priority (compounds first)


def check(narrative, hours, rules):
    text = str(narrative)
    expanded = text
    hits, flags = [], set()

    for r in rules:
        if r["rx"].search(expanded):
            hits.append(r["shorthand"])
            flags.add("SHORTHAND")
            if r["ambiguous"] == "Y":
                flags.add("AMBIGUOUS_SHORTHAND")
            expanded = r["rx"].sub(r["expansion"], expanded)

    expanded = re.sub(r"\s{2,}", " ", expanded).strip()
    expanded = expanded[:1].upper() + expanded[1:]
    words = len(expanded.split())

    if words < MIN_WORDS or any(re.search(p, expanded, re.I) for p in VAGUE_PHRASES):
        flags.add("VAGUE")
    if any(re.search(p, expanded, re.I) for p in ADMIN_PHRASES):
        flags.add("ADMIN_TASK")
    tasks = [t for t in TASK_SPLIT.split(expanded) if t and len(t.split()) > 2]
    if len(tasks) >= 3 or (len(tasks) >= 2 and _num(hours) > 0.5):
        flags.add("BLOCK_BILLING")
    if text and text[0].islower():
        flags.add("FORMATTING")

    return {
        "narrative_expanded": expanded,
        "shorthand_found": "|".join(hits),
        "flags": "|".join(sorted(flags)),
        "n_flags": len(flags),
    }


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main(entries_path, bank_path, out_path):
    rules = load_bank(bank_path)
    df = pd.read_csv(entries_path)
    res = df.apply(lambda r: check(r["Narrative"], r.get("Hours"), rules),
                   axis=1, result_type="expand")
    pd.concat([df, res], axis=1).to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
