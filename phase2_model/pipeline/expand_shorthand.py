"""Step 2 of the pipeline: expand shorthand on each split line.

Runs AFTER split_block.py, so expansions that contain "and" (R&R -> "review and revise")
can never be mistaken for a task boundary.

Policy
  * unambiguous shorthand (bank column ambiguous=N) is expanded deterministically
  * ambiguous shorthand (ambiguous=Y, e.g. "rev" = review/revise, "cl" = client/?) is LEFT IN the text
    and reported per line, so the rewriter model (or the timekeeper) resolves it from context.
    This matches the inputs make_pairs.py builds for training.

  from expand_shorthand import load_bank, expand_line
  rules = load_bank()                      # data/raw/shorthand_bank_v2.csv
  expand_line("R&R APA; tcw J. Smith re rev", rules)

  python pipeline/expand_shorthand.py split.jsonl expanded.jsonl [--bank data/raw/shorthand_bank_v2.csv]
     input : JSONL from split_block.py (one entry per line, each with `lines`)
     output: same records; every line gains text_expanded, shorthand_expanded, ambiguous_shorthand,
             and the entry gains flag AMBIGUOUS_SHORTHAND if any line still holds ambiguous shorthand
"""
import argparse
import csv
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANK = os.path.normpath(os.path.join(HERE, "..", "data", "raw", "shorthand_bank_v2.csv"))


def load_bank(path=DEFAULT_BANK):
    """Rules in file order (= priority: compounds like TCW come before CW)."""
    rules = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r = {k: (v or "") for k, v in r.items()}
            r["rx"] = re.compile(r["pattern"], 0 if r["case_sensitive"] == "Y" else re.I)
            rules.append(r)
    return rules


def _tidy(text):
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:1].upper() + text[1:]


def expand_unambiguous(text, rules):
    """Expand every unambiguous rule. Returns the text only (used by make_pairs.py)."""
    return expand_line(text, rules)["text_expanded"]


def expand_line(text, rules):
    out, used = str(text), []
    for r in rules:
        if r["ambiguous"] == "Y":
            continue
        out, n = r["rx"].subn(r["expansion"], out)
        if n:
            used.append(r["shorthand"])
    out = _tidy(out)
    amb = [dict(shorthand=r["shorthand"], found=m.group(0), default=r["expansion"], note=r["notes"])
           for r in rules if r["ambiguous"] == "Y" for m in r["rx"].finditer(out)]
    return dict(text_expanded=out, shorthand_expanded=used, ambiguous_shorthand=amb)


def expand_record(rec, rules):
    for ln in rec["lines"]:
        ln.update(expand_line(ln["text"], rules))
    if any(ln["ambiguous_shorthand"] for ln in rec["lines"]) and "AMBIGUOUS_SHORTHAND" not in rec["flags"]:
        rec["flags"].append("AMBIGUOUS_SHORTHAND")
    return rec


def main():
    ap = argparse.ArgumentParser(description="Expand shorthand on split task lines.")
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--bank", default=DEFAULT_BANK)
    a = ap.parse_args()
    rules = load_bank(a.bank)
    n_entries = n_lines = n_changed = n_amb = 0
    used, amb = Counter(), Counter()
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as g:
        for line in f:
            rec = expand_record(json.loads(line), rules)
            n_entries += 1
            for ln in rec["lines"]:
                n_lines += 1
                n_changed += ln["text_expanded"] != ln["text"]
                n_amb += bool(ln["ambiguous_shorthand"])
                used.update(ln["shorthand_expanded"])
                amb.update(x["shorthand"] for x in ln["ambiguous_shorthand"])
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {n_entries} entries ({n_lines} lines) -> {a.out}")
    print(f"  lines changed by expansion:        {n_changed}")
    print(f"  lines with ambiguous shorthand left: {n_amb}")
    print("  expanded:", dict(used.most_common(15)))
    print("  ambiguous left:", dict(amb.most_common()))


if __name__ == "__main__":
    main()
