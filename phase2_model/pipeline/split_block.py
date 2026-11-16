"""Step 1 of the pipeline: split a block-billed time entry into task lines with hours.

Splitting only. Shorthand expansion (expand_shorthand.py), classification, rewriting and
assembly (assemble.py) happen in later steps.

  from split_block import split_entry
  r = split_entry("Review draft (0.4); call with J. Smith re plan (0.3).", 0.7)

  python pipeline/split_block.py in.csv out.jsonl [--text-col Narrative --hours-col Hours --id-col "Time ID"]

Result dict:
  status   single | split_ok | split_hours_mismatch | split_inferred_remainder | split_hours_missing
  lines    [{text, hours, hours_source}]   hours_source: itemized | inferred | missing
  heading  billing-category heading removed from the text (e.g. 'Case Administration'), or ''
  total_hours, itemized_sum, flags[]

Rules (nothing is invented; any hours not written in the entry stay None unless exactly one task is untimed,
in which case the remainder is assigned and marked 'inferred'):
  1. strip invoice noise (headers, footers, docket ids)
  2. split on semicolons; within a piece, 2+ times split after each time
  3. no semicolons but 2+ times -> split after each time
  4. a time is a parenthesised number <= 24, optionally 'hours' or '- comment', e.g. (.4) (0.25) (2) (1.0 - partial)
"""
import argparse
import csv
import json
import re
from collections import Counter

HEADINGS = re.compile(
    r"^(?:Case Administration|Asset Sales|Asset Disposition|Court Hearings|Retention and Fee Applications|"
    r"Fee/Employment Applications(?: - Retained Professionals)?)\s+(?=[A-Za-z])")
NOISE_HEAD = re.compile(r"^\s*ITEMIZED SERVICES\s*-\s*\S+\s*-.*?\bIndex\b\s*", re.I)
NOISE_TAIL = re.compile(
    r"\s+(?:Invoice Date:|Total Hours\b|SUBTOTAL\b|Case:\s*\d|Doc#|TIME SUMMARY\b|Matter Number:|Matter Name:|"
    r"Client/Matter|Client:|TOTAL COURTESY DISCOUNT|Total Fees\b|Total Adjusted|BILL NO\b)", re.S)
LABEL = re.compile(r"\b(?:Matter Name|Matter Number|Client/Matter|Client):\s*[^();]*")
EXPENSE_START = re.compile(r"^\W*(?:meals?|taxi|cab|uber|airfare|airplane|lodging|hotel|parking|mileage|"
                           r"local transportation|ground transportation|out of town|long distance|"
                           r"travel (?:to|from))\b", re.I)
VERB = re.compile(
    r"^(?:review|draft|prepar|attend|call|e-?mail|revis|confer|telephon|research|analy[sz]|correspond|particip|"
    r"update|finaliz|edit|coordinat|conduct|discuss|meet|fil|send|respond|negotiat|communicat|summari[sz]|"
    r"evaluat|investigat|assist|monitor|perform|complet|exchang|circulat|compil|organi[sz]|retriev|repl|speak|"
    r"interview|work|continu|follow|attention|plan|read|check|schedul)\w*\b"
    # shorthand that may still be unexpanded when splitting happens before expansion
    r"|^(?:rev|rvw|revw|drft|dft|prep|corr|corresp|conf|tel|tc|tcw|tcf|tct|ocw|vcw|mtw|ecw|emt|emf|lmw|lmf|rsrch|"
    r"ck|chk|ems?|mtgs?|disc|r&r|d&r|r&a|p&f)\b|^\w+ing\b", re.I)
DANGLING = {"for", "in", "with", "to", "of", "on", "at", "from", "by", "re", "regarding", "concerning", "about",
            "between", "and"}
DOC_ID = re.compile(r"\(\s*\d{6,}\s*\)\.?")
TIME = re.compile(r"(?<![A-Za-z0-9)])\(\s*(\d{0,2}\.?\d{1,2})\s*(?:h|hr|hrs|hours)?\s*(?:[-–]\s*[A-Za-z][^()]*)?\)")
EDGE = re.compile(r"^[\s;,.\-–]+|[\s;,]+$")
LEAD_AND = re.compile(r"^(?:and|also)\s+", re.I)
WEAK_MULTI = re.compile(r"\band\s+(?:review|draft|prepare|attend|call|email|revise|confer|research|analy[sz]e)\b", re.I)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def clean_noise(text):
    t = re.sub(r"\s+", " ", str(text)).strip()
    t = NOISE_HEAD.sub("", t)
    m = NOISE_TAIL.search(t)
    if m and len(t[:m.start()].split()) >= 3:
        # a footer runs to the end; a label sitting between tasks (a ';' follows) is removed on its own
        t = LABEL.sub("", t) if ";" in t[m.start():] else t[:m.start()]
    return DOC_ID.sub("", t).strip()


def _times(piece):
    out = []
    for m in TIME.finditer(piece):
        v = float(m.group(1))
        if not 0 < v <= 24:
            continue
        if "." not in m.group(1):                      # bare integers like "(3)" are often counts, not hours
            nxt = piece[m.end():].lstrip()
            if nxt and nxt[0] not in ";,.:" and not nxt[0].isupper():
                continue
        out.append((m, v))
    return out


def _tidy(s):
    s = EDGE.sub("", s)
    s = LEAD_AND.sub("", s)
    return EDGE.sub("", s).strip()


def _split_piece(piece):
    """Split one semicolon piece after each time when it holds 2+ times. Returns [(text, hours|None)]."""
    ts = _times(piece)
    if len(ts) <= 1:
        out = [(TIME.sub("", piece) if ts else piece, ts[0][1] if ts else None)]
        return [(_tidy(t), h) for t, h in out]
    out, start = [], 0
    for m, h in ts:
        out.append((_tidy(piece[start:m.start()]), h))
        start = m.end()
    rest = _tidy(piece[start:])
    if rest:
        ends_dangling = all(t.split() and t.split()[-1].lower() in DANGLING for t, _ in out)
        if ends_dangling and not VERB.match(rest):     # "prepare for (.2) and participate in (.5) call with X"
            out = [(f"{t} {rest}", h) for t, h in out]
        else:
            out.append((rest, None))
    return out


def split_entry(narrative, hours):
    total = round(_num(hours), 4)
    text = clean_noise(narrative)
    heading = ""
    flags = []
    pieces = [p for p in re.split(r";\s*", text) if p.strip()]
    lines = []
    for p in pieces:
        m = HEADINGS.match(p.strip())
        if m:
            heading = heading or m.group(0).strip()
            p = p.strip()[m.end():]
        lines += _split_piece(p)
    cleaned = []
    for t, h in lines:
        m = HEADINGS.match(t)
        if m:
            heading = heading or m.group(0).strip()
            t = t[m.end():]
        if h is None and len(t.split()) < 2:            # untimed scraps such as a bare ".", "re same"
            continue
        if not t:
            continue
        cleaned.append((t, h))
    lines = cleaned
    if (len(lines) >= 2 and lines[-1][1] is None and any(h is not None for _, h in lines)
            and not VERB.match(lines[-1][0])):
        flags.append("trailing text dropped: " + lines[-1][0][:60])   # letterhead or header debris
        lines = lines[:-1]
    if EXPENSE_START.match(text):
        flags.append("expense_line")
        lines = []
    if len(lines) < 2:
        t = _tidy(TIME.sub("", HEADINGS.sub("", text)))
        if WEAK_MULTI.search(t):
            flags.append("possible_multi_task")
        return dict(status="single", heading=heading, total_hours=total, itemized_sum=None,
                    lines=[dict(text=t, hours=total or None, hours_source="entry_total")], flags=flags)

    timed = [h for _, h in lines if h is not None]
    untimed = [i for i, (_, h) in enumerate(lines) if h is None]
    s = round(sum(timed), 4)
    out = [dict(text=t, hours=h, hours_source="itemized" if h is not None else "missing") for t, h in lines]
    if not untimed:
        status = "split_ok" if abs(s - total) <= 0.051 else "split_hours_mismatch"
        if status == "split_hours_mismatch":
            flags.append(f"times sum to {s:g} but entry total is {total:g}")
    elif len(untimed) == 1 and timed and total - s > 0.051:
        i = untimed[0]
        out[i].update(hours=round(total - s, 4), hours_source="inferred")
        status = "split_inferred_remainder"
        flags.append("one untimed task; hours taken as total minus the other tasks")
    else:
        status = "split_hours_missing"
        flags.append(f"{len(untimed)} of {len(lines)} tasks have no time")
    if any(len(t.split()) < 2 for t, h in lines if h is not None):
        flags.append("fragment: a timed task has fewer than 2 words (entry may start or end mid-sentence)")
    if any(h is not None and abs(h * 10 - round(h * 10)) > 1e-6 for h in timed):
        flags.append("time not in tenths of an hour")
    return dict(status=status, heading=heading, total_hours=total, itemized_sum=s if timed else None,
                lines=out, flags=flags)


def main():
    ap = argparse.ArgumentParser(description="Split time entries into task lines.")
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--text-col", default="Narrative")
    ap.add_argument("--hours-col", default="Hours")
    ap.add_argument("--id-col", default="Time ID")
    a = ap.parse_args()
    n, status = 0, Counter()
    with open(a.inp, newline="", encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as g:
        for row in csv.DictReader(f):
            r = split_entry(row[a.text_col], row[a.hours_col])
            r = dict(id=row.get(a.id_col, str(n)), narrative=row[a.text_col], **r)
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            status[r["status"]] += 1
    print(f"wrote {n} entries -> {a.out}")
    for k, v in status.most_common():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
