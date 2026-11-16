"""Build (input, target) training pairs for the line classifier/rewriter (pipeline steps 3-4).

Usage:
  python training/make_pairs.py balanced_clean.csv shorthand_bank.csv out_dir
         [--sep ,] [--seed 0] [--identity-rate 0.2] [--mask-rate 0.6]
         [--abbrev-rate 0.6] [--amb-rate 0.4] [--context-rate 0.6] [--merge-rate 0.8]

Input : the DECONTAMINATED balanced file (run decontaminate.py first). Columns used:
        Time ID, Timekeeper, Date, Hours, Narrative, SOURCE: Case, SOURCE: URL
Output: out_dir/pairs_train.jsonl, pairs_dev.jsonl   training pairs (one task line each)
        out_dir/split_eval_blocks.jsonl              synthetic blocks for eval_splitter.py (NOT training data)
        out_dir/rt_failures.jsonl, stats.json

The model only ever sees ONE task line. Splitting (split_block.py) and re-assembling into a block or
separate entries (assemble.py) are deterministic, so no training pair asks the model to split.

Pair format
  prompt = "[FLAGS: ...] [HOURS: h] [CONTEXT: other line || other line | none] [LINE: text]"
  target = the rewritten LINE only. `label` = the flags (classifier target).

Source lines are prepared the way production sees them: split_block.split_entry -> strip scraper prefixes
(matter codes, timekeeper initials, 'Heading - ') -> true-case ALL CAPS from corpus capitalisation ->
expand unambiguous shorthand. Each line is then sorted:
  * junk (travel, expenses, invoice/report debris)      -> `junk` pairs, label JUNK, target = unchanged (flag only)
  * passes is_good_source()                             -> source for the pair types below
  * otherwise                                           -> unused
is_good_source does NOT gate on a start-word list (that would make every training line start the same way);
it asks for an action in the first 3 words in any form (verb stem, -ing, action noun, irregular past).

Pair types: identity (NONE), ambiguous_shorthand, missing_participant / missing_subject (INCOMPLETE),
vague_object (VAGUE), context_suggested, all_caps (FORMATTING: caps original -> true-cased), junk (JUNK).

Principles
  * Targets come only from rule-selected "good" source lines (no labels exist in the balanced file,
    so audit a sample by hand before trusting them: about 1 in 5 accepted gold lines was not GOOD/SPECIFIC).
  * Recoverable degradations (shorthand) -> target is the original text.
  * Deleted information (participant, subject, object) -> target uses [PARTICIPANT] / [SUBJECT] /
    [DOCUMENT] placeholders, never the deleted specifics, so the model learns not to invent facts.
  * If the exact deleted specific appears in a context line, the target fills it in but marks it:
    "Call with [SUGGESTED: J. Smith] re plan" (type context_suggested). The timekeeper must confirm it, like a
    placeholder, because in production a name in a neighbouring task is a likely guess, not a fact.
    If only part of it is visible, the pair is dropped (stats: dropped_context_leak).
  * Context lines are real same-day entries by the same timekeeper in the same case (what a block
    entry would hold), sent through the same shorthand round trip as the target line.
  * Model input is what the production pipeline would produce: raw text after deterministic
    expansion of UNAMBIGUOUS shorthand (ambiguous shorthand is left for the model).
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # phase2_model/
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, os.path.join(_ROOT, "..", "shorthand_bank"))
from expand_shorthand import expand_unambiguous, load_bank  # noqa: E402  (same expansion as production)
from rule_baseline import check  # noqa: E402
from split_block import EXPENSE_START, VERB, split_entry  # noqa: E402

COLS = dict(id="Time ID", tk="Timekeeper", date="Date", hours="Hours", text="Narrative",
            case="SOURCE: Case", url="SOURCE: URL")

# ---- "does the line say what was done?" Not a start-word gate: any verb form (stem match from split_block.VERB,
# any -ing word), an action noun, or an irregular past tense, anywhere in the first 3 words.
ACTION_NOUN = re.compile(r"^(?:preparation|analysis|analyses|revisions?|correspondence|communications?|discussions?|"
                         r"review|draft|drafting|research|negotiations?|coordination|investigation|evaluation|"
                         r"consultation|conference|teleconference|calls?|e-?mails?|meetings?|attendance|"
                         r"participation|summary|memorandum|letter|edits|comments|outline|interview|"
                         r"follow-?up|reply|response)$", re.I)
IRREGULAR_VERBS = set("met spoke sent wrote read led began took gave made held ran drew sought went oversaw "
                      "undertook brought built found told thought taught understood".split())
# ---- junk: lines that are not billable legal work. They become JUNK-labeled pairs, not good sources.
TRAVEL_JUNK = re.compile(r"^(?:non-?working\s+|local\s+)?travel(?:led|ed|ing)?\b(?!\s+arrangements)", re.I)
# expense records: transport/lodging words are junk anywhere; meal words only when no work action is stated
EXPENSE_ALWAYS = re.compile(r"\b(?:uber|lyft|taxi|cab fare|limousine|limo|wi-?fi|airfare|hotel|lodging|mileage|"
                            r"subway|parking|car service|transportation type|travel (?:dinner|lunch|breakfast))\b",
                            re.I)
EXPENSE_MEAL = re.compile(r"\b(?:dinner|lunch|breakfast|meals?|dining|food service|catering)\b", re.I)
FIELD_JUNK = re.compile(r"\b(?:Type|Starting Point|End Point|Quantity|Hours Value):", re.I)
MONEY = re.compile(r"\$\s?\d")                         # junk only when no work action (else a scraped amount)
REPORT_JUNK = re.compile(r"\b(?:Run Date|Worked thru|Page \d+ of|Invoice (?:Date|No)|Total (?:Hours|Fees)|"
                         r"SUBTOTAL|Timekeeper Summary|BILL NO|Balance Due|Amount Due|Hours Value)\b", re.I)
# ---- prefixes the scraper leaves in front of the real entry
MATTER_CODE = re.compile(r"^(?:\d[\w.-]*/\S*|\d[\w.-]*:|\d{4,}\w*)\s+")
INITIALS = re.compile(r"^([A-Z]{2,4})\s+(?=[A-Z][a-z])")
HEADING_DASH = re.compile(r"^[A-Z][\w/&,-]*(?:\s+[A-Z/&][\w/&,-]*){0,4}\s+[-–]\s+")
DANGLING_END = set("and or of the to re with for in on a an regarding from by at about between".split())
LETTER_OK_BEFORE = set("exhibit schedule class tranche phase part series appendix annex item plan".split())
NEED_SUBJECT = re.compile(r"^(telephone|phone|email|e-mail|conference|confer|correspondence|call|meeting|meet|"
                          r"video|discussion|communicat|attend|participat)", re.I)
SPECIFIC_NOUNS = set("""motion application agreement brief response order plan objection stipulation declaration
report schedule complaint affidavit subpoena claim claims settlement hearing deposition discovery disclosure
statement notice petition budget analysis proposal term sheet contract lease exhibit exhibits opinion retention
sale bid reply joinder adversary proceeding""".split())
NOISE = re.compile(r"Case:|Doc#|Filed:|Entered:|SUBTOTAL|Page \d|\d{6,}|\bTotal\b", re.I)
VAGUE_WORDS = {"matters", "issues", "various", "pending", "workplan", "items", "general"}
SENT_BREAK = re.compile(r"(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bDr)(?<!\bInc)(?<!\bCo)(?<!\bCorp)(?<!\bLtd)"
                        r"(?<!\bNo)(?<![A-Z])[.!?]\s+[A-Z]")
HEADER_NOISE = re.compile(r"\b[A-Z]{4,}\s+[A-Z]{4,}\b")
COMM_SUBJECT_CUE = re.compile(r"\b(regarding|re|concerning|about|on|to discuss|in connection|relating|related|for)\b",
                              re.I)
# Ambiguous shorthand also gets reversed from its other meanings, so the model sees the contrast.
ALT_EXPANSIONS = {"rev": ["revise", "revised", "revision"], "disc": ["discovery"],
                  "admin": ["administration"]}
GENERIC_OBJECTS = ["documents", "materials", "file", "various documents", "matter"]
INITIAL_STOP = set("FTX SEC UST DOJ IRS UCC BSA LTL DIP USA LLC LLP INC NYC PHV".split())

# A counterparty: person ("J. Smith", "Mr. Goodman", "Abramowitz"), or org/role ("Committee", "UST"),
# optionally followed by an affiliation in parentheses ("(ST Law)").
_TOK = r"(?:[A-Z]\.|[A-Z][A-Za-z&'’-]+)"
NAME = rf"(?:(?:Mr|Ms|Mrs|Dr)\.?\s+)?{_TOK}(?:\s+{_TOK}){{0,2}}(?:\s+\([^)]{{2,30}}\))?"
NAMES = rf"{NAME}(?:(?:,\s*(?:and\s+)?|\s+and\s+){NAME})*"
LOOK = (r"(?=\s+(?:re\b|regarding|concerning|about|on\b|to\b|in\b|for\b|relating|related|that\b|whether)"
        r"|\s*[.;:,)]|\s*$)")
PART_RX = re.compile(r"\s+(?P<lead>with|from|to|between)\s+(?P<who>" + NAMES + r")" + LOOK)
COMM_START = re.compile(r"^(telephone|phone|email|e-mail|conference|confer|correspondence|call|meeting|meet|"
                        r"video|discussion|communicat)", re.I)
SUBJ_RX = re.compile(r"\s+(?:regarding|concerning|relating to|related to|in connection with|re:?)\s+"
                     r"(?P<subj>[^.;(]+?)(?P<tail>\s*(?:\([^)]*\))?\.?\s*)$")
OBJ_RX = re.compile(r"^(?P<verb>(?:Review(?:ed)?|Draft(?:ed)?|Revise[d]?|Analy[sz]e[d]?|Prepare[d]?|"
                    r"Research(?:ed)?|Finalize[d]?|Edit(?:ed)?|Update[d]?))\s+(?P<obj>[^.;(,]+?)"
                    r"(?P<tail>\s*(?:\([^)]*\))?\.?\s*)$")
OBJ_BLOCK = re.compile(r"\b(regarding|re|concerning|for|with|from|to)\b", re.I)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def clean(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def key(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", t.lower())).strip()


# ---------------------------------------------------------------- line preparation
_WORD = re.compile(r"[A-Za-z][A-Za-z'&./-]*")


def strip_prefix(t):
    """Remove matter codes, timekeeper initials and 'Heading - ' prefixes left by the scraper."""
    for _ in range(3):
        before = t
        for rx in (MATTER_CODE, HEADING_DASH, INITIALS):
            m = rx.match(t)
            if m and not (rx is INITIALS and m.group(1) in INITIAL_STOP):
                t = t[m.end():]
        if t == before:
            break
    return t.strip()


def build_case_forms(texts):
    """How each word is usually capitalised in non-ALL-CAPS text (ignoring sentence-initial words)."""
    forms = defaultdict(Counter)
    for t in texts:
        if t.isupper():
            continue
        for i, w in enumerate(_WORD.findall(t)):
            if i:
                forms[w.lower()][w] += 1
    return forms


def truecase(t, forms):
    """ALL CAPS -> normal case, keeping a word's usual capitalisation (names, acronyms) from the corpus."""
    if not t.isupper():
        return t

    def fix(m):
        w = m.group(0)
        c = forms.get(w.lower())
        if c:
            best, n = c.most_common(1)[0]
            return best if n / sum(c.values()) >= 0.6 else w.lower()
        before = t[:m.start()].rstrip()
        if re.search(r"\b[A-Z]\.$", before):           # unseen word after an initial: a surname ("A. Rizkalla")
            return w[:1] + w[1:].lower()
        return w.lower()
    out = _WORD.sub(fix, t)
    return out[:1].upper() + out[1:]


def is_junk(t):
    """Non-billable or non-work lines: travel, expenses, invoice/report debris, no real words."""
    letters = sum(ch.isalpha() for ch in t)
    return bool(TRAVEL_JUNK.match(t) or EXPENSE_START.match(t) or REPORT_JUNK.search(t)
                or EXPENSE_ALWAYS.search(t) or FIELD_JUNK.search(t)
                or ((EXPENSE_MEAL.search(t) or MONEY.search(t)) and not has_action(t))
                or letters < 0.5 * max(len(t.replace(" ", "")), 1))


def has_action(t):
    for w in [w.strip(",.:;()").lower() for w in t.split()[:3]]:
        if w and (VERB.match(w) or ACTION_NOUN.match(w) or w in IRREGULAR_VERBS):
            return True
    return False


def looks_truncated(t, vocab):
    words = t.rstrip(" .;").split()
    if not words:
        return True
    last = words[-1]
    if re.search(r"[&/\-–,(]$", last) or last.lower() in DANGLING_END:
        return True
    if re.fullmatch(r"[A-Z]", last) and (len(words) < 2 or words[-2].lower() not in LETTER_OK_BEFORE):
        return True                                    # "... re KERP and M"
    if last.isalpha() and last.islower() and len(last) >= 3 and vocab is not None and vocab[last] < 2:
        return True                                    # "... status and mana" (cut mid-word)
    return False


# ---------------------------------------------------------------- source selection
def is_good_source(text, hours, rules, vocab=None):
    """Rule guess at 'already compliant'. Run on split, prefix-stripped, true-cased, expanded lines."""
    t = clean(text)
    words = t.split()
    if not 6 <= len(words) <= 40 or t.isupper():
        return False
    if NOISE.search(t) or ";" in t or re.search(r"\(\s*\d*\.\d+\s*\)", t) or is_junk(t):
        return False
    if not has_action(t):
        return False
    if not 0 < num(hours) <= 6:
        return False
    if SENT_BREAK.search(t) or HEADER_NOISE.search(t):   # concatenated entries / table headers
        return False
    if looks_truncated(t, vocab):
        return False
    if check(t.rstrip(" .;:"), hours, rules)["flags"]:    # any rule flag (incl. shorthand left over)
        return False
    lw = {w.lower().strip(".,") for w in words}
    if lw & VAGUE_WORDS:
        return False
    if NEED_SUBJECT.match(t) and not COMM_SUBJECT_CUE.search(t):   # call/meeting/attend with no stated subject
        return False
    return bool(lw & SPECIFIC_NOUNS) or any(w[0].isupper() for w in words[2:])


# ---------------------------------------------------------------- shorthand reverse (forward lives in expand_shorthand.py)
def reverse_table(rules):
    by_exp = defaultdict(list)
    for r in rules:
        exp = r["expansion"].strip()
        if exp:
            by_exp[exp.lower()].append(r)
        if r["ambiguous"] == "Y":                       # only ambiguous rules can stand for their other meanings
            for alt in ALT_EXPANSIONS.get(r["shorthand"], []):
                by_exp[alt].append(r)
    return sorted(by_exp.items(), key=lambda kv: -len(kv[0]))     # longest expansion first


def match_case(orig, short):
    if len(orig) > 1 and orig.isupper():
        return short.upper()
    if orig[:1].isupper() and short[:1].islower():
        return short[:1].upper() + short[1:]
    return short


def abbreviate(text, table, rng, rate, amb_rate):
    taken, subs, used = [], [], []
    for exp, rs in table:
        rx = re.compile(r"(?<![\w/&])" + re.escape(exp) + r"(?![\w/&])", re.I)
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))                       # block inner matches even if declined
            r = rng.choice(rs)
            amb = r["ambiguous"] == "Y"
            if rng.random() >= (amb_rate if amb else rate):
                continue
            subs.append((s, e, match_case(m.group(0), r["shorthand"])))
            used.append((r["shorthand"], amb))
    for s, e, rp in sorted(subs, reverse=True):
        text = text[:s] + rp + text[e:]
    return text, used


# ---------------------------------------------------------------- information-removing degradations
def mask_participant(t):
    if not COMM_START.match(t):
        return None
    m = PART_RX.search(t)
    if not m:
        return None
    who = m.group("who")
    return (t[:m.start()] + t[m.end():],
            t[:m.start("who")] + "[PARTICIPANT]" + t[m.end("who"):], [who])


def mask_subject(t):
    m = SUBJ_RX.search(t)
    if not m or not 3 <= len(m.group("subj").split()) <= 25:
        return None
    return (t[:m.start()] + m.group("tail"),
            t[:m.start("subj")] + "[SUBJECT]" + m.group("tail"), [m.group("subj")])


def mask_object(t, rng):
    m = OBJ_RX.match(t)
    if not m:
        return None
    obj = m.group("obj")
    if not 2 <= len(obj.split()) <= 10 or OBJ_BLOCK.search(obj):
        return None
    generic = rng.choice(GENERIC_OBJECTS)
    if key(obj) == key(generic):
        return None
    return (f"{m.group('verb')} {generic}{m.group('tail')}",
            f"{m.group('verb')} [DOCUMENT]{m.group('tail')}", [obj])


def leaks(removed, model_in):
    """True if any removed specific (or a distinctive word of it) is still visible in the input."""
    for span in removed:
        if span in model_in:
            return True
        for w in re.findall(r"[A-Za-z]{4,}", span):
            if w[0].isupper() and re.search(rf"\b{re.escape(w)}\b", model_in):
                return True
    return False


# ---------------------------------------------------------------- pair assembly
class Ctx:
    pass


def degrade_shorthand(ctx, text):
    """Production round trip for one line: maybe abbreviate, then expand unambiguous shorthand."""
    used, raw = [], text
    if ctx.rng.random() < ctx.abbrev_rate:
        raw, used = abbreviate(text, ctx.table, ctx.rng, ctx.rate, ctx.amb_rate)
    return raw, expand_unambiguous(raw, ctx.rules), used


def build(ctx, typ, inp, tgt, base_flags, hours, src_ids, group, context=(), removed=(), abbrev=True,
          force_amb=False):
    used, raw = [], inp
    if abbrev and ctx.rng.random() < ctx.abbrev_rate:
        raw, used = abbreviate(inp, ctx.table, ctx.rng, ctx.rate, ctx.amb_rate)
    model_in = expand_unambiguous(raw, ctx.rules)
    amb_used = [s for s, a in used if a]
    if force_amb and not amb_used:
        return None
    rt_ok = True
    if used and not amb_used:                           # deterministic path must reproduce the original
        rt_ok = key(model_in) == key(inp)
    if removed and leaks(removed, model_in):
        ctx.stats["dropped_leaky"] += 1
        return None
    if removed and context and leaks(removed, " ".join(context)):
        ctx_text = " ".join(context).lower()
        if len(removed) == 1 and removed[0].strip().lower() in ctx_text:
            # the exact missing detail is in a neighbouring line: suggest it, marked for the timekeeper to confirm
            tgt = re.sub(r"\[(PARTICIPANT|SUBJECT|DOCUMENT)\]", lambda _: f"[SUGGESTED: {removed[0].strip()}]",
                         tgt, count=1)
            typ = "context_suggested"
        else:                                          # only partly visible: ambiguous supervision, drop
            ctx.stats["dropped_context_leak"] += 1
            return None
    flags = list(base_flags) + (["AMBIGUOUS_SHORTHAND"] if amb_used else [])
    ctx_str = " || ".join(context) if context else "none"
    return dict(type=typ,
                prompt=f"[FLAGS: {'|'.join(flags)}] [HOURS: {hours:g}] [CONTEXT: {ctx_str}] [LINE: {model_in}]",
                input_model=model_in, input_raw=raw, context=list(context), target=tgt, label=flags,
                hours=hours, flags=flags, shorthand_used=[s for s, _ in used], rt_ok=rt_ok,
                source_ids=src_ids, group=group)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("balanced"); ap.add_argument("bank"); ap.add_argument("out_dir")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--identity-rate", type=float, default=0.2)
    ap.add_argument("--mask-rate", type=float, default=0.6)
    ap.add_argument("--abbrev-rate", type=float, default=0.6)
    ap.add_argument("--amb-rate", type=float, default=0.4)
    ap.add_argument("--context-rate", type=float, default=0.6,
                    help="chance a line with same-day siblings is shown with 1-3 of them as context")
    ap.add_argument("--merge-rate", type=float, default=0.8, help="chance a group yields a split-eval block")
    ap.add_argument("--max-junk", type=int, default=1500, help="cap on JUNK-labeled pairs")
    ap.add_argument("--copies", type=int, default=3,
                    help="stochastic variants per source entry (duplicates are removed)")
    a = ap.parse_args()

    ctx = Ctx()
    ctx.rng = random.Random(a.seed)
    ctx.rules = load_bank(a.bank)
    ctx.table = reverse_table(ctx.rules)
    ctx.abbrev_rate, ctx.rate, ctx.amb_rate = a.abbrev_rate, 0.6, a.amb_rate
    ctx.stats = Counter(dropped_leaky=0, dropped_context_leak=0, dropped_duplicate=0, dropped_roundtrip_failure=0)
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.balanced, sep=a.sep.replace("\\t", "\t"))
    ctx.stats["rows_total"] = len(df)
    # Same order as production: split the entry, then work line by line.
    split_rows = []
    for r in df.to_dict("records"):
        res = split_entry(r[COLS["text"]], r[COLS["hours"]])
        for i, ln in enumerate(res["lines"]):
            split_rows.append((r, i, len(res["lines"]), strip_prefix(clean(ln["text"])), ln["hours"]))
    ctx.stats["lines_total"] = len(split_rows)
    forms = build_case_forms(t for *_, t, _ in split_rows)
    vocab = Counter(w.lower() for *_, t, _ in split_rows for w in _WORD.findall(t))

    sources, junk, seen = [], [], set()
    for r, i, n_lines, t, h in split_rows:
        if not t:
            continue
        caps = t if t.isupper() and sum(ch.isalpha() for ch in t) >= 8 else None
        t_exp = expand_unambiguous(truecase(t, forms), ctx.rules)
        if key(t_exp) in seen:
            continue
        meta = dict(id=f"{r[COLS['id']]}#{i}" if n_lines > 1 else str(r[COLS["id"]]), hours=num(h),
                    tk=r[COLS["tk"]], date=r[COLS["date"]], case=r[COLS["case"]], url=str(r[COLS["url"]]))
        if is_junk(t_exp) and len(t_exp.split()) >= 2:
            seen.add(key(t_exp))
            junk.append(dict(meta, text=t_exp))
        elif h is not None and is_good_source(t_exp, h, ctx.rules, vocab):
            seen.add(key(t_exp))
            sources.append(dict(meta, text=t_exp, caps_original=caps))
    ctx.stats["sources_selected"] = len(sources)
    ctx.stats["sources_from_block_entries"] = sum("#" in s["id"] for s in sources)
    ctx.stats["sources_from_all_caps"] = sum(bool(s["caps_original"]) for s in sources)
    ctx.stats["junk_lines_found"] = len(junk)

    pairs = []
    rng = ctx.rng
    groups = defaultdict(list)                         # same case + timekeeper + day = one possible block
    for s in sources:
        groups[(s["case"], s["tk"], s["date"])].append(s)
    siblings = {s["id"]: [o for o in groups[(s["case"], s["tk"], s["date"])] if o["id"] != s["id"]]
                for s in sources}

    for s in [s for _ in range(a.copies) for s in sources]:
        t, h, g = s["text"], s["hours"], s["url"]
        sib = siblings[s["id"]]
        context = []
        if sib and rng.random() < a.context_rate:     # same context for every pair from this copy
            context = [degrade_shorthand(ctx, o["text"])[1] for o in rng.sample(sib, rng.randint(1, min(3, len(sib))))]
        if rng.random() < a.identity_rate:
            pairs.append(build(ctx, "identity", t, t, ["NONE"], h, [s["id"]], g, context=context, abbrev=False))
        if rng.random() < a.identity_rate:               # ambiguous shorthand only
            pairs.append(build(ctx, "ambiguous_shorthand", t, t, [], h, [s["id"]], g, context=context,
                               force_amb=True))
        for typ, fn, fl in (("missing_participant", mask_participant, ["INCOMPLETE"]),
                            ("missing_subject", mask_subject, ["INCOMPLETE"]),
                            ("vague_object", lambda x: mask_object(x, rng), ["VAGUE"])):
            if rng.random() >= a.mask_rate:
                continue
            res = fn(t)
            if not res or res[0] == t:
                continue
            inp, tgt, removed = res
            pairs.append(build(ctx, typ, inp, tgt, fl, h, [s["id"]], g, context=context, removed=removed))

    # ALL CAPS formatting: the original caps line as input, the true-cased line as target
    for s in sources:
        if s["caps_original"]:
            pairs.append(build(ctx, "all_caps", s["caps_original"], s["text"], ["FORMATTING"], s["hours"],
                               [s["id"]], s["url"], abbrev=False))

    # junk: flag, do not rewrite (target = the line unchanged)
    for j in rng.sample(junk, min(a.max_junk, len(junk))):
        pairs.append(build(ctx, "junk", j["text"], j["text"], ["JUNK"], j["hours"], [j["id"]], j["url"],
                           abbrev=False))

    # synthetic blocks for eval_splitter.py only. Raw (unexpanded) text, because in production the
    # splitter runs BEFORE shorthand expansion.
    blocks = []
    strip = lambda x: x.rstrip(" .")
    for rows in [g for _ in range(a.copies) for g in groups.values()]:
        if len(rows) < 2 or rng.random() >= a.merge_rate:
            continue
        rows = rng.sample(rows, rng.randint(2, min(4, len(rows))))
        with_h = rng.random() < 0.5
        raws = [strip(degrade_shorthand(ctx, r["text"])[0]) for r in rows]
        parts = [f"{x} ({r['hours']:g})" if with_h else x for x, r in zip(raws, rows)]
        blocks.append(dict(input="; ".join(parts) + ".", hours=round(sum(r["hours"] for r in rows), 2),
                           with_times=with_h,
                           lines=[dict(text=x, hours=r["hours"] if with_h else None) for x, r in zip(raws, rows)],
                           source_ids=[r["id"] for r in rows], group=min(r["url"] for r in rows)))
    blocks = list({b["input"]: b for b in blocks}.values())
    with open(os.path.join(a.out_dir, "split_eval_blocks.jsonl"), "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    ctx.stats["split_eval_blocks"] = len(blocks)

    good, bad, dedup = [], [], set()
    for p in pairs:
        if p is None:
            continue
        k = (p["prompt"], p["target"])
        if k in dedup:
            ctx.stats["dropped_duplicate"] += 1
            continue
        dedup.add(k)
        (good if p["rt_ok"] else bad).append(p)
    ctx.stats["dropped_roundtrip_failure"] = len(bad)
    fail_by_sh = Counter(sh for p in bad for sh in p["shorthand_used"])

    train, dev = [], []
    for p in good:
        in_dev = int(hashlib.md5(p["group"].encode()).hexdigest(), 16) % 10 == 0
        p["split"] = "dev" if in_dev else "train"
        (dev if in_dev else train).append(p)

    for name, rows in (("pairs_train.jsonl", train), ("pairs_dev.jsonl", dev), ("rt_failures.jsonl", bad)):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = dict(ctx.stats)
    stats.update(train=len(train), dev=len(dev),
                 with_context=sum(bool(p["context"]) for p in good),
                 by_type=dict(Counter(p["type"] for p in good)),
                 roundtrip_failures_by_shorthand=dict(fail_by_sh.most_common(10)),
                 shorthand_used=dict(Counter(sh for p in good for sh in p["shorthand_used"]).most_common(15)))
    with open(os.path.join(a.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()