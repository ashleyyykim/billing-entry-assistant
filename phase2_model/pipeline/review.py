"""Narrative review at serving time: the whole phase-2 workflow for one time entry.

  1. split      split_block.split_entry          block entry -> task lines with hours
  2. expand     expand_shorthand.expand_line     unambiguous shorthand expanded, ambiguous shorthand reported
  3. classify   classifier.joblib + two rules    flags per line; the entry gets the worst line's flag
  4. rewrite    rewriter/ (flan-t5) or rules     placeholders where information is missing
  5. assemble   assemble.py                      one block entry (default) or separate entries
  6. re-check   open placeholders + candidates   what the timekeeper still has to fill in

  from review import Reviewer
  r = Reviewer("models")                     # models/classifier.joblib, optional models/rewriter/
  r.review("Rev docs (0.5); call re APA (0.3)", 0.8, block=True, history=["Call with J. Smith re APA"])

Candidates for a placeholder come from the entry's other lines and from `history` (earlier narratives on the
same matter, when the caller has them). They are shown as options; nothing is filled in automatically.
The model never guesses a name: its training targets always use the placeholder.
"""
import json
import os
import re
from collections import Counter

import joblib

from assemble import assemble_block, assemble_separate
from expand_shorthand import DEFAULT_BANK, expand_line, load_bank
from split_block import split_entry

# Flags the classifier learns; the other two are exact rules.
MODEL_FLAGS = ["MISSING_PARTICIPANT", "MISSING_SUBJECT", "VAGUE", "JUNK"]
RULE_FLAGS = ["AMBIGUOUS_SHORTHAND", "FORMATTING"]
# Worst first. The entry gets the first flag in this list that any of its lines has.
# Placeholder order until the label set and severities are decided.
SEVERITY = ["JUNK", "MISSING_PARTICIPANT", "MISSING_SUBJECT", "VAGUE", "AMBIGUOUS_SHORTHAND", "FORMATTING"]
PLACEHOLDER_RX = re.compile(r"\[(PARTICIPANT|SUBJECT|DOCUMENT)\]")
FLAG_FOR = {"PARTICIPANT": "MISSING_PARTICIPANT", "SUBJECT": "MISSING_SUBJECT", "DOCUMENT": "VAGUE"}

# ---- regexes shared with make_pairs.py (copied, so the service does not import the training code)
_TOK = r"(?:[A-Z]\.|[A-Z][A-Za-z&'’-]+)"
NAME = rf"(?:(?:Mr|Ms|Mrs|Dr)\.?\s+)?{_TOK}(?:\s+{_TOK}){{0,2}}(?:\s+\([^)]{{2,30}}\))?"
NAMES = rf"{NAME}(?:(?:,\s*(?:and\s+)?|\s+and\s+){NAME})*"
LOOK = (r"(?=\s+(?:re\b|regarding|concerning|about|on\b|to\b|in\b|for\b|relating|related|that\b|whether)"
        r"|\s*[.;:,)]|\s*$)")
PART_RX = re.compile(r"\s+(?P<lead>with|from|to|between)\s+(?P<who>" + NAMES + r")" + LOOK)
COMM_START = re.compile(r"^(?:telephone conference|telephone call|phone call|conference call|video conference|"
                        r"e-?mails?(?: exchange| correspondence)?|correspondence|correspond|conferences?|conferred|"
                        r"confer|calls?|meetings?|meet|discussions?|discuss|communications?|communicate)\b", re.I)
SUBJ_RX = re.compile(r"\s+(?:regarding|concerning|relating to|related to|in connection with|re:?)\s+"
                     r"(?P<subj>[^.;(]+?)(?P<tail>\s*(?:\([^)]*\))?\.?\s*)$", re.I)
OBJ_RX = re.compile(r"^(?P<verb>(?:Review(?:ed)?|Draft(?:ed)?|Revise[d]?|Analy[sz]e[d]?|Prepare[d]?|"
                    r"Research(?:ed)?|Finalize[d]?|Edit(?:ed)?|Update[d]?))\s+(?P<obj>[^.;(,]+?)"
                    r"(?P<tail>\s*(?:\([^)]*\))?\.?\s*)$", re.I)
VAGUE_OBJ_RX = re.compile(r"\b(?:various documents|documents|materials|file|matters?|issues|items)\b", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def tidy(t):
    """Scrape artifacts: spaces before punctuation, doubled spaces."""
    t = re.sub(r"\s+([.,;:)])", r"\1", str(t))
    return re.sub(r"\s{2,}", " ", t).strip()


def is_caps(t):
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) >= 0.8


def rewriter_prompt(flags, line):
    """Input format for the rewriter. train_rewriter.py builds its training inputs with this same function."""
    return f"fix billing line | flags: {', '.join(flags) or 'NONE'} | line: {line}"


# ---------------------------------------------------------------- rule rewriter (fallback)
COMMON_SHORT = set("AND THE WITH FOR RE TO OF ON IN AT BY OR AS IS ARE WAS ALL ANY OUR NEW OLD PER VIA "
                   "CALL MEET SEND FILE PLAN SALE CASE DRAFT EMAIL NOTE".split())


def _sentence_case(t):
    """Rule fallback for ALL CAPS: lower-case, keep initials and 2-3 letter acronyms (UCC, DOJ, POR),
    capitalise the word after an initial (C. ANDERSON -> C. Anderson). Imperfect; the timekeeper checks it."""
    out, prev_initial = [], False
    for w in t.split():
        core = w.strip(".,;:()")
        if re.fullmatch(r"[A-Z]\.", w):
            out.append(w); prev_initial = True; continue
        if 2 <= len(core) <= 3 and core.isalpha() and core.isupper() and core not in COMMON_SHORT:
            out.append(w)
        elif prev_initial and core.isalpha():
            out.append(w[:1].upper() + w[1:].lower())
        else:
            out.append(w.lower())
        prev_initial = False
    s = " ".join(out)
    return s[:1].upper() + s[1:]


def rule_rewrite(line, flags):
    t = line
    if "FORMATTING" in flags:
        t = _sentence_case(t)
    if "MISSING_PARTICIPANT" in flags and "[PARTICIPANT]" not in t:
        m = COMM_START.match(t)
        if m:
            t = t[:m.end()] + " with [PARTICIPANT]" + t[m.end():]
    if "VAGUE" in flags and "[DOCUMENT]" not in t:
        t = VAGUE_OBJ_RX.sub("[DOCUMENT]", t, count=1)
    if "MISSING_SUBJECT" in flags and "[SUBJECT]" not in t and not SUBJ_RX.search(t):
        m = re.search(r"\s*\.?\s*$", t)
        t = t[:m.start()] + " regarding [SUBJECT]" + t[m.start():]
    return t


CONNECTORS = set("with regarding re concerning about to and the a an for of in on from between".split())


ALT_WORDS = {"rev": {"revised", "revision"}, "admin": {"administration"}}


def allowed_words(rules, raw=None):
    """Words a rewrite may add: connectors for placeholders, plus the expansions of shorthand that appears in
    this line's raw text (the rewriter resolves ambiguous shorthand such as rev -> review / revise).
    raw=None allows every expansion in the bank (used only for offline scoring)."""
    words = set(CONNECTORS)
    for r in rules:
        if raw is None or r["rx"].search(raw):
            words |= {w.lower() for w in WORD.findall(r.get("expansion", ""))}
            words |= ALT_WORDS.get(r["shorthand"].lower(), set())
    return words


def safe_rewrite(inp, out, allowed=None):
    """Reject rewriter output that invents content. Every word must come from the input (any case), the
    connectors, or a shorthand expansion: a seq2seq model can swap 'testimony' for 'counsel' fluently.
    Also rejects new numbers and large changes in length."""
    if not out or not out.strip():
        return False
    base = {w.lower() for w in WORD.findall(inp)}
    allowed = CONNECTORS if allowed is None else allowed
    for w in WORD.findall(PLACEHOLDER_RX.sub(" ", out)):
        if w.lower() not in base and w.lower() not in allowed:
            return False
    if set(re.findall(r"\d+", out)) - set(re.findall(r"\d+", inp)):
        return False
    n_in, n_out = len(inp.split()), len(out.split())
    return n_out <= 2 * n_in + 8 and n_out >= 0.5 * n_in


# ---------------------------------------------------------------- candidates for placeholders
def _participants(t):
    out = []
    for m in PART_RX.finditer(t):
        for who in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group("who")):
            if who and who.lower() not in {"same", "team"}:
                out.append(who.strip())
    return out


def _subjects(t):
    m = SUBJ_RX.search(t)
    if not m:
        return []
    s = m.group("subj").strip()
    return [] if s.lower() in {"same", "the same", "same issues"} else [s]


def _documents(t):
    m = OBJ_RX.match(t)
    if not m or VAGUE_OBJ_RX.fullmatch(m.group("obj").strip()):
        return []
    obj = re.split(r"\s+(?:regarding|re|concerning|for|with|from|to)\s+", m.group("obj"), maxsplit=1)[0]
    return [obj.strip()] if 1 <= len(obj.split()) <= 8 else []


EXTRACT = {"PARTICIPANT": _participants, "SUBJECT": _subjects, "DOCUMENT": _documents}


def candidates(kind, same_entry, history, k=5):
    """Ranked options: the entry's other lines first, then history, most frequent first within each."""
    out, seen = [], set()
    for source, texts in (("same entry", same_entry), ("history", history)):
        counts = Counter(x for t in texts for x in EXTRACT[kind](t))
        for value, n in counts.most_common():
            if value.lower() not in seen:
                seen.add(value.lower())
                out.append(dict(value=value, source=source, count=n))
    return out[:k]


# ---------------------------------------------------------------- the reviewer
class Reviewer:
    def __init__(self, model_dir, bank_path=None, use_rewriter=True):
        bank_path = bank_path or (os.path.join(model_dir, "shorthand_bank.csv")
                                  if os.path.exists(os.path.join(model_dir, "shorthand_bank.csv")) else DEFAULT_BANK)
        self.rules = load_bank(bank_path)
        clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        self.vec, self.clf, self.flags = clf["vectorizer"], clf["model"], clf["flags"]
        self.thresholds = clf.get("thresholds", {f: 0.5 for f in self.flags})
        meta = os.path.join(model_dir, "metadata.json")
        self.meta = json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}
        self.tok = self.seq2seq = None
        rw = os.path.join(model_dir, "rewriter")
        if use_rewriter and os.path.isdir(rw):
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(rw)
            self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(rw).eval()

    @property
    def rewriter_name(self):
        return "flan-t5 (fine-tuned)" if self.seq2seq is not None else "rules"

    def classify(self, texts):
        probs = self.clf.predict_proba(self.vec.transform(texts))
        return [{f: float(p[i]) for i, f in enumerate(self.flags)} for p in probs]

    def generate(self, prompts):
        import torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=192)
        with torch.no_grad():
            out = self.seq2seq.generate(**enc, max_new_tokens=128, num_beams=2)
        return [tidy(x) for x in self.tok.batch_decode(out, skip_special_tokens=True)]

    def review(self, narrative, hours=None, block=True, history=()):
        return self.review_many([dict(narrative=narrative, hours=hours, history=list(history or []))], block)[0]

    def review_many(self, entries, block=True):
        """entries: [{narrative, hours, history?}]. Batched so the models run once per request."""
        recs, lines = [], []
        for e in entries:
            rec = split_entry(e.get("narrative") or "", e.get("hours"))
            rec["lines"] = [ln for ln in rec["lines"] if tidy(ln["text"]).strip(" .;,")]   # nothing to review
            for ln in rec["lines"]:
                ln["text"] = tidy(ln["text"])
                ln.update(expand_line(ln["text"], self.rules))
                lines.append(ln)
            recs.append(rec)
        texts = [ln["text_expanded"] for ln in lines]
        probs = self.classify(texts) if texts else []

        to_rewrite = []
        for ln, p in zip(lines, probs):
            flags = [f for f in self.flags if p[f] >= self.thresholds.get(f, 0.5)]
            if ln["ambiguous_shorthand"]:
                flags.append("AMBIGUOUS_SHORTHAND")
            if is_caps(ln["text"]):
                flags.append("FORMATTING")
            ln["flags"] = [f for f in SEVERITY if f in flags]
            ln["scores"] = {f: round(v, 3) for f, v in p.items()}
            ln["rewrite_source"] = "unchanged"
            ln["suggestion"] = ln["text_expanded"]
            if ln["flags"] and "JUNK" not in ln["flags"]:
                to_rewrite.append(ln)

        if to_rewrite and self.seq2seq is not None:
            outs = self.generate([rewriter_prompt(ln["flags"], ln["text_expanded"]) for ln in to_rewrite])
        else:
            outs = [None] * len(to_rewrite)
        for ln, out in zip(to_rewrite, outs):
            if out is not None and safe_rewrite(ln["text_expanded"], out, allowed_words(self.rules, ln["text"])):
                ln["suggestion"], ln["rewrite_source"] = out, "model"
            else:
                ln["suggestion"] = rule_rewrite(ln["text_expanded"], ln["flags"])
                ln["rewrite_source"] = "rules" if out is None else "rules (model output rejected)"
            # a flag whose placeholder never appeared is still shown; make sure every placeholder has its flag
            for kind in PLACEHOLDER_RX.findall(ln["suggestion"]):
                if FLAG_FOR[kind] not in ln["flags"]:
                    ln["flags"] = [f for f in SEVERITY if f in ln["flags"] + [FLAG_FOR[kind]]]

        results = []
        for rec, e in zip(recs, entries):
            history = [h for h in (e.get("history") or []) if isinstance(h, str)]
            open_ph = []
            for i, ln in enumerate(rec["lines"]):
                others = [o["text_expanded"] for j, o in enumerate(rec["lines"])
                          if j != i and "JUNK" not in o["flags"]]       # "travel to Wilmington" is not a participant
                ln["placeholders"] = [dict(kind=k, candidates=candidates(k, others, history))
                                      for k in PLACEHOLDER_RX.findall(ln["suggestion"])]
                ln["ambiguous_shorthand"] = [dict(found=a["found"], default=a["default"], note=a["note"])
                                             for a in ln["ambiguous_shorthand"]]
                open_ph += [dict(line=i, kind=p["kind"]) for p in ln["placeholders"]]
            flags_all = {f for ln in rec["lines"] for f in ln["flags"]}
            worst = next((f for f in SEVERITY if f in flags_all), None)
            rewritten = dict(heading=rec["heading"],
                             lines=[dict(text=ln["suggestion"], hours=ln["hours"], hours_source=ln["hours_source"])
                                    for ln in rec["lines"]])
            results.append(dict(
                narrative=e.get("narrative"), hours=e.get("hours"),
                split_status=rec["status"], split_notes=rec["flags"], heading=rec["heading"],
                worst_flag=worst, ok=worst is None,
                lines=[{k: ln[k] for k in ("text", "text_expanded", "hours", "hours_source", "flags", "scores",
                                            "suggestion", "rewrite_source", "placeholders", "ambiguous_shorthand")}
                       for ln in rec["lines"]],
                suggested_block=assemble_block(rewritten) if rec["lines"] else "",
                suggested_separate=assemble_separate(rewritten),
                output_mode="block" if block else "separate",
                open_placeholders=open_ph))
        return results
