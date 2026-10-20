"""Parse time entries sent by email into the same fields as a typed entry: CMID, client, matter, hours, narrative.

The rules come from general email and billing conventions, not from any labeled entries:
- email clutter: reply chains ("On ... wrote:", "-----Original Message-----", Outlook From/Sent/To blocks),
  quoted lines ("> ..."), greetings, sign-offs, mobile footers ("Sent from my iPhone"), confidentiality notices
- requests to the person entering time ("Please enter...", "My time for today:")
- hours written as decimals (".3", "1.5 hrs"), minutes ("45 min"), "1 hr 15 min", or "1:15"
- client-matter numbers ("7215.003", "7215-003", "C/M 7215/003")
- client names from the firm's own client list (names and aliases), allowing typos and missing spaces,
  with "Client / Matter" and "Client: ..." headers for anything not on the list
"""
import re

import pandas as pd
from rapidfuzz import fuzz, process

from clean import is_blank

# ---------------- Email clutter ----------------

REPLY_MARKERS = re.compile(
    r"^\s*(On\s.{4,160}wrote:|-{2,}\s*(Original|Forwarded) Message\s*-{2,}|Begin forwarded message:"
    r"|From:\s.+|_{10,})\s*$", re.I | re.M)
HEADER_LINE = re.compile(r"^\s*(From|Sent|Date|To|Cc|Bcc):.*$", re.I | re.M)
SUBJECT_LINE = re.compile(r"^\s*Subject:\s*(?:(?:re|fw|fwd)\s*:\s*)*(.*)$", re.I | re.M)
QUOTED_LINE = re.compile(r"^\s*>.*$", re.M)
SIGNATURE_DELIMITER = re.compile(r"^--\s*$", re.M)
DISCLAIMER = re.compile(
    r"(\*{0,2}\s*(CONFIDENTIALITY NOTICE|CONFIDENTIAL|PRIVILEGED|NOTICE)\s*[:*-]"
    r"|This (e-?mail|message|communication|transmission)[^.]{0,80}(confidential|privileged|intended)).*",
    re.I | re.S)
MOBILE_FOOTER = re.compile(
    r"(Sent from my \w+(\s\w+)?|Sent from (Outlook|Mail|Yahoo Mail|Gmail) for \w+|Get Outlook for \w+"
    r"|Sent (via|from) (the )?\w+ (app|mobile))\.?", re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening))\b[^\n,!:]{0,40}[,!:]?\s*", re.I)
SIGN_OFF = re.compile(
    r"(^|[\s,.;:!)-])(thanks( so much| again)?|thank you|thx|tx|ty|best( regards)?|regards|cheers|sincerely"
    r"|many thanks)\s*[!.,]*(\s*[-–—]?\s*[A-Z][\w.]{0,15}){0,3}\s*$", re.I)
CLOSING_LINE = re.compile(r"(thanks( so much| again)?|thank you|thx|tx|ty|best( regards)?|regards|kind regards|cheers"
                          r"|sincerely|many thanks)\s*[!.,]*", re.I)
TRAILING_INITIALS = re.compile(r"\s[-–—~]\s?[A-Z]{1,3}\.?\s*$")
REQUEST = re.compile(
    r"\b(please|pls|plz|kindly|can you|could you)?\s*(enter|bill|record|add|log|put in|input|book)\b"
    r"(\s+(my|the following|this|these|today'?s|yesterday'?s))?(\s+(time|hours|entry|entries))?\s*:?", re.I)
TIME_LABEL = re.compile(
    r"\b(my\s+)?(time|hours)(\s+entr(y|ies))?(\s+for)?\s+(today|yesterday|this week|last week|"
    r"\d{1,2}/\d{1,2}(/\d{2,4})?|(mon|tues|wednes|thurs|fri|satur|sun)day)\s*:?|\btime\s+entr(y|ies)\s*:"
    r"|^\s*(for\s+)?(today|yesterday|\d{1,2}/\d{1,2}(/\d{2,4})?)\s*:", re.I)


def strip_clutter(text):
    """The part of an email that holds the entry: no reply chain, quotes, signature, disclaimer, or greeting."""
    text = str(text).replace("\r\n", "\n")
    subject = SUBJECT_LINE.search(text)
    marker = REPLY_MARKERS.search(text)
    if marker:
        above = text[:marker.start()]
        # A forward with nothing written above it: the entry is in the forwarded part
        text = above if above.strip() else text[marker.end():]
    text = SUBJECT_LINE.sub("", HEADER_LINE.sub("", text))
    text = QUOTED_LINE.sub("", text)
    text = SIGNATURE_DELIMITER.split(text)[0]
    text = DISCLAIMER.sub("", text)
    text = MOBILE_FOOTER.sub("", text)
    text = GREETING.sub("", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # Everything from a closing line ("Thanks,", "Best,") down is the signature
    closing = [i for i, ln in enumerate(lines) if i > 0 and CLOSING_LINE.fullmatch(ln)]
    if closing:
        lines = lines[:closing[0]]
    # Sign-off on its own line(s) at the end ("Thanks,\nAlex"), or at the end of a one-line email
    while lines and (SIGN_OFF.fullmatch(" " + lines[-1]) or (len(lines) > 1 and len(lines[-1].split()) <= 2
                                                             and not re.search(r"\d", lines[-1]))):
        lines.pop()
    if lines:
        lines[-1] = TRAILING_INITIALS.sub("", SIGN_OFF.sub("", lines[-1])).strip()
    subject_text = subject.group(1).strip() if subject else ""
    return [ln for ln in lines if ln], subject_text


# ---------------- Hours ----------------

NOT_HOURS_BEFORE = re.compile(r"(rule|section|sec|item|§|no|v|version|form|article|art|reg|schedule|exhibit|ex)\.?\s*$", re.I)
HOURS_PATTERNS = [
    ("hm", re.compile(r"(?<![\d:])(\d{1,2}):([0-5]\d)(?!\d|:\d)(?!\s*(am|pm|a\.m|p\.m))", re.I)),
    ("unit", re.compile(r"(?<![\w.])(\d+(?:\.\d+)?|\.\d+)\s*(?:hours?|hrs?|h)\b\.?"
                        r"(?:\s*(?:and\s*)?(\d{1,2})\s*(?:minutes?|mins?|m)\b\.?)?", re.I)),
    ("min", re.compile(r"(?<![\w.])(\d{1,3})\s*(?:minutes?|mins?)\b\.?", re.I)),
    ("tenths", re.compile(r"(?<![\w.,§/\-])(\d{0,2}\.\d)(?![\w.%/\-]|\d)")),
]


def find_hours(text):
    """Every amount of time in the text: list of (hours, start, end, in_parentheses).

    Billing is in tenths, so a bare number like "2.2" or ".5" counts as hours, unless it follows
    a word like "Rule" or "Item" ("Rule 1.5", "Item 5.02" aren't hours).
    """
    found, taken = [], [False] * len(text)
    for kind, pattern in HOURS_PATTERNS:
        for m in pattern.finditer(text):
            if any(taken[m.start():m.end()]) or NOT_HOURS_BEFORE.search(text[:m.start()]):
                continue
            if kind == "hm":
                hours = int(m.group(1)) + int(m.group(2)) / 60
            elif kind == "unit":
                hours = float(m.group(1)) + (int(m.group(2)) / 60 if m.group(2) else 0)
            elif kind == "min":
                hours = int(m.group(1)) / 60
            else:
                hours = float(m.group(1))
            if not 0 < hours <= 24:
                continue
            in_parens = text[:m.start()].rstrip().endswith("(") and text[m.end():].lstrip().startswith(")")
            found.append((round(hours, 2), m.start(), m.end(), in_parens))
            taken[m.start():m.end()] = [True] * (m.end() - m.start())
    return sorted(found, key=lambda f: f[1])


def total_hours(amounts):
    """One amount -> that amount. Several -> their sum, unless one of them is already the total of the others."""
    if not amounts:
        return None
    values = [a[0] for a in amounts]
    for i, v in enumerate(values):
        rest = values[:i] + values[i + 1:]
        if rest and abs(v - sum(rest)) < 0.05:
            return round(v, 2)
    return round(sum(values), 2)


# ---------------- Client-matter numbers ----------------

CMID_IN_EMAIL = re.compile(
    r"(?:\b(?:c\s*/\s*m|client[\s-]*matter|matter)\s*(?:no\.?|#|number)?\s*:?\s*#?\s*)?"
    r"(?<![\d.\-/])(\d{3,6})\s*[.\-/]\s*(\d{1,4})(?![\d.\-/])", re.I)


def find_cmid(text):
    """The first client-matter number as written ("7215/003"), with its position, or None."""
    m = CMID_IN_EMAIL.search(text)
    return (f"{m.group(1)}.{m.group(2)}", m.start(), m.end()) if m else None


# ---------------- Client and matter ----------------

LABELED = re.compile(r"^\s*(client|matter|c\s*/\s*m|time|hours|description|narrative|task|work)\s*:\s*(.*)$", re.I)
SPLIT_MATTER = re.compile(r"^\s+[/|]\s+([^/|:;,()\[\]\n]{2,60}?)\s*(?=[:;,(\[\]]|\s[-–—]{1,2}\s|$)")


def squash(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def matter_lookup(matter_names):
    """Matter names and codenames to recognize a "[Project Atlas]" header, from fuzzy.matter_name_table."""
    names = matter_names.copy()
    names["squashed"] = names["name"].map(squash)
    return names[names["squashed"].str.len() > 4].reset_index(drop=True)


def client_lookup(client_names, skip=()):
    """Names to look for: each client's name and aliases, from fuzzy.client_name_table."""
    names = client_names[~client_names["id"].isin(skip)].copy()
    names["squashed"] = names["name"].map(squash)
    return names[names["squashed"].str.len() > 1].reset_index(drop=True)


def find_client(text, lookup, min_score=85):
    """Best client name in the text: (client, start, end, score) or None.

    Compares runs of 1 to 5 words with spaces and punctuation removed, so "HalvorsenIndustrial",
    "Cobalt Ridge Partners" and "CobaltRidge Partners" all match. Short names (4 letters or fewer, like "JPC")
    must match exactly and be written in capitals, and names under 10 letters need a score of 90+.
    """
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[\w&'.]+", text)]
    squashed = lookup["squashed"].tolist()
    best = None
    for n in range(1, 6):
        for i in range(len(words) - n + 1):
            start, end = words[i][1], words[i + n - 1][2]
            window = text[start:end]
            query = squash(window)
            if len(query) < 2:
                continue
            hit = process.extractOne(query, squashed, scorer=fuzz.ratio, score_cutoff=min_score)
            if not hit:
                continue
            if len(hit[0]) <= 4 and not (query == hit[0] and window.strip(".").isupper()):
                continue
            if len(hit[0]) < 10 and hit[1] < 90:
                continue
            score = (hit[1], len(hit[0]))
            if best is None or score > best[0]:
                best = (score, lookup.loc[hit[2], "id"], start, end)
    return (best[1], best[2], best[3], best[0][0]) if best else None


def header_name(text):
    """A short name-like header at the start ("Quill & Anchor - ...", "[Acme] ...", "Acme: ..."), or None."""
    m = (re.match(r"^\s*\[([^\]]{2,60})\]", text)
         or re.match(r"^\s*(?:for\s+)?([A-Z][\w&.,' ]{1,50}?)\s*(?::|\s[-–—]{1,2}\s|\s/\s)", text))
    if not m:
        return None
    words = m.group(1).replace(",", " ").split()
    caps = sum(w[0].isupper() or w == "&" for w in words)
    if len(words) <= 5 and caps / len(words) >= 0.5:
        return m.group(1).strip(), m.start(1), m.end(1)
    return None


# ---------------- Putting it together ----------------

def tidy(text):
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = re.sub(r"(\s[-–—]{1,2})(\s+[-–—]{1,2})+(?=\s)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\-–—:;,|/.]+|[\s\-–—:;,|/]+$", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def cut(text, spans):
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]
    return text


def parse_line(line, lookup, context=None, matters=None):
    """One entry from one piece of text. context: client/matter/cmid from a header line above it."""
    fields = {"cmid": None, "client": None, "matter": None, "hours": None}
    line = TIME_LABEL.sub(" ", REQUEST.sub(" ", line))
    amounts = find_hours(line)
    fields["hours"] = total_hours(amounts)
    breakdown = len(amounts) > 1 and all(a[3] for a in amounts)
    spans = [] if breakdown else [(a[1], a[2]) for a in amounts]
    line = tidy(cut(line, spans))

    cmid = find_cmid(line)
    if cmid:
        fields["cmid"] = cmid[0]
        line = tidy(cut(line, [(cmid[1], cmid[2])]))

    line = re.sub(r"^(?:for|to|on|re:?)\s+", "", line, flags=re.I)
    client = find_client(line, lookup)
    if client:
        fields["client"] = line[client[1]:client[2]]
        rest = line[client[2]:]
        matter = SPLIT_MATTER.match(rest)
        if matter:
            fields["matter"] = matter.group(1).strip()
            rest = rest[matter.end():]
        line = line[:client[1]] + " " + rest
    else:
        header = header_name(line)
        is_matter = matters is not None and header and process.extractOne(
            squash(header[0]), matters["squashed"].tolist(), scorer=fuzz.ratio, score_cutoff=90)
        if header and is_matter:
            fields["matter"] = header[0]
            line = line[:header[1]] + " " + line[header[2]:]
        elif header:
            fields["client"] = header[0]
            rest = line[header[2]:]
            matter = SPLIT_MATTER.match(rest)
            if matter:
                fields["matter"] = matter.group(1).strip()
                rest = rest[matter.end():]
            line = line[:header[1]] + " " + rest

    for key, value in (context or {}).items():
        if fields.get(key) is None:
            fields[key] = value
    fields["narrative"] = tidy(re.sub(r"^(?:for|re:?)\s+", "", tidy(line), flags=re.I))
    return fields


def parse_email(text, lookup, matters=None):
    """All entries in one email: a list of dicts with cmid, client, matter, hours, narrative.

    Most emails hold one entry. When several lines each have their own hours, each line is its own entry,
    and lines without hours above them (a client header, "Client: ..." labels) carry over to the entries below.
    """
    if is_blank(text):
        return []
    lines, subject = strip_clutter(text)
    labels, kept = {}, []
    for ln in lines:
        m = LABELED.match(ln)
        key = m.group(1).lower().replace(" ", "") if m else None
        if key in ("client", "matter"):
            labels[key] = m.group(2).strip() or None
        elif key == "c/m":
            cmid = find_cmid(m.group(2))
            labels["cmid"] = cmid[0] if cmid else m.group(2).strip()
        else:
            kept.append(m.group(2) if m else ln)
    lines = kept
    if not lines:
        return [{"cmid": None, "client": None, "matter": None, "hours": None, "narrative": "", **labels}] if labels else []
    timed = [ln for ln in lines if find_hours(TIME_LABEL.sub(" ", ln))]
    if len(timed) < 2:
        entry = parse_line(" ".join(lines), lookup, labels, matters)
        if subject and entry["client"] is None:
            entry["client"] = parse_line(subject, lookup)["client"]
        return [entry]

    entries, context = [], dict(labels)
    for ln in lines:
        parsed = parse_line(re.sub(r"^\s*([-*•·]|\d+[.)])\s+", "", ln), lookup, context, matters)
        if parsed["hours"] is None:
            context.update({k: v for k, v in parsed.items() if k in ("cmid", "client", "matter") and v})
        else:
            entries.append(parsed)
    return entries


def parse_email_entries(entries, client_names, matter_names=None, skip=()):
    """Parse every EMAIL row of the entries table into its cmid, client, matter and narrative columns.

    Other formats are left as they are. Adds parsed_hours and n_parsed (entries found in the email).
    An email with several entries keeps the first one's fields and all the narratives, like a block-billed entry.
    """
    lookup = client_lookup(client_names, skip)
    matters = matter_lookup(matter_names) if matter_names is not None else None
    out = entries.copy()
    out["parsed_hours"], out["n_parsed"] = None, 0
    is_email = out["source_format"].str.upper() == "EMAIL"
    for idx in out.index[is_email]:
        found = parse_email(out.at[idx, "narrative"], lookup, matters)
        if not found:
            continue
        first = found[0]
        for col in ["cmid", "client", "matter"]:
            if is_blank(out.at[idx, col]) and first[col]:
                out.at[idx, col] = first[col]
        out.at[idx, "narrative"] = "; ".join(f["narrative"] for f in found if f["narrative"]) or out.at[idx, "narrative"]
        hours = [f["hours"] for f in found if f["hours"] is not None]
        out.at[idx, "parsed_hours"] = round(sum(hours), 2) if hours else None
        out.at[idx, "n_parsed"] = len(found)
    return out
