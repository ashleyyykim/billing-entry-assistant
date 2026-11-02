"""Free text into entry rows: loose handwritten notes (read by Textract), typed notes, emails, and voice transcripts.

None of these have columns, so each one goes through text_parser, which finds hours, CMIDs, clients, and matters
in plain text. Every entry it finds becomes one row.

Emails go to the parser whole, since it was built for them (reply chains, signatures, labeled fields). Notes are split
into entries first, line by line, because a small note wraps one entry over several lines:
- A line with hours starts an entry, and the lines below it without hours join it.
- A line that's only a date ("Tues 10/27") applies to the entries below it.
- A short line that's just a client name, or lines above the first entry, are headers for the entries below them.
- A line starting with "same" or a ditto mark gets the client of the entry above, flagged as copied.
Photographed notes also get a few fixes for small marks Textract often misreads ("05" for ".5"), flagged so a
reviewer checks the hours, and a confidence score for each entry from its own lines.
"""
import re

import pandas as pd

from text_parser import client_lookup, find_client, find_hours, matter_lookup, parse_text
from textract import page_lines

# "10/13" or "10/13/26", but not a CMID like "4590/003" or a fraction like "1/2 hr"
DATE_IN_TEXT = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?(?![\d/])(?!\s*(?:h|hr|hrs|hours?)\b)")
DITTO_START = re.compile(r"^\s*(same as above|same|ditto|\"|''|〃|↓)\s*[-–:,]?\s*", re.I)
WEEKDAY = re.compile(r"\b(mon|tues?|wed|thur?s?|fri|sat|sun)\b\.?", re.I)

# Small marks Textract often misreads in photos
SPACED_DECIMAL = re.compile(r"(?<![\w.])[,.]\s+(\d)(?![\d.])")        # ", 8" for ".8"
LEADING_ZERO = re.compile(r"(?<![\w.,/:])0([1-9])(?![\w.,/:])")      # "05" for ".5"
SLASH_AS_ONE = re.compile(r"^(\D*?)(\d{1,2})1(\d{2})\s*$")           # "10127" for "10/27"


def parser_lookups(client_names, matter_names, skip=("Firm",)):
    """The client and matter lookups the text parser needs, from matching's client and matter name tables."""
    return client_lookup(client_names, skip), matter_lookup(matter_names)


def find_date(text):
    """The first real date written in the text ("10/13", "10/13/26"), as written, or None.
    Checked against a leap year, so "2/29" counts and "9/31" doesn't."""
    for m in DATE_IN_TEXT.finditer(text):
        if pd.notna(pd.to_datetime(f"{m.group(1)}/{m.group(2)}/2024", format="%m/%d/%Y", errors="coerce")):
            return m.group()
    return None


def is_date_line(text):
    """True for a line that's only a date, maybe with a weekday ("Tues 10/27")."""
    date = find_date(text)
    return bool(date) and not WEEKDAY.sub(" ", text.replace(date, " ")).strip(" .,-:")


def fix_misreads(text):
    """The line with common photo misreads fixed: a decimal point read as a comma or dropped before a leading zero,
    and a date's slash read as a 1. Returns (text, whether the hours were changed)."""
    fixed = LEADING_ZERO.sub(r".\1", SPACED_DECIMAL.sub(r".\1", text))
    hours_changed = fixed != text
    slash = SLASH_AS_ONE.match(fixed)
    if slash and not find_date(fixed) and not WEEKDAY.sub("", slash.group(1)).strip():
        date = f"{slash.group(1)}{slash.group(2)}/{slash.group(3)}"
        fixed = date if find_date(date) else fixed
    return fixed, hours_changed


def is_header(text, lookup):
    """A short line that's just a client name ("HLP", "Cobalt Ridge Partners"), for the entries below it."""
    return len(text.split()) <= 4 and find_client(text, lookup) is not None


def group_lines(lines, lookup, fix=False):
    """A note's lines split into entries. lines: (text, confidence) pairs in reading order.

    Returns one dict per entry: its lines (headers first), lowest confidence, date written above it, and whether a
    misread was fixed in its hours.
    """
    headers, groups, date = [], [], None
    for text, conf in lines:
        text, changed = fix_misreads(text) if fix else (text, False)
        if is_date_line(text):
            date = find_date(text)
        elif find_hours(text):
            groups.append({"lines": headers + [text], "confs": [conf], "date": date, "fixed": changed})
        elif groups and not is_header(text, lookup):
            groups[-1]["lines"].append(text)
            groups[-1]["confs"].append(conf)
        else:
            headers = [text] if groups else headers + [text]
    if not groups and headers:
        groups.append({"lines": headers, "confs": [c for _, c in lines], "date": date, "fixed": False})
    return groups


def copy_down(rows):
    """A line that starts with "same" or a ditto mark, with no client or CMID of its own, gets the line above's."""
    for above, row in zip(rows, rows[1:]):
        ditto = DITTO_START.match(row["narrative"] or "")
        if ditto and not row["client"] and not row["cmid"]:
            row.update(client=above["client"], cmid=above["cmid"], matter=above["matter"],
                       narrative=row["narrative"][ditto.end():])
            row["reasons"].append("client_copied")
    return rows


def parse_group(text, lookup, matters, default_date=None):
    """Entries in one piece of text, each with the date written in it (or default_date)."""
    date = find_date(text)
    if date:
        text = text.replace(date, " ", 1)
    return [{**f, "work_date": date or default_date, "reasons": []} for f in parse_text(text, lookup, matters)]


def rows_from_lines(lines, lookup, matters, input_type, source=None, upload_date=None, fix=False, low_conf=80):
    """One row per entry in a note, from its lines ((text, confidence) pairs)."""
    rows = []
    for group in group_lines(lines, lookup, fix):
        confs = [c for c in group["confs"] if c is not None and pd.notna(c)]
        # one line per entry, so the parser doesn't mistake a wrapped line for an email signature
        for row in parse_group(" ".join(group["lines"]), lookup, matters, group["date"] or upload_date):
            row["min_conf"] = min(confs) if confs else None
            if confs and min(confs) < low_conf:
                row["reasons"].append("low_conf")
            if group["fixed"]:
                row["reasons"].append("hours_10x")
            rows.append(row)
    return finish(copy_down(rows), input_type, source)


def finish(rows, input_type, source):
    for row in rows:
        row.update(input_type=input_type, source=source, check_reasons="; ".join(row.pop("reasons")))
    return rows


def text_rows(text, lookup, matters, input_type, source=None, upload_date=None):
    """One row per entry in an email, typed note, or transcript. Emails are parsed whole; the rest line by line."""
    if input_type == "EMAIL":
        return finish(copy_down(parse_group(text, lookup, matters, upload_date)), input_type, source)
    lines = [(line, None) for line in text.splitlines() if line.strip()]
    return rows_from_lines(lines, lookup, matters, input_type, source, upload_date)


def note_rows(pages, lookup, matters, source=None, upload_date=None):
    """One row per entry on a photographed note. Printed text on the paper (a notebook's "Date" label) is skipped."""
    lines = pd.concat([page_lines(p) for p in pages], ignore_index=True)
    if lines["handwritten"].any():
        lines = lines[lines["handwritten"]]
    return rows_from_lines(list(zip(lines["text"], lines["conf"])), lookup, matters, "NOTE", source, upload_date,
                           fix=True)
