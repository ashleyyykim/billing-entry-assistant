"""Free text into entry rows: loose handwritten notes (read by Textract), typed notes, emails, and voice transcripts.

None of these have columns, so each one goes through text_parser, which finds hours, CMIDs, clients, and matters
in plain text. Every entry it finds becomes one row.

Emails go to the parser whole, since it was built for them (reply chains, signatures, labeled fields). Notes are split
into entries first, line by line, because a small note wraps one entry over several lines:
- A line with hours starts an entry, and the lines below it without hours join it.
- A line that's only a date ("Tues 10/27"), or a date at the start of an entry, applies to the entries below it.
- A short line that's just a client name, or lines above the first entry, are headers for the entries below them,
  except entries after the first that name their own client or CMID.
- A line without hours that names a different client or CMID than the entry above starts a new entry (one whose
  hours weren't read), and hours on their own line just below it fill it in.
- A line starting with "same" or a ditto mark gets the client of the entry above, flagged as copied.
Photos are first put back in reading order from where each line sits on the page: pieces at the same height are one
line, read left to right ("same" and ".5 board minutes" written apart), a piece read twice is dropped, and lines go
top to bottom.
Photographed notes (not screenshots) also get a few fixes for small marks Textract often misreads ("05" for ".5"),
flagged so a reviewer checks the hours, and a confidence score for each entry from its own lines.
"""
import re

import pandas as pd

from text_parser import (HEADER_LINE, SUBJECT_LINE, client_lookup, cut, find_client, find_cmid, find_hours,
                         matter_lookup, parse_text, strip_clutter)
from intake import to_date
from textract import page_lines

# "10/13" or "10/13/26", but not a CMID like "4590/003" or a fraction like "1/2 hr"
DATE_IN_TEXT = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?(?![\d/])(?!\s*(?:h|hr|hrs|hours?)\b)")
# "11-3" or "11-3-26" at the start of an entry, maybe after a weekday ("Tues 11-3"). Only at the start, since "3-5"
# elsewhere is usually a range (pages, days, hours)
DASH_DATE = re.compile(r"^\s*(?:(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[a-z]*\.?,?\s+)?"
                       r"((\d{1,2})-(\d{1,2})(?:-(?:\d{4}|\d{2}))?)(?![\d/:.-])(?!\s*(?:h|hr|hrs|hours?|days?|"
                       r"weeks?|months?|min\w*)\b)", re.I)
DITTO_START = re.compile(r"^\s*(same as above|same|ditto|\"|''|〃|↓|-+>|=>|→|>)\s*[-–:,]?\s*", re.I)
FIRM_TIME = re.compile(r"^\s*firm\b", re.I)   # "Firm admin": no client on purpose, matching reads it
WEEKDAY = re.compile(r"\b(mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|"
                     r"sat(?:urday)?|sun(?:day)?)\b\.?", re.I)
WEEKDAY_NUMBER = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

# Small marks Textract often misreads in photos
SPACED_DECIMAL = re.compile(r"(?<![\w.])[,.]\s+(\d)(?![\d.])")        # ", 8" for ".8"
LEADING_ZERO = re.compile(r"(?<![\w.,/:])0([1-9])(?![\w.,/:])")      # "05" for ".5"
SLASH_AS_ONE = re.compile(r"^(\D*?)(\d{1,2})1(\d{2})\s*$")           # "10127" for "10/27"

# Several entries written as one paragraph, hours said in words, and CMIDs typed wrong
BREAK = re.compile(r";|,|\.(?=\s)|\s(?:and|then|also|plus)\s|\s\d+[.)]\s", re.I)   # where one entry can end
CONNECTOR = re.compile(r"^\s*(and|then|also|plus)\s+", re.I)
CLOCK = re.compile(r"\d{1,2}:\d{2}")
MAX_RANGE = 12            # hours; a longer time range is left alone
CMID_TOKEN = re.compile(r"(?<![\w.])[A-Za-z0-9]{4}[.\-/][A-Za-z0-9]{1,3}(?![\w.])")
CMID_SHAPE = re.compile(r"\d{4}\.\d{1,3}")
LOOKALIKES = "ODQoIliZzSsbGBgq"
OCR_DIGITS = str.maketrans(LOOKALIKES, "0000111225566899")
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
NUMBER_WORDS.update({"a": 1, "an": 1, "oh": 0})
NUMBER = "|".join(NUMBER_WORDS)
POINT = re.compile(rf"\bpoint\s+({NUMBER}|\d)\b", re.I)
WHOLE = "|".join(w for w in NUMBER_WORDS if w not in ("a", "an"))
SPOKEN_DECIMAL = re.compile(rf"\b({WHOLE}|\d{{1,2}})\s+point\s+({NUMBER}|\d)\b", re.I)   # "one point two"
LEADING_BREAK = re.compile(r"^(?:[\s;,]|\.(?=\s))+")
QUARTER_HOUR = re.compile(r"\bquarter\s+(?:of\s+an?\s+)?hours?\b", re.I)
FRACTION_HOURS = re.compile(r"(?<![\w./])([1-9])/([2-9])\s*(?:hours?|hrs?)\b", re.I)
COMMA_CMID = re.compile(r"(?<![\w.,])(\d{4}),(\d{3})(?![\w.,])")
HALF_HOUR = re.compile(r"\bhalf\s+(?:an?\s+)?hours?\b|\ba\s+half[- ]hour\b", re.I)
AND_A_HALF = re.compile(rf"\b({NUMBER})\s+(?:hours?\s+)?and\s+a\s+half(?:\s+hours?)?\b", re.I)
WORD_HOURS = re.compile(rf"\b({NUMBER})\s+hours?\b", re.I)

# The header block of a screenshotted email, and where a line wraps
ADDRESS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
STAMP = re.compile(rf"\b{MONTH}\.?\s+\d{{1,2}}\b.*\d{{1,2}}:\d{{2}}\s?(am|pm)\b", re.I)   # "Oct 28, 6:05 PM"
ENDS_IN_HOURS = re.compile(r"(\d|hrs?|hours?|mins?)\.?\s*$", re.I)   # a finished entry, so the next line is new
HEADER_SPAN = 8    # the header is in the first lines of the screenshot
WRAP_EDGE = 0.75   # a line ending past this share of the page width ran to the margin, so the next line continues it


def parser_lookups(client_names, matter_names, skip=("Firm",)):
    """The client and matter lookups the text parser needs, from matching's client and matter name tables."""
    return client_lookup(client_names, skip), matter_lookup(matter_names)


def find_date(text):
    """The first real date written in the text ("10/13", "10/13/26", or "11-3" at the start), as written, or None.
    Checked against a leap year, so "2/29" counts and "9/31" doesn't."""
    for m in DATE_IN_TEXT.finditer(text):
        if pd.notna(pd.to_datetime(f"{m.group(1)}/{m.group(2)}/2024", format="%m/%d/%Y", errors="coerce")):
            return m.group()
    dash = DASH_DATE.match(text)
    if dash and pd.notna(pd.to_datetime(f"{dash.group(2)}/{dash.group(3)}/2024", format="%m/%d/%Y", errors="coerce")):
        return dash.group(1)
    return None


def is_date_line(text):
    """True for a line that's only a date, maybe with a weekday ("Tues 10/27")."""
    date = find_date(text)
    return bool(date) and not WEEKDAY.sub(" ", text.replace(date, " ")).strip(" .,-:")


def weekday_mismatch(text, year):
    """True when the text names a weekday and a date that falls on another day ("Tues 10/21" in 2026, a Wednesday),
    which usually means the date was misread."""
    day, date = WEEKDAY.search(text), find_date(text)
    if not (day and date):
        return False
    when = to_date(date, year)
    return pd.notna(when) and when.weekday() != WEEKDAY_NUMBER[day.group(1).lower()[:3]]


def page_amounts(texts):
    """Every amount of hours written on a page, read the way entries read them (", 8" is .8), leaving out dates and
    CMIDs. A bare whole number ("4") isn't counted, since it may be a lost decimal point or not hours at all."""
    amounts = []
    for text in texts:
        text = fix_misreads(text)[0]
        date = find_date(text)
        text = CMID_TOKEN.sub(" ", text.replace(date, " ") if date else text)
        amounts += [hours for hours, *_ in find_hours(spoken_hours(text))]
    return amounts


def unused_hours(amounts, hours):
    """The amounts on a page that no entry's hours account for, each entry used once."""
    left = [h for h in hours if h is not None and pd.notna(h)]
    unused = []
    for amount in amounts:
        match = next((h for h in left if abs(h - amount) < 0.01), None)
        if match is None:
            unused.append(amount)
        else:
            left.remove(match)
    return unused


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


def spoken_hours(text):
    """The text with hours said in words or as a fraction written as numbers: "point four" to ".4", "one point two"
    to "1.2", "half an hour" to ".5 hrs", "quarter of an hour" to ".25 hrs", "an hour and a half" to "1.5 hrs",
    "two hours" to "2 hrs", "1/2 hr" to ".5 hrs"."""
    number = lambda word: NUMBER_WORDS.get(word.lower(), word)
    text = SPOKEN_DECIMAL.sub(lambda m: f"{number(m.group(1))}.{number(m.group(2))}", text)
    text = POINT.sub(lambda m: f".{number(m.group(1))}", text)
    text = HALF_HOUR.sub(" .5 hrs ", text)
    text = QUARTER_HOUR.sub(" .25 hrs ", text)
    text = FRACTION_HOURS.sub(lambda m: f" {int(m.group(1)) / int(m.group(2))} hrs ", text)
    text = AND_A_HALF.sub(lambda m: f" {number(m.group(1)) + 0.5} hrs ", text)
    return WORD_HOURS.sub(lambda m: f" {number(m.group(1))} hrs ", text)


def fix_cmid_letters(text):
    """The text with a CMID typed with letters for digits ("S520.OO1") or a comma ("5520,001") written as
    "5520.001", and whether anything was fixed."""
    def fix(match):
        word = match.group()
        letters = [c for c in word if c.isalpha()]
        digits = sum(c.isdigit() for c in word)
        return word.translate(OCR_DIGITS) if letters and digits >= 3 and all(c in LOOKALIKES for c in letters) else word
    fixed = COMMA_CMID.sub(r"\1.\2", CMID_TOKEN.sub(fix, text))
    return fixed, fixed != text


def drop_clock_times(text):
    """The text without clock times ("2:30") on lines that also hold another amount of hours, and whether any were
    dropped. A lone "1:15" is still hours."""
    lines, dropped = [], False
    for line in text.split("\n"):
        amounts = find_hours(line)
        clocks = [a for a in amounts if CLOCK.fullmatch(line[a[1]:a[2]].strip())]
        if clocks and len(clocks) < len(amounts):
            line, dropped = cut(line, [(a[1], a[2]) for a in clocks]), True
        lines.append(line)
    return "\n".join(lines), dropped


def find_clients(text, lookup):
    """Each client named in the text as (client id, start), by position."""
    found = []
    while hit := find_client(text, lookup):
        found.append((hit[0], hit[1]))
        text = text[:hit[1]] + " " * (hit[2] - hit[1]) + text[hit[2]:]
    return sorted(found, key=lambda f: f[1])


def one_entry(line, amounts):
    """True when several amounts in a line still mean one entry: a total with its parts, all in parentheses."""
    values = [a[0] for a in amounts]
    has_total = len(values) >= 3 and any(abs(v - (sum(values) - v)) < 0.05 for v in values)
    return has_total or all(a[3] for a in amounts) or bool(re.search(r"\btotal\b", line, re.I))


def split_amounts(line, lookup):
    """A line holding several entries as one piece for each amount of hours, or the line whole when it can't be cut.

    Hours either follow the words ("call with Halvorsen .4, memo for Tidewater .3") or lead them ("spent .4 on a call
    with Halvorsen, then .3 on Tidewater"), told apart by whether the first client comes before the first amount.
    Pieces are cut at the first break after an amount (words first) or at the last break before one (hours first).
    A line that starts with a client, the way people dictate ("Halvorsen, .4, call with lender counsel. Then
    Ridgewater, 1.2, revised the LPA"), is cut at the last break before each next client instead.
    """
    amounts = find_hours(line)
    if len(amounts) < 2 or one_entry(line, amounts):
        return [line]
    clients = find_clients(line, lookup)
    words_first = not clients or clients[0][1] < amounts[0][1]
    client_first = words_first and clients and not re.search(r"[A-Za-z]{2,}", line[:clients[0][1]])
    cuts = []
    for before, after in zip(amounts, amounts[1:]):
        breaks = list(BREAK.finditer(line[before[2]:after[1]]))
        between = [c for _, c in clients if before[2] <= c < after[1]]
        if client_first and between:
            ahead = list(BREAK.finditer(line[before[2]:between[0]]))
            cuts.append(before[2] + ahead[-1].start() if ahead else between[0])
        elif words_first:
            cuts.append(before[2] + (breaks[0].start() if breaks else 0))
        elif breaks:
            cuts.append(before[2] + breaks[-1].start())
        else:
            return [line]
    pieces = [line[a:b] for a, b in zip([0] + cuts, cuts + [len(line)])]
    return [CONNECTOR.sub("", LEADING_BREAK.sub("", piece)) for piece in pieces]


def is_unsure(line, lookup):
    """True when a line may hold more than one entry: several amounts of hours, or several different clients."""
    amounts = find_hours(line)
    several = len(amounts) > 1 and not one_entry(line, amounts)
    return several or len({client for client, _ in find_clients(line, lookup)}) > 1


def is_header(text, lookup):
    """A short line that's just a client name ("HLP", "Cobalt Ridge Partners"), for the entries below it."""
    return len(text.split()) <= 4 and find_client(text, lookup) is not None


SAME_ROW = 0.5   # pieces whose heights overlap by half of the shorter one sit on the same line


def page_order(lines, same_row=SAME_ROW):
    """A page's lines put back in reading order from where they sit, as one row per written line: text, conf
    (lowest piece), and box. Textract can split a line into pieces ("same" and ".5 board minutes" written apart),
    read a piece twice (". 9" and "9"), or list a slanted line's end after the line below it."""
    rows = []
    for line in lines.sort_values("top").to_dict("records"):
        line["bottom"] = line["top"] + line["height"]
        line["right"] = line["left"] + line["width"]
        for row in rows:
            overlap = min(row["bottom"], line["bottom"]) - max(row["top"], line["top"])
            if overlap < same_row * min(row["bottom"] - row["top"], line["height"]):
                continue
            pieces = row["pieces"]
            if any(p["left"] <= line["right"] and line["left"] <= p["right"] for p in pieces):
                inside = [p for p in pieces if p["left"] <= line["left"] and line["right"] <= p["right"]]
                if inside:
                    break   # read twice: the narrower copy is dropped
                continue
            pieces.append(line)
            row.update(top=min(row["top"], line["top"]), bottom=max(row["bottom"], line["bottom"]))
            break
        else:
            rows.append({"pieces": [line], "top": line["top"], "bottom": line["bottom"]})
    out = []
    for row in sorted(rows, key=lambda r: r["top"]):
        pieces = sorted(row["pieces"], key=lambda p: p["left"])
        confs = [p["conf"] for p in pieces if pd.notna(p["conf"])]
        numbers = [p["number_conf"] for p in pieces if pd.notna(p.get("number_conf"))]
        out.append({"text": " ".join(p["text"] for p in pieces), "conf": min(confs) if confs else None,
                    "number_conf": min(numbers) if numbers else None, "left": pieces[0]["left"], "top": row["top"],
                    "height": row["bottom"] - row["top"]})
    return pd.DataFrame(out, columns=["text", "conf", "number_conf", "left", "top", "height"])


def names_own(text, lookup):
    """The client or CMID the text names, if any (client first)."""
    client = find_client(text, lookup)
    if client:
        return client[0]
    cmid = find_cmid(text)
    return cmid[0] if cmid else None


def group_lines(lines, lookup, fix=False, split_on_names=True):
    """A note's lines split into entries. lines: (text, confidence) pairs in reading order.

    Returns one dict per entry: its lines (headers first), lowest confidence, date written above it, and whether a
    misread was fixed in its hours.
    """
    headers, groups, date, used = [], [], None, False
    for text, conf in lines:
        text, changed = fix_misreads(text) if fix else (text, False)
        last = groups[-1] if groups else None
        if is_date_line(text):
            date = find_date(text)
        elif find_hours(spoken_hours(text)):   # "Northgate two hours" starts an entry, not a header
            date = find_date(text) or date   # "11/2 Ridgewater .8 ..." dates the entries below it too
            own = names_own(text, lookup)
            if last and not last["hours"] and not own and not DITTO_START.match(text):
                # hours on their own line under an entry that had none
                last.update(lines=last["lines"] + [text], confs=last["confs"] + [conf], hours=True,
                            fixed=last["fixed"] or changed)
                continue
            # a header written above one entry ("2201.004" over "Harborlight 1.0") isn't the next client's too, and
            # "same" takes the entry above, not the header
            top = [] if (used and own) or DITTO_START.match(text) else headers
            groups.append({"lines": top + [text], "confs": [conf], "date": date, "fixed": changed, "hours": True})
            used = True
        elif groups and not is_header(text, lookup):
            own = names_own(text, lookup) if split_on_names else None
            above = names_own(" ".join(last["lines"]), lookup) if own else None
            if own and above and own != above:
                # a different client or CMID under an entry that has one: an entry whose hours weren't read
                groups.append({"lines": [text], "confs": [conf], "date": date, "fixed": False, "hours": False})
            else:
                last["lines"].append(text)
                last["confs"].append(conf)
        else:
            headers = [text] if groups else headers + [text]
            used = False   # a new header is for the entries below it
    if not groups and headers:
        groups.append({"lines": headers, "confs": [c for _, c in lines], "date": date, "fixed": False})
    return groups


def copy_down(rows):
    """An entry with no client, matter, or CMID of its own gets the one above's, flagged as copied. It starts with
    "same", a ditto mark, or an arrow, or just has nothing written. Firm time ("Firm admin") is left blank."""
    for above, row in zip(rows, rows[1:]):
        ditto = DITTO_START.match(row["narrative"] or "")
        own = row["client"] or row["cmid"] or row["matter"] or FIRM_TIME.match(row["narrative"] or "")
        blank = not own
        if blank and (ditto or above["client"] or above["cmid"]):
            row.update(client=above["client"], cmid=above["cmid"], matter=above["matter"])
            if ditto:
                row["narrative"] = row["narrative"][ditto.end():]
            row["reasons"].append("client_copied")
    return rows


MONTH_NUMBER = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
                "nov": 11, "dec": 12}
MONTH_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
                        r"(?:,?\s+(\d{4}))?\b", re.I)
TIME_RANGE = re.compile(r"(?<![\d/.:$])(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?m?\.?\s*(?:-|\u2013|to)\s*"
                        r"(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?m?\.?(?![\d/:])", re.I)
BILLING_CODE = re.compile(r"\b[LABE]\d{3}\b")
MONEY = re.compile(r"\$\s?[\d,]+(?:\.\d\d)?")
MAIL_HEADER = re.compile(r"^\s*(>|on\s.+\bwrote:|(date|sent|from|to|cc|subject):)", re.I)   # dates here aren't work dates
PHONE = re.compile(r"(?<![\d.])(?:\(?\d{3}\)?[\s.-])?\d{3}[-.]\d{4}\b")   # "555-0100", "(415) 555-0100"
TWO_DECIMALS = re.compile(r"(?<![\w$,.\-/])(\d{0,2}\.\d\d)(?![\w.%/\-]|\d|\s*(?:hrs?|hours?)\b)", re.I)


def month_dates(text):
    """Dates written with a month name as numbers ("Monday, October 26" to "Monday, 10/26"). A lowercase "may" is
    left alone, since it's usually the verb."""
    def number(m):
        if m.group(1).lower() == "may" and not m.group(1)[0].isupper():
            return m.group()
        day = int(m.group(2))
        if not 1 <= day <= 31:
            return m.group()
        date = f"{MONTH_NUMBER[m.group(1)[:3].lower()]}/{day}"
        return f"{date}/{m.group(3)}" if m.group(3) else date
    return MONTH_DATE.sub(number, text)


def range_hours(m):
    """Hours between the two clock times of a time range ("9:30-10:15am" is 0.75), or None if it isn't one: one side
    needs minutes or am/pm, and the span has to be more than 0 and at most MAX_RANGE hours."""
    h1, m1, p1, h2, m2, p2 = m.groups()
    if not (m1 or m2 or p1 or p2):
        return None
    start, end = int(h1) * 60 + int(m1 or 0), int(h2) * 60 + int(m2 or 0)
    if (p1 or "").lower() == "p" and int(h1) < 12 or (not p1 and (p2 or "").lower() == "p" and int(h1) < 12
                                                       and start + 720 < end + (720 if int(h2) < 12 else 0)):
        start += 720
    if (p2 or "").lower() == "p" and int(h2) < 12:
        end += 720
    while end <= start and end < start + 1440:
        end += 720
    hours = (end - start) / 60
    return hours if 0 < hours <= MAX_RANGE else None


def typed_fixes(text, two_decimals=True):
    """Typed time written in ways the parser doesn't read, rewritten line by line (except email header and quoted
    lines, whose dates are when a message was sent, not when the work was done):
    - month-name dates as numbers ("October 26" as "10/26")
    - task codes ("L110", "A103"), dollar amounts, and phone numbers dropped
    - a time range as its hours ("9:30-10:15" as "0.75 hrs"), unless its clause (the line up to a ";") also gives
      hours, in which case the range is only clock times and is dropped
    - two-decimal hours ("0.50") given a unit, so they're read. two_decimals is off for photos, where a stray mark
      can read as ".42"."""
    lines = []
    for line in text.split("\n"):
        if MAIL_HEADER.match(line):
            lines.append(line)
            continue
        line = PHONE.sub(" ", MONEY.sub(" ", BILLING_CODE.sub(" ", month_dates(line))))
        clauses = []
        for clause in line.split(";"):
            for m in reversed(list(TIME_RANGE.finditer(clause))):
                hours = range_hours(m)
                if hours is None:
                    continue
                rest = clause[:m.start()] + " " + clause[m.end():]
                clause = rest if find_hours(spoken_hours(rest)) else \
                    f"{clause[:m.start()]} {hours:.2f} hrs {clause[m.end():]}"
            clauses.append(clause)
        line = ";".join(clauses)
        if two_decimals:
            line = TWO_DECIMALS.sub(r"\1 hrs", line)
        lines.append(line)
    return "\n".join(lines)


def parse_group(text, lookup, matters, default_date=None):
    """Entries in one piece of text, each with the date written in it (or default_date).

    Hours said in words are written as numbers, letters in a CMID are fixed, a clock time next to another amount is
    dropped, and a paragraph holding several entries is cut into one line for each. Fixes are flagged, and so is
    anything that may still hold more than one entry.
    """
    date = find_date(text)
    if date:
        text = text.replace(date, " ", 1)
    text, fixed = fix_cmid_letters(spoken_hours(text))
    text, clock = drop_clock_times(text)
    lines = [line if line.lstrip().startswith(">") else "\n".join(split_amounts(line, lookup))
             for line in text.split("\n")]
    text = "\n".join(lines)
    unsure = any(is_unsure(line, lookup) for line in strip_clutter(text)[0])
    reasons = [r for r, found in [("cmid_fixed", fixed), ("clock_time", clock), ("multiple_entries", unsure)] if found]
    entries = parse_text(text, lookup, matters)
    return [{**f, "work_date": date or default_date, "reasons": list(reasons)} for f in entries]


def rows_from_lines(lines, lookup, matters, input_type, source=None, upload_date=None, fix=False, low_conf=80,
                    split_on_names=True):
    """One row per entry in a note, from its lines ((text, confidence) pairs)."""
    year = pd.Timestamp(upload_date or "today").year
    wrong_days = {find_date(text) for text, _ in lines if weekday_mismatch(text, year)}
    rows = []
    for group in group_lines(lines, lookup, fix, split_on_names):
        confs = [c for c in group["confs"] if c is not None and pd.notna(c)]
        # one line per entry, so the parser doesn't mistake a wrapped line for an email signature
        for row in parse_group(" ".join(group["lines"]), lookup, matters, group["date"] or upload_date):
            row["min_conf"] = min(confs) if confs else None
            if confs and min(confs) < low_conf:
                row["reasons"].append("low_conf")
            if group["fixed"]:
                row["reasons"].append("hours_10x")
            if row["work_date"] in wrong_days:
                row["reasons"].append("date_weekday")
            rows.append(row)
    return finish(copy_down(rows), input_type, source)


def finish(rows, input_type, source):
    for row in rows:
        if row["cmid"] and not CMID_SHAPE.fullmatch(row["cmid"]):
            row["reasons"].append("cmid_shape")
        row.update(input_type=input_type, source=source, check_reasons="; ".join(row.pop("reasons")))
    return rows


def text_rows(text, lookup, matters, input_type, source=None, upload_date=None):
    """One row per entry in an email, typed note, or transcript. Emails are parsed whole; the rest line by line.
    Typed habits the parser doesn't read are rewritten first (typed_fixes).
    A forwarded email's note above the forward and the forwarded message are parsed apart, and only the ones with
    hours in them, so "Adding to the queue." above an attorney's forwarded time isn't an entry."""
    text = typed_fixes(text)
    if input_type == "EMAIL":
        pieces = [piece for piece in split_forward(text) if piece.strip()]
        timed = [piece for piece in pieces if find_hours(spoken_hours(piece))]
        rows = [row for piece in (timed or pieces) for row in parse_group(piece, lookup, matters, upload_date)]
        return finish(copy_down(rows), input_type, source)
    lines = [(line, None) for line in text.splitlines() if line.strip()]
    return rows_from_lines(lines, lookup, matters, input_type, source, upload_date)


FORWARD_MARK = re.compile(r"^\s*(-{3,}\s*(forwarded message|original message)\s*-{3,}|begin forwarded message:|"
                          r"_{10,}\s*$(?=\s*^\s*From:))", re.I | re.M)   # Gmail, Apple Mail, and Outlook forwards
FROM_LINE = re.compile(r"^\s*From:\s*(.+?)\s*$", re.I | re.M)


FORWARD_HEADER = re.compile(r"^\s*(From|Sent|Date|To|Cc):", re.I)


def split_forward(text):
    """A forwarded email as [the note above the forward, the forwarded message], with the forwarded message's
    From/Date/To lines dropped (its Subject is kept, since it can name the client). Not forwarded: [text]."""
    mark = FORWARD_MARK.search(text)
    if not mark:
        return [text]
    rest = text[mark.end():].splitlines()
    start = 0
    while start < len(rest) and (not rest[start].strip() or FORWARD_HEADER.match(rest[start])):
        start += 1
    return [text[:mark.start()], "\n".join(rest[start:])]


def original_sender(text):
    """Who wrote a forwarded email: the first "From:" line after a forward marker, or None if it wasn't forwarded."""
    mark = FORWARD_MARK.search(text)
    sender = FROM_LINE.search(text, mark.end()) if mark else None
    return sender.group(1) if sender else None


def is_email_header(text):
    """A line of the header block above an email: a From/Sent/To/Subject label, an address, or a dated timestamp."""
    return bool(HEADER_LINE.match(text) or SUBJECT_LINE.match(text) or ADDRESS.search(text) or STAMP.search(text))


def screenshot_rows(pages, lookup, matters, source=None, upload_date=None, low_conf=80):
    """One row per entry in a screenshot of a typed email.

    The header block at the top (sender, date, recipients, subject) is cut off, a line that runs to the right margin
    is joined to the next one (a wrapped line), and the rest is parsed like an email. The subject names the client
    when the body doesn't.
    """
    lines = pd.concat([page_lines(p) for p in pages], ignore_index=True)
    is_header = lines["text"].head(HEADER_SPAN).map(is_email_header)
    cut = is_header[is_header].index.max() + 1 if is_header.any() else 0
    head, body = lines.iloc[:cut], lines.iloc[cut:]
    subject = next((m.group(1) for t in head["text"] if (m := SUBJECT_LINE.match(t))), None)
    if subject is None and cut and not is_email_header(head["text"].iloc[0]):
        subject = head["text"].iloc[0]   # the title above the sender, as Gmail shows it

    text = []
    for line, right in zip(body["text"], body["left"] + body["width"]):
        if text and text[-1][1] >= WRAP_EDGE and not ENDS_IN_HOURS.search(text[-1][0]):
            text[-1] = (text[-1][0] + " " + line, right)
        else:
            text.append((line, right))
    email = "\n".join(t for t, _ in text) + (f"\n\nSubject: {subject}" if subject else "")

    rows = finish(copy_down(parse_group(email, lookup, matters, upload_date)), "SCREENSHOT", source)
    conf = body["conf"].min()
    for row in rows:
        row["min_conf"] = conf
        if pd.notna(conf) and conf < low_conf:
            row["check_reasons"] = "; ".join(filter(None, [row["check_reasons"], "low_conf"]))
    return rows


def note_rows(pages, lookup, matters, source=None, upload_date=None, layout=True, split_on_names=True,
              confidence="line", low_conf=80):
    """One row per entry on a photographed note. Printed text on the paper (a notebook's "Date" label) is skipped.

    layout: put lines back in reading order from where they sit on the page (page_order). False keeps Textract's
    order. layout and split_on_names are on by default.
    confidence: which words decide low_conf. "line" is every word on the entry's lines; "numbers" is only the words
    with a digit (hours, dates, CMIDs), since a misread client or description goes to matching anyway.
    low_conf: the cutoff (0 to 100) below which an entry is flagged low_conf.
    """
    per_page = [page_lines(p, drop_repeats=layout) for p in pages]
    handwritten = pd.concat(per_page)["handwritten"].any()
    per_page = [page[page["handwritten"]] if handwritten else page for page in per_page]
    lines = pd.concat([page_order(page) if layout else page for page in per_page], ignore_index=True)
    conf = lines["number_conf"] if confidence == "numbers" else lines["conf"]
    rows = rows_from_lines(list(zip(lines["text"], conf)), lookup, matters, "NOTE", source, upload_date,
                           fix=True, low_conf=low_conf, split_on_names=split_on_names)
    if unused_hours(page_amounts(lines["text"]), [row["hours"] for row in rows]):
        for row in rows:
            row["check_reasons"] = "; ".join(filter(None, [row["check_reasons"], "unused_hours"]))
    return rows
