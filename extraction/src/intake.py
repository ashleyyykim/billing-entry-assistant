"""The shared entry table that every input type fills before matching.

Each converter (sheets.py, notes.py) returns a list of rows, one dict per entry. make_entries turns them into one
table: the columns matching's decide() needs, plus where each entry came from and why it may need a check.
add_day_check reruns the daily hours check on a combined table (several uploads from the same day), and
for_matching keeps only the columns decide() takes.
"""
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from text_parser import is_blank

MATCH_COLS = ["entry_id", "tk_id", "source_format", "work_date", "cmid", "client", "matter", "narrative", "hours"]
SOURCE_COLS = ["input_type", "tk_name", "source", "page", "rows", "hours_read", "min_conf"]
CHECK_COLS = ["check", "check_reasons"]
ENTRY_COLS = MATCH_COLS + SOURCE_COLS + CHECK_COLS

# Matching only knows these four formats (TYPED is its baseline), so every input type maps to one of them
FORMAT_FOR_INPUT = {
    "SHEET": "HANDWRITTEN",    # handwritten time sheet with columns
    "NOTE": "HANDWRITTEN",     # loose handwritten note: sticky note, napkin, notepad
    "PDF": "PDF",              # printed or typed time report
    "SPREADSHEET": "TYPED",    # Excel or CSV export
    "EMAIL": "EMAIL",
    "TEXT": "EMAIL",           # typed note or text message, parsed like an email
    "VOICE": "EMAIL",          # transcribed voice memo, parsed like an email
}

CHECK_REASONS = {
    "unknown_tk": "Timekeeper not found",
    "no_hours": "No hours found",
    "no_date": "No work date found",
    "day_over_max": "Over 14 hours on one day",
    "low_conf": "Hard to read",
    "hours_10x": "Hours may be missing a decimal",
    "client_copied": "Client copied from the line above",
    "date_guessed": "Date taken from the week, not the row",
}
DAY_MAX = 14


def timekeeper_table(timekeepers):
    """The firm's timekeeper list, ready for find_timekeeper.

    timekeepers: columns "TK ID", "Timekeeper" ("R. Patel"), and optionally "Email".
    """
    table = pd.DataFrame({"tk_id": timekeepers["TK ID"], "name": timekeepers["Timekeeper"]})
    table["email"] = timekeepers["Email"].str.lower() if "Email" in timekeepers else None
    parts = table["name"].map(name_parts)
    table["initial"], table["last"] = parts.str[0], parts.str[1]
    return table


def name_parts(name):
    """(first initial, last name) from "R. Patel", "Raj Patel", "Patel, R.", or "Patel"."""
    name = str(name).strip()
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    words = re.findall(r"[a-z]+", name.lower())
    if not words:
        return None, None
    initial = words[0][0] if len(words) > 1 else None
    return initial, words[-1]


def find_timekeeper(text, timekeepers, min_score=80):
    """TK ID for a name or email as written, or None.

    An email address is looked up in the Email column. A name is matched on its last name, allowing OCR typos
    ("Pate1"), and a first initial, when given, has to agree. None when no one, or more than one person, fits.
    """
    if is_blank(text):
        return None
    text = str(text).strip()
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    if email:
        hit = timekeepers.loc[timekeepers["email"] == email.group().lower(), "tk_id"]
        if len(hit) == 1:
            return hit.iloc[0]
        text = text[:email.start()].strip(' <"')
    initial, last = name_parts(text)
    if last is None:
        return None
    scores = timekeepers["last"].map(lambda x: fuzz.ratio(last, x))
    if initial:
        scores = scores.where(timekeepers["initial"] == initial, 0)
    best = timekeepers[(scores >= min_score) & (scores == scores.max())]
    return best["tk_id"].iloc[0] if len(best) == 1 else None


def to_date(value, year):
    """A work date from "08/11", "08/11/2026", "2026-08-11", or a date. A date with no year gets the given year."""
    if is_blank(value):
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}/\d{1,2}", text):
        text = f"{text}/{year}"
    return pd.to_datetime(text, errors="coerce")


def entry_ids(entries):
    """IDs that also point back to the source: "2399_001-p1-r4" for page 1, row 4, or "note_0412-1" with no page."""
    base = entries["source"].map(lambda s: "entry" if is_blank(s) else Path(str(s)).stem)
    page = entries["page"].map(lambda p: "" if is_blank(p) else f"-p{int(p)}")
    rows = entries["rows"].map(lambda r: "" if is_blank(r) else f"-r{r}")
    ids = base + page + rows
    count = ids.groupby(ids).cumcount() + 1
    return ids.where(~ids.duplicated(keep=False), ids + "-" + count.astype(str))


def add_reason(entries, mask, reason):
    """Add a check reason to the entries where mask is True, and update the check flag."""
    out = entries.copy()
    for i in out.index[mask]:
        reasons = [r for r in out.at[i, "check_reasons"].split("; ") if r]
        if reason not in reasons:
            out.at[i, "check_reasons"] = "; ".join(reasons + [reason])
    out["check"] = out["check_reasons"] != ""
    return out


def add_day_check(entries):
    """Flag entries on a day where the timekeeper's hours add up to more than DAY_MAX, usually a misread hour.

    Run it on everything entered for the day, since one upload alone may look fine. Dates taken from the week
    instead of the row are left out, since they aren't the real day.
    """
    real_date = ~entries["check_reasons"].str.contains("date_guessed")
    totals = entries[real_date].groupby(["tk_id", "work_date"])["hours"].transform("sum")
    over = (totals > DAY_MAX).reindex(entries.index, fill_value=False)
    return add_reason(entries, over, "day_over_max")


def make_entries(rows, timekeepers, year):
    """The shared table from a list of rows (dicts with any of ENTRY_COLS).

    Fills source_format from input_type, tk_id from tk_name, the entry ID, numeric hours and dates,
    and the checks that apply to every input type (unknown timekeeper, no hours, no date, over 14 hours in a day).
    year: for dates written without one ("08/11").
    """
    entries = pd.DataFrame(rows).reindex(columns=ENTRY_COLS).astype(object)
    entries["source_format"] = entries["source_format"].fillna(entries["input_type"].map(FORMAT_FOR_INPUT))
    missing_tk = entries["tk_id"].map(is_blank)
    entries.loc[missing_tk, "tk_id"] = entries.loc[missing_tk, "tk_name"].map(lambda n: find_timekeeper(n, timekeepers))
    entries["work_date"] = entries["work_date"].map(lambda d: to_date(d, year))
    entries["hours"] = pd.to_numeric(entries["hours"], errors="coerce")
    entries["min_conf"] = pd.to_numeric(entries["min_conf"], errors="coerce")
    entries["check_reasons"] = entries["check_reasons"].fillna("")
    entries["entry_id"] = entries["entry_id"].fillna(entry_ids(entries))

    entries = add_reason(entries, entries["tk_id"].map(is_blank), "unknown_tk")
    entries = add_reason(entries, entries["hours"].isna(), "no_hours")
    entries = add_reason(entries, entries["work_date"].isna(), "no_date")
    return add_day_check(entries).reset_index(drop=True)


def for_matching(entries):
    """Just the columns matching's decide() takes."""
    return entries[MATCH_COLS].copy()
