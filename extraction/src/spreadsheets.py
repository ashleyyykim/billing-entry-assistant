"""Excel and CSV time exports into entry rows.

Every firm's export names its columns differently ("Atty", "Dt Worked", "Hrs"), so each column is matched to the role
it plays (timekeeper, date, hours, narrative, client, matter, CMID) by its header, and by its values when the header
isn't recognized. Values are passed on as typed: matching already splits "Client / Matter" in one field, fixes OCR
letters in CMIDs, and handles misspelled clients.

Hours can come as hours, as tenths of an hour ("Units"), or as minutes, and are converted to hours. A title above the
header, blank rows, and a Total row are skipped.
"""
import datetime
import re

import pandas as pd
from rapidfuzz import fuzz, process

from text_parser import is_blank

ALIASES = {
    "tk": ["tk", "tkid", "timekeeper", "timekeepername", "tkinitials", "initials", "tkpr", "timekeeperinitials",
           "atty", "attorney", "user", "feeearner", "biller", "author", "name"],
    "date": ["date", "dt", "dateworked", "dtworked", "workdate", "entrydate", "transactiondate", "servicedate"],
    "cmid": ["cmid", "cm", "cmno", "matterno", "matternumber", "matternum", "matterid", "clientmatterno"],
    "client": ["client", "clientname", "clientmatter", "clientmatterdescription"],
    "matter": ["matter", "mattername", "matterdescription", "matterdesc"],
    "hours": ["hours", "hrs", "hour", "time", "timespent", "duration", "billablehours"],
    "units": ["units", "unit", "tenths"],
    "minutes": ["minutes", "mins", "min"],
    "narrative": ["narrative", "description", "desc", "workdescription", "worknarrative", "notes", "note", "details",
                  "entrydescription", "activity", "workperformed"],
}
ROLE_OF = {alias: role for role, aliases in ALIASES.items() for alias in aliases}
DATE_LIKE = re.compile(r"^\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}")
BILLING_CODE = re.compile(r"^\s*[LABE]\d{3}\s*$")   # task and activity codes ("L110", "A103")
TOTAL = re.compile(r"(grand\s+)?totals?:?", re.I)
HEADER_SPAN = 10   # the header is in the first rows
MIN_HEADER_HITS = 2
SIMILAR_NAME = 85  # fuzzy score for a header that is close to a known one ("Hours Worked")
MAX_DAY_HOURS = 24
MIN_SHARE = 0.8    # share of a column's values that must look like a date (or a number) to give it that role


def squash(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_spreadsheet(path):
    """The raw cells of an .xlsx or .csv file, with no header row assumed."""
    path = str(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, header=None, dtype=object)
    return pd.read_excel(path, header=None, dtype=object)


def role_of_header(name):
    """(role, how) for a column header, or (None, None)."""
    name = squash(name)
    if name in ROLE_OF:
        return ROLE_OF[name], "name"
    close = process.extractOne(name, list(ROLE_OF), scorer=fuzz.ratio, score_cutoff=SIMILAR_NAME) if name else None
    return (ROLE_OF[close[0]], "similar name") if close else (None, None)


def find_header(raw):
    """Index of the header row: the row in the first few with the most recognized headers (at least two)."""
    rows = range(min(HEADER_SPAN, len(raw)))
    hits = {i: sum(role_of_header(v)[0] is not None for v in raw.iloc[i].dropna()) for i in rows}
    best = max(hits, key=lambda i: (hits[i], -i), default=None)
    if best is None or hits[best] < MIN_HEADER_HITS:
        raise ValueError("No header row found in the first rows")
    return best


def looks_like(column, kind):
    values = column.dropna()
    if values.empty:
        return False
    if kind == "date":
        ok = values.map(lambda v: isinstance(v, datetime.datetime) or bool(DATE_LIKE.match(str(v))))
    elif kind == "code":
        ok = values.astype(str).str.match(BILLING_CODE)
    else:
        numbers = pd.to_numeric(values, errors="coerce")
        ok = numbers.notna() & (numbers > 0) & (numbers <= MAX_DAY_HOURS)
    return ok.mean() >= MIN_SHARE


def map_columns(raw):
    """(table, mapping): the data rows with named columns, and one row per column saying the role it was given.

    A role goes to the leftmost column with a recognized header, except a narrative-named column ("Activity") that
    holds task or activity codes. A date, hours, or narrative column with an unrecognized header is found by its
    values (dates, small numbers, the longest text). Other columns are ignored.
    """
    top = find_header(raw)
    table = raw.iloc[top + 1:].copy()
    table.columns = [str(c) if not is_blank(c) else f"column {i + 1}" for i, c in enumerate(raw.iloc[top])]
    table = table.dropna(how="all")

    mapping = []
    taken = set()
    for col in table.columns:
        role, how = role_of_header(col)
        if role is not None and role in taken:
            role, how = None, "ignored (another column has this role)"
        elif role == "narrative" and looks_like(table[col], "code"):
            role, how = None, "ignored (task or activity codes)"
        taken.add(role)
        mapping.append({"column": col, "role": role, "how": how})
    mapping = pd.DataFrame(mapping)

    free = mapping["role"].isna()
    for role, kind in [("date", "date"), ("hours", "number"), ("narrative", "text")]:
        filled = role in set(mapping["role"]) or (role == "hours" and {"units", "minutes"} & set(mapping["role"]))
        candidates = [c for c in mapping.loc[free, "column"]]
        if filled or not candidates:
            continue
        if kind == "text":
            lengths = {c: table[c].dropna().astype(str).str.len().mean() for c in candidates if table[c].notna().any()}
            pick = max(lengths, key=lengths.get, default=None)
        else:
            pick = next((c for c in candidates if looks_like(table[c], kind)), None)
        if pick:
            mapping.loc[mapping["column"] == pick, ["role", "how"]] = [role, "values"]
            free = mapping["role"].isna()
    mapping["role"] = mapping["role"].where(mapping["role"].notna(), None)
    mapping["how"] = mapping["how"].fillna("ignored")
    return table, mapping


def to_hours(table, mapping):
    """Hours from the hours column, or the units column (tenths), or the minutes column."""
    roles = dict(zip(mapping["role"], mapping["column"]))
    for role, divide in [("hours", 1), ("units", 10), ("minutes", 60)]:
        if role in roles:
            text = table[roles[role]].astype(str).str.replace(r"(?i)\s*h(ou)?rs?\.?$", "", regex=True)
            return pd.to_numeric(text, errors="coerce") / divide
    return pd.Series(float("nan"), index=table.index)


def cmid_text(value):
    """A CMID as text. A number from Excel keeps its three decimals (5602.030, not 5602.03)."""
    if is_blank(value):
        return None
    return f"{value:.3f}" if isinstance(value, float) else str(value).strip()


def spreadsheet_rows(raw, source=None, mapping=None):
    """One row per entry in an export. mapping: a confirmed (table, mapping) from map_columns, to use instead."""
    table, mapping = mapping or map_columns(raw)
    roles = dict(zip(mapping["role"], mapping["column"]))
    table = table[~table.apply(lambda row: any(TOTAL.fullmatch(str(v).strip()) for v in row.dropna()), axis=1)]
    hours = to_hours(table, mapping)
    cell = lambda role, i: table.at[i, roles[role]] if role in roles else None
    clean = lambda v: None if is_blank(v) else str(v).strip()
    return [{
        "input_type": "SPREADSHEET", "source": source, "rows": str(i + 1), "tk_name": clean(cell("tk", i)),
        "work_date": cell("date", i), "cmid": cmid_text(cell("cmid", i)), "client": clean(cell("client", i)),
        "matter": clean(cell("matter", i)), "narrative": clean(cell("narrative", i)) or "", "hours": hours[i],
        "reasons": [],
    } for i in table.index]
