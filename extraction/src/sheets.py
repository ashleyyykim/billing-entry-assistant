"""Tables into entry rows: scanned time sheets, and notes with columns drawn on them.

Textract's table keeps each line of the page as one row, but it doesn't know what the columns mean or where one entry
ends and the next begins. The rules:
- Header: only Sheet ID, Timekeeper, and Week of, read by name (forms mode also finds false fields inside tables).
- Columns: named by the printed header ("Hours", "Client", "Description") when there is one, otherwise by what they
  hold (mostly dates, or the longest text, which is the narrative). Other columns are searched for a date, CMID,
  and hours, and whatever is left is the client.
- Margins: a date or CMID written beside the table belongs to the row at the same height.
- Rows: a row with hours starts an entry, and a row without hours joins the entry above it, unless it has its own
  client or CMID where the entry above already has one (then it's an entry whose hours weren't read). Rows without
  hours above the first entry (a first line that wrapped, or a date written on its own) join the first entry.
- Dates: carried down the page. Rows before any date get the Week of date, flagged date_guessed.
- Copy-down: an entry with hours but no client or CMID gets the ones above it (a drawn arrow or ditto mark reads
  as blank), flagged client_copied.
- CMIDs written in the Client column move to cmid.
- Hours: flagged low_conf when Textract was unsure, and hours_10x when there's no decimal point or the value is too
  high for one entry, since a small decimal point is easy to miss. A whole number of 5 or more on a one-task line is
  divided by 10 (still flagged); the reading is kept in hours_read.
"""
from notes import WEEKDAY, find_date
from text_parser import find_cmid, find_hours
from textract import outside_table, page_cells, page_fields, page_lines

HEADER_FIELDS = {"sheet id": "sheet_id", "timekeeper": "tk_name", "week of": "week_of"}
HEADER_WORDS = {"hours": "hours", "hrs": "hours", "client": "client", "description": "narrative", "date": "date"}
LOW_CONF = 80
MAX_HOURS = 12      # more than any single entry in the firm's billing history
FIX_FROM = 5        # a whole number this high on a one-task line is almost always a missed decimal point


def has_table(pages):
    return any(b["BlockType"] == "TABLE" for page in pages for b in page["Blocks"])


def header(page):
    """Sheet ID, timekeeper name, and Week of from the form's printed header, if the page has one."""
    fields = page_fields(page)
    fields["name"] = fields["key"].str.lower().str.strip(" :").map(HEADER_FIELDS)
    found = fields.dropna(subset=["name"]).drop_duplicates("name")
    return dict(zip(found["name"], found["value"]))


def main_table(cells):
    """The page's biggest table, in case Textract also finds a small stray one."""
    if cells.empty:
        return cells
    return cells[cells["table"] == cells.groupby("table").size().idxmax()]


def column_roles(cells):
    """What each column holds, and the header rows to skip.

    A printed header names its columns. Columns without a name are a date column when most of their cells are dates,
    the narrative when they have the longest text, and "other" otherwise.
    """
    first = cells[cells["row"] == cells["row"].min()]
    roles = {}
    for col, text in zip(first["col"], first["text"].str.lower()):
        role = next((r for word, r in HEADER_WORDS.items() if word in text), None)
        if role:
            roles[col] = role
    header_rows = [first["row"].iloc[0]] if roles else []
    body = cells[~cells["row"].isin(header_rows) & (cells["text"] != "")]
    for col, texts in body.groupby("col")["text"]:
        if col not in roles and texts.map(find_date).notna().mean() > 0.5:
            roles[col] = "date"
    if "narrative" not in roles.values():
        unnamed = body[~body["col"].isin(roles.keys())]
        if not unnamed.empty:
            roles[unnamed.groupby("col")["text"].apply(lambda t: t.str.len().mean()).idxmax()] = "narrative"
    return roles, header_rows


def margin_by_row(lines, cells):
    """Text written beside the table, joined by the table row at the same height ({row: text})."""
    left, right = cells["left"].min(), (cells["left"] + cells["width"]).max()
    out = outside_table(lines, cells)
    x, y = out["left"] + out["width"] / 2, out["top"] + out["height"] / 2
    beside = out[(x < left) | (x > right)].assign(y=y)
    bounds = cells.assign(bottom=cells["top"] + cells["height"]).groupby("row").agg(top=("top", "min"),
                                                                                   bottom=("bottom", "max"))
    margin = {}
    for _, line in beside.sort_values("top").iterrows():
        hit = bounds[(bounds["top"] <= line["y"]) & (line["y"] < bounds["bottom"])]
        if len(hit):
            margin.setdefault(hit.index[0], []).append(line["text"])
    # a CMID written sideways can come back in pieces ("5520", "002")
    return {row: (".".join(t) if all(p.isdigit() for p in t) else " ".join(t)) for row, t in margin.items()}


def to_hours(text):
    """Hours from an hours cell ("0.4", ".4", "1,5"), or None. Spaces are dropped, so "9 9" reads as 99."""
    text = text.replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        found = find_hours(text)
        return found[0][0] if found else None


def read_row(cells, roles, margin=""):
    """One table row's pieces: date, cmid, hours (with the text it came from and its confidence), client, narrative."""
    row = {"date": None, "cmid": None, "hours": None, "hours_text": None, "hours_conf": None,
           "client": [], "narrative": []}
    texts = [(roles.get(col, "other"), text.strip(), conf) for col, text, conf in
             zip(cells["col"], cells["text"], cells["conf"])] + [("margin", margin, None)]
    for role, text, conf in texts:
        if not text:
            continue
        if role == "narrative":
            row["narrative"].append(text)
            continue
        if role == "hours":
            row["hours"], row["hours_text"], row["hours_conf"] = to_hours(text), text, conf
            continue
        date = find_date(text)
        if date:
            row["date"] = date
            text = text.replace(date, " ")
        text = WEEKDAY.sub(" ", text)
        cmid = find_cmid(text)
        if cmid:
            row["cmid"] = cmid[0]
            text = text[:cmid[1]] + " " + text[cmid[2]:]
        if role in ("other", "date") and row["hours"] is None:
            found = find_hours(text)
            if found:
                hours, start, end, _ = found[-1]
                row["hours"], row["hours_text"], row["hours_conf"] = hours, text[start:end], conf
                text = text[:start] + " " + text[end:]
        text = " ".join(text.split())
        if role in ("client", "other") and len(text.strip(" .,-")) > 1:   # skips a lone mark like "X"
            row["client"].append(text)
    return row


def check_hours(row):
    """The hours to use and their check reasons. A whole number of FIX_FROM to MAX_HOURS on a one-task line ("7" for
    a phone call) is divided by 10, since a small decimal point is easy to miss. Other whole numbers are left as read.
    Either way the entry is flagged hours_10x, so a reviewer sees what was read."""
    hours, reasons = row["hours"], []
    if row["hours_conf"] is not None and row["hours_conf"] < LOW_CONF:
        reasons.append("low_conf")
    if hours is not None and ("." not in row["hours_text"] or hours > MAX_HOURS):
        reasons.append("hours_10x")
        one_task = ";" not in " ".join(row["narrative"])
        if FIX_FROM <= hours <= MAX_HOURS and hours == int(hours) and one_task:
            hours = hours / 10
    return hours, reasons


def join_rows(rows):
    """Several table rows' pieces as one row (a first line that wrapped onto the next)."""
    joined = dict(rows[-1])
    joined["narrative"] = [t for r in rows for t in r["narrative"]]
    joined["client"] = [t for r in rows for t in r["client"]]
    joined["cmid"] = next((r["cmid"] for r in rows if r["cmid"]), None)
    return joined


def new_entry(row, first, last, date, info, **source):
    """An entry from a row's pieces, with its hours flags and date."""
    entry = {**source, "first": first, "last": last, "tk_name": info.get("tk_name"), "cmid": row["cmid"],
             "client": " ".join(row["client"]) or None, "matter": None, "narrative": " ".join(row["narrative"]),
             "hours": row["hours"], "hours_read": row["hours_text"], "min_conf": row["hours_conf"],
             "work_date": date or info.get("week_of") or source.get("default_date")}
    entry.pop("default_date", None)
    entry["hours"], entry["reasons"] = check_hours(row)
    if not date and info.get("week_of"):
        entry["reasons"].append("date_guessed")
    return entry


def page_entries(page, page_no, source=None, input_type="SHEET", default_date=None):
    """The entries on one page, as rows for intake.make_entries."""
    cells = main_table(page_cells(page))
    if cells.empty:
        return []
    info = header(page)
    roles, header_rows = column_roles(cells)
    margin = margin_by_row(page_lines(page), cells)
    where = {"input_type": input_type, "source": source, "page": page_no, "default_date": default_date}
    entries, date, waiting = [], None, []
    for r, row_cells in cells[~cells["row"].isin(header_rows)].groupby("row"):
        row = read_row(row_cells, roles, margin.get(r, ""))
        if not (row["narrative"] or row["client"] or row["cmid"] or row["hours"] is not None):
            continue
        line = r - len(header_rows)
        date = row["date"] or date
        above = entries[-1] if entries else {}
        own_entry = (row["client"] and above.get("client")) or (row["cmid"] and above.get("cmid"))
        if row["hours"] is None and not entries:
            waiting.append((line, row))
        elif row["hours"] is not None or own_entry:
            first = waiting[0][0] if waiting else line
            entry = new_entry(join_rows([w for _, w in waiting] + [row]), first, line, date, info, **where)
            waiting = []
            if entries and not entry["client"] and not entry["cmid"]:
                above = entries[-1]
                if above["client"] or above["cmid"]:
                    entry.update(client=above["client"], cmid=above["cmid"], matter=above["matter"])
                    entry["reasons"].append("client_copied")
            entries.append(entry)
        else:
            entry = entries[-1]
            entry["last"] = line
            entry["narrative"] = " ".join([entry["narrative"]] + row["narrative"]).strip()
            entry["cmid"] = entry["cmid"] or row["cmid"]
            if row["client"]:
                entry["client"] = " ".join([entry["client"] or ""] + row["client"]).strip()
    if waiting:
        rows = [w for _, w in waiting]
        entries.append(new_entry(join_rows(rows), waiting[0][0], waiting[-1][0], date, info, **where))
    for e in entries:
        first, last = e.pop("first"), e.pop("last")
        e["rows"] = str(first) if first == last else f"{first}-{last}"
        e["check_reasons"] = "; ".join(e.pop("reasons"))
    return entries


def sheet_rows(pages, source=None, input_type="SHEET", default_date=None):
    """One row per entry across all pages of a sheet or a column note."""
    return [e for n, page in enumerate(pages, 1) for e in page_entries(page, n, source, input_type, default_date)]
