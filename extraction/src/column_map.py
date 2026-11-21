"""A spreadsheet read from its column map: a model says what each column holds, and code reads every row.

The model sees the title and header rows and the first SAMPLE_ROWS data rows, column by column, and gives each column
one role. Code then reads every data row with that map, so no row can be dropped or reworded, and a 200-row export
takes one call per reader instead of one per 25 rows. Tenths and minutes become hours in code. The entries go through
the same checks as a model's (one_flow.answer_rows): hours units the header doesn't name, dates, clients, CMIDs,
timekeepers, and rows no entry uses.
"""
import hashlib
import json
import re

import pandas as pd

from spreadsheets import TOTAL, cmid_text

ROLES = ["timekeeper", "date", "cmid", "client", "matter", "hours", "tenths", "minutes", "narrative", "billable",
         "ignore"]
DIVIDE = {"hours": 1, "tenths": 10, "minutes": 60}
SAMPLE_ROWS = 5          # data rows the model sees
MIN_NUMBERS = 0.8        # share of the time column's filled cells that must be numbers for the map to be used
NOT_BILLABLE = {"n", "no", "false", "0", "nb", "nc", "non-billable", "nonbillable", "not billable", "no charge"}
HOURS_WORD = re.compile(r"(?i)\s*h(ou)?rs?\.?$")
CLOCK = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
CLIENT_AND_MORE = re.compile(r"\s+/\s+|\s*/\s*(?=\d{4}\b)")
CMID = re.compile(r"\d{4}\.\d{3}")

SYSTEM = """# Task
You map the columns of a law firm's time export (a spreadsheet) so that code can read every row. You see the title
and header rows, then each column with its header and its first values. Give every column exactly one role:
- timekeeper: who did the work: a name, initials, an email, or a timekeeper ID.
- date: the day the work was done (an entry or work date; not a run date or a billing date when there is a work date).
- cmid: the client-matter number alone (4 digits, a dot, 3 digits, like 4590.003).
- client: the client's name, or the client and matter together in one cell ("Northgate Bank / 7215.003").
- matter: the matter's name, apart from the client ("Series B Financing").
- hours: time worked in hours (1.5, 0.25, "2 hrs").
- tenths: time worked in tenths of an hour (6 means 0.6 hours), often headed "Units".
- minutes: time worked in minutes.
- narrative: the description of the work.
- billable: whether the entry is billable (Y/N, Yes/No).
- ignore: anything else: rates, amounts billed, task or activity codes (L110, A103), entry IDs, and empty columns.

# Rules
- Read the header first and use the values to settle it. A column with an unclear header gets the role its values show.
- Only one column holds the time worked; give it hours, tenths, or minutes, and ignore any other time column (a rate
  or an amount billed is not time).
- When two columns could hold the same role, give it to the one that fits best and ignore the other, except
  timekeeper, which can go to both a name column and an ID column.
- "unsure": say in a few words when you can't tell which column holds the time worked or its unit (a header like
  "Time" or "Duration" with values that could be hours, tenths, or minutes), or which column names the client.
  Otherwise leave it "".
"""
SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {"type": "array", "description": "Every column, in order.", "items": {
            "type": "object",
            "properties": {"column": {"type": "integer", "description": "The column's number, as shown."},
                           "role": {"type": "string", "enum": ROLES}},
            "required": ["column", "role"]}},
        "unsure": {"type": "string", "description": "What isn't clear about the time column or the client, or \"\"."},
    },
    "required": ["columns", "unsure"],
}


def prompt_hash():
    """A short fingerprint of what the model is told for a column map."""
    return hashlib.sha256((SYSTEM + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:10]


def cell_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)) or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp) or hasattr(value, "strftime"):
        return value.strftime("%m/%d/%Y")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def column_text(raw, top, body):
    """What the model reads: the title and header rows, then each column with its header and first values."""
    texts = lambda i: [cell_text(v) for v in raw.iloc[i]]
    header = texts(top[-1])
    samples = [texts(i) for i in body[:SAMPLE_ROWS]]
    out = ["Title and header rows:"] + [" | ".join(texts(i)) for i in top] + ["", "Columns:"]
    for n, name in enumerate(header):
        values = " | ".join(row[n] or "(blank)" for row in samples) or "(no data rows)"
        out.append(f"Column {n + 1}, header \"{name or '(none)'}\": {values}")
    return "\n".join(out)


def ask_columns(runtime, text, lines, model_id, force_tool=True):
    """The model's column map for one spreadsheet, forced into SCHEMA by a tool call when the model allows it."""
    response = runtime.converse(
        modelId=model_id, system=[{"text": SYSTEM}], messages=[{"role": "user", "content": [{"text": text}]}],
        toolConfig={"tools": [{"toolSpec": {"name": "record_columns", "description": "Record each column's role.",
                                            "inputSchema": {"json": SCHEMA}}}],
                    **({"toolChoice": {"tool": {"name": "record_columns"}}} if force_tool else {})},
        inferenceConfig={"temperature": 0, "maxTokens": 1000})
    blocks = response["output"]["message"]["content"]
    found = next((block["toolUse"]["input"] for block in blocks if "toolUse" in block), None)
    if found is None:   # a model that wasn't forced can answer in its text
        said = " ".join(block.get("text", "") for block in blocks)
        start, end = said.find("{"), said.rfind("}")
        try:
            found = json.loads(said[start:end + 1]) if start >= 0 else None
        except json.JSONDecodeError:
            found = None
    return {"model": model_id, "map": found or {}, "lines": lines, "stop_reason": response["stopReason"],
            "usage": response["usage"], "latency_ms": response["metrics"]["latencyMs"], "entries": []}


def roles_of(answer, width):
    """{column index: role} from a column map answer, leaving out unknown roles and columns."""
    out = {}
    for item in (answer or {}).get("map", {}).get("columns") or []:
        try:
            n = int(item.get("column")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= n < width and item.get("role") in ROLES and item["role"] != "ignore":
            out.setdefault(n, item["role"])
    return out


def time_column(roles):
    """(column index, divide by) of the first column holding the time worked, or None."""
    return next(((n, DIVIDE[role]) for n, role in sorted(roles.items()) if role in DIVIDE), None)


def read_time(value, divide):
    """(as written, hours) for a time cell: "1.5", "2 hrs", "1:30", or a whole number of tenths or minutes."""
    written = cell_text(value)
    text = HOURS_WORD.sub("", written)
    clock = CLOCK.match(text)
    if clock:
        return written, round(int(clock.group(1)) + int(clock.group(2)) / 60, 2)
    number = pd.to_numeric(text, errors="coerce")
    return written, (None if pd.isna(number) else round(float(number) / divide, 4))


def usable(roles, raw, body):
    """True when the map names a time column and that column's filled cells are mostly numbers."""
    found = time_column(roles)
    if found is None:
        return False
    times = [read_time(raw.iat[i, found[0]], found[1]) for i in body]
    filled = [hours for written, hours in times if written]
    return bool(filled) and sum(h is not None and h > 0 for h in filled) / len(filled) >= MIN_NUMBERS


def split_client(text):
    """(client, cmid, matter) for a client cell that can hold the matter too ("Northgate Bank / 7215.003")."""
    parts = CLIENT_AND_MORE.split(text, maxsplit=1)
    if len(parts) < 2:
        return text, "", ""
    client, rest = parts[0].strip(), parts[1].strip()
    return (client, rest, "") if CMID.fullmatch(rest) else (client, "", rest)


def row_entry(cells, roles, line_id):
    """The entry for one data row, in the shape a model's answer has, or None for a total row or one with no time and
    no narrative."""
    if any(TOTAL.fullmatch(cell_text(v)) for v in cells if cell_text(v)):
        return None
    first = lambda role: next((cell_text(cells[n]) for n, r in sorted(roles.items())
                               if r == role and cell_text(cells[n])), "")
    n, divide = time_column(roles)
    written, hours = read_time(cells[n], divide)
    narrative = first("narrative")
    if not written and not narrative:
        return None
    cmid = next((cmid_text(cells[n]) for n, r in sorted(roles.items()) if r == "cmid" and cell_text(cells[n])), "")
    client, more_cmid, more_matter = split_client(first("client"))
    billable = first("billable")
    return {"lines": [line_id], "hours_written": written, "hours": hours, "narrative": narrative, "client": client,
            "cmid": cmid or more_cmid, "date": first("date"), "timekeeper": first("timekeeper"),
            "matter": first("matter") or more_matter, "unsure": "",
            "billable": billable.lower() not in NOT_BILLABLE if billable else True}
