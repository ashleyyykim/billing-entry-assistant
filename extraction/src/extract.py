"""The one function the app calls for each upload: any input type in, the shared entry table out.

On AWS, the upload lands in S3 and Textract runs on it (textract.read_page, or Textract's background mode for long
PDFs). extract then takes Textract's pages, or the text itself for emails and typed notes, and returns the table
that goes to matching's decide() (through intake.for_matching).

Sheets and photographed notes go through the table reader (sheets.py) when Textract found a table on the page, and
through the free-text reader (notes.py) when it didn't, so a note with columns drawn on it keeps each row together.
Textract only finds tables in "sheet" mode, which also returns every line, so reading photos in that mode lets one
call serve both paths.
"""
import pandas as pd

from intake import make_entries
from notes import note_rows, parser_lookups, text_rows
from sheets import has_table, sheet_rows

SKIP_CLIENTS = ("Firm",)   # an ordinary word, so "firm meeting" doesn't name a client (same as matching)
TEXT_TYPES = ("EMAIL", "TEXT", "VOICE")


def extract(content, input_type, ref, timekeepers, tk_id=None, tk_name=None, upload_date=None, source=None,
            year=None):
    """The shared entry table for one upload.

    content: Textract's pages (a list of saved JSON answers) for SHEET and NOTE; the text itself for EMAIL, TEXT,
        and VOICE.
    input_type: a key of intake.FORMAT_FOR_INPUT. PDF and SPREADSHEET come next.
    ref: matching's load_reference output (only client_names and matter_names are used here).
    timekeepers: intake.timekeeper_table.
    tk_id: who uploaded, when the app knows it. tk_name: a name or email to look up instead, like an email's sender.
    upload_date: the work date when none is written. source: file name or email ID, for the link back to the original.
    year: for dates written without one. Defaults to the upload date's year.
    """
    lookup, matters = parser_lookups(ref["client_names"], ref["matter_names"], SKIP_CLIENTS)
    if input_type in ("SHEET", "NOTE") and has_table(content):
        rows = sheet_rows(content, source, input_type, upload_date)
    elif input_type == "NOTE":
        rows = note_rows(content, lookup, matters, source, upload_date)
    elif input_type in TEXT_TYPES:
        rows = text_rows(content, lookup, matters, input_type, source, upload_date)
    else:
        raise NotImplementedError(f"{input_type} isn't built yet")
    for row in rows:
        row["tk_id"] = tk_id
        row["tk_name"] = tk_name or row.get("tk_name")
    year = year or pd.Timestamp(upload_date or "today").year
    return make_entries(rows, timekeepers, year)
