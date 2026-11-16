"""The rules: any input type in, the shared entry table out. The app calls one_flow.read for every upload, and this
is its backup when the model can't answer (one_flow.py).

On AWS, the upload lands in S3 and Textract runs on it (textract.read_page, or Textract's background mode for long
PDFs). extract then takes Textract's pages, or the text itself for emails and typed notes, and returns the table
that goes to matching's decide() (through intake.for_matching).

Sheets and photographed notes go through the table reader (tables.py) when Textract found a table on the page, and
through the free-text reader (free_text.py) when it didn't, so a note with columns drawn on it keeps each row together.
Textract only finds tables in "sheet" mode, which also returns every line, so reading photos in that mode lets one
call serve both paths.
"""
import pandas as pd

from documents import document_rows
from free_text import note_rows, original_sender, parser_lookups, screenshot_rows, text_rows
from intake import add_reason, find_timekeeper, make_entries
from spreadsheets import spreadsheet_rows
from tables import has_table, sheet_rows

SKIP_CLIENTS = ("Firm",)   # an ordinary word, so "firm meeting" doesn't name a client (same as matching)
TEXT_TYPES = ("EMAIL", "TEXT", "VOICE")


def email_writer(body, sender, timekeepers):
    """TK ID of who wrote an email: the original sender of a forwarded email when they're a timekeeper (an assistant
    forwarding an attorney's email), else the sender (an attorney sending their own). None when neither is one."""
    original = original_sender(body)
    return (find_timekeeper(original, timekeepers) if original else None) or find_timekeeper(sender, timekeepers)


def extract(content, input_type, ref, timekeepers, tk_id=None, tk_name=None, upload_date=None, source=None,
            year=None, note_options=None):
    """The shared entry table for one upload.

    content: Textract's pages (a list of saved JSON answers) for SHEET, NOTE, and SCREENSHOT; the text itself for
        EMAIL, TEXT, and VOICE; the raw cells (spreadsheets.read_spreadsheet) for SPREADSHEET; the file path for PDF
        (with a text layer; a scanned PDF is a SHEET) and DOC (.docx or .txt).
    input_type: a key of intake.FORMAT_FOR_INPUT.
    ref: matching's load_reference output (only client_names and matter_names are used here).
    timekeepers: intake.timekeeper_table.
    tk_id: whose time it is, when the app knows it (an assistant picks the attorney).
    tk_name: a name or email to look up instead, like an email's sender. A timekeeper written in the upload itself
        (a sheet's or report's "Timekeeper:" field, an export's name column) wins over both. A forwarded email is
        credited to the original sender when they're a timekeeper (an assistant forwarding an attorney's email), and
        to the sender otherwise (an attorney forwarding their own). When both are timekeepers and differ, every entry
        is flagged tk_forwarded.
    upload_date: the work date when none is written.
    source: file name or email ID, for the link back to the original.
    year: for dates written without one. Defaults to the upload date's year.
    note_options: settings passed to free_text.note_rows for photos without a table.
    """
    lookup, matters = parser_lookups(ref["client_names"], ref["matter_names"], SKIP_CLIENTS)
    if input_type in ("SHEET", "NOTE") and has_table(content):
        rows = sheet_rows(content, source, input_type, upload_date)
    elif input_type == "NOTE":
        rows = note_rows(content, lookup, matters, source, upload_date, **(note_options or {}))
    elif input_type == "SCREENSHOT":
        rows = screenshot_rows(content, lookup, matters, source, upload_date)
    elif input_type == "SPREADSHEET":
        rows = spreadsheet_rows(content, source)
    elif input_type in ("PDF", "DOC"):
        rows = document_rows(content, lookup, matters, source, upload_date)
    elif input_type in TEXT_TYPES:
        rows = text_rows(content, lookup, matters, input_type, source, upload_date)
    else:
        raise NotImplementedError(f"{input_type} isn't built yet")
    for row in rows:
        own = row.get("tk_name")
        row["tk_id"] = None if own else tk_id
        row["tk_name"] = own or tk_name
    year = year or pd.Timestamp(upload_date or "today").year
    forwarder = None
    if input_type == "EMAIL" and tk_id is None:
        original = original_sender(content)
        if original and find_timekeeper(original, timekeepers):
            forwarder = find_timekeeper(tk_name, timekeepers)
            for row in rows:
                row["tk_name"] = original
    entries = make_entries(rows, timekeepers, year)
    if forwarder and len(entries):
        entries = add_reason(entries, entries["tk_id"] != forwarder, "tk_forwarded")
    return entries
