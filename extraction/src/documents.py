"""Typed documents into entry rows: PDF reports with a text layer (exports from a billing system, typed time sheets),
Word files (.docx, including Google Docs downloaded as Word), and plain text files (a shared doc downloaded as text).

A document can hold a table of entries, lines of entries, or both:
- A table goes through spreadsheets.py like an Excel export: its header row gives each column a role. A table that
  continues on the next page without its header is joined to the one before it.
- Text outside tables goes through free_text.py like a typed note, one entry per amount of hours. Before that, page
  numbers and total lines are dropped, a labeled timekeeper ("Timekeeper: R. Patel") is kept for every entry, and a
  matter heading (a CMID with no hours, as on a pre-bill) is put in front of each entry below it.
A PDF with no text layer is a scan (is_scanned). It goes to Textract instead.
"""
import re
from pathlib import Path

import pandas as pd

from free_text import CMID_SHAPE, is_date_line, names_own, spoken_hours, text_rows, typed_fixes
from text_parser import find_hours
from spreadsheets import MIN_HEADER_HITS, find_header, role_of_header, spreadsheet_rows

SAME_LINE = 3            # words whose tops are within this many points are on one line
COLUMN_GAP = 6           # a gap wider than this (points) between words separates two columns
TABLE_GAP = 36           # a gap taller than this (points) below a table row ends the table
MIN_TEXT = 20            # a PDF with fewer characters than this is a scan
PAGE_NUMBER = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I)
PAGE_IN_LINE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.I)
TOTAL_LINE = re.compile(r"\btotals?\b", re.I)
TIMEKEEPER_LABEL = re.compile(r"^\s*(timekeeper|attorney|prepared for|fee earner)\s*[:\-]\s*(.+?)\s*$", re.I)
TK_ID_IN_NAME = re.compile(r"\s*\(\s*TK\d+\s*\)\s*$", re.I)
HEADING = re.compile(r"^\s*(\d{4}\.\d{1,3})\b")
DATE_THEN_INITIALS = re.compile(r"^(\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+[A-Z]{2,3}\s+")


def is_scanned(path):
    """True for a PDF with no text layer (a scan or photo saved as PDF)."""
    import pymupdf
    with pymupdf.open(path) as pdf:
        return sum(len(page.get_text().strip()) for page in pdf) < MIN_TEXT


def pdf_lines(page):
    """A PDF page's words grouped into lines, top to bottom: (top, words), with words as (left, right, text) left to
    right. Words whose tops are within SAME_LINE points are one line, so text drawn in pieces stays together."""
    lines = []
    for x0, y0, x1, _, text, *_ in sorted(page.get_text("words"), key=lambda w: (w[1], w[0])):
        if lines and abs(lines[-1]["top"] - y0) <= SAME_LINE:
            lines[-1]["words"].append((x0, x1, text))
        else:
            lines.append({"top": y0, "words": [(x0, x1, text)]})
    return [(line["top"], sorted(line["words"])) for line in lines]


def word_groups(words):
    """A line's words split into groups wherever the gap between two words is wider than COLUMN_GAP points."""
    groups = []
    for x0, x1, text in words:
        if groups and x0 - groups[-1]["right"] <= COLUMN_GAP:
            groups[-1].update(right=x1, text=groups[-1]["text"] + " " + text)
        else:
            groups.append({"left": x0, "right": x1, "text": text})
    return groups


def header_columns(words):
    """Column starts if the line is a table header (two or more groups name a known column, as in an export), or
    None. A word belongs to the last column that starts at or before it (with COLUMN_GAP points to spare), since
    values sit under their header's left edge and a long description stays inside its own column."""
    groups = word_groups(words)
    if sum(role_of_header(g["text"])[0] is not None for g in groups) < MIN_HEADER_HITS:
        return None
    return {"names": [g["text"] for g in groups], "starts": [g["left"] - COLUMN_GAP for g in groups]}


def column_of(word, columns):
    return max(sum(word[0] >= start for start in columns["starts"]) - 1, 0)


def pdf_parts(path):
    """(tables, lines) in a PDF with a text layer.

    A line that names two or more known columns ("Date", "Matter No.", "Hours") starts a table. Each line below it
    becomes a row, with each word in the column under the header word it sits beneath. A line that starts right of
    the first column continues the row above (a wrapped description). The table ends at a gap wider than TABLE_GAP
    points, or at the end of the page. Page numbers and total lines are dropped. Everything else is text lines.
    """
    import pymupdf
    tables, lines = [], []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            columns, rows, last_top = None, None, None
            for top, words in pdf_lines(page):
                text = " ".join(w[2] for w in words)
                if PAGE_IN_LINE.search(text) or TOTAL_LINE.search(text):
                    continue
                found = header_columns(words)
                if found:
                    columns, rows, last_top = found, [found["names"]], top
                    tables.append(rows)
                    continue
                if columns is not None and top - last_top > TABLE_GAP:
                    columns = None
                if columns is not None:
                    last_top = top
                    cells = [""] * len(columns["names"])
                    for word in words:
                        i = column_of(word, columns)
                        cells[i] = (cells[i] + " " + word[2]).strip()
                    if cells[0]:
                        rows.append(cells)
                        continue
                    if len(rows) > 1:
                        rows[-1] = [(a + " " + b).strip() for a, b in zip(rows[-1], cells)]
                        continue
                    columns = None
                lines.append(text)
    return join_continued([pd.DataFrame(t).replace({"": None}) for t in tables]), lines


def docx_parts(path):
    """(tables, lines) in a Word file: each table as raw cells, and every paragraph outside tables as a line."""
    from docx import Document
    document = Document(path)
    tables = [pd.DataFrame([[cell.text.strip() or None for cell in row.cells] for row in table.rows])
              for table in document.tables]
    lines = [p.text for p in document.paragraphs if p.text.strip()]
    return join_continued(tables), lines


def text_parts(path):
    """(no tables, lines) in a plain text file."""
    return [], [line for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def has_header(raw):
    try:
        find_header(raw)
        return True
    except ValueError:
        return False


def join_continued(tables):
    """Tables with a header, with a table that has none (it continues on the next page) joined to the one above."""
    joined = []
    for raw in tables:
        if joined and not has_header(raw) and raw.shape[1] == joined[-1].shape[1]:
            joined[-1] = pd.concat([joined[-1], raw], ignore_index=True)
        else:
            joined.append(raw)
    return joined


def clean_lines(lines):
    """(lines, timekeeper): page numbers and totals dropped, a labeled timekeeper taken out, and each entry under a
    matter heading ("3455.002 Aster & Vale - Board & Governance") given the heading's CMID and client, but not the
    matter name, which isn't part of the description ("3455.002 Aster & Vale 10/27/26 0.50 Prepare minutes").
    Initials after a date ("10/27/26 EB") are dropped. Two-decimal hours ("0.50") are read later (typed_fixes)."""
    timekeeper, heading, out = None, None, []
    for line in lines:
        label = TIMEKEEPER_LABEL.match(line)
        if label:
            timekeeper = timekeeper or TK_ID_IN_NAME.sub("", label.group(2))
            continue
        if PAGE_NUMBER.match(line) or TOTAL_LINE.search(line):
            continue
        if HEADING.match(line) and not re.search(r"(?<![\d.])\d*\.\d+(?![\d.])", HEADING.sub("", line)):
            heading = re.split(r"\s+[-\u2013|]\s+", line.strip(), maxsplit=1)[0]
            continue
        line = DATE_THEN_INITIALS.sub(r"\1 ", line)
        out.append(f"{heading} {line.strip()}" if heading else line)
    return out, timekeeper


def has_hours(line):
    return bool(find_hours(spoken_hours(line)))


def drop_titles(lines, lookup):
    """The lines from the first one with hours, keeping lines above it that are only a date or name a client or CMID
    ("Mon 10/26", "Kestrel Foods"). Others above it are titles or labels ("Week of Oct 26", "Time - M. Alvarez").
    With no hours anywhere, there are no entries in the text, so no lines."""
    first = next((i for i, line in enumerate(lines) if has_hours(line)), None)
    if first is None:
        return []
    above = [line for line in lines[:first] if is_date_line(line) or names_own(line, lookup)]
    return above + lines[first:]


def document_rows(path, lookup, matters, source=None, upload_date=None):
    """One row per entry in a typed document (.pdf with a text layer, .docx, or .txt)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if is_scanned(path):
            raise ValueError(f"{path.name} is a scan with no text layer: read it with Textract as a sheet")
        tables, lines = pdf_parts(path)
        input_type = "PDF"
    elif suffix == ".docx":
        tables, lines = docx_parts(path)
        input_type = "DOC"
    elif suffix in (".txt", ".md"):
        tables, lines = text_parts(path)
        input_type = "DOC"
    else:
        raise ValueError(f"{path.name}: typed documents are .pdf, .docx, or .txt")

    lines, timekeeper = clean_lines(lines)
    lines = drop_titles([typed_fixes(line) for line in lines], lookup)
    rows = []
    for t, raw in enumerate(tables, 1):
        if not has_header(raw):
            lines += [" ".join(str(v) for v in row if v) for row in raw.itertuples(index=False)]
            continue
        for row in spreadsheet_rows(raw, source):
            row.update(rows=f"{t}.{row['rows']}")
            if row["cmid"] and not CMID_SHAPE.fullmatch(row["cmid"]):
                row["check_reasons"] = "cmid_shape"
            rows.append(row)
    if lines:
        rows += text_rows("\n".join(lines), lookup, matters, "TEXT", source, upload_date)
    for row in rows:
        row["input_type"] = input_type
        row["tk_name"] = row.get("tk_name") or timekeeper
    return rows
