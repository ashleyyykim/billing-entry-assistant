# Extraction

Turns time notes in any format into the table matching takes: one row per entry, with a timekeeper, work date, CMID, client, matter, narrative, and hours. Timekeepers write time anywhere (time sheets, sticky notes, napkins, emails), so every input lands in the same table.

Extraction gets what was written into the right columns. It doesn't correct misspelled clients or OCR letters in CMIDs, since matching already handles those.

## How it works

```
time sheet, note photo ─► Textract ─┬─ table found ─► sheets.py ─┐
                                     └─ no table ────► notes.py  ─┤
typed note, text, email ─────────────────────────────► notes.py  ─┴─► intake.py ─► matching's decide()
```

1. **Read.** Amazon Textract reads scans and photos. Its full answer is saved, so each page is paid for once.
2. **Tables** (printed sheets, or columns drawn on a notepad). Header fields, margin dates, rows that wrap, arrows that mean "same as above", and CMIDs written under Client.
3. **Free text** (notes without columns, typed notes, emails). Lines are grouped into entries and parsed for hours, CMIDs, and clients.
4. **Check.** Each entry gets a `check` flag with reasons when extraction isn't sure, and a link back to its source (file, page, and row).

| Check reason | Meaning |
|---|---|
| `hours_10x` | Hours may be missing a decimal (`hours_read` keeps what was read) |
| `no_hours` | No hours found |
| `day_over_max` | Over 14 hours on one day |
| `low_conf` | Hard to read |
| `client_copied` | Client copied from the line above |
| `date_guessed` | Date taken from the week, not the row |
| `unknown_tk`, `no_date` | Timekeeper or date not found |

## Using it in the app

```python
entries = extract(content, "NOTE", ref, timekeepers, tk_id="TK102", upload_date="2026-11-02", source="IMG_0412.jpg")
pairs, top, decisions = decide(for_matching(entries), ref)
```

`content` is Textract's saved pages for a scan or photo, or the text itself for an email or typed note. `decide()` doesn't look at hours or the `check` flag, so the app handles those separately.

## Notebooks

| Notebook | What it does |
|---|---|
| `00_textract_samples` | Runs Textract on the sample sheets and photos and looks at what comes back (needs AWS) |
| `01_handwritten_sheets` | Converts scanned sheets and notes with drawn columns, graded against the answer key |
| `02_free_text_notes` | Typed notes, emails, and note photos, graded against the answer key |
| `03_end_to_end` | Extracted entries through matching's `decide()`, compared with the intended matters (read only) |

## Folders

| Folder | What it is |
|---|---|
| `src/` | `extract.py` (the one function the app calls), `textract.py`, `sheets.py`, `notes.py`, `text_parser.py`, `intake.py`, `grade.py` |
| `data/sheets/` | Scanned sample sheets and the worksheets their writers copied from |
| `data/notes/` | Note photos (JPEG or PNG; Textract can't read HEIC) |
| `data/textract/` | Saved Textract answers, one JSON per page |
| `data/` | Answer keys (what is written), intended matters, and the timekeeper list with emails |

Matching imports `text_parser.py` from here to parse the synthetic emails, so both `src/` folders need to be on the path.
