# Extraction

Turns time notes in any format into one table, the input matching takes. Timekeepers can write time anywhere (a time sheet, a sticky note, a napkin, an email), so every input type has to land in the same table: one row per entry, with a timekeeper, a work date, the client, matter, or CMID if written, the narrative, and the hours.

## How it works

```
time sheet, note photo  ─► Textract (tables + forms) ─► saved JSON ─┬─ table found ─► sheets.py ─┐
                                                                      └─ no table ────► notes.py  ─┤
typed note, text, email ───────────────────────────────────────────────────────────► notes.py  ─┼─► intake.py ─► matching's decide()
spreadsheet export      ─► column mapping (later) ──────────────────────────────────────────────┘
```

What decides the path is whether the writing has columns, not what it's written on. When Textract finds a table (a printed time sheet, or columns drawn on a notepad page), `sheets.py` reads it row by row. Otherwise the page's lines go through the free-text reader in `notes.py`, the same one used for emails.

Every entry also gets a link back to its source (file, page, and row, or email ID) and a **check flag** with plain reasons when extraction isn't sure. Flagged entries go to a reviewer instead of being matched automatically:

| Reason | Shown to the reviewer |
|---|---|
| `unknown_tk` | Timekeeper not found |
| `no_hours` | No hours found |
| `no_date` | No work date found |
| `day_over_max` | Over 14 hours on one day |
| `low_conf` | Hard to read |
| `hours_10x` | Hours may be missing a decimal |
| `client_copied` | Client copied from the line above |
| `date_guessed` | Date taken from the week, not the row |

## The shared table

| Column | Example | Goes to `decide()` |
|---|---|---|
| `entry_id` | `2399_001-p1-r4` | yes |
| `tk_id` | `TK102` | yes |
| `source_format` | `HANDWRITTEN` | yes |
| `work_date` | 2026-08-11 | yes |
| `cmid`, `client`, `matter` | `5520.001`, `Halvorsen Ind.` | yes |
| `narrative` | `Tel conference w/ CEO re hiring questions` | yes |
| `hours` | 0.4 | yes (feedback loop) |
| `input_type` | `SHEET`, `NOTE`, `PDF`, `SPREADSHEET`, `EMAIL`, `TEXT`, `VOICE` | |
| `tk_name` | name or sender as written | |
| `source`, `page`, `rows` | link back to the original | |
| `hours_read` | hours as Textract read them, before any correction (`7` when `hours` is 0.7) | |
| `min_conf` | lowest Textract confidence (0 to 100) | |
| `check`, `check_reasons` | `True`, `client_copied; low_conf` | |

Matching only knows four formats (`TYPED`, `PDF`, `HANDWRITTEN`, `EMAIL`), so `input_type` maps onto one of them: loose notes count as `HANDWRITTEN`, and typed notes and voice transcripts as `EMAIL`.

## Using it in the app

```python
from extract import extract
from intake import for_matching, timekeeper_table

timekeepers = timekeeper_table(firm_timekeepers)   # once at startup, like matching's load_reference
entries = extract(content, "NOTE", ref, timekeepers, tk_id="TK102", upload_date="2026-11-02", source="IMG_0412.jpg")
pairs, top, decisions = decide(for_matching(entries), ref)
```

`content` is Textract's saved pages for a photo, or the text itself for an email or typed note. The Textract call is a separate step, so the service can run it however it likes: `textract.read_page` for one page, or Textract's background mode for long PDFs. AWS credentials come from the machine (`aws configure` locally, the server's own permissions on AWS), never from the code.

## Notebooks

| Notebook | What it does |
|---|---|
| `00_textract_samples` | Runs Textract on the sample sheets and note photos, saves the JSON, and looks at what comes back. Needs AWS. |
| `01_handwritten_sheets` | Converts the scanned sheets and a note with drawn columns, and grades them against the answer key |
| `02_free_text_notes` | Typed notes, emails, and note photos through the free-text converter, graded |
| `03_end_to_end` | Extracted entries through matching's `decide()` (later) |

## Folders

| Path | What it is |
|---|---|
| `src/intake.py` | The shared table, timekeeper lookup, checks |
| `src/textract.py` | Calls Textract, saves its JSON, reads lines, table cells, and labeled fields back |
| `src/text_parser.py` | Finds hours, CMIDs, clients, and matters in free text (emails, notes, texts). Matching also uses it to parse the synthetic emails |
| `src/sheets.py` | Tables to rows: header, column roles, margin dates and CMIDs, merging wrapped rows, copy-down, hours checks |
| `src/notes.py` | Free text to rows: notes are split into entries line by line (an entry starts at its hours; wrapped lines and dates written above apply to it), with fixes for common photo misreads, then each entry goes through the text parser |
| `src/grade.py` | Comparing extracted entries with an answer key (similar clients, CMIDs by digits) |
| `src/extract.py` | `extract()`, the one function the app calls |
| `data/timekeepers.csv` | The synthetic firm's timekeepers with email addresses |
| `data/sheets/` | The 4 scanned sample time sheets (`2399_001.pdf`, a copy of `ocr/2399_001.pdf`) and the worksheets the writers copied from (`HW_worksheets.pdf`) |
| `data/sheets_answer_key.csv` | What is actually written on the 4 sample sheets and the notepad note (`notes/mon_fri_sample.jpg`), one row per entry, for grading the sheet converter |
| `data/notes_answer_key.csv` | The same for free-text note photos: `photo`, `entry`, `work_date` (YYYY-MM-DD, blank for none), `hours`, `client`, `cmid`, `narrative`, `note` |
| `data/notes/` | Note photos (JPEG or PNG; set the iPhone camera to Most Compatible, since Textract can't read HEIC) |
| `data/textract/` | Saved Textract answers, one JSON per page |

Matching imports the text parser from `extraction/src` (to parse the synthetic emails), so both folders need to be on the path. Extraction doesn't import from matching: the firm's client and matter names come in as `ref`, from matching's `load_reference` in the app.
