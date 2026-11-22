"""One flow for every upload: get the text out, a model reads it into entries, and the same checks run on every entry.

This follows how document AI products are built: parse each input into text, extract with an LLM into a fixed
schema, validate with business rules, and send what fails to a person. Only
the first step depends on the input:

| Input | What the model reads | Image too |
|---|---|---|
| NOTE: a photo of a note, columns or not | Textract's handwritten lines, each with where it sits | the photo |
| SHEET: a scanned time sheet | every Textract line (printed headings and handwriting), with where it sits | the page |
| SCREENSHOT: a picture of an email | Textract's lines | the screenshot |
| EMAIL, TEXT, VOICE | the text (for voice, Transcribe's transcript) | |
| PDF, DOC: typed documents | the text in reading order, with table cells and wide gaps shown as " | " | |
| SPREADSHEET: Excel or CSV | the rows, cells joined by " | ", the header first (column_map: header and first rows) | |

Then the same steps for every input:
1. Nova Pro reads the lines into the entry schema with a forced tool call at temperature 0, naming the lines each
   entry uses (model_readers.ask_nova). Note photos get the photo prompt (model_readers.system_prompt); every other
   input gets prompt v6 (prompts_v6.py, built on prompts.py): the same task and fields, what is and isn't an entry, how to read hours, the habits of
   that input type, and worked examples of it. For documents, scans, and spreadsheets the model also says whose time
   each entry is, when the text names a timekeeper.
2. Long inputs are read in pieces (a page of a PDF or scan, SPREADSHEET_ROWS rows with the header repeated). Each
   later piece starts with the first lines of the document, so a heading like "Timekeeper: R. Patel" still applies.
3. A second model (Qwen3 VL) reads every piece too (ModelReader.second, combine): an entry the two read
   differently goes to review, and one they read the same way loses model_unsure.
4. Code cleans up what the models wrote: a matter name moved out of the CMID, a CMID's other spellings fixed, dates
   worked out from "yesterday", a day name, or "Week of", units converted only where the header names them.
5. The same checks run on every entry (model_readers.model_rows, intake.make_entries, and read): hours not on the
   lines the model named, hours that don't match how they're written, a likely lost decimal point, a CMID with the
   wrong shape or not in Matter Master, a client the firm doesn't have or that could be more than one of its
   clients, a client that doesn't match its CMID, marked not billable, the model unsure, an unknown timekeeper, no
   date or a date left blank after dated entries, a weekday that doesn't match its date, no hours, over 14 hours in
   a day, and a photo that failed the upload check. On free-text inputs, an amount of hours that no entry uses flags
   every entry from that piece (unused_hours, total and quoted lines left out); on spreadsheets, a row no entry
   cites does the same (rows_unread).
6. When the model can't answer, the rules read the upload instead (extract.extract) and every entry gets
   read_by_rules, so a person looks.
7. review.add_review then splits the reasons into review and note.

Answers are saved as JSON (the lines shown, the answer, tokens, and time), so each piece is paid for once and every
entry can be traced back to the lines it came from.
"""
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as NoAnswerYet
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rapidfuzz import fuzz, process

from extract import TEXT_TYPES, extract
from free_text import WEEKDAY_NUMBER, is_date_line, original_sender, page_amounts, page_order, weekday_mismatch
from intake import ENTRY_COLS, FORMAT_FOR_INPUT, add_day_check, add_reason, entry_ids, find_timekeeper, make_entries, \
    name_parts, to_date
from model_first import drop_reasons
import prompts
import prompts_v6
from column_map import ROLES as COLUMN_ROLES, SAMPLE_ROWS, ask_columns, cell_text, column_text, prompt_hash as column_prompt_hash, roles_of, \
    row_entry, usable as usable_map
from model_readers import (NOVA_MODELS, ask_nova, blank, cross_check, entry_schema, nova_rows, page_text_lines, saved,
                           system_prompt)
from spreadsheets import find_header, role_of_header
from textract import page_lines
from upload_check import add_photo_check, check_photo

FREE_TEXT = {"NOTE", "SHEET", "SCREENSHOT", "EMAIL", "TEXT", "VOICE"}   # inputs checked for hours no entry uses
NAMES_TIMEKEEPERS = {"SHEET", "PDF", "DOC", "SPREADSHEET"}              # inputs that can name a timekeeper per entry
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TOTAL_LINE = re.compile(r"\b(sub)?totals?\b", re.I)
CLOCK_RANGE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*[-\u2013to]+\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$",
                         re.I)
CLOCK_SPAN = re.compile(r"(?<![\d/.])(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*[-\u2013]\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
                        r"(?![\d/])", re.I)
CMID_DIGITS = re.compile(r"\D{0,2}(\d{4})\s*[.\-_,/ ]?\s*(\d{3})\D{0,2}")
CHECKS_A_CONVERSION_EXPLAINS = ("hours_mismatch", "hours_10x")
CLIENT_KNOWN = 80        # rapidfuzz WRatio to one of the firm's clients or aliases for a client to count as known
AGREE_CLEARS = ("model_unsure",)   # reasons dropped when a second, different model reads the entry the same way
MAVERICK = "us.meta.llama4-maverick-17b-instruct-v1:0"   # the second reader until Nov 20, 2026; Legacy, end of life
                                                        # Mar 30, 2027 (its answers are kept outside the repository)
SECOND_MODEL = "qwen.qwen3-vl-235b-a22b"   # the second reader since Nov 20: Qwen3 VL, a different family from Nova
MAX_LINES = 150          # a typed input with more lines is read in pieces (without the long_pieces fix)
MAX_AMOUNTS = 25         # with long_pieces: amounts of hours per piece of a typed input, so about 25 entries, far
                         # under the answer limit (model_readers.MAX_OUTPUT_TOKENS)
CARRIED_LINES = 3        # with long_pieces: lines just above a cut repeated at the top of the next piece
SPREADSHEET_ROWS = 25    # data rows per piece of a spreadsheet. An entry takes about 110 output tokens (measured on the
                         # saved answers), so 50 rows would need about 5,500 against ask_nova's 4,000-token limit
CONTEXT_LINES = 5        # lines from the start of a document repeated at the top of each later piece
COLUMN_GAP = 12          # points between two words of a PDF line that mark separate cells
PAGE_DPI = 150           # for a scanned PDF page shown to the model

# Checks and clean-ups that can be turned off one at a time (ModelReader.fixes); all are on by default:
# - dates: the date as the model wrote it is worked out in code (resolve_date): "today", "yesterday", and a day name
#   count back from the upload date (date_relative, a note); "Week of: 11/2" gives that date (date_guessed, a note);
#   "Tues 10/27" is read; a date with no year that would fall over FUTURE_DAYS after the upload is last year's
#   (December work sent in January); and a weekday that doesn't match its date, in the model's answer or in a line it
#   cites, goes to review (date_weekday, which only the rule readers ran before).
# - date_gap: an entry with no date after a dated entry in the same upload goes to review (it used to take the
#   upload date with nothing shown).
# - client_gap: an entry with no client and no CMID after an entry that has one, in the same upload, goes to review:
#   it is usually more work for the client above (a ditto mark, an arrow, or nothing at all), and filling that in
#   unseen would be a guess.
# - date_agree: the two readers must also agree on the work date (model_readers.cross_check).
# - client_cmid: a client written next to a CMID that belongs to a different client (client_cmid_mismatch).
# - client_ambiguous: a client that is the start of more than one of the firm's clients, with no CMID to settle it.
# - rows: a spreadsheet data row that no entry cites flags the entries from that piece (rows_unread), the way
#   unused_hours does for free text.
# - units: a spreadsheet number divided by 10 or 60 only counts as explained when the header names the unit.
# - empty: no entries from text that has amounts in it is treated as no answer, so the rules read the upload.
# - cut_off: an answer that stopped at the token limit flags its entries (answer_cut_off).
# - one_reader: when the second reader gave no answer, the entries say so (a note) instead of looking cross-checked.
NOV15_FIXES = ("dates", "date_gap", "client_gap", "date_agree", "client_cmid", "client_ambiguous", "rows", "units", "empty",
               "cut_off", "one_reader")
# Added Nov 20 (NOV20_FIXES), also on by default:
# - long_pieces: emails, typed notes, transcripts, PDFs, and Word files are read in pieces of at most MAX_AMOUNTS
#   amounts of hours instead of 150 lines, since a 150-line report can hold more entries than one answer fits. Each
#   later piece starts with context lines (the document's first lines, the last date line and CMID heading above it,
#   and the lines just above it); the model is told to record no entries from them, and any entry that cites only
#   them is dropped. An input with MAX_AMOUNTS amounts or fewer is one piece, as before.
# - piece_fallback: when a reader fails on a piece (a timeout, an error, no answer), that piece is taken from the other
#   reader with one_reader (a note), and the rules read only the pieces both readers failed on, with read_by_rules.
#   Before, one failed piece sent the whole upload to the rules, and the second reader wasn't used.
# - parallel_pieces: a reader asks for up to PIECE_WORKERS pieces of an upload at the same time, instead of one after
#   another. The answers are the same; only the wait changes.
# - stamp: each entry records the model that read it (model_id) and a fingerprint of the prompt it was given
#   (prompt_hash), so a change in results can be traced to a model or prompt change. Rules entries say "rules".
# - matter_narrative: an entry with no narrative whose "matter" isn't one of the firm's matter names had its description
#   put in the wrong field ("Bluefin Robotics - equity plan grant tracker update 1.2" read as client and matter). The text
#   moves to the narrative (narrative_from_matter, a note). A short form of a matter name ("Series B" for "Series B
#   Financing") stays a matter. Without the firm's matter names (ModelReader.matters), nothing moves.
# - no_narrative: an entry with hours but still no narrative goes to review; it can't be billed without one (firm time,
#   which no client is billed for, is left out: "Firm admin .3" has the matter Admin and nothing more). On the
#   samples it reviews 2 entries the key counts right but that have no description (note_08 "1:15 Ironwood", note_18
#   Halverson .8), so right with no flag goes from 197 to 195. Turned on by Ashley on Nov 20; the pass rule moved to 195.
# - hours_reused: more entries use an amount than it is written on their lines ("Meridian and Kestrel Foods, 2.5 hours
#   total" read as 2.5 for each, so 5 hours billed for 2.5): those entries go to review. Two readers can agree on it.
# - column_map: a spreadsheet with a header row is read from a column map instead of row by row. Each reader is shown
#   the title and header rows and the first rows, column by column, and gives each column a role (column_map.py);
#   code then reads every data row, so no row is dropped or reworded, and an export takes one call per reader instead
#   of one per SPREADSHEET_ROWS rows. A map with no numeric time column isn't used. When the two maps read a row
#   differently, or a reader is unsure of the time column or the client, that row gets columns_unsure. With no header
#   row, or no usable map from either reader, the model reads the rows as before. On the samples (Nov 20) both readers
#   mapped all 5 spreadsheets the same way and the results matched reading row by row, with about 70% fewer tokens.
# - line_order: a reader's entries are put in the order of the lines they cite (their last line), before the checks
#   and the cross-check, which pair the two readers' entries in order. Nova once listed "Firm admin .3" (line 7) before
#   "Ember Grid ... 2.2" (line 6), so each was compared with the wrong entry of the other reader.
# - prompt_cache: Nova's system prompt is marked for Bedrock's prompt cache (model_readers.ask_nova), so the pieces
#   of an upload, and uploads within 5 minutes of each other, pay about a quarter of the input price for it. Qwen3 VL
#   has no prompt cache on Bedrock, and a prompt under CACHE_MIN_CHARS isn't marked. On the samples (Nov 20) Nova's
#   typed inputs went from 380,948 input tokens at full price to 27,012 at full price and 283,106 from the cache.
#   Nova Pro answers identical requests differently now and then, cache or not (4 of 10 on v4, 1 of 5 on v6), so a
#   few entries' difference between two Nova runs is noise.
# - narrative_agree: the two readers must also agree on the narrative (model_readers.same_narrative, at
#   NARRATIVE_AGREE), or the entry goes to review as readers_disagree. On the samples it flags exactly the 2 entries
#   whose narrative was wrong with nothing else flagged (Nova split the description of note_03 and cut note_12's short;
#   Qwen3 VL read both right), so silent narrative errors go from 2 to 0 and right with no flag from 198 to 196.
# - writer: a note photo, memo, or text that says whose it is gets that timekeeper (note_writer) when the upload has
#   none: a name or initials on a line of their own at the top or bottom of the photo ("C. Dunn", "- B. Quinn",
#   "FC 9/22"), or a memo that opens "this is Sato". The readers aren't asked for a timekeeper on these inputs, since
#   a note names other people too ("call w/ O. Grant"); this reads only the signature, with code, and only names on
#   the firm's list. An entry a reader made of the signature alone (no hours, nothing but the name) is dropped, its
#   matter number going to the entry before it (drop_signature). Added by Ashley's request on Nov 20 after the app's
#   sample notes asked "Whose time is this?" on all three photos and the memo. The extraction test is unchanged with it
#   (every test upload comes with its timekeeper, so it never runs there); on the app's samples it named C. Dunn,
#   F. Castillo and H. Sato.
# - header_roles: a column a reader's column map leaves out ("ignore") gets the role its header names, when the header
#   is a known one ("CMID", "Matter No."; spreadsheets.role_of_header). On the app's samples (Nov 20) Nova left out the
#   CMID column of the Excel sheet and the CSV, whose first rows hold blanks and typos ("Zgbo.O03", "9O2B.003"), while
#   Qwen3 VL read it, so every row with a CMID was read differently and got columns_unsure: 34 of 54 Excel rows and 3
#   of 6 CSV rows; with it, none. The extraction test is unchanged with it (Nov 20, run by Ashley: 196 right with no
#   flag, 87 to review, 3 silent errors), since both readers map the test's spreadsheets the same way. On by default since.
NOV20_FIXES = ("long_pieces", "piece_fallback", "parallel_pieces", "stamp", "matter_narrative", "no_narrative",
               "hours_reused", "column_map", "line_order", "prompt_cache", "narrative_agree", "writer", "header_roles")
# Off by default (OPTIONAL_FIXES):
# - note_narrative: both photo prompts (model_readers.system_prompt for Nova, open_models.system_prompt_v2 for Qwen3 VL)
#   get model_readers.NARRATIVE_RULE: every word of the description through the lines it wraps onto, up to the next
#   entry, and a check that none is cut off, left over, or given to the entry below. For the 2 silent narrative errors
#   left on note photos (note_03, note_12). Needs new answers for every note photo.
#   Measured Nov 20 (01_evaluation, NOTE_NARRATIVE, new answers in data/model_answers/note_narrative): 185 right with no
#   flag against 196, 99 to review against 87, the same 3 silent errors. Most of the extra review is on our own note
#   photos (17 to 27). Left off.
# - client_shortlist: every input but note photos gets the SHORTLIST firm names closest to its own text instead of the
#   whole client list. A real firm has thousands of clients; the list is in every call. Note photos keep the whole
#   list, since the model uses it to read handwritten names. Measured Nov 20 with Qwen3 VL, which answers identical
#   requests the same way: 5 of 110 pieces read differently (2 spellings closer to the note, 2 dates filled in, an
#   entry with no hours dropped) for about 6% fewer input tokens at 76 names. Worth trying again with a real firm's
#   client list.
OPTIONAL_FIXES = ("note_narrative", "client_shortlist")
PROMPTS = {"v4": prompts, "v6": prompts_v6}   # ModelReader.prompt: what every input but note photos is told
MATTER_KNOWN = 85        # rapidfuzz ratio to one of the firm's matter names for a "matter" to count as a real matter
SHORTLIST = 20           # firm names sent with each piece, with client_shortlist
CACHE_MIN_CHARS = 4000   # a system prompt this long (about 1,000 tokens) can be cached
PIECE_WORKERS = 4        # pieces of one upload asked at the same time, per reader (the app also caps calls overall)
FUTURE_DAYS = 60         # a date with no year this far after the upload is read as the year before
RELATIVE_DAYS = {"today": 0, "tonight": 0, "this morning": 0, "this afternoon": 0, "this evening": 0,
                 "yesterday": -1}
DAY_NAME = re.compile(r"^\s*(last\s+|this\s+|on\s+)?(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\s*$", re.I)
DAY_PREFIX = re.compile(r"^\s*(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+(?=\S*\d)", re.I)
WEEK_OF = re.compile(r"^\s*(?:week\s+(?:of|ending|beginning|starting)|w/[eo])\s*:?\s*(?=\S)", re.I)
WRITTEN_YEAR = re.compile(r"\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2}\b")
CMID_HEADING = re.compile(r"^\s*\d{4}\.\d{3}\b")
CONTEXT_NOTE = ("Lines L1 to L{n} repeat earlier lines of the document (its first lines and the headings above this "
                "part) for context only: use them for dates, clients, and headings, but record no entries from them.")
TENTHS_HEADER = re.compile(r"unit|tenth|1/10|\b0?\.1\b", re.I)
MINUTES_HEADER = re.compile(r"\bmin", re.I)

INTROS = {
    "SHEET": ("Lines read from a scanned time sheet: a printed form filled in, often by hand. Each line shows its id, "
              "where it starts on the page (percent across and down), the reader's confidence (0 to 100), and its "
              "text. The form's printed headings say what each column holds; they are not entries."),
    "SCREENSHOT": ("Lines read from a screenshot of an email or message about time worked, each with its id, where "
                   "it starts on the screen (percent across and down), the reader's confidence, and its text. "
                   "Headers, signatures, and quoted earlier messages are not entries."),
    "EMAIL": ("Lines of an email to a law firm's billing assistant about time worked, each with its id. Greetings, "
              "signatures, contact details, and quoted earlier messages are not entries."),
    "TEXT": ("Lines of a typed time note or text messages to a billing assistant, each with its id. Messages that "
             "aren't about time worked are not entries."),
    "VOICE": ("Lines of an automatic transcript of a voice memo in which a lawyer describes time worked, each with "
              "its id. The transcript can mishear names and run numbers together; a CMID is 4 digits, a dot, and "
              "3 digits."),
    "PDF": ("Lines of a typed document (a time report, pre-bill, or time log), each with its id, in reading order. "
            "Cells of a table row are separated by \" | \". Headings, page numbers, rates, amounts billed, and totals "
            "are not entries."),
    "DOC": ("Lines of a typed document (a time log or report), each with its id. Cells of a table row are separated "
            "by \" | \". Headings, rates, amounts billed, and totals are not entries."),
    "SPREADSHEET": ("Rows of a spreadsheet of time entries, each with its id, cells separated by \" | \" in column "
                    "order. The first rows shown are the title and header. Each data row is usually one entry. Use "
                    "the header to tell hours from minutes or tenths of an hour, and say in \"unsure\" when it isn't "
                    "clear. Total rows are not entries."),
}


@dataclass
class ModelReader:
    """How one_flow asks the model, and where its answers are saved.

    runtime: a bedrock-runtime client, or None to use saved answers only (a missing answer then falls back to the
        rules, with read_by_rules).
    answers_dir: where answers are saved, one JSON per piece, named after the upload's source.
    answer_path: optional function (source, piece) -> Path, to reuse answers saved elsewhere.
    prompt: "v6" (prompts_v6.py for every input but note photos, since Nov 20), "v4" (prompts.py, before), or "v1"
        (the photo prompt for every input).
    new_checks: the clean-ups and checks described in read; False turns them off.
    force_tool, open_note_prompt: for a model that can't be forced to call a tool (Llama 4 Maverick): offer the tool,
        and read note photos with open_models.ask_model's photo prompt.
    second: another ModelReader (a different model) that reads every piece too; entries the two read differently go
        to review, and entries they read the same way lose AGREE_CLEARS.
    catch_errors: turn a failed call (a timeout, a dropped connection) into an answer with an "error", which isn't
        saved, so the upload goes on without this reader and the next run asks again. For the second model, whose
        failure only means no second reading.
    fixes: the checks named in NOV15_FIXES and NOV20_FIXES to apply; () turns them all off, and NOV15_FIXES alone
        reproduces the Nov 15 reader.
    client_of, aliases_of: {CMID: client} and {client: [aliases]} from Matter Master (firm_lookup), for the
        client_cmid and client_ambiguous checks; without them those two checks don't run.
    matters: the firm's matter names (firm_lookup), for matter_narrative.
    """
    clients: list
    cmids: set
    runtime: object = None
    answers_dir: Path = Path("../data/model_answers/one_flow")
    model_id: str = NOVA_MODELS["Nova Pro"]
    answer_path: object = None
    images_dir: Path = field(default=None)
    prompt: str = "v6"
    new_checks: bool = True
    force_tool: bool = True
    open_note_prompt: bool = False
    second: object = None
    catch_errors: bool = False
    fixes: tuple = NOV15_FIXES + NOV20_FIXES
    client_of: dict = None
    aliases_of: dict = None
    matters: list = None
    made: dict = field(default_factory=dict)   # answers made in code (a spreadsheet read from its column map), by
                                               # source, for source_lines

    def path(self, source, piece):
        if self.answer_path is not None:
            found = self.answer_path(source, piece)
            if found is not None:
                return Path(found)
        stem = re.sub(r"[^\w.-]+", "_", Path(str(source)).name)
        if isinstance(piece, str):
            return Path(self.answers_dir) / f"{stem}_{piece}.json"
        return Path(self.answers_dir) / (f"{stem}.json" if piece is None else f"{stem}_p{piece + 1}.json")

    def answer(self, piece, source, number):
        if not self.catch_errors:
            return self.ask(piece, source, number)
        try:
            return self.ask(piece, source, number)
        except Exception as error:   # botocore's timeouts and connection errors
            return {"model": self.model_id, "entries": [], "lines": piece["lines"], "stop_reason": "call_failed",
                    "error": f"{type(error).__name__}: {error}"[:300], "usage": {"inputTokens": 0, "outputTokens": 0},
                    "latency_ms": 0}

    def prompt_hash(self, piece):
        """A short fingerprint of what this reader tells the model for a piece: its system prompt and schema."""
        if piece.get("columns"):
            return column_prompt_hash()
        kind, tk = piece["input_type"], piece["with_timekeeper"]
        narrative = "note_narrative" in (self.fixes or ())
        if kind == "NOTE" and self.open_note_prompt:
            from open_models import system_prompt_v2
            text = system_prompt_v2(piece["photo"] is not None, False, "tool", narrative)
        elif self.prompt in PROMPTS and kind != "NOTE":
            version = PROMPTS[self.prompt]
            text = version.system_prompt(kind, tk) + json.dumps(version.entry_schema(tk), sort_keys=True)
        else:
            text = system_prompt(tk, narrative) + json.dumps(entry_schema(tk), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:10]

    def ask(self, piece, source, number):
        path = self.path(source, number)
        if not path.exists() and self.runtime is None:
            return None
        if piece.get("columns"):
            return saved(path, lambda: {**ask_columns(self.runtime, piece["text"], piece["lines"], self.model_id,
                                                      self.force_tool), "prompt_hash": column_prompt_hash()})
        kind, tk = piece["input_type"], piece["with_timekeeper"]
        if kind == "NOTE" and self.open_note_prompt:
            from open_models import ask_model
            ask = lambda: {**ask_model(self.runtime, piece["lines"], self.clients, self.model_id, photo=piece["photo"],
                                       prompt="v2", narrative="note_narrative" in (self.fixes or ())),
                           "prompt_hash": self.prompt_hash(piece)}
            return saved(path, ask)
        version = PROMPTS.get(self.prompt) if kind != "NOTE" else None
        intro = version.intro(kind) if version else piece["intro"]
        if piece.get("context"):
            intro = f"{intro} {CONTEXT_NOTE.format(n=piece['context'])}"
        fixes = set(self.fixes or ())
        system = (version.system_prompt(kind, tk) if version
                  else system_prompt(tk, True) if "note_narrative" in fixes else None)   # None: ask_nova's photo prompt
        clients = (client_shortlist(self.clients, piece["lines"]) if "client_shortlist" in fixes and kind != "NOTE"
                   else self.clients)
        cache = ("prompt_cache" in fixes and self.model_id in NOVA_MODELS.values()
                 and len(system or system_prompt(tk)) >= CACHE_MIN_CHARS)
        ask = lambda: {**ask_nova(self.runtime, piece["lines"], clients, self.model_id, photo=piece["photo"],
                                  intro=intro, with_timekeeper=tk, system=system,
                                  schema=version.entry_schema(tk) if version else None, force_tool=self.force_tool,
                                  cache=cache),
                       "prompt_hash": self.prompt_hash(piece)}
        return saved(path, ask)


def client_shortlist(clients, lines, n=SHORTLIST):
    """The n firm names (clients and aliases) closest to a piece's text: each name's best partial match to one line."""
    texts = [str(line["text"]).lower() for line in lines] or [""]
    score = {name: max(fuzz.partial_ratio(str(name).lower(), text) for text in texts) for name in clients}
    return sorted(clients, key=lambda name: -score[name])[:n]


def numbered(texts, start=1):
    return [{"id": n, "text": t} for n, t in enumerate(texts, start)]


def all_text_lines(page):
    """Every line on a Textract page in reading order (printed and handwritten), with position and confidence, as
    model_readers.page_text_lines gives handwritten lines."""
    lines = page_order(page_lines(page, drop_repeats=True))
    return [{"id": n, "across": round(r.left * 100), "down": round(r.top * 100), "conf": round(r.conf),
             "number_conf": None if pd.isna(r.number_conf) else round(r.number_conf), "text": r.text}
            for n, r in enumerate(lines.itertuples(), 1)]


def pdf_text_lines(path):
    """A typed PDF's lines, page by page: words left to right, with " | " where the gap between two words is wider
    than COLUMN_GAP points (a table cell boundary)."""
    import pymupdf
    from documents import pdf_lines
    pages = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            out = []
            for _, words in pdf_lines(page):
                text = words[0][2]
                for (_, right, _), (left, _, word) in zip(words, words[1:]):
                    text += (" | " if left - right > COLUMN_GAP else " ") + word
                out.append(text)
            pages.append(out)
    return pages


def docx_text_lines(path):
    """A Word file's paragraphs and table rows in document order, cells joined by " | "."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document = Document(path)
    out = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, document).text.strip()
            if text:
                out.append(text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    out.append(" | ".join(cells))
    return out


def document_text_lines(path):
    """[page lines] for a PDF with a text layer, a .docx, or a .txt file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_text_lines(path)
    if path.suffix.lower() == ".docx":
        return [docx_text_lines(path)]
    return [[line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]]


def spreadsheet_texts(raw):
    """Each row of a spreadsheet's raw cells as text, cells joined by " | "."""
    return [" | ".join(cell_text(v) for v in row) for row in raw.itertuples(index=False)]


def spreadsheet_indexes(raw):
    """(title and header row numbers, data row numbers) of a spreadsheet's raw cells, leaving out empty rows."""
    texts = spreadsheet_texts(raw)
    keep = [i for i, t in enumerate(texts) if t.replace("|", "").strip()]
    try:
        header = find_header(raw)
    except ValueError:
        header = keep[0] if keep else 0
    return [i for i in keep if i <= header], [i for i in keep if i > header]


def spreadsheet_rows(raw):
    """(title and header rows, data rows) of a spreadsheet's raw cells, as text with cells joined by " | "."""
    texts = spreadsheet_texts(raw)
    top, body = spreadsheet_indexes(raw)
    return [texts[i] for i in top], [texts[i] for i in body]


def spreadsheet_piece_raw(raw, n, rows_per_piece=SPREADSHEET_ROWS):
    """The raw cells behind spreadsheet piece n (spreadsheet_pieces): the title and header rows, then its data rows."""
    top, body = spreadsheet_indexes(raw)
    return raw.iloc[top + body[n * rows_per_piece:(n + 1) * rows_per_piece]].reset_index(drop=True)


def spreadsheet_pieces(raw, rows_per_piece=SPREADSHEET_ROWS):
    """[lines] for a spreadsheet's raw cells: the title and header rows, then up to rows_per_piece data rows."""
    top, body = spreadsheet_rows(raw)
    return [top + body[i:i + rows_per_piece] for i in range(0, max(len(body), 1), rows_per_piece)]


def column_piece(raw):
    """What a reader is asked for a spreadsheet's column map (column_map.py), or None when no header row is found."""
    try:
        find_header(raw)
    except ValueError:
        return None
    top, body = spreadsheet_indexes(raw)
    texts = spreadsheet_texts(raw)
    lines = numbered([texts[i] for i in top + body[:SAMPLE_ROWS]])
    return {"columns": True, "text": column_text(raw, top, body), "lines": lines, "photo": None,
            "input_type": "SPREADSHEET", "with_timekeeper": True, "intro": None}


def spreadsheet_calls(raw, fixes):
    """Model calls per reader for a spreadsheet: one for its column map (column_map), or one per piece."""
    return 1 if "column_map" in fixes and column_piece(raw) is not None else len(spreadsheet_pieces(raw))


def mapped_answers(raw, parts, answer, model):
    """An answer for each spreadsheet piece made from a column map: one entry per data row (column_map.row_entry)."""
    _, body = spreadsheet_indexes(raw)
    roles = roles_of(answer, raw.shape[1])
    out = []
    for n, part in enumerate(parts):
        rows = body[n * SPREADSHEET_ROWS:(n + 1) * SPREADSHEET_ROWS]
        entries = [row_entry(list(raw.iloc[i]), roles, part["data_start"] + k) for k, i in enumerate(rows)]
        out.append({"model": answer.get("model") or model.model_id, "entries": [e for e in entries if e is not None],
                    "lines": part["lines"], "stop_reason": "column_map", "map": answer.get("map"),
                    "used_prompt": answer.get("prompt_hash") or column_prompt_hash()})
    return out


def header_roles(answer, raw):
    """A column map answer with each column it left out given the role its header names, if the header is a known one
    (the header_roles fix)."""
    if not answer or "error" in answer or not isinstance(answer.get("map"), dict):
        return answer
    try:
        header = raw.iloc[find_header(raw)]
    except ValueError:
        return answer
    columns = []
    for item in answer["map"].get("columns") or []:
        item = dict(item)
        try:
            n = int(item.get("column")) - 1
        except (TypeError, ValueError):
            n = -1
        if item.get("role") == "ignore" and 0 <= n < len(header):
            role = role_of_header(cell_text(header.iloc[n]))[0]
            role = "timekeeper" if role == "tk" else role
            if role in COLUMN_ROLES and role != "ignore":
                item["role"] = role
        columns.append(item)
    return {**answer, "map": {**answer["map"], "columns": columns}}


def mark_unsure(made, unsure):
    """columns_unsure on every entry when a reader was unsure, else on the rows the two readers read differently."""
    read = [answers for answers in made if answers[0] is not None]
    by_line = [{e["lines"][0]: json.dumps(e, sort_keys=True) for piece in answers for e in piece["entries"]}
               for answers in read]
    differ = {line for line in set().union(*by_line) if len({found.get(line) for found in by_line}) > 1}
    for answers in read:
        for piece in answers:
            for entry in piece["entries"]:
                entry["columns_unsure"] = unsure or entry["lines"][0] in differ


def by_columns(raw, parts, first, second, source, deadline=None):
    """(first reader's answers, second's) for a spreadsheet read from column maps (the column_map fix), or None when
    there is no header row or neither reader gave a map with a numeric time column, so the model reads the rows as
    before. A reader without a usable map gets None for every piece (piece_fallback then takes the other's)."""
    piece = column_piece(raw)
    if piece is None:
        return None
    readers = [first] + ([second] if second is not None else [])
    pool = ThreadPoolExecutor(max_workers=len(readers))
    jobs = [pool.submit(model.answer, piece, source, "columns") for model in readers]
    answers = [wait_for(job, deadline) for job in jobs]
    pool.shutdown(wait=False, cancel_futures=True)
    if "header_roles" in (first.fixes or ()):
        answers = [header_roles(a, raw) for a in answers]
    _, body = spreadsheet_indexes(raw)
    good = [a if a and "error" not in a and usable_map(roles_of(a, raw.shape[1]), raw, body) else None for a in answers]
    if all(a is None for a in good):
        return None
    made = [mapped_answers(raw, parts, a, model) if a else [None] * len(parts) for a, model in zip(good, readers)]
    mark_unsure(made, any(str(a["map"].get("unsure") or "").strip() for a in good if a))
    for model, answers in zip(readers, made):
        if answers[0] is not None:
            model.made[source] = answers
    return made[0], made[1] if second is not None else None


def split_long(texts, max_lines=MAX_LINES):
    """A long typed input in pieces of at most max_lines, each later piece starting with the first CONTEXT_LINES."""
    if len(texts) <= max_lines:
        return [texts]
    head, step = texts[:CONTEXT_LINES], max_lines - CONTEXT_LINES
    return [texts[:max_lines]] + [head + texts[i:i + step] for i in range(max_lines, len(texts), step)]


def line_amounts(text):
    """How many amounts of hours a line holds. A clock range counts once; a total or quoted line counts none."""
    if TOTAL_LINE.search(text) or text.lstrip().startswith(">"):
        return 0
    return len(page_amounts([clock_ranges_as_hours(text)]))


def best_cut(counts, start, end):
    """Where to end a piece that would go past MAX_AMOUNTS at line `end`: between two lines that both have amounts, in
    the second half of the piece, so no heading or wrapped line is parted from its entry; else just before `end`."""
    for i in range(end, max(start + (end - start) // 2, start + 1) - 1, -1):
        if counts[i] and counts[i - 1]:
            return i
    return end


def amount_cuts(counts, max_amounts=MAX_AMOUNTS):
    """The line numbers where each new piece starts, given how many amounts each line holds."""
    cuts, start, total = [], 0, 0
    for i, n in enumerate(counts):
        if total and total + n > max_amounts:
            start = best_cut(counts, start, i)
            cuts.append(start)
            total = sum(counts[start:i])
        total += n
    return cuts


def context_lines(texts, counts, cut):
    """The lines repeated at the top of the piece that starts at `cut`: the document's first CONTEXT_LINES, the last
    date line and the last CMID heading above the cut, and the CARRIED_LINES just above it."""
    keep = set(range(min(CONTEXT_LINES, cut))) | set(range(max(cut - CARRIED_LINES, 0), cut))
    dates = [i for i in range(cut) if is_date_line(texts[i])]
    headings = [i for i in range(cut) if not counts[i] and CMID_HEADING.match(texts[i])]
    return [texts[i] for i in sorted(keep | set(dates[-1:]) | set(headings[-1:]))]


def amount_pieces(texts, max_amounts=MAX_AMOUNTS):
    """A typed input as [(context lines, own lines)], each piece holding at most max_amounts amounts of hours. One
    piece, with no context, when the whole input has max_amounts or fewer."""
    counts = [line_amounts(t) for t in texts]
    cuts = amount_cuts(counts, max_amounts)
    return [(context_lines(texts, counts, start) if start else [], texts[start:end])
            for start, end in zip([0] + cuts, cuts + [len(texts)])]


def pdf_page_image(path, page, images_dir):
    """A scanned PDF's page as a PNG (saved once), for the model to see."""
    import pymupdf
    out = Path(images_dir) / f"{Path(path).stem}_p{page + 1}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(path) as pdf:
            pdf[page].get_pixmap(dpi=PAGE_DPI).save(str(out))
    return out


def pieces(content, input_type, image=None, images_dir=None, by_amounts=False):
    """What the model reads for one upload: a list of dicts with lines, intro, photo, and with_timekeeper, and with
    by_amounts (the long_pieces fix) "context": how many lines at the top repeat earlier ones.

    content is what extract takes: Textract's pages for NOTE, SHEET, and SCREENSHOT; the text for EMAIL, TEXT, and
    VOICE; the file path for PDF and DOC; the raw cells for SPREADSHEET. image: the photo, screenshot, or scanned
    file the Textract pages came from, when there is one.
    """
    names_tk = input_type in NAMES_TIMEKEEPERS
    piece = lambda lines, photo=None: {"lines": lines, "intro": INTROS.get(input_type), "photo": photo,
                                       "with_timekeeper": names_tk, "input_type": input_type}
    with_context = lambda context, own: {**piece(numbered(context + own)), "context": len(context)}
    if input_type in ("NOTE", "SHEET", "SCREENSHOT"):
        out = []
        for n, page in enumerate(content):
            lines = all_text_lines(page) if input_type == "SHEET" else page_text_lines(page)
            photo = None
            if image is not None and Path(image).suffix.lower() in IMAGE_SUFFIXES and n == 0:
                photo = Path(image)
            elif image is not None and Path(image).suffix.lower() == ".pdf" and images_dir is not None:
                photo = pdf_page_image(image, n, images_dir)
            out.append(piece(lines, photo))
        return out
    if input_type in ("EMAIL", "TEXT", "VOICE"):
        texts = [line.strip() for line in str(content).splitlines() if line.strip()]
        if by_amounts:
            return [with_context(context, own) for context, own in amount_pieces(texts)]
        return [piece(numbered(t)) for t in split_long(texts)]
    if input_type in ("PDF", "DOC"):
        pages = document_text_lines(content)
        texts = [t for page in pages for t in page]
        if by_amounts:
            return [with_context(context, own) for context, own in amount_pieces(texts)]
        if len(texts) <= MAX_LINES:
            return [piece(numbered(texts))]
        head = texts[:CONTEXT_LINES]
        return [piece(numbered(page if n == 0 else head + page)) for n, page in enumerate(pages) if page]
    if input_type == "SPREADSHEET":
        data_start = len(spreadsheet_rows(content)[0]) + 1   # the id of the first data row in every piece
        return [{**piece(numbered(lines)), "data_start": data_start} for lines in spreadsheet_pieces(content)]
    raise NotImplementedError(f"{input_type} isn't built yet")


def clock_range_hours(written):
    """How long a clock range lasts ("2-3:30pm" is 1.5, "11:30am-1pm" is 1.5), or None if it isn't one."""
    found = CLOCK_RANGE.match(str(written or ""))
    if not found:
        return None
    h1, m1, s1, h2, m2, s2 = found.groups()
    start, end = int(h1) + int(m1 or 0) / 60, int(h2) + int(m2 or 0) / 60
    if s2 and s2.lower() == "pm" and int(h2) < 12:
        end += 12
    if s1:
        start += 12 if s1.lower() == "pm" and int(h1) < 12 else 0
    elif s2 and s2.lower() == "pm" and start + 12 <= end:
        start += 12
    length = end - start if end > start else end + 12 - start
    return round(length, 2) if 0 < length <= 12 else None


def conversion_explains(written, hours, input_type, header=None):
    """True when the difference between the hours as written and the model's hours is a conversion the text explains:
    a clock range's length, or a spreadsheet's bare number in tenths or minutes (the column header says which).

    header: the spreadsheet's title and header rows as text. When given (the "units" fix), dividing by 10 only counts
    as explained if the header names units or tenths, and dividing by 60 only if it names minutes, so an "Hours"
    column the model divided anyway keeps its flag. None: any division by 10 or 60 counts."""
    if hours is None or pd.isna(hours):
        return False
    span = clock_range_hours(written)
    if span is not None:
        return abs(span - hours) < 0.01
    bare = re.fullmatch(r"\s*(\d+)\s*", str(written or ""))
    if input_type != "SPREADSHEET" or not bare:
        return False
    allowed = [d for d, names in ((10, TENTHS_HEADER), (60, MINUTES_HEADER)) if header is None or names.search(header)]
    return any(abs(int(bare.group(1)) / d - hours) < 0.01 for d in allowed)


def drop_explained(row, input_type, header=None):
    if conversion_explains(row.get("hours_read"), row.get("hours"), input_type, header):
        row["check_reasons"] = "; ".join(r for r in str(row["check_reasons"] or "").split("; ")
                                         if r and r not in CHECKS_A_CONVERSION_EXPLAINS)
    return row


def clean_cmid(value):
    """(cmid, matter, fixed) for what a model put in "cmid": a matter name with no digits is a matter, not a CMID
    ("Fund IV Formation"); 4 and 3 digits with a dash, space, or nothing between them are a CMID written another way
    ("4590-003", "7222001"), fixed and noted (cmid_fixed). Anything else is left for the CMID checks."""
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    if not text:
        return None, None, False
    if not re.search(r"\d", text):
        return None, text, False
    found = CMID_DIGITS.fullmatch(text)
    if found:
        cmid = f"{found.group(1)}.{found.group(2)}"
        return cmid, None, cmid != text
    return text, None, False


def clean_entries(entries):
    """A model's entries with clean_cmid applied: (entries, [whether each CMID was fixed])."""
    out, fixed = [], []
    for entry in entries:
        cmid, matter, was_fixed = clean_cmid(entry.get("cmid"))
        out.append({**entry, "cmid": cmid or "", "matter": entry.get("matter") or matter or ""})
        fixed.append(was_fixed)
    return out, fixed


def clock_ranges_as_hours(text):
    """The text with each clock range ("9:30-11:00", "2-3:30pm") written as its length ("1.5 hours"), so the
    unused-hours check counts a range once instead of reading its two ends as two amounts. A range needs a colon or
    am/pm, so "11/2-11/6" or "2-3 calls" isn't one."""
    def length(found):
        span = found.group(0)
        if ":" not in span and not re.search(r"[ap]m", span, re.I):
            return span
        hours = clock_range_hours(span)
        return f" {hours} hours " if hours else span
    return CLOCK_SPAN.sub(length, text)


def matter_known(matter, matters, min_score=MATTER_KNOWN):
    """True when a matter as written is one of the firm's matter names: every word of it is in one name ("Series B" in
    "Series B Financing"), or it is spelled close to one."""
    words = set(name_words(matter))
    if any(words and words <= set(name_words(name)) for name in matters):
        return True
    best = process.extractOne(str(matter).strip().lower(), matters, scorer=fuzz.ratio)
    return best is not None and best[1] >= min_score


def client_known(client, clients, min_score=CLIENT_KNOWN):
    """True when a client is blank, firm time, or close to one of the firm's clients or aliases."""
    if client is None or (isinstance(client, float) and pd.isna(client)) or not str(client).strip():
        return True
    text = str(client).strip().lower()
    if text.startswith("firm") or not clients:
        return True
    best = process.extractOne(text, [c.lower() for c in clients], scorer=fuzz.WRatio)
    return best is not None and best[1] >= min_score


def add_client_check(entries, model):
    """client_unknown on entries whose client isn't one of the firm's (and whose CMID isn't in Matter Master):
    a client the firm doesn't have, or one misread past recognizing. Matching can't place either, and on the samples it
    billed one to the wrong matter without a flag."""
    if not len(entries):
        return entries
    known_cmid = entries["cmid"].map(lambda c: isinstance(c, str) and c in model.cmids)
    unknown = entries["client"].map(lambda c: not client_known(c, model.clients))
    return add_reason(entries, unknown & ~known_cmid, "client_unknown")


def firm_lookup(matter_master):
    """What a ModelReader needs from Matter Master (columns Client, CMID, Matter, Known Aliases), as its keyword
    arguments: clients (names and aliases, for the prompt), cmids, matters (matter names), client_of ({CMID: client}),
    and aliases_of ({client: [aliases]})."""
    aliases_of = {}
    for client, known in zip(matter_master["Client"], matter_master["Known Aliases"]):
        names = aliases_of.setdefault(client, set())
        if isinstance(known, str):
            names.update(a.strip() for a in known.split(",") if a.strip())
    clients = sorted((set(aliases_of) | {a for names in aliases_of.values() for a in names}) - {""})
    return {"clients": clients, "cmids": set(matter_master["CMID"]),
            "matters": sorted({str(m).lower() for m in matter_master.get("Matter", pd.Series(dtype=str)).dropna()}),
            "client_of": dict(zip(matter_master["CMID"], matter_master["Client"])),
            "aliases_of": {client: sorted(names) for client, names in aliases_of.items()}}


def name_words(name):
    return re.sub(r"[^a-z0-9&]+", " ", str(name).lower()).split()


def client_candidates(written, aliases_of):
    """The firm's clients a written name could mean: the client it names exactly (its name or an alias), or else
    every client whose name or alias starts with it, word for word ("Northgate" for Northgate Bank and Northgate
    Capital Partners). Firm time is left out."""
    words = name_words(written)
    exact, starts = set(), set()
    for client, aliases in (aliases_of or {}).items():
        if not words or str(client).strip().lower() == "firm":
            continue
        for name in [client, *aliases]:
            other = name_words(name)
            if other == words:
                exact.add(client)
            elif other[:len(words)] == words:
                starts.add(client)
    return exact or starts


def add_firm_checks(entries, model, fixes):
    """client_cmid_mismatch: the entry has a CMID in Matter Master and a client that isn't that CMID's client (one of
    the two was misread, or the CMID belongs to the entry next to it). client_ambiguous: no known CMID, and the
    client as written could be more than one of the firm's clients. Both go to review."""
    if not len(entries) or not model.aliases_of:
        return entries
    mismatch, ambiguous = [], []
    for row in entries.itertuples():
        client = "" if blank(row.client) else str(row.client).strip()
        owner = (model.client_of or {}).get(row.cmid) if isinstance(row.cmid, str) else None
        firm = client.lower().startswith("firm")
        candidates = client_candidates(client, model.aliases_of) if client and not firm else set()
        if owner and str(owner).lower() != "firm" and client and not firm:
            names = [owner, *model.aliases_of.get(owner, [])]
            close = process.extractOne(client.lower(), [n.lower() for n in names], scorer=fuzz.WRatio)
            mismatch.append(owner not in candidates and (close is None or close[1] < CLIENT_KNOWN))
        else:
            mismatch.append(False)
        ambiguous.append(owner is None and len(candidates) > 1)
    if "client_cmid" in fixes:
        entries = add_reason(entries, pd.Series(mismatch, index=entries.index), "client_cmid_mismatch")
    if "client_ambiguous" in fixes:
        entries = add_reason(entries, pd.Series(ambiguous, index=entries.index), "client_ambiguous")
    return entries


def resolve_date(written, upload_date, year):
    """(the work date or None, [reasons]) for a date as the model wrote it. None leaves the date to intake.to_date.

    - "today", "yesterday", and a day name alone ("Monday", "last Tues") count back from upload_date: a day name is
      the latest such day up to the upload day (date_relative, a note). With no upload date they stay unread.
    - "Week of: 11/2" is that date (date_guessed, a note), as the table reader does for a sheet's rows.
    - "Tues 10/27" and "Thu 11/19" are read without the weekday; a weekday that doesn't match goes to review
      (date_weekday).
    - A date with no year more than FUTURE_DAYS after the upload is the year before."""
    text = "" if blank(written) else str(written).strip()
    if not text:
        return None, []
    upload = pd.Timestamp(upload_date).normalize() if upload_date is not None and not blank(upload_date) else None
    said = text.lower().strip(" .,:")
    day_name = DAY_NAME.match(text)
    if said in RELATIVE_DAYS or day_name:
        if upload is None:
            return None, []
        if said in RELATIVE_DAYS:
            return upload + pd.Timedelta(days=RELATIVE_DAYS[said]), ["date_relative"]
        back = (upload.weekday() - WEEKDAY_NUMBER[day_name.group(2).lower()]) % 7
        if back == 0 and (day_name.group(1) or "").lower().startswith("last"):
            back = 7
        return upload - pd.Timedelta(days=back), ["date_relative"]
    reasons = []
    week = WEEK_OF.match(text)
    if week:
        text, reasons = text[week.end():].strip(), ["date_guessed"]
    weekday = DAY_PREFIX.match(text)
    rest = text[weekday.end():].strip() if weekday else text
    when = to_date(rest, year)
    if pd.isna(when):
        return None, []
    if upload is not None and not WRITTEN_YEAR.search(rest) and when > upload + pd.Timedelta(days=FUTURE_DAYS):
        when = when - pd.DateOffset(years=1)
    if weekday and when.weekday() != WEEKDAY_NUMBER[weekday.group(1).lower()]:
        reasons.append("date_weekday")
    return when, reasons


def line_ids(entry):
    """The line ids an entry cites, as integers (a model can write "L3" or "3")."""
    ids = entry.get("lines")
    out = []
    for i in ids if isinstance(ids, list) else []:
        try:
            out.append(int(str(i).strip().lstrip("Ll")))
        except ValueError:
            pass
    return out


def unread_rows(part, entries):
    """The ids of a spreadsheet piece's data rows that no entry cites: rows from data_start on with a number and at
    least two filled cells, leaving out total rows."""
    start = part.get("data_start")
    if not start:
        return []
    cited = {i for entry in entries for i in line_ids(entry)}
    return [line["id"] for line in part["lines"]
            if line["id"] >= start and line["id"] not in cited and not TOTAL_LINE.search(line["text"])
            and re.search(r"\d", line["text"]) and sum(bool(c.strip()) for c in line["text"].split("|")) >= 2]


def has_amounts(part, input_type):
    """True when a piece the model found no entries in has something that looks like time: an amount of hours in free
    text (total and quoted lines left out), or a data row in a spreadsheet."""
    if input_type == "SPREADSHEET":
        return bool(unread_rows(part, []))
    if input_type not in FREE_TEXT:
        return False
    texts = [clock_ranges_as_hours(line["text"]) for line in part["lines"] if line["id"] > part.get("context", 0)
             and not TOTAL_LINE.search(line["text"]) and not line["text"].lstrip().startswith(">")]
    return bool(page_amounts(texts))


def is_photo(image, input_type):
    return (image is not None and input_type in ("NOTE", "SHEET")
            and Path(str(image)).suffix.lower() in IMAGE_SUFFIXES and Path(str(image)).exists())


def amount_count(written, text):
    """How many times an amount as written ("2.5", ".4", "1 hr 15 min") appears in the text, not as part of a longer
    number."""
    pattern = r"(?<![\d.])" + re.escape(re.sub(r"\s+", " ", str(written).strip())) + r"(?![\d])"
    return len(re.findall(pattern, re.sub(r"\s+", " ", text), re.I))


def reused_amounts(entries, lines):
    """The positions of entries whose amount is used by more entries than it is written on their cited lines."""
    by_id = {line["id"]: line["text"] for line in lines}
    groups = {}
    for n, entry in enumerate(entries):
        written = str(entry.get("hours_written") or "").strip()
        if written and not blank(written):
            groups.setdefault(written.lower(), []).append(n)
    flagged = []
    for written, members in groups.items():
        cited = sorted({i for n in members for i in line_ids(entries[n]) if i in by_id})
        found = amount_count(written, " \n ".join(by_id[i] for i in cited))
        if len(members) > 1 and 0 < found < len(members):
            flagged += members
    return flagged


def only_context(entry, context):
    """True when an entry cites only the context lines at the top of a piece (it was read in an earlier piece)."""
    ids = line_ids(entry)
    return bool(context and ids and all(i <= context for i in ids))


def combine(first, second, clears=AGREE_CLEARS, dates=False, note_missing=False, narratives=False):
    """First reader's entries checked against a second reader's (model_readers.cross_check): readers_disagree and
    second_reader_only go to review, and entries both read the same way lose the reasons in clears.

    dates: the two readers must agree on the work date too. note_missing: when the second reader gave no answer (no
    entries, or the rules read the upload in its place), the first reader's entries get one_reader, a note."""
    if not len(second) or second["check_reasons"].str.contains("read_by_rules").all():
        if note_missing and len(first):
            return add_reason(first, pd.Series(True, index=first.index), "one_reader")
        return first
    both = cross_check(first, second, dates=dates, narratives=narratives)
    agree = ~both["check_reasons"].str.contains("readers_disagree|second_reader_only")
    both.loc[agree, "check_reasons"] = both.loc[agree, "check_reasons"].map(
        lambda r: "; ".join(x for x in str(r).split("; ") if x and x not in clears))
    both["check"] = both["check_reasons"] != ""
    extra = both["check_reasons"].str.contains("second_reader_only") & both["entry_id"].isin(set(first["entry_id"]))
    both.loc[extra, "entry_id"] = both.loc[extra, "entry_id"] + "-second"
    return both.reset_index(drop=True)


def usable(answer, part, fixes):
    """The answer, or None when the reader failed on this piece: no answer, an error, or (the empty fix) no entries
    from text that has amounts in it."""
    if answer is None or "error" in answer:
        return None
    if "empty" in fixes and not answer["entries"] and has_amounts(part, part["input_type"]):
        return None
    return answer


def wait_for(job, deadline):
    """A piece's answer, or an error answer when it isn't back by the deadline (a time.monotonic() value)."""
    try:
        return job.result(timeout=None if deadline is None else max(deadline - time.monotonic(), 0))
    except NoAnswerYet:
        return {"entries": [], "error": "no answer before the upload's deadline"}


def ask_pieces(model, parts, source, deadline=None):
    """One reader's answer for each piece, in order, with None where it failed. With parallel_pieces the pieces are
    asked at the same time. deadline: a time.monotonic() value; a piece with no answer by then counts as failed, and
    its call is left to finish in the background."""
    fixes = set(model.fixes or ())
    number = lambda n: None if len(parts) == 1 else n
    ask = lambda n: model.answer(parts[n], source, number(n))
    if "parallel_pieces" not in fixes and deadline is None:
        answers = [ask(n) for n in range(len(parts))]
    else:
        pool = ThreadPoolExecutor(max_workers=PIECE_WORKERS if "parallel_pieces" in fixes else 1)
        jobs = [pool.submit(ask, n) for n in range(len(parts))]
        answers = [wait_for(job, deadline) for job in jobs]
        pool.shutdown(wait=False, cancel_futures=True)
    return [usable(answer, part, fixes) for answer, part in zip(answers, parts)]


def answer_rows(model, parts, answers, input_type, source, upload_date, year, tk_id, tk_name):
    """(rows, piece of each row): entry rows from every piece that has an answer (None skips a piece), with the
    checks made per entry."""
    fixes = set(model.fixes or ())
    rows, piece_of, dated_before, named_before = [], [], False, False
    for n, (part, answer) in enumerate(zip(parts, answers)):
        if answer is None:
            continue
        context = part.get("context", 0)
        amounts = ([line["text"] for line in part["lines"] if line["id"] > context
                    and not TOTAL_LINE.search(line["text"]) and not line["text"].lstrip().startswith(">")]
                   if input_type in FREE_TEXT else [])
        kept = [i for i, entry in enumerate(answer["entries"]) if not only_context(entry, context)]
        if "line_order" in fixes:   # the order of the text, whatever order the model listed them in
            kept = sorted(kept, key=lambda i: max(line_ids(answer["entries"][i]), default=0))
        model_entries = [answer["entries"][i] for i in kept]
        fixed = [False] * len(model_entries)
        if model.new_checks:
            model_entries, fixed = clean_entries(model_entries)
            amounts = [clock_ranges_as_hours(t) for t in amounts]
        found = nova_rows({**answer, "entries": model_entries}, source, upload_date, model.cmids,
                          input_type=input_type, lines=part["lines"], amount_lines=amounts)
        for row, i in zip(found, kept):
            row["rows"] = i + 1   # the entry's place in the saved answer, which source_lines reads back
        by_id = {line["id"]: line["text"] for line in part["lines"]}
        cut_off = "cut_off" in fixes and answer.get("stop_reason") == "max_tokens"
        reused = set(reused_amounts(model_entries, part["lines"])) if "hours_reused" in fixes else set()
        unread = "rows" in fixes and bool(unread_rows(part, model_entries))
        header = (" ".join(line["text"] for line in part["lines"][:part["data_start"] - 1])
                  if "units" in fixes and part.get("data_start") else None)
        for position, (row, entry, was_fixed) in enumerate(zip(found, model_entries, fixed)):
            more = ["hours_reused"] if position in reused else []
            if "dates" in fixes:
                when, said = resolve_date(entry.get("date"), upload_date, year)
                more = more + said
                if when is not None:
                    row["work_date"] = when
                if "date_weekday" not in more and any(weekday_mismatch(by_id[i], year) for i in line_ids(entry)
                                                      if i in by_id):
                    more = more + ["date_weekday"]
            if "date_gap" in fixes:
                if blank(entry.get("date")) and dated_before:
                    more = more + ["date_gap"]
                dated_before = dated_before or not blank(entry.get("date"))
            if "client_gap" in fixes:
                unnamed = blank(entry.get("client")) and blank(entry.get("cmid"))
                if unnamed and named_before:
                    more = more + ["client_gap"]
                named_before = named_before or not unnamed
            more = more + (["answer_cut_off"] if cut_off else []) + (["rows_unread"] if unread else [])
            more = more + (["columns_unsure"] if entry.get("columns_unsure") else [])
            row["check_reasons"] = "; ".join(dict.fromkeys(filter(None, [*str(row["check_reasons"]).split("; "),
                                                                          *more])))
            own = str(entry.get("timekeeper") or "").strip() if part["with_timekeeper"] else ""
            row["tk_id"] = None if own else tk_id
            row["tk_name"] = own or tk_name
            row["page"] = n + 1 if len(parts) > 1 else None
            if "stamp" in fixes:
                row["model_id"] = answer.get("model") or model.model_id
                row["prompt_hash"] = answer.get("used_prompt") or model.prompt_hash(part)
            if model.new_checks:
                row["matter"] = entry.get("matter") or None
                firm = str(entry.get("client") or "").strip().lower().startswith("firm")
                extra = (["cmid_fixed"] if was_fixed else []) + (
                    ["not_billable"] if entry.get("billable") is False and not firm else [])
                row["check_reasons"] = "; ".join(filter(None, [row["check_reasons"], *extra]))
            more = []
            if ("matter_narrative" in fixes and model.matters and blank(row["narrative"])
                    and not blank(row.get("matter")) and not matter_known(row["matter"], model.matters)):
                row["narrative"], row["matter"] = row["matter"], None
                more.append("narrative_from_matter")
            firm_time = str(row.get("client") or "").strip().lower().startswith("firm")
            if "no_narrative" in fixes and blank(row["narrative"]) and row["hours"] is not None and not firm_time:
                more.append("no_narrative")
            row["check_reasons"] = "; ".join(filter(None, [row["check_reasons"], *more]))
        found = [drop_explained(row, input_type, header) for row in found]
        rows += found
        piece_of += [n] * len(found)
    return rows, piece_of


def finish_entries(rows, content, input_type, timekeepers, model, tk_id, tk_name, year, image):
    """The entry table from one reader's rows, with the checks made across entries."""
    fixes = set(model.fixes or ())
    forwarder = None
    if input_type == "EMAIL" and tk_id is None:
        original = original_sender(content)
        if original and find_timekeeper(original, timekeepers):
            forwarder = find_timekeeper(tk_name, timekeepers)
            for row in rows:
                row["tk_name"] = original
    entries = drop_reasons(make_entries(rows, timekeepers, year))
    if forwarder and len(entries):
        entries = add_reason(entries, entries["tk_id"] != forwarder, "tk_forwarded")
    if model.new_checks:
        entries = add_client_check(entries, model)
        if is_photo(image, input_type):
            entries = add_photo_check(entries, check_photo(image))
    return add_firm_checks(entries, model, fixes)


def by_rules(entries, stamp=False):
    if stamp:
        entries = entries.assign(model_id="rules", prompt_hash=None)
    return add_reason(entries, pd.Series(True, index=entries.index), "read_by_rules")


def piece_rules(content, parts, n, input_type, ref, timekeepers, rules_content=None, **common):
    """The rules' entries for piece n alone, when both readers failed on it but not on the whole upload."""
    if input_type in ("NOTE", "SHEET", "SCREENSHOT"):
        pages = rules_content() if rules_content else content
        found = extract(pages[n:n + 1], input_type, ref, timekeepers, **common)
    elif input_type == "SPREADSHEET":
        found = extract(spreadsheet_piece_raw(content, n), input_type, ref, timekeepers, **common)
    else:
        part = parts[n]
        own = "\n".join(line["text"] for line in part["lines"] if line["id"] > part.get("context", 0))
        found = extract(own, input_type if input_type in TEXT_TYPES else "TEXT", ref, timekeepers, **common)
        found["input_type"] = input_type
        found["source_format"] = FORMAT_FOR_INPUT[input_type]
    found["page"] = n + 1
    found["entry_id"] = entry_ids(found)
    return by_rules(found, stamp=True)


WRITER_TYPES = {"NOTE", "VOICE", "TEXT"}   # one person's own note: no timekeeper field, but it may say whose it is
SPOKEN_NAME = re.compile(r"\b(?i:this is|it's|it is|i'm|i am)\s+((?:[A-Z]\.\s?)?[A-Z][\w'\u2019-]+(?:\s+[A-Z][\w'\u2019-]+)?)|\b([A-Z][\w'\u2019-]+)\s+here\b")
NOT_A_NAME = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b(?:mon|tues?|wed|thu|thurs?|fri|sat|sun)(?:day)?\b\.?|\b(?:time|by|from)\b|[-–—~:,]", re.I)
NAME_LIKE = re.compile(r"[A-Z]{2,3}|[A-Za-z]\.\s?[A-Za-z'’-]{2,}|[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?")


def writer_id(name, timekeepers):
    """A timekeeper for a name written as a note's signature: as find_timekeeper reads it, or by the last name alone
    when it's written "X. Last" and the initial was misread ("O. Dunn" for "C. Dunn"), if only one timekeeper has it."""
    tk = find_timekeeper(name, timekeepers)
    if tk or not re.fullmatch(r"[A-Za-z]\.\s?[A-Za-z'’-]{2,}", name):
        return tk
    return find_timekeeper(name_parts(name)[1], timekeepers, min_score=90)


def note_writer(content, input_type, timekeepers):
    """The firm's name for whoever a note says it's from, or None (the writer fix): a name or initials on a line of
    their own among the first two or last two lines of a note photo ("C. Dunn", "- B. Quinn", "FC 9/22"), or a memo or
    text that opens "this is Sato" or "Sato here". A name inside a task ("call w/ O. Grant") never counts. None when no
    one, or more than one person, on the firm's list fits."""
    if input_type == "NOTE":
        lines = [line["text"] for line in page_text_lines(content[0])] if content else []
        said = [NOT_A_NAME.sub(" ", text) for text in lines[:2] + lines[-2:]]
    else:
        said = [a or b for a, b in SPOKEN_NAME.findall(str(content)[:300])]
    names = [" ".join(n.split()).strip(" .") for n in said]
    found = {writer_id(n, timekeepers) for n in names if n and NAME_LIKE.fullmatch(n)} - {None}
    if len(found) != 1:
        return None
    return timekeepers.loc[timekeepers["tk_id"] == found.pop(), "name"].iloc[0]


def drop_signature(entries, writer):
    """The entries without one a reader made of the note's signature alone ("- B. Quinn" read as an entry): no hours,
    and no word in its narrative but the writer's name. A matter number on it goes to the entry before it, if that
    one has none."""
    own = set(re.findall(r"[a-z]+", writer.lower()))
    drop = []
    for n in range(len(entries)):
        row = entries.iloc[n]
        words = set(re.findall(r"[a-z]+", str(row.get("narrative") or "").lower()))
        if words and words <= own and (pd.isna(row.get("hours")) or not row.get("hours")):
            if n and not blank(row.get("cmid")) and blank(entries.iloc[n - 1].get("cmid")):
                entries.iloc[n - 1, entries.columns.get_loc("cmid")] = row["cmid"]
            drop.append(entries.index[n])
    return entries.drop(index=drop) if drop else entries


def read_pair(content, input_type, ref, timekeepers, first, second=None, tk_id=None, tk_name=None, **kw):
    """(entries, first reader's entries, second reader's entries) for one upload, as both_readers gives them, with the
    writer fix: an upload of one person's own note that names no timekeeper gets the one it's signed with
    (note_writer), and an entry made of the signature alone is dropped (drop_signature)."""
    writer = None
    if tk_id is None and blank(tk_name) and "writer" in (first.fixes or ()) and input_type in WRITER_TYPES:
        writer = note_writer(content, input_type, timekeepers)
    entries, ones, twos = both_readers(content, input_type, ref, timekeepers, first, second, tk_id=tk_id,
                                       tk_name=writer or tk_name, **kw)
    return (drop_signature(entries, writer) if writer and len(entries) else entries), ones, twos


def both_readers(content, input_type, ref, timekeepers, first, second=None, tk_id=None, tk_name=None, upload_date=None,
                 source=None, year=None, image=None, rules_content=None, deadline=None):
    """(entries, first reader's entries, second reader's entries) for one upload. The arguments are read's, with the
    two readers given apart; second=None reads with one model. Both readers are asked at the same time.
    deadline: a time.monotonic() value after which a piece with no answer counts as failed (the app sets one per
    upload; None waits for every answer).

    Without piece_fallback in first.fixes, a piece the first reader fails on sends the whole upload to the rules
    (read_by_rules), and a piece the second fails on leaves every entry with one reader (one_reader). With it, each
    piece is settled on its own (NOV20_FIXES)."""
    year = year or pd.Timestamp(upload_date or "today").year
    fixes = set(first.fixes or ())
    common = dict(tk_id=tk_id, tk_name=tk_name, upload_date=upload_date, source=source, year=year)
    split = lambda model: pieces(content, input_type, image, model.images_dir,
                                 by_amounts="long_pieces" in (model.fixes or ()))
    parts = split(first)
    mapped = (by_columns(content, parts, first, second, source, deadline)
              if input_type == "SPREADSHEET" and "column_map" in fixes else None)
    if mapped is not None:
        (firsts, seconds), parts_second = mapped, parts if second is not None else None
    elif second is None:
        firsts, seconds, parts_second = ask_pieces(first, parts, source, deadline), None, None
    else:
        parts_second = split(second)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(ask_pieces, first, parts, source, deadline),
                    pool.submit(ask_pieces, second, parts_second, source, deadline)]
            firsts, seconds = [job.result() for job in jobs]
    empty = pd.DataFrame(columns=ENTRY_COLS)
    rows_of = lambda model, model_parts, answers: answer_rows(model, model_parts, answers, input_type, source,
                                                              upload_date, year, tk_id, tk_name)
    finish = lambda model, rows: finish_entries(rows, content, input_type, timekeepers, model, tk_id, tk_name, year,
                                                image)
    whole_rules = lambda: by_rules(extract(rules_content() if rules_content else content, input_type, ref, timekeepers,
                                           **common), stamp="stamp" in fixes)
    note_missing, dates, narratives = "one_reader" in fixes, "date_agree" in fixes, "narrative_agree" in fixes

    if "piece_fallback" not in fixes:
        if any(a is None for a in firsts):
            entries = whole_rules()
            return entries, entries, empty
        ones = finish(first, rows_of(first, parts, firsts)[0])
        if second is None:
            return ones, ones, empty
        twos = empty if any(b is None for b in seconds) else finish(second, rows_of(second, parts_second, seconds)[0])
        return combine(ones, twos, dates=dates, note_missing=note_missing, narratives=narratives), ones, twos

    seconds = seconds or [None] * len(parts)
    taken = []
    for a, b, part in zip(firsts, seconds, parts_second or parts):
        if a is None and b is not None:
            b = {**b, "used_prompt": b.get("used_prompt") or second.prompt_hash(part)}   # the second reader's prompt
        taken.append(a if a is not None else b)
    if all(a is None for a in taken):
        entries = whole_rules()
        return entries, entries, empty
    rows, piece_of = rows_of(first, parts, taken)
    ones = finish(first, rows)
    checked = [a is not None and b is not None for a, b in zip(firsts, seconds)]
    rows_second, piece_second = rows_of(second, parts_second, [b if ok else None for b, ok in zip(seconds, checked)]) \
        if second is not None else ([], [])
    twos = finish(second, rows_second) if second is not None else empty
    if second is None and all(a is not None for a in taken):
        return ones, ones, empty
    if all(checked):
        return combine(ones, twos, dates=dates, note_missing=note_missing, narratives=narratives), ones, twos
    if second is not None and all(a is not None for a in taken) and not any(checked):
        return combine(ones, empty, dates=dates, note_missing=note_missing, narratives=narratives), ones, twos
    ones_piece, twos_piece = pd.Series(piece_of, index=ones.index), pd.Series(piece_second, index=twos.index)
    found = []
    for n in range(len(parts)):
        mine = ones[ones_piece == n]
        if taken[n] is None:
            found.append(piece_rules(content, parts, n, input_type, ref, timekeepers, rules_content, **common))
        elif checked[n]:
            found.append(combine(mine, twos[twos_piece == n], dates=dates, note_missing=note_missing, narratives=narratives))
        elif second is not None and note_missing and len(mine):
            found.append(add_reason(mine, pd.Series(True, index=mine.index), "one_reader"))
        else:
            found.append(mine)
    found = [f for f in found if len(f)]
    entries = add_day_check(pd.concat(found, ignore_index=True)) if found else empty
    return entries, ones, twos


def read(content, input_type, ref, timekeepers, model, tk_id=None, tk_name=None, upload_date=None, source=None,
         year=None, image=None, rules_content=None):
    """The shared entry table for one upload, the same way for every input type. The arguments are extract's, plus
    model (a ModelReader) and image (the photo or scan the Textract pages came from, if any).
    rules_content: a function giving what the rules read instead of content when the model can't. The app reads a
    scanned sheet with Textract's plain text (the model only uses the lines) and asks for its tables only here.

    With model.new_checks:
    - a matter name the model put in "cmid" is moved to "matter", and a CMID written with a dash or without its dot
      is fixed (cmid_fixed, a note);
    - a clock range counts as one amount in the unused-hours check, not two;
    - client_unknown: a client the firm doesn't have;
    - not_billable: an entry the text marks not billable, unless it's firm time;
    - a photo that fails upload_check gets poor_photo (the app asks for a retake first).
    With model.second, a second model reads every piece too (combine).
    With model.fixes, the checks named there (NOV15_FIXES and NOV20_FIXES say what each does)."""
    return read_pair(content, input_type, ref, timekeepers, model, model.second, tk_id=tk_id, tk_name=tk_name,
                     upload_date=upload_date, source=source, year=year, image=image, rules_content=rules_content)[0]
