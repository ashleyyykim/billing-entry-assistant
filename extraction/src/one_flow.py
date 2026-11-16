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
| SPREADSHEET: Excel or CSV | the rows, cells joined by " | ", the header first | |

Then the same steps for every input:
1. Nova Pro reads the lines into the entry schema with a forced tool call at temperature 0, naming the lines each
   entry uses (model_readers.ask_nova). Note photos get the photo prompt (model_readers.system_prompt); every other
   input gets prompts.py: the same task and fields, what is and isn't an entry, how to read hours, the habits of
   that input type, and worked examples of it. For documents, scans, and spreadsheets the model also says whose time
   each entry is, when the text names a timekeeper.
2. Long inputs are read in pieces (a page of a PDF or scan, SPREADSHEET_ROWS rows with the header repeated). Each
   later piece starts with the first lines of the document, so a heading like "Timekeeper: R. Patel" still applies.
3. A second model (Llama 4 Maverick) reads every piece too (ModelReader.second, combine): an entry the two read
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
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rapidfuzz import fuzz, process

from extract import extract
from free_text import WEEKDAY_NUMBER, original_sender, page_amounts, page_order, weekday_mismatch
from intake import add_reason, find_timekeeper, make_entries, to_date
from model_first import drop_reasons
import prompts
from model_readers import NOVA_MODELS, ask_nova, blank, cross_check, nova_rows, page_text_lines, saved
from spreadsheets import find_header
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
MAVERICK = "us.meta.llama4-maverick-17b-instruct-v1:0"
MAX_LINES = 150          # a typed input with more lines is read in pieces
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
FUTURE_DAYS = 60         # a date with no year this far after the upload is read as the year before
RELATIVE_DAYS = {"today": 0, "tonight": 0, "this morning": 0, "this afternoon": 0, "this evening": 0,
                 "yesterday": -1}
DAY_NAME = re.compile(r"^\s*(last\s+|this\s+|on\s+)?(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?\s*$", re.I)
DAY_PREFIX = re.compile(r"^\s*(mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+(?=\S*\d)", re.I)
WEEK_OF = re.compile(r"^\s*(?:week\s+(?:of|ending|beginning|starting)|w/[eo])\s*:?\s*(?=\S)", re.I)
WRITTEN_YEAR = re.compile(r"\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2}\b")
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
    prompt: "v4" (prompts.py for every input but note photos) or "v1" (the photo prompt for every input).
    new_checks: the clean-ups and checks described in read; False turns them off.
    force_tool, open_note_prompt: for a model that can't be forced to call a tool (Llama 4 Maverick): offer the tool,
        and read note photos with open_models.ask_model's photo prompt.
    second: another ModelReader (a different model) that reads every piece too; entries the two read differently go
        to review, and entries they read the same way lose AGREE_CLEARS.
    catch_errors: turn a failed call (a timeout, a dropped connection) into an answer with an "error", which isn't
        saved, so the upload goes on without this reader and the next run asks again. For the second model, whose
        failure only means no second reading.
    fixes: the checks named in NOV15_FIXES to apply; () turns them all off.
    client_of, aliases_of: {CMID: client} and {client: [aliases]} from Matter Master (firm_lookup), for the
        client_cmid and client_ambiguous checks; without them those two checks don't run.
    """
    clients: list
    cmids: set
    runtime: object = None
    answers_dir: Path = Path("../data/model_answers/one_flow")
    model_id: str = NOVA_MODELS["Nova Pro"]
    answer_path: object = None
    images_dir: Path = field(default=None)
    prompt: str = "v4"
    new_checks: bool = True
    force_tool: bool = True
    open_note_prompt: bool = False
    second: object = None
    catch_errors: bool = False
    fixes: tuple = NOV15_FIXES
    client_of: dict = None
    aliases_of: dict = None

    def path(self, source, piece):
        if self.answer_path is not None:
            found = self.answer_path(source, piece)
            if found is not None:
                return Path(found)
        stem = re.sub(r"[^\w.-]+", "_", Path(str(source)).name)
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

    def ask(self, piece, source, number):
        path = self.path(source, number)
        if not path.exists() and self.runtime is None:
            return None
        kind, tk = piece["input_type"], piece["with_timekeeper"]
        if kind == "NOTE" and self.open_note_prompt:
            from open_models import ask_model
            ask = lambda: ask_model(self.runtime, piece["lines"], self.clients, self.model_id, photo=piece["photo"],
                                    prompt="v2")
            return saved(path, ask)
        version = prompts if self.prompt == "v4" and kind != "NOTE" else None
        ask = lambda: ask_nova(self.runtime, piece["lines"], self.clients, self.model_id, photo=piece["photo"],
                               intro=version.intro(kind) if version else piece["intro"], with_timekeeper=tk,
                               system=version.system_prompt(kind, tk) if version else None,
                               schema=version.entry_schema(tk) if version else None, force_tool=self.force_tool)
        return saved(path, ask)


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


def cell_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)) or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp) or hasattr(value, "strftime"):
        return value.strftime("%m/%d/%Y")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def spreadsheet_rows(raw):
    """(title and header rows, data rows) of a spreadsheet's raw cells, as text with cells joined by " | "."""
    texts = [" | ".join(cell_text(v) for v in row) for row in raw.itertuples(index=False)]
    keep = [i for i, t in enumerate(texts) if t.replace("|", "").strip()]
    try:
        header = find_header(raw)
    except ValueError:
        header = keep[0] if keep else 0
    return [texts[i] for i in keep if i <= header], [texts[i] for i in keep if i > header]


def spreadsheet_pieces(raw, rows_per_piece=SPREADSHEET_ROWS):
    """[lines] for a spreadsheet's raw cells: the title and header rows, then up to rows_per_piece data rows."""
    top, body = spreadsheet_rows(raw)
    return [top + body[i:i + rows_per_piece] for i in range(0, max(len(body), 1), rows_per_piece)]


def split_long(texts, max_lines=MAX_LINES):
    """A long typed input in pieces of at most max_lines, each later piece starting with the first CONTEXT_LINES."""
    if len(texts) <= max_lines:
        return [texts]
    head, step = texts[:CONTEXT_LINES], max_lines - CONTEXT_LINES
    return [texts[:max_lines]] + [head + texts[i:i + step] for i in range(max_lines, len(texts), step)]


def pdf_page_image(path, page, images_dir):
    """A scanned PDF's page as a PNG (saved once), for the model to see."""
    import pymupdf
    out = Path(images_dir) / f"{Path(path).stem}_p{page + 1}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(path) as pdf:
            pdf[page].get_pixmap(dpi=PAGE_DPI).save(str(out))
    return out


def pieces(content, input_type, image=None, images_dir=None):
    """What the model reads for one upload: a list of dicts with lines, intro, photo, and with_timekeeper.

    content is what extract takes: Textract's pages for NOTE, SHEET, and SCREENSHOT; the text for EMAIL, TEXT, and
    VOICE; the file path for PDF and DOC; the raw cells for SPREADSHEET. image: the photo, screenshot, or scanned
    file the Textract pages came from, when there is one.
    """
    names_tk = input_type in NAMES_TIMEKEEPERS
    piece = lambda lines, photo=None: {"lines": lines, "intro": INTROS.get(input_type), "photo": photo,
                                       "with_timekeeper": names_tk, "input_type": input_type}
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
        return [piece(numbered(t)) for t in split_long(texts)]
    if input_type in ("PDF", "DOC"):
        pages = document_text_lines(content)
        texts = [t for page in pages for t in page]
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
    """What a ModelReader needs from Matter Master (columns Client, CMID, Known Aliases), as its keyword arguments:
    clients (names and aliases, for the prompt), cmids, client_of ({CMID: client}), and aliases_of
    ({client: [aliases]})."""
    aliases_of = {}
    for client, known in zip(matter_master["Client"], matter_master["Known Aliases"]):
        names = aliases_of.setdefault(client, set())
        if isinstance(known, str):
            names.update(a.strip() for a in known.split(",") if a.strip())
    clients = sorted((set(aliases_of) | {a for names in aliases_of.values() for a in names}) - {""})
    return {"clients": clients, "cmids": set(matter_master["CMID"]),
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
    texts = [clock_ranges_as_hours(line["text"]) for line in part["lines"]
             if not TOTAL_LINE.search(line["text"]) and not line["text"].lstrip().startswith(">")]
    return bool(page_amounts(texts))


def is_photo(image, input_type):
    return (image is not None and input_type in ("NOTE", "SHEET")
            and Path(str(image)).suffix.lower() in IMAGE_SUFFIXES and Path(str(image)).exists())


def combine(first, second, clears=AGREE_CLEARS, dates=False, note_missing=False):
    """First reader's entries checked against a second reader's (model_readers.cross_check): readers_disagree and
    second_reader_only go to review, and entries both read the same way lose the reasons in clears.

    dates: the two readers must agree on the work date too. note_missing: when the second reader gave no answer (no
    entries, or the rules read the upload in its place), the first reader's entries get one_reader, a note."""
    if not len(second) or second["check_reasons"].str.contains("read_by_rules").all():
        if note_missing and len(first):
            return add_reason(first, pd.Series(True, index=first.index), "one_reader")
        return first
    both = cross_check(first, second, dates=dates)
    agree = ~both["check_reasons"].str.contains("readers_disagree|second_reader_only")
    both.loc[agree, "check_reasons"] = both.loc[agree, "check_reasons"].map(
        lambda r: "; ".join(x for x in str(r).split("; ") if x and x not in clears))
    both["check"] = both["check_reasons"] != ""
    extra = both["check_reasons"].str.contains("second_reader_only") & both["entry_id"].isin(set(first["entry_id"]))
    both.loc[extra, "entry_id"] = both.loc[extra, "entry_id"] + "-second"
    return both.reset_index(drop=True)


def read(content, input_type, ref, timekeepers, model, tk_id=None, tk_name=None, upload_date=None, source=None,
         year=None, image=None):
    """The shared entry table for one upload, the same way for every input type. The arguments are extract's, plus
    model (a ModelReader) and image (the photo or scan the Textract pages came from, if any).

    With model.new_checks:
    - a matter name the model put in "cmid" is moved to "matter", and a CMID written with a dash or without its dot
      is fixed (cmid_fixed, a note);
    - a clock range counts as one amount in the unused-hours check, not two;
    - client_unknown: a client the firm doesn't have;
    - not_billable: an entry the text marks not billable, unless it's firm time;
    - a photo that fails upload_check gets poor_photo (the app asks for a retake first).
    With model.second, a second model reads every piece too (combine).
    With model.fixes, the checks named there (NOV15_FIXES says what each does)."""
    year = year or pd.Timestamp(upload_date or "today").year
    fixes = set(model.fixes or ())
    parts = pieces(content, input_type, image, model.images_dir)
    rows, dated_before, named_before = [], False, False
    for n, part in enumerate(parts):
        answer = model.answer(part, source, None if len(parts) == 1 else n)
        failed = answer is None or "error" in answer
        if not failed and "empty" in fixes and not answer["entries"] and has_amounts(part, input_type):
            failed = True   # nothing found in text that has amounts in it: the rules read it and a person looks
        if failed:
            entries = extract(content, input_type, ref, timekeepers, tk_id=tk_id, tk_name=tk_name,
                              upload_date=upload_date, source=source, year=year)
            return add_reason(entries, pd.Series(True, index=entries.index), "read_by_rules")
        amounts = ([line["text"] for line in part["lines"]
                    if not TOTAL_LINE.search(line["text"]) and not line["text"].lstrip().startswith(">")]
                   if input_type in FREE_TEXT else [])
        model_entries, fixed = answer["entries"], [False] * len(answer["entries"])
        if model.new_checks:
            model_entries, fixed = clean_entries(model_entries)
            amounts = [clock_ranges_as_hours(t) for t in amounts]
        found = nova_rows({**answer, "entries": model_entries}, source, upload_date, model.cmids,
                          input_type=input_type, lines=part["lines"], amount_lines=amounts)
        by_id = {line["id"]: line["text"] for line in part["lines"]}
        cut_off = "cut_off" in fixes and answer.get("stop_reason") == "max_tokens"
        unread = "rows" in fixes and bool(unread_rows(part, model_entries))
        header = (" ".join(line["text"] for line in part["lines"][:part["data_start"] - 1])
                  if "units" in fixes and part.get("data_start") else None)
        for row, entry, was_fixed in zip(found, model_entries, fixed):
            more = []
            if "dates" in fixes:
                when, more = resolve_date(entry.get("date"), upload_date, year)
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
            row["check_reasons"] = "; ".join(dict.fromkeys(filter(None, [*str(row["check_reasons"]).split("; "),
                                                                          *more])))
            own = str(entry.get("timekeeper") or "").strip() if part["with_timekeeper"] else ""
            row["tk_id"] = None if own else tk_id
            row["tk_name"] = own or tk_name
            row["page"] = n + 1 if len(parts) > 1 else None
            if model.new_checks:
                row["matter"] = entry.get("matter") or None
                firm = str(entry.get("client") or "").strip().lower().startswith("firm")
                extra = (["cmid_fixed"] if was_fixed else []) + (
                    ["not_billable"] if entry.get("billable") is False and not firm else [])
                row["check_reasons"] = "; ".join(filter(None, [row["check_reasons"], *extra]))
        rows += [drop_explained(row, input_type, header) for row in found]
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
    entries = add_firm_checks(entries, model, fixes)
    if model.second is not None:
        second = read(content, input_type, ref, timekeepers, model.second, tk_id=tk_id, tk_name=tk_name,
                      upload_date=upload_date, source=source, year=year, image=image)
        entries = combine(entries, second, dates="date_agree" in fixes, note_missing="one_reader" in fixes)
    return entries
