"""An email with attachments into entries: the body and every attachment, all credited to the email's writer.

This is how assistants usually receive time: the attorney emails it, with the entries typed in the body or attached as
photos, PDFs, spreadsheets, or Word files. Each part goes to the reader for its kind:

| Attachment | Read as |
|---|---|
| .jpg, .jpeg, .png | a photo: the table reader if Textract found a table on it, else the free-text reader |
| .pdf | a typed document if it has a text layer (documents.py), else a scan read by Textract as a sheet |
| .xlsx, .xls, .csv | a spreadsheet export |
| .docx, .txt | a typed document |

Anything else (an iPhone .heic photo, a .zip, a calendar invite) is listed as not read, with the reason, so a person
can ask for it in another format. Signature logos and icons (Outlook's image001.png) are skipped before Textract, and an
attachment with no time in it (the term sheet the attorney worked on) is listed as "no time found". Photos and scans need Textract: answers already saved are used, and run_ocr=True reads new ones
(a paid call). Every entry is credited to the email's writer (see email_writer), unless the attachment names its own
timekeeper (a report's "Timekeeper:" field, an export's name column). The same work sent twice, in the body and in an
attachment, is flagged by the duplicate check.

A message is a dict: id, sender, sent (date), subject, body, and attachments (file paths). read_message reads one
saved as JSON; read_eml reads an .eml file and saves its attachments.
"""
import json
import re
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

from documents import is_scanned
from extract import email_writer, extract
from intake import ENTRY_COLS, add_duplicate_check
from spreadsheets import read_spreadsheet
from tables import has_table
from textract import load_pages, run_textract

KINDS = {".jpg": "photo", ".jpeg": "photo", ".png": "photo", ".pdf": "pdf", ".xlsx": "spreadsheet",
         ".xls": "spreadsheet", ".csv": "spreadsheet", ".docx": "document", ".txt": "document"}
NOT_READ = {".heic": "an iPhone photo: ask for it as a JPEG, or convert it",
            ".doc": "an old Word file: save it as .docx",
            ".zip": "a zip file: unzip it and send the files",
            ".ics": "a calendar invite", ".vcf": "a contact card", ".gif": "a logo or animated image"}
SIGNATURE_NAME = re.compile(r"^(image|logo|outlook)[-_]?\d*\.(png|jpe?g)$", re.I)   # Outlook's inline image001.png
MIN_PHOTO_SIDE = 400     # pixels; a smaller image is a logo or icon, not a photo of a note
MIN_PHOTO_BYTES = 20_000
TAGS = re.compile(r"<[^>]+>")


def read_message(path, data_dir):
    """A message saved as JSON, with its attachment paths relative to data_dir."""
    message = json.loads(Path(path).read_text())
    message["attachments"] = [Path(data_dir) / a for a in message.get("attachments", [])]
    return message


def read_eml(path, out_dir):
    """A message from an .eml file. Its attachments are saved in out_dir/<file name>/. An encrypted email can't be read
    here, which is why forwarding (the body and attachments arrive readable) is the way in."""
    with open(path, "rb") as f:
        mail = BytesParser(policy=policy.default).parse(f)
    if mail.get_content_type() in ("application/pkcs7-mime", "multipart/encrypted"):
        raise ValueError(f"{Path(path).name} is encrypted: forward it instead")
    body = mail.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body else ""
    if body is not None and body.get_content_type() == "text/html":
        text = TAGS.sub(" ", re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text))
    folder = Path(out_dir) / Path(path).stem
    attachments = []
    for part in mail.iter_attachments():
        name = Path(part.get_filename() or "attachment").name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(part.get_payload(decode=True) or b"")
        attachments.append(folder / name)
    sent = parsedate_to_datetime(mail["Date"]).date().isoformat() if mail["Date"] else None
    return {"id": Path(path).stem, "sender": str(mail["From"] or ""), "sent": sent,
            "subject": str(mail["Subject"] or ""), "body": text, "attachments": attachments}


def looks_like_logo(path):
    """True for an image that's almost surely a signature logo or icon rather than a photo of a note: named like an
    Outlook inline image, tiny on disk, or small in pixels. Checked before Textract, so logos aren't paid for."""
    if SIGNATURE_NAME.match(path.name) or path.stat().st_size < MIN_PHOTO_BYTES:
        return True
    from PIL import Image
    try:
        with Image.open(path) as image:
            return min(image.size) < MIN_PHOTO_SIDE
    except OSError:
        return False


def textract_pages(path, textract_dir, mode, run_ocr):
    pages = load_pages(textract_dir, path, mode)
    if not pages and run_ocr:
        run_textract(path, textract_dir, mode)
        pages = load_pages(textract_dir, path, mode)
    return pages


def rules_reader(ref, timekeepers):
    """The default reader for message parts: extract (the rules), which takes no image."""
    return lambda content, input_type, image=None, **common: extract(content, input_type, ref, timekeepers, **common)


def read_attachment(path, ref, timekeepers, textract_dir, tk_id, upload_date, source, run_ocr, read=None):
    """(entries, how it was read) for one attachment. entries is None when it couldn't be read.

    read: the reader for each part, called as read(content, input_type, image=..., tk_id=..., upload_date=...,
    source=...). Default: the rules (extract), which send a photo with a table to the table reader. one_flow passes
    its own reader, which reads every photo the same way."""
    kind = KINDS.get(path.suffix.lower())
    common = dict(tk_id=tk_id, upload_date=upload_date, source=source)
    rules = read is None
    read = read or rules_reader(ref, timekeepers)
    if kind == "photo":
        if looks_like_logo(path):
            return None, "skipped: looks like a signature logo or icon"
        sheet = load_pages(textract_dir, path, "sheet")
        if rules and has_table(sheet):
            return read(sheet, "NOTE", **common), "photo with a table (Textract)"
        pages = textract_pages(path, textract_dir, "note", run_ocr) or ([] if rules else sheet)
        if not pages:
            return None, "photo not read yet: run Textract on it (run_ocr=True)"
        return read(pages, "NOTE", image=path, **common), "photo (Textract)"
    if kind == "pdf":
        if is_scanned(path):
            pages = textract_pages(path, textract_dir, "sheet", run_ocr)
            if not pages:
                return None, "scanned PDF not read yet: run Textract on it (run_ocr=True)"
            return read(pages, "SHEET", image=path, **common), "scanned PDF (Textract)"
        return read(path, "PDF", **common), "typed PDF"
    if kind == "spreadsheet":
        return read(read_spreadsheet(path), "SPREADSHEET", **common), "spreadsheet"
    if kind == "document":
        return read(path, "DOC", **common), "typed document"
    return None, f"not read: {NOT_READ.get(path.suffix.lower(), f'{path.suffix or path.name} files are not supported')}"


def message_entries(message, ref, timekeepers, textract_dir, tk_id=None, run_ocr=False, read=None):
    """(entries, parts): the entries from a message's body and attachments, and one row per part saying how it was
    read and how many entries it gave.

    tk_id: the attorney, when the assistant picks one. Otherwise the email's writer (email_writer).
    read: the reader for every part (see read_attachment); default the rules.
    """
    writer = tk_id or email_writer(message["body"], message["sender"], timekeepers)
    upload_date = message["sent"]
    found, parts = [], []
    body = (read or rules_reader(ref, timekeepers))(message["body"], "EMAIL", tk_id=tk_id, tk_name=message["sender"],
                                                    upload_date=upload_date, source=f"{message['id']}-body")
    how = "email body"
    if message["attachments"] and not body["hours"].notna().any():
        body, how = body.iloc[0:0], "email body (no time in it)"
    found.append(body.assign(part="body"))
    parts.append({"part": "body", "read as": how, "entries": len(body)})
    for path in message["attachments"]:
        path = Path(path)
        source = f"{message['id']}-{path.name}"
        try:
            entries, how = read_attachment(path, ref, timekeepers, textract_dir, writer, upload_date, source, run_ocr,
                                           read)
        except (ValueError, OSError, KeyError) as error:
            entries, how = None, f"not read: {error}"
        if entries is not None and not len(entries):
            how += " (no time found)"
        parts.append({"part": path.name, "read as": how, "entries": 0 if entries is None else len(entries)})
        if entries is not None:
            found.append(entries.assign(part=path.name))
    entries = pd.concat([f for f in found if len(f)], ignore_index=True) if any(len(f) for f in found) else \
        pd.DataFrame(columns=ENTRY_COLS + ["part"])
    entries["message"] = message["id"]
    return add_duplicate_check(entries) if len(entries) else entries, pd.DataFrame(parts)
