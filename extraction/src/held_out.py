"""Samples from outside writers: notes and voice memos written by people who never saw the readers.

Writers get task cards (data/held_out/cards.csv): for each entry the date, the hours, the task, and whether to write
the client's name, its CMID, or neither. A page can also say how to write it (a week on one page, one paragraph). How they lay it out, abbreviate, and write the hours is up to
them. Because the facts come from the card, the answer key is built from it (build_key), so no one reads a photo to
write the key.

freeze saves a fingerprint of every reader file before a batch of samples arrives, so a later run can show which
files changed since.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from extract import extract
from intake import make_entries
from model_first import read_photo
from model_readers import nova_rows, page_text_lines
from tables import has_table

READER_FILES = ["extract.py", "textract.py", "tables.py", "free_text.py", "spreadsheets.py", "documents.py",
                "messages.py", "text_parser.py", "intake.py", "model_readers.py", "model_first.py",
                "voice.py", "open_models.py"]
FORMS = {"photo", "typed", "voice"}
DIGIT_CUTOFF = 61.63   # Textract confidence cutoff for digits, set on our own notes
STRUCTURE_FLAGS = ["no_hours", "multiple_entries", "cmid_shape", "hours_10x"]   # photos with these go to Nova
PHOTO_SETUPS = ["Rules", "Rules, digit flags", f"Rules, digit flags under {DIGIT_CUTOFF}", "Nova Pro + photo",
                "Rules, then Nova Pro + photo", "Model first"]   # Model first: model_first.read_photo
TYPED_SETUPS = ["Rules", "Nova Pro"]
CARD_COLS = ["page", "entry", "tk_id", "date", "write", "client", "cmid", "intended_cmid", "hours", "task", "page_note"]
MIN_GAIN = 2   # a setup replaces a simpler one only with at least this many more entries right with no flag
FILE_COLS = ["file", "page", "writer", "form"]


def fingerprint(src_dir, files=READER_FILES):
    """The md5 of each reader file."""
    return {name: hashlib.md5((Path(src_dir) / name).read_bytes()).hexdigest() for name in files}


def freeze(path, src_dir):
    """Save the readers' fingerprint the first time; afterwards, the reader files that changed since.

    Returns (frozen_on, changed files)."""
    path = Path(path)
    now = fingerprint(src_dir)
    if not path.exists():
        path.write_text(json.dumps({"frozen_on": date.today().isoformat(), "files": now}, indent=1))
    saved = json.loads(path.read_text())
    changed = [name for name in READER_FILES if saved["files"].get(name) != now[name]]
    return saved["frozen_on"], changed


def load_cards(path):
    cards = pd.read_csv(path, dtype={"cmid": str, "intended_cmid": str, "date": str})
    assert list(cards.columns) == CARD_COLS, f"cards.csv columns should be {CARD_COLS}"
    return cards


def load_files(path):
    """Which card page each held-out file was written from, by whom (a label, not a name), in what form, and in which
    batch. Each batch is scored against the readers frozen before its notes arrived; a missing batch is batch 1."""
    files = pd.read_csv(path, dtype=str)
    assert set(FILE_COLS) <= set(files.columns), f"files.csv needs the columns {FILE_COLS} (and optionally batch)"
    assert files["form"].isin(FORMS).all(), f"form should be one of {sorted(FORMS)}"
    assert files["file"].is_unique
    files["batch"] = pd.to_numeric(files["batch"], errors="coerce").fillna(1).astype(int) if "batch" in files else 1
    return files


def build_key(cards, files, upload_date):
    """The answer key, one row per entry of every file, in the order on the card.

    The photo column holds the file name, so grade.grade_photos works on typed files too. A page with no date
    written takes the upload date, as extract does."""
    if "batch" not in files:
        files = files.assign(batch=1)
    key = files.merge(cards, on="page", how="left", validate="many_to_many")
    missing = key.loc[key["entry"].isna(), "page"].unique()
    assert not len(missing), f"Pages not on the cards: {list(missing)}"
    key = key.rename(columns={"file": "photo", "task": "narrative"})
    key["entry"] = key["entry"].astype(int)
    key["work_date"] = pd.to_datetime(key["date"].fillna(upload_date))
    key["note"] = ""
    return key[["photo", "entry", "work_date", "hours", "client", "cmid", "narrative", "note", "page", "writer",
                "form", "batch", "tk_id", "intended_cmid"]]


def when(date):
    return pd.Timestamp(date).strftime("%a %-m/%-d") if pd.notna(date) else "no date (don't write one)"


def card_text(cards):
    """The cards as a writer sees them: one block per page, with any page note, and each entry's date, client or
    CMID, hours, and task."""
    blocks = []
    for page, entries in cards.groupby("page", sort=True):
        note = entries["page_note"].dropna()
        lines = [f"Page {page}" + (f"  |  {note.iloc[0]}" if len(note) else "")]
        for e in entries.itertuples():
            lines.append(f"  {e.entry}. {when(e.date)}  |  {e.write}  |  {e.hours:.1f} hours  |  {e.task}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def with_tk(rows, tk_id, timekeepers, year):
    for row in rows:
        row["tk_id"] = tk_id
    return make_entries(rows, timekeepers, year)


def photo_setups(name, note_pages, sheet_pages, nova_answer, ref, timekeepers, cmids, tk_id, upload_date):
    """One photo's entries under every setup in PHOTO_SETUPS (setup name to entries).

    A photo with columns drawn on it goes through the table reader in every setup (Nova only reads free text).
    nova_answer: a saved Nova Pro + photo answer, or None to leave the Nova setups out.
    The Nova setups use digit flags (under 80), as measured in the model readers experiment."""
    common = dict(tk_id=tk_id, upload_date=upload_date, source=name)
    if has_table(sheet_pages):
        rules = extract(sheet_pages, "NOTE", ref, timekeepers, **common)
        return {setup: rules for setup in PHOTO_SETUPS}
    rules = extract(note_pages, "NOTE", ref, timekeepers, **common)
    digits = extract(note_pages, "NOTE", ref, timekeepers, note_options={"confidence": "numbers"}, **common)
    out = {"Rules": rules, "Rules, digit flags": digits,
           PHOTO_SETUPS[2]: extract(note_pages, "NOTE", ref, timekeepers,
                                    note_options={"confidence": "numbers", "low_conf": DIGIT_CUTOFF}, **common)}
    if nova_answer is not None:
        lines = page_text_lines(note_pages[0])
        nova = with_tk(nova_rows(nova_answer, name, upload_date, cmids, confidence="numbers", lines=lines), tk_id,
                       timekeepers, pd.Timestamp(upload_date).year)
        structural = rules["check_reasons"].str.split("; ").explode().isin(STRUCTURE_FLAGS).any()
        out["Nova Pro + photo"] = nova
        out["Rules, then Nova Pro + photo"] = nova if structural else digits
        out["Model first"] = read_photo(name, note_pages, sheet_pages, nova_answer, ref, timekeepers, cmids, tk_id,
                                        upload_date)
    return out


def typed_setups(name, text, nova_answer, ref, timekeepers, cmids, tk_id, upload_date):
    """One typed page's entries under every setup in TYPED_SETUPS. The rules read it as an email body."""
    out = {"Rules": extract(text, "EMAIL", ref, timekeepers, tk_id=tk_id, upload_date=upload_date, source=name)}
    if nova_answer is not None:
        out["Nova Pro"] = with_tk(nova_rows(nova_answer, name, upload_date, cmids, input_type="EMAIL"), tk_id,
                                  timekeepers, pd.Timestamp(upload_date).year)
    return out


def stack(by_file):
    """{file: {setup: entries}} as {setup: one table with a photo column}. A setup missing for any file is left out."""
    setups = [s for s in next(iter(by_file.values())) if all(s in found for found in by_file.values())]
    return {s: pd.concat([found[s].assign(photo=name) for name, found in by_file.items()], ignore_index=True)
            for s in setups}


def intended_for(found, key):
    """Each found entry's intended matter, from the key entry it lines up with (grade.align). Entries with no key
    entry get None."""
    from grade import align
    out = []
    for name, got in found.groupby("photo", sort=False):
        got = got.reset_index()
        wanted = key[key["photo"] == name].reset_index(drop=True)
        for i, j in align(got, wanted):
            if i is not None:
                out.append((got.loc[i, "index"], None if j is None else wanted.loc[j, "intended_cmid"]))
    return pd.Series(dict(out)).reindex(found.index)


def summarize_ignoring(graded, ignore=("date",)):
    """Like grade.summarize, but an entry wrong only in the `ignore` fields counts as right. For diagnosis after the
    results are recorded (how much one gap explains), never for the decision."""
    from grade import summarize
    wrong_fields = graded["wrong"].fillna("").map(lambda s: {f for f in s.split(", ") if f})
    only_ignored = (graded["outcome"].isin(["wrong, flagged", "wrong, silent"])
                    & wrong_fields.map(lambda fields: bool(fields) and fields <= set(ignore)))
    regraded = graded.copy()
    regraded.loc[only_ignored, "outcome"] = "right"
    return summarize(regraded)
