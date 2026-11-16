"""The model-first flow for note photos: a model reads the note, and the rules check what it read.

1. The upload check (upload_check.check_photo) runs first; a photo that fails and is sent anyway gets poor_photo.
2. A photo with columns drawn on it goes to the table reader, as before: it's structured, and code reads it well.
3. Any other photo is read by Nova Pro, which sees the photo and Textract's lines with their positions
   (model_readers.ask_nova). Its entries go through model_readers.model_rows and intake.make_entries, so they get
   the model's own checks (hours not on the lines it named, hours that don't match how they're written, a CMID
   with the wrong shape or not in Matter Master, hours on the page no entry used, the model unsure) and the checks
   every entry gets (no hours, unknown timekeeper, over 14 hours in a day).
4. Textract's word confidence (low_conf) is left out. The model read the photo itself, so a hard-to-read word for
   Textract isn't a reason to doubt the model.
5. review.add_review splits the reasons into review and note, as for every other input.

If the model fails (no answer, or an error), the rules read the photo instead, with their own checks.
"""
import pandas as pd

from extract import extract
from intake import make_entries
from model_readers import nova_rows, page_text_lines
from tables import has_table
from upload_check import add_photo_check

LEFT_OUT = ["low_conf"]


def drop_reasons(entries, reasons=LEFT_OUT):
    """The entries with the given check reasons removed (and `check` updated)."""
    out = entries.copy()
    out["check_reasons"] = out["check_reasons"].map(
        lambda r: "; ".join(x for x in str(r or "").split("; ") if x and x != "nan" and x not in reasons))
    out["check"] = out["check_reasons"] != ""
    return out


def model_ok(answer):
    return answer is not None and "error" not in answer


def read_photo(name, note_pages, sheet_pages, nova_answer, ref, timekeepers, cmids, tk_id, upload_date, check=None):
    """One note photo's entries, model first.

    note_pages, sheet_pages: Textract's saved answers for the photo (textract.load_pages, "note" and "sheet").
    nova_answer: Nova Pro's answer for the photo (model_readers.ask_nova or a saved one), or None.
    check: upload_check.check_photo's result for the photo, or None to skip that step.
    Returns the entries with every check reason; review.add_review or review.as_reviewed applies the two tiers.
    """
    common = dict(tk_id=tk_id, upload_date=upload_date, source=name)
    if has_table(sheet_pages):
        entries = extract(sheet_pages, "NOTE", ref, timekeepers, **common)
    elif not model_ok(nova_answer):
        entries = extract(note_pages, "NOTE", ref, timekeepers, **common)
    else:
        rows = nova_rows(nova_answer, name, upload_date, cmids, confidence="numbers",
                         lines=page_text_lines(note_pages[0]))
        for row in rows:
            row["tk_id"] = tk_id
        entries = drop_reasons(make_entries(rows, timekeepers, pd.Timestamp(upload_date).year))
    return add_photo_check(entries, check) if check is not None else entries
