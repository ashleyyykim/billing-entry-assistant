"""Which check reasons send an entry to a person, and which are only shown as a note.

Extraction flags anything it isn't sure of (intake.CHECK_REASONS). Some reasons are only worth showing: a client
copied from the line above, a date taken from the week or from a day name, a CMID with a letter fixed, a clock time
left out of the hours, a forwarded email credited to its writer, and a second reader that gave no answer. Those are
shown to the reviewer as notes and don't send the entry to review on their own. Every other reason does.
"""
from intake import CHECK_REASONS

NOTE_ONLY = ["client_copied", "date_guessed", "cmid_fixed", "clock_time", "tk_forwarded", "date_relative", "one_reader",
             "narrative_from_matter"]
REVIEW_REASONS = {**CHECK_REASONS, "poor_photo": "The photo looked blurry, dark, or too small, and was sent anyway"}


def split_reasons(check_reasons):
    """(reasons that send the entry to review, reasons shown only as notes) for one entry's check_reasons."""
    reasons = [r for r in str(check_reasons or "").split("; ") if r and r != "nan"]
    return [r for r in reasons if r not in NOTE_ONLY], [r for r in reasons if r in NOTE_ONLY]


def add_review(entries):
    """The entries with a `review` column (True when any reason sends it to a person) and a `notes` column (the
    reasons shown only as notes). check and check_reasons are left as they are."""
    out = entries.copy()
    split = out["check_reasons"].map(split_reasons)
    out["review"] = split.str[0].str.len() > 0
    out["notes"] = split.str[1].str.join("; ")
    return out


def as_reviewed(entries):
    """The entries with only the reasons that send them to review kept in check_reasons, so the graders
    (grade.summarize, grade.case_outcome) count what a reviewer would see under the two tiers."""
    out = entries.copy()
    out["check_reasons"] = out["check_reasons"].map(lambda r: "; ".join(split_reasons(r)[0]))
    out["check"] = out["check_reasons"] != ""
    return out
