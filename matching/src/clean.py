"""Clean up typed fields: normalize text, handle ditto marks and one-field entries, and fix CMIDs."""
import re

import pandas as pd
from rapidfuzz.distance import OSA

LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "llp", "lp", "corp", "corporation",
                  "co", "company", "ltd", "limited"}
DITTO_MARKS = {'"', "''", "“", "”", "〃", "ditto", "same", "↓"}
FIELD_SEPARATOR = re.compile(r"\s*[/\-–—]\s*")
CMID_PATTERN = re.compile(r"^(\d{4})\s*[.\-/,]?\s*(\d{1,3})$")
CMID_IN_TEXT = re.compile(r"\b(\d{4})\s*[./\-]?\s*(\d{3})\b")

# Letters that OCR and handwriting confuse with digits ("61D3.004" -> "6103.004")
OCR_DIGITS = str.maketrans("ODQoIli|!ZzSsbGBgq", "000011111225566899")


def is_blank(value):
    return value is None or pd.isna(value) or str(value).strip() == ""


def normalize(text, drop_suffixes=True):
    """Lowercase, turn '&' into 'and', drop punctuation and trailing legal endings.

    "Halvorsen Ind., Inc." -> "halvorsen ind"
    """
    if is_blank(text):
        return ""
    text = str(text).lower().replace("&", " and ").replace(".", "")
    words = re.sub(r"[^\w\s]", " ", text).split()
    while drop_suffixes and words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def is_ditto(value):
    return not is_blank(value) and str(value).strip().lower() in DITTO_MARKS


def split_client_field(value):
    """Split "client / matter" typed into the client field. Returns (client, matter or None)."""
    if is_blank(value):
        return None, None
    parts = FIELD_SEPARATOR.split(str(value).strip(), maxsplit=1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return str(value).strip(), None


def find_cmid_in_text(text):
    """First CMID-shaped number in a narrative ("7215/003" in an email body), or None."""
    if is_blank(text):
        return None
    found = CMID_IN_TEXT.search(str(text))
    return f"{found.group(1)}.{found.group(2)}" if found else None


def _parse_cmid(text):
    found = CMID_PATTERN.match(text.strip())
    return f"{found.group(1)}.{found.group(2).zfill(3)}" if found else None


def clean_cmid(text):
    """Rewrite a typed CMID as "####.###" ("4471.1" -> "4471.001"), fixing OCR letters if needed.

    Returns None when the text doesn't look like a CMID. Doesn't check that the CMID exists.
    """
    if is_blank(text):
        return None
    return _parse_cmid(str(text)) or _parse_cmid(str(text).translate(OCR_DIGITS))


def cmid_status(typed, cleaned, valid_cmids):
    """missing, unparseable, invalid (no such matter), reformatted, or valid."""
    if is_blank(typed):
        return "missing"
    if cleaned is None:
        return "unparseable"
    if cleaned not in valid_cmids:
        return "invalid"
    return "valid" if str(typed).strip() == cleaned else "reformatted"


def flip_cmid(cleaned):
    """Read the matter number with its zeros on the other side: "4471.100" -> "4471.001"."""
    if is_blank(cleaned):
        return None
    client_no, matter_no = cleaned.split(".")
    flipped = matter_no.rstrip("0").zfill(3) if matter_no.strip("0") else matter_no
    return f"{client_no}.{flipped}" if flipped != matter_no else None


def cmid_similarity(a, b):
    """0-100 similarity of two CMIDs' digits. Swapping two neighboring digits counts as one typo."""
    if is_blank(a) or is_blank(b):
        return 0.0
    return OSA.normalized_similarity(re.sub(r"\D", "", a), re.sub(r"\D", "", b)) * 100


def clean_entries(entries):
    """The typed fields after cleanup. A ditto counts as blank, a one-field entry is split,
    and a CMID found in the narrative fills a blank CMID field.

    entries needs columns: cmid, client, matter, narrative.
    """
    ditto = entries["client"].map(is_ditto)
    split = entries["client"].where(~ditto).map(split_client_field)
    matter_blank = entries["matter"].map(is_blank)
    cmid_blank = entries["cmid"].map(is_blank)
    cmid_in_text = entries["narrative"].map(find_cmid_in_text)

    return pd.DataFrame({
        "typed_cmid": entries["cmid"].where(~cmid_blank, cmid_in_text),
        "typed_client": split.str[0],
        "typed_matter": entries["matter"].where(~matter_blank, split.str[1]),
        "client_is_ditto": ditto.astype(int),
        "client_had_matter": split.str[1].notna().astype(int),
        "cmid_from_text": (cmid_blank & cmid_in_text.notna()).astype(int),
    }, index=entries.index)
