"""Comparing extracted entries with a hand-typed answer key.

Clients are compared by similarity rather than exactly, since misread names ("Halwren Ind.") are left for matching
to sort out. CMIDs are compared by their digits, so "4590-003" and "4590.003" count as the same.
"""
import pandas as pd
from rapidfuzz import fuzz


def similar(found, truth, min_score=80):
    """True when both are blank, or the found text is at least min_score (0 to 100) similar to the truth."""
    if pd.isna(found) or pd.isna(truth) or found == "" or truth == "":
        return (pd.isna(found) or found == "") and (pd.isna(truth) or truth == "")
    return fuzz.ratio(str(found).lower(), str(truth).lower()) >= min_score


def digits(cmids):
    """A column of CMIDs with everything but the digits removed."""
    return cmids.fillna("").astype(str).str.replace(r"\D", "", regex=True)
