"""Comparing extracted entries with a hand-typed answer key.

Clients are compared by similarity rather than exactly, since misread names ("Halwren Ind.") are left for matching
to sort out. A full client name for a short one written ("Cobalt Ridge Capital" for "Cobalt Ridge") counts, and so
does "Firm" on firm time, where the key leaves the client blank. CMIDs are compared by their digits, so "4590-003" and "4590.003" count as the same.

Photos are graded entry by entry after lining each photo's entries up with its answer key in order (align), so a page
that loses an entry doesn't throw off the entries after it. Each key entry gets one outcome: right, wrong but flagged,
wrong and silent (no check reason), or missing. Found entries with no match in the key are extra.
"""
import pandas as pd
from rapidfuzz import fuzz

OUTCOMES = ["right", "wrong, flagged", "wrong, silent", "missing", "extra"]


def similar(found, truth, min_score=80):
    """True when the found text is close enough to the truth for matching to use it.

    A blank truth is matched by a blank or by "Firm..." (firm time). Otherwise the found text must be at least
    min_score (0 to 100) similar to the truth, or start with it (a full name for a short one).
    """
    found_blank = pd.isna(found) or found == ""
    if pd.isna(truth) or truth == "":
        return found_blank or str(found).lower().startswith("firm")
    if found_blank:
        return False
    found, truth = str(found).lower(), str(truth).lower()
    return fuzz.ratio(found, truth) >= min_score or found.startswith(truth + " ")


def digits(cmids):
    """A column of CMIDs with everything but the digits removed."""
    return cmids.astype(object).where(cmids.notna(), "").astype(str).str.replace(r"\D", "", regex=True)


def same_cmid(found, truth):
    return digits(pd.Series([found]))[0] == digits(pd.Series([truth]))[0]


def score(found, truth):
    """How much a found entry agrees with a key entry (0 to 4): hours, client, CMID, narrative."""
    same_hours = pd.notna(found["hours"]) and pd.notna(truth["hours"]) and round(found["hours"], 2) == round(
        truth["hours"], 2)
    return (same_hours + similar(found["client"], truth["client"]) + (pd.notna(truth["cmid"]) and
            same_cmid(found["cmid"], truth["cmid"])) + similar(found["narrative"], truth["narrative"], 60))


def align(found, wanted):
    """Pairs (found position or None, key position or None) in order, matching entries that agree on something."""
    n, m = len(found), len(wanted)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            s = score(found.iloc[i], wanted.iloc[j])
            best[i][j] = max(best[i + 1][j], best[i][j + 1], best[i + 1][j + 1] + s if s else 0)
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        s = score(found.iloc[i], wanted.iloc[j])
        if s and best[i][j] == best[i + 1][j + 1] + s:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            pairs.append((i, None))
            i += 1
        else:
            pairs.append((None, j))
            j += 1
    return pairs + [(a, None) for a in range(i, n)] + [(None, b) for b in range(j, m)]


def is_flagged(entry):
    return pd.notna(entry["check_reasons"]) and entry["check_reasons"] != ""


def grade_photos(found, key):
    """One row per key entry (plus extras) with its outcome.

    found: extracted entries with a photo column. key: the answer key, one row per entry with photo and entry,
    work_date already filled in. Any other key columns (like batch) are carried along.
    """
    extra_cols = [c for c in key.columns if c not in ("entry", "work_date", "hours", "client", "cmid", "narrative",
                                                       "note")]
    out = []
    for photo, wanted in key.groupby("photo", sort=False):
        got = found[found["photo"] == photo].reset_index(drop=True)
        wanted = wanted.reset_index(drop=True)
        carried = {c: wanted[c][0] for c in extra_cols}
        for i, j in align(got, wanted):
            f = got.iloc[i] if i is not None else None
            k = wanted.iloc[j] if j is not None else None
            if k is None:
                out.append({**carried, "outcome": "extra", "found": f"{f['client']} {f['hours']}",
                            "check": f["check_reasons"], "flagged": is_flagged(f), "min_conf": f.get("min_conf")})
            elif f is None:
                out.append({**carried, "entry": k["entry"], "outcome": "missing"})
            else:
                right = {"hours": pd.notna(f["hours"]) and round(f["hours"], 2) == round(k["hours"], 2),
                         "client": similar(f["client"], k["client"]),
                         "cmid": same_cmid(f["cmid"], k["cmid"]),
                         "date": f["work_date"] == k["work_date"]}
                wrong = [name for name, ok in right.items() if not ok]
                flagged = is_flagged(f)
                outcome = "right" if not wrong else "wrong, flagged" if flagged else "wrong, silent"
                out.append({**carried, "entry": k["entry"], "outcome": outcome, "wrong": ", ".join(wrong),
                            "found": f"{f['client']} {f['cmid']} {f['hours']}", "check": f["check_reasons"],
                            "flagged": flagged, "min_conf": f.get("min_conf")})
    graded = pd.DataFrame(out)
    if "entry" in graded:
        graded["entry"] = graded["entry"].astype("Int64")
    return graded


COUNTS = ["right, no flag", "right, flagged", "wrong, flagged", "wrong, silent", "missing", "extra, flagged",
          "extra, silent", "to review", "silent errors"]


def summarize(graded):
    """What a reviewer would see: entries that go straight through right, entries sent to review (right or wrong),
    and silent errors (wrong with no flag, missed, or made up with no flag)."""
    flagged = graded["flagged"].fillna(False).astype(bool)
    outcome = graded["outcome"]
    counts = {
        "right, no flag": (outcome == "right") & ~flagged,
        "right, flagged": (outcome == "right") & flagged,
        "wrong, flagged": outcome == "wrong, flagged",
        "wrong, silent": outcome == "wrong, silent",
        "missing": outcome == "missing",
        "extra, flagged": (outcome == "extra") & flagged,
        "extra, silent": (outcome == "extra") & ~flagged,
    }
    out = pd.Series({name: int(mask.sum()) for name, mask in counts.items()})
    out["to review"] = out["right, flagged"] + out["wrong, flagged"] + out["extra, flagged"]
    out["silent errors"] = out["wrong, silent"] + out["missing"] + out["extra, silent"]
    return out


def case_outcome(entries, expected):
    """One stress case's outcome: right when every expected entry is found with its client, hours, and
    CMID, and nothing else; a case expected to be unreadable (expected None) is right when it's flagged."""
    from clean import clean_cmid, find_cmid_in_text

    def same(found, want):
        typed = found["cmid"] if pd.notna(found["cmid"]) else find_cmid_in_text(found["narrative"])
        return (similar(found["client"], want["client"]) and pd.notna(found["hours"])
                and abs(found["hours"] - want["hours"]) < 0.01 and clean_cmid(typed) == want["cmid"])

    flagged = len(entries) > 0 and bool((entries["check_reasons"].fillna("") != "").any())
    if expected is None:
        right = flagged
    else:
        right = len(entries) == len(expected) and all(same(f, w) for f, w in zip(entries.to_dict("records"), expected))
    return "right" if right else ("wrong, flagged" if flagged else "wrong, silent")


def pick_cutoff(confs, right, share_if_none_wrong=0.1):
    """A confidence cutoff picked on tuning notes: just above the most confident wrong value, so every wrong one there
    is caught. With none wrong, the least confident share of right ones is flagged."""
    confs, right = pd.Series(confs, dtype=float), pd.Series(right, dtype=bool)
    wrong = confs[~right].dropna()
    if len(wrong):
        return round(wrong.max() + 0.01, 2)
    return round(confs.dropna().quantile(share_if_none_wrong), 2)
