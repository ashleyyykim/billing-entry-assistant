"""Benchmark metrics for graded entries (grade.grade_photos output), the same for every input and reader.

Per key entry: found (not missing), right (hours, client, CMID, and date all right), right with no flag, and silent
error (wrong with no flag, or missing). Extras (entries with no key entry) are counted separately: a flagged extra
costs review, an unflagged one is a silent error. Field accuracy is over found entries only, so a reader that finds
fewer entries isn't rewarded for it there; "found" shows that.

Ranges are 95% bootstrap intervals that resample whole photos or files (the unit column), since entries on one page
succeed or fail together. With a few dozen entries the ranges are wide, and they should be reported with the numbers.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from grade import summarize

FIELDS = ["hours", "client", "cmid", "date"]
MIN_UNITS = 3    # fewer photos or files than this and a resampled range means nothing


def field_right(graded):
    """For each found key entry, whether each field is right (a DataFrame of booleans, one column per field)."""
    found = graded[graded["outcome"].isin(["right", "wrong, flagged", "wrong, silent"])]
    wrong = found["wrong"].fillna("").map(lambda s: {f.strip() for f in s.split(",") if f.strip()})
    return pd.DataFrame({f: ~wrong.map(lambda w, f=f: f in w) for f in FIELDS}, index=found.index)


def rates(graded, unit="photo"):
    """The headline numbers for one reader on one input, as percents of key entries (extras as a count)."""
    key = graded[graded["outcome"] != "extra"]
    n = len(key)
    c = summarize(graded)
    fields = field_right(graded)
    out = {"entries": n, "units": graded[unit].nunique(),
           "found %": 100 * (key["outcome"] != "missing").mean() if n else np.nan,
           "right %": 100 * (key["outcome"] == "right").mean() if n else np.nan,
           "right, no flag %": 100 * c["right, no flag"] / n if n else np.nan,
           "to review %": 100 * c["to review"] / n if n else np.nan,
           "silent errors %": 100 * c["silent errors"] / n if n else np.nan,
           "extras": int((graded["outcome"] == "extra").sum())}
    out.update({f"{f} right %": 100 * fields[f].mean() if len(fields) else np.nan for f in FIELDS})
    return out


def interval(graded, metric, n=1000, seed=0, unit="photo"):
    """A 95% range for one of rates()'s percents, resampling whole units (photos or files) with replacement. None
    with fewer than MIN_UNITS units."""
    units = graded[unit].unique()
    if len(units) < MIN_UNITS:
        return None
    groups = {u: graded[graded[unit] == u] for u in units}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        pick = rng.choice(units, size=len(units), replace=True)
        sample = pd.concat([groups[u] for u in pick], ignore_index=True)
        values.append(rates(sample, unit)[metric])
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return round(low), round(high)


def with_ranges(graded, metrics=("right, no flag %", "silent errors %"), unit="photo", **kwargs):
    """rates() with a "low-high" range next to each of the given metrics."""
    out = rates(graded, unit)
    for metric in metrics:
        found = interval(graded, metric, unit=unit, **kwargs)
        out[f"{metric} range"] = f"{found[0]}-{found[1]}" if found else "n/a"
    return out


def model_cost(answers, prices):
    """Average dollars and seconds per page from saved model answers (usage and latency), given (input, output)
    dollars per million tokens. None for answers without usage."""
    used = [a for a in answers if a and "usage" in a and "error" not in a]
    if not used:
        return {"$ per page": None, "seconds per page": None}
    tokens_in = sum(a["usage"]["inputTokens"] for a in used) / len(used)
    tokens_out = sum(a["usage"]["outputTokens"] for a in used) / len(used)
    dollars = (tokens_in * prices[0] + tokens_out * prices[1]) / 1e6 if prices else None
    return {"$ per page": None if dollars is None else round(dollars, 4),
            "seconds per page": round(sum(a.get("latency_ms", 0) for a in used) / len(used) / 1000, 1)}


# ---- Are two readers really different? (McNemar's test, as Dietterich 1998 recommends for one shared test set) ----

def per_entry(graded, metric, unit="photo"):
    """For each answer-key entry (photo, entry), whether it counts for the metric: "right, no flag" (right and sent
    straight through) or "silent error" (wrong with no flag, or missing). Extras have no key entry and are left out."""
    key = graded[graded["outcome"] != "extra"]
    flagged = key["flagged"].fillna(False).astype(bool)
    if metric == "right, no flag":
        hit = (key["outcome"] == "right") & ~flagged
    elif metric == "silent error":
        hit = key["outcome"].isin(["wrong, silent", "missing"])
    else:
        raise ValueError(metric)
    return pd.Series(hit.values, index=pd.MultiIndex.from_arrays([key[unit], key["entry"].astype(int)]))


def mcnemar(graded_a, graded_b, metric="right, no flag", unit="photo"):
    """McNemar's exact test on the key entries both readers were graded on. Only entries where the two disagree count:
    only_a (A hits, B doesn't) and only_b. The p-value is two-sided, from the binomial distribution, so it's valid for
    small counts. A small p (under 0.05) means the difference is unlikely to be luck."""
    a, b = per_entry(graded_a, metric, unit), per_entry(graded_b, metric, unit)
    a, b = a.align(b, join="inner")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
    return {"entries": int(len(a)), "only first": only_a, "only second": only_b, "p": float(f"{min(1.0, 2 * tail):.2g}")}


def entries_needed(discordant, difference, alpha=0.05, power=0.8):
    """About how many paired entries a McNemar test needs to detect a difference (as a share, 0.10 for 10 points)
    when the two readers disagree on a share `discordant` of entries (the usual normal approximation)."""
    if difference <= 0 or discordant <= difference:
        return None
    z_a, z_b = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    n = (z_a * math.sqrt(discordant) + z_b * math.sqrt(discordant - difference ** 2)) ** 2 / difference ** 2
    return math.ceil(n)


# ---- Through matching: what reaches the bill ----

BILL_OUTCOMES = ["billed right", "to a person", "billed wrong, unseen", "lost"]


def pair_found(found, key):
    """Each found entry next to the key entry it lines up with (grade.align), with whether its hours and date are
    right and whether extraction sends it to review. Key entries nobody found come back with no entry_id (lost); found
    entries with no key entry come back with no key entry (extra)."""
    from grade import align, is_flagged
    rows = []
    for photo, wanted in key.groupby("photo", sort=False):
        got = found[found["photo"] == photo].reset_index(drop=True)
        wanted = wanted.reset_index(drop=True)
        for i, j in align(got, wanted):
            f = got.iloc[i] if i is not None else None
            k = wanted.iloc[j] if j is not None else None
            rows.append({
                "photo": photo, "entry": None if k is None else int(k["entry"]),
                "entry_id": None if f is None else f["entry_id"],
                "hours_ok": bool(f is not None and k is not None and pd.notna(f["hours"])
                                 and round(f["hours"], 2) == round(k["hours"], 2)),
                "date_ok": bool(f is not None and k is not None and f["work_date"] == k["work_date"]),
                "extraction_review": bool(f is not None and is_flagged(f))})
    return pd.DataFrame(rows)


def bill_outcome(row):
    """What happens to one entry once matching has decided: "lost" (extraction never found it), "to a person"
    (extraction or matching sends it to review), "billed right" (matched to the intended matter with the right hours
    and date and nobody looked), or "billed wrong, unseen" (anything else that went straight through, including a
    made-up entry). row needs entry, entry_id, hours_ok, date_ok, extraction_review, action, top_pick, intended."""
    if pd.isna(row["entry_id"]):
        return "lost"
    if row["extraction_review"] or row["action"] != "AUTO_MATCH":
        return "to a person"
    if pd.notna(row["entry"]) and row["top_pick"] == row["intended"] and row["hours_ok"] and row["date_ok"]:
        return "billed right"
    return "billed wrong, unseen"
