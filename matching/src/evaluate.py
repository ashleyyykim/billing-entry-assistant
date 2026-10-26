"""Evaluation: metrics for each stage, slices, failure types, checks on every decision, stress tests, and gates.

Every function here takes decisions already scored against the answer key (rules.score), so the same code
grades VAL now and TEST and the holdout file in the final run.
"""
import hashlib
import operator

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from classifier import CLUES, confidence
from clean import is_blank
from fuzzy import match
from rules import ACTIONS, OUTCOMES, precision_ci, summarize


# Gates, set before the final run (notebook 05, section 9). Each: (metric, operator, line, where the line comes from, why).
# Blockers have to pass; tracked lines are reported but don't fail the run.
GATES = {
    "decided precision, low end of range": ("ci_low", ">=", 0.95, "97% target", "the 97% target, allowing for about 600 entries"),
    "silent errors per auto-match": ("silent_rate", "<=", 0.03, "97% target", "at most 3 in 100 auto-matches go on the bill wrong"),
    "decided precision above the rules": ("ci_low", ">", "rules_decided_precision", "baseline", "clearly better than the rules, not by noise"),
    "fewer silent errors than the rules": ("silent_errors", "<", "rules_silent_errors", "baseline", "the main reason to replace the rules"),
    "decisions breaking a check": ("violations", "<=", 0, "principle", "every decision follows the checks in check_decisions"),
    "decisions changed by formatting": ("formatting_changed", "<=", 0, "principle", "case and spacing shouldn't change anything"),
    "new-client calls right": ("no_match_precision", ">=", 0.90, "team judgment", "a wrong call sends a real client to intake"),
    "multi-matter entries caught": ("multi_caught", ">=", 0.70, "team judgment", "a missed second matter is misbilled or lost time"),
    "right matter in review shortlist": ("shortlist_hit", ">=", 0.80, "team judgment", "a review should be a quick pick from three"),
    "top-pick calibration error": ("calibration_error", "<=", 0.05, "team judgment", "a pick shown as 90% sure is right about 9 in 10 times"),
}
TRACKED = {
    "worst stress test, decided precision": ("stress_worst_precision", ">=", 0.95, "97% target", "messier input should mean more review, not more errors"),
    "worst source format, decided precision": ("format_worst_precision", ">=", 0.90, "team judgment", "no input format far behind the rest"),
}


# ---------------- Stage metrics ----------------

def calibration_error(y, p, bins=10):
    """Expected calibration error: the average gap between predicted probability and how often it was right,
    weighted by how many pairs fall in each probability bin (0 = perfectly calibrated)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return float((g["n"] * (g["predicted"] - g["actual"]).abs()).sum() / g["n"].sum())


def parse_metrics(parsed, answers, client_names, min_score=80):
    """Extraction (emails): share with hours, share with a client filled in, and whether that client points to the
    right one (same fuzzy match as notebook 01). parsed: notebook 00's email rows (client, parsed_hours)."""
    df = parsed.merge(answers[["entry_id", "key_true_client", "key_true_cmid"]], on="entry_id")
    top = df["client"].map(lambda c: match(c, client_names, k=1) if not is_blank(c) else [])
    pick = top.map(lambda t: t[0][0] if t and t[0][1] >= min_score else None)
    known = df["key_true_cmid"] != "NONE"
    return {"emails": len(df), "hours_found": float(df["parsed_hours"].notna().mean()),
            "client_filled": float(df["client"].notna().mean()),
            "right_client": float((pick[known] == df.loc[known, "key_true_client"]).mean()),
            "wrong_client": float((pick[known].notna() & (pick[known] != df.loc[known, "key_true_client"])).mean())}


def pair_metrics(pairs, prob_col="prob"):
    """Step 1 (pair model): how often the right matter is ranked first or in the top 3, and how good the probabilities are.
    Calibration uses each entry's top pick, the probability a reviewer sees. Across all pairs it would look
    near perfect only because most pairs are obvious no's close to 0."""
    df = pairs[["entry_id", "is_true_matter", prob_col]].copy()
    df["rank"] = df.groupby("entry_id")[prob_col].rank(ascending=False, method="first")
    has_true = df.groupby("entry_id")["is_true_matter"].transform("max") == 1
    best_rank = df[has_true & (df["is_true_matter"] == 1)].groupby("entry_id")["rank"].min()
    top = df[df["rank"] == 1]
    return {"entries": int(df["entry_id"].nunique()), "top_1": float((best_rank <= 1).mean()),
            "top_3": float((best_rank <= 3).mean()), "log_loss": float(log_loss(df["is_true_matter"], df[prob_col])),
            "calibration_error": calibration_error(top["is_true_matter"], top[prob_col])}


def action_table(scored):
    """Step 2 (decisions): for each action, how often the call is the expected one (precision), how often entries
    that should get it do (recall), and for calls that suggest a matter, how often that matter is right."""
    rows = {}
    for action in ACTIONS:
        called = scored["action"] == action
        expected = scored["key_expected_action"] == action
        rows[action] = {"calls": int(called.sum()), "expected": int(expected.sum()),
                        "precision": (called & expected).sum() / called.sum() if called.any() else np.nan,
                        "recall": (called & expected).sum() / expected.sum() if expected.any() else np.nan,
                        "matter_right": scored.loc[called, "matter_right"].mean() if action != "NO_MATCH" and called.any() else np.nan}
    return pd.DataFrame(rows).T


def headline(scored):
    """The numbers every slice, stress test, and gate uses."""
    s = summarize(scored)
    decided = scored["action"].isin(["AUTO_MATCH", "NO_MATCH"])
    low, high = precision_ci(scored) if decided.sum() >= 2 else (np.nan, np.nan)
    auto = scored["action"] == "AUTO_MATCH"
    return {"entries": s["entries"], "decided_precision": s["decided_precision"], "ci_low": float(low), "ci_high": float(high),
            "straight_through": s["straight_through"], "review_share": s["review_share"],
            "matter_accuracy": float(scored["matter_right"].mean()), "action_accuracy": s["action_accuracy"],
            "silent_errors": s["silent_errors"], "silent_rate": s["silent_errors"] / auto.sum() if auto.any() else np.nan,
            "multi_caught": s["multi_caught"], "no_match_precision": s["no_match_precision"],
            "shortlist_hit": s["shortlist_hit"]}


def slice_table(scored, by, min_entries=10):
    """headline() for each group of entries (source format, entry type, ...). Small groups are left out."""
    rows = {name: headline(g) for name, g in scored.groupby(by) if len(g) >= min_entries}
    return pd.DataFrame(rows).T


# ---------------- Failure types (error analysis) ----------------

FAILURE_TYPES = {
    "silent error": "accepted with no review, but the matter is wrong",
    "wrong new-client call": "sent to intake as a new client or matter, but it's a known one",
    "missed second matter": "covers two matters, but wasn't called multi-matter",
    "false multi-matter alarm": "called multi-matter, but covers one matter",
    "wrong flag suggestion": "flagged, but the suggested matter is wrong",
    "review, right matter not in shortlist": "sent to review, and the right matter isn't in the top 3",
    "review, top pick right": "sent to review, but the top pick was right (extra work, not an error)",
}


def failure_type(row):
    """The first failure type that applies to one scored decision, or None if the decision did its job."""
    action, expected = row["action"], row["key_expected_action"]
    if action == "AUTO_MATCH" and row["outcome"] == OUTCOMES[1]:
        return "silent error"
    if action == "NO_MATCH" and expected != "NO_MATCH":
        return "wrong new-client call"
    if expected == "MULTI_MATTER" and action != "MULTI_MATTER":
        return "missed second matter"
    if action == "MULTI_MATTER" and expected != "MULTI_MATTER":
        return "false multi-matter alarm"
    if action == "FLAG_MISMATCH" and expected != "NO_MATCH" and not row["matter_right"]:
        return "wrong flag suggestion"
    if action == "NEEDS_REVIEW" and expected != "NO_MATCH":
        return "review, top pick right" if row["top_right"] else "review, right matter not in shortlist" if not row["in_shortlist"] else None
    return None


def add_failure_types(scored):
    top_pick = scored["shortlist"].str.split(" | ", regex=False).str[0]
    out = scored.assign(top_right=top_pick == scored["key_true_cmid"])
    out["failure_type"] = out.apply(failure_type, axis=1)
    return out


# ---------------- Checks on every decision ----------------

def _row_clues(row, phrase_min):
    """The clue values reasons are built from, for one decision's top pick (same names as classifier.explain)."""
    return {**row, "client_named": row["client_in_text"] >= phrase_min}


def check_decisions(scored, pairs, valid_cmids, client_col, matter_col, phrase_min=90):
    """Rules every decision has to follow, checked one by one. Returns one row per check with the entries that break it.

    pairs: the pairs table (with prob) the decisions came from. scored needs reason_short and reason_full.
    """
    top = scored["shortlist"].str.split(" | ", regex=False).str[0]
    pick = pairs.merge(pd.DataFrame({"entry_id": scored["entry_id"], "cmid": top}), on=["entry_id", "cmid"])
    pick = pick.rename(columns={client_col: "client_score", matter_col: "matter_score"}).set_index("entry_id")
    df = scored.set_index("entry_id").join(pick[["prob", "not_open", "is_typed_cmid", "cmid_valid", "cmid_clean",
                                                 "past_narrative_sim", "in_history", "own_keyterm_hits", "client_in_text",
                                                 "client_score", "matter_score", "cue_share"]])
    auto = df["action"] == "AUTO_MATCH"
    preds = df["pred_cmid"].fillna("").str.split(" | ", regex=False)
    typed_valid = df["cmid_clean"].isin(valid_cmids)

    clue_check = {short: present for present, short, _ in CLUES}
    reason_ok = [clue_check[r["reason_short"]](_row_clues(r, phrase_min)) if r["reason_short"] in clue_check else True
                 for r in df[auto].reset_index().to_dict("records")]
    stated = df["reason_full"].str.extract(r"(high|medium|low) confidence")[0]
    expected_conf = df["prob"].map(confidence)

    checks = {
        "suggested matters exist in Matter Master": ~preds.map(lambda p: all(c in valid_cmids for c in p if c)),
        "auto-matched matter was open on the work date": auto & (df["not_open"] == 1),
        "auto-match against a valid typed CMID says it corrected it": (auto & typed_valid & (df["cmid_clean"] != df["pred_cmid"])
                                                                       & ~df["reason_full"].str.contains("correcting the typed CMID", na=False)),
        "NO_MATCH suggests no matter": (df["action"] == "NO_MATCH") & df["pred_cmid"].notna(),
        "every decision has a reason": df["reason_short"].map(is_blank) | df["reason_full"].map(is_blank),
        "auto-match reason is true for that entry": pd.Series(~np.array(reason_ok, dtype=bool), index=df.index[auto]).reindex(df.index, fill_value=False),
        "stated confidence matches the probability": stated.notna() & (stated != expected_conf),
        "review shortlist has 3 different matters": (df["action"] == "NEEDS_REVIEW") & (df["shortlist"].str.split(" | ", regex=False).map(lambda s: len(set(s))) < 3),
    }
    rows = [{"check": name, "entries_checked": int(len(df)), "violations": int(bad.sum()),
             "examples": ", ".join(bad[bad].index[:3])} for name, bad in checks.items()]
    return pd.DataFrame(rows).set_index("check")


# ---------------- Stress tests ----------------

OCR_SWAPS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8", "m": "rn", "e": "c", "a": "o", "i": "l"}


def _ocr(text, rate, rng):
    if is_blank(text):
        return text
    return "".join(OCR_SWAPS[c] if c in OCR_SWAPS and rng.random() < rate else c for c in str(text))


def _typo(text, rng):
    """One typo in a word of 4+ letters: two neighboring letters swapped or one letter dropped."""
    if is_blank(text):
        return text
    words = str(text).split()
    long = [i for i, w in enumerate(words) if len(w) >= 4]
    if not long:
        return text
    i = rng.choice(long)
    w, j = words[i], rng.integers(1, len(words[i]) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:] if rng.random() < 0.5 else w[:j] + w[j + 1:]
    return " ".join(words)


def _swap_digits(cmid, rng):
    """Two neighboring digits swapped ("6103.004" -> "6013.004"), like a typing slip."""
    if is_blank(cmid):
        return cmid
    s = str(cmid)
    spots = [i for i in range(len(s) - 1) if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1]]
    if not spots:
        return cmid
    i = rng.choice(spots)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def stress_variants(entries, seed=210, ocr_rate=0.1):
    """Changed copies of the entries, each testing one kind of messiness. Same entry ids, so the answer key still applies.

    The first two shouldn't change any decision (invariance); the rest should make things harder, not break them.
    """
    rng = np.random.default_rng(seed)
    e = entries
    return {
        "same, upper case narrative": e.assign(narrative=e["narrative"].str.upper()),
        "same, extra spaces": e.assign(narrative=e["narrative"].str.replace(" ", "  ", regex=False) + " "),
        "OCR noise (10% of letters)": e.assign(narrative=e["narrative"].map(lambda t: _ocr(t, ocr_rate, rng)),
                                              client=e["client"].map(lambda t: _ocr(t, ocr_rate, rng)),
                                              cmid=e["cmid"].map(lambda t: _ocr(t, ocr_rate, rng))),
        "typo in client name": e.assign(client=e["client"].map(lambda t: _typo(t, rng))),
        "typo in narrative": e.assign(narrative=e["narrative"].map(lambda t: _typo(t, rng))),
        "CMID digits swapped": e.assign(cmid=e["cmid"].map(lambda t: _swap_digits(t, rng))),
        "CMID left blank": e.assign(cmid=None),
        "client left blank": e.assign(client=None, matter=None),
        "new timekeeper (no history)": e.assign(tk_id="TK-NEW"),
    }


def changed_decisions(base, other):
    """Share of entries whose action or suggested matter changed between two runs."""
    m = base[["entry_id", "action", "pred_cmid"]].merge(other[["entry_id", "action", "pred_cmid"]], on="entry_id")
    same = (m["action_x"] == m["action_y"]) & (m["pred_cmid_x"].fillna("") == m["pred_cmid_y"].fillna(""))
    return float(1 - same.mean())


# ---------------- Gates and the saved report ----------------

OPERATORS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt}


def check_gates(results, gates):
    """Pass or fail for each gate. gates: {name: (metric key in results, operator, limit, source, why)}.
    limit is a number, or the key of another result to compare with (like the rules' precision)."""
    rows = []
    for name, (key, op, limit, source, why) in gates.items():
        value = results[key]
        if isinstance(limit, str):
            line, rule = results[limit], f"{op} {limit} ({results[limit]:.3f})"
        else:
            line, rule = limit, f"{op} {limit}"
        rows.append({"gate": name, "value": round(float(value), 3), "rule": rule, "passed": bool(OPERATORS[op](value, line)),
                     "source": source, "why": why})
    return pd.DataFrame(rows).set_index("gate")


def dataset_hash(df, cols):
    """A short fingerprint of an evaluation set, so a changed set can't pass as the same one."""
    text = df[cols].astype(str).sort_values(cols[0]).to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def compare_reports(new, old, keys):
    """New vs. saved numbers side by side, for the regression check after any change."""
    return pd.DataFrame({"saved": {k: old.get(k) for k in keys}, "now": {k: new.get(k) for k in keys}}).assign(
        change=lambda d: d["now"] - d["saved"])
