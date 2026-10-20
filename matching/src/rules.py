"""Rule baseline: route each entry to AUTO_MATCH, FLAG_MISMATCH, NEEDS_REVIEW, MULTI_MATTER, or NO_MATCH.

Works on the pairs table from notebook 02. Each entry is decided from its own rows only (g below).
"""
import numpy as np
import pandas as pd

ACTIONS = ["AUTO_MATCH", "FLAG_MISMATCH", "NEEDS_REVIEW", "MULTI_MATTER", "NO_MATCH"]
OUTCOMES = ["Accepted, right", "Accepted, wrong", "Flagged", "Sent to review", "Multi-matter", "No match"]

# Settings are explained in notebook 03, section 3
DEFAULTS = {
    "client_col": "client_ratio",
    "matter_col": "matter_token_sort_ratio",
    "client_min": 80,
    "matter_min": 90,
    "no_match_max": 80,
    "phrase_min": 90,
    "multi_other_client": False,
    "accept_cue_only": True,
    "require_history": False,
    "flag_cue_conflict": False,
    "cue_fit_agrees": False,
    "history_within_client": False,
    "bad_cmid_use_text": True,
    "bad_cmid_accept_closest": False,
    "bad_cmid_fallback": "NEEDS_REVIEW",
    "shortlist_size": 3,
}

RULE_COLS = ["entry_id", "cmid", "client", "typed_client", "cmid_status", "has_cmid", "has_client", "has_matter",
             "cmid_valid", "cmid_client_known", "is_typed_cmid", "is_flipped_cmid", "cmid_similarity",
             "same_client_number", "keyterm_hits", "own_keyterm_hits", "n_matters_named", "cue_hits", "cue_share",
             "n_parts", "client_in_text", "n_clients_named", "not_open",
             "opened_after_history", "tk_known", "in_history"]


def _top(g, col, floor=0.0):
    """Rows tied for the highest value of col (none if the best is 0 or below floor)."""
    best = g[col].max() if len(g) else 0
    return g[g[col] == best] if best > 0 and best >= floor else g.iloc[0:0]


def _one(rows):
    return rows["cmid"].iloc[0] if len(rows) == 1 else None


def _row(g, cmid):
    return g[g["cmid"] == cmid].iloc[0]


def looks_like_acronym(typed_client):
    """A one-word client field of 2-4 letters ("prh"): shorthand for a client, not a new one."""
    words = str(typed_client).split()
    return len(words) == 1 and 2 <= len(words[0]) <= 4


def find_client(g, cfg):
    """The candidates at the client the entry seems to be for, and where that came from.
    Checks the typed client, the CMID's client number, then a client named in the narrative."""
    e = g.iloc[0]
    best_client = g[cfg["client_col"]].max()
    if e["has_client"] and best_client >= cfg["client_min"]:
        tied = g.loc[g[cfg["client_col"]] == best_client, "client"].unique()
        return g[g["client"].isin(tied)], "typed client"
    if e["has_cmid"] and g["same_client_number"].max() == 1:
        return g[g["same_client_number"] == 1], "CMID client number"
    named = g.loc[g["client_in_text"] >= cfg["phrase_min"], "client"].unique()
    if len(named) == 1:
        return g[g["client"] == named[0]], "client named in narrative"
    return g.iloc[0:0], None


def typed_vote(g, cfg):
    """The matter the typed fields point to: a valid CMID, else the typed names. Returns (cmid, source)."""
    e = g.iloc[0]
    if e["cmid_valid"]:
        return _one(g[g["is_typed_cmid"] == 1]), "typed CMID"
    if not e["has_client"] and not e["has_matter"]:
        return None, None
    rows = g
    if e["has_client"]:
        rows = _top(rows, cfg["client_col"], cfg["client_min"])
    if e["has_matter"]:
        rows = _top(rows, cfg["matter_col"], cfg["matter_min"])
    pick = _one(rows)
    return (pick, "typed names") if pick else (None, None)


def text_vote(g, cfg):
    """The matter the narrative points to. Returns (cmid, source).

    1. A key term listed for one matter only.
    2. A shared key term ("Fund II") within the client.
    3. Task cues within the client (or all matters if the client is unknown), ties broken by TK history.
    4. Optional: no cues, and the timekeeper billed only one of the client's matters.
    """
    own = g[g["own_keyterm_hits"] > 0]
    if len(own) == 1:
        return own["cmid"].iloc[0], "key term"
    if len(own) > 1:
        return None, None

    client_rows, _ = find_client(g, cfg)
    pool = client_rows if len(client_rows) else g
    shared = pool[pool["keyterm_hits"] > 0]
    if len(shared) == 1 and len(client_rows):
        return shared["cmid"].iloc[0], "key term + client"

    cued = _top(pool, "cue_share")
    if len(cued) > 1 and cued["in_history"].any():
        cued = cued[cued["in_history"] == 1]
    if _one(cued):
        return _one(cued), "task cue"
    if cfg["history_within_client"] and cued.empty and len(client_rows):
        pick = _one(client_rows[client_rows["in_history"] == 1])
        if pick:
            return pick, "client + TK history"
    return None, None


def text_disagrees(g, cmid, cfg):
    """The narrative argues against cmid: task cues were found, but none fit its type of work."""
    row = _row(g, cmid)
    if cfg["flag_cue_conflict"] and g["cue_hits"].max() > 0 and row["cue_hits"] == 0:
        return True
    return False


def closest_cmid(g):
    """Best guess for a CMID that doesn't exist: the flipped reading, else the closest by digits."""
    return _one(g[g["is_flipped_cmid"] == 1]) or _one(_top(g, "cmid_similarity"))


def typed_name_unknown(g, cfg):
    """A client or matter was typed, nothing in Matter Master is close to it, and nothing else
    (CMID, key term, client in the narrative) points anywhere. The positive signal for NO_MATCH."""
    e = g.iloc[0]
    if e["has_cmid"] or not (e["has_client"] or e["has_matter"]):
        return False
    if g["keyterm_hits"].max() > 0 or e["n_clients_named"] > 0:
        return False
    if e["has_client"] and looks_like_acronym(e["typed_client"]):
        return False
    client_unknown = not e["has_client"] or g[cfg["client_col"]].max() < cfg["no_match_max"]
    matter_unknown = not e["has_matter"] or g[cfg["matter_col"]].max() < cfg["no_match_max"]
    return client_unknown and matter_unknown


def other_client_named(g, cfg):
    """A multi-part narrative names a matter (by its own key term) at a different client. Hints at a second matter."""
    if g["n_parts"].iloc[0] < 2:
        return False
    client_rows, _ = find_client(g, cfg)
    named_clients = set(g.loc[g["own_keyterm_hits"] > 0, "client"])
    return len(client_rows) > 0 and bool(named_clients - set(client_rows["client"]))


def shortlist(g, cfg):
    """Top candidates for a reviewer, strongest evidence first, as "cmid | cmid | cmid"."""
    order = ["own_keyterm_hits", "keyterm_hits", "is_typed_cmid", cfg["client_col"], cfg["matter_col"],
             "client_in_text", "cue_share", "in_history", "cmid_similarity"]
    top = g.sort_values(order + ["cmid"], ascending=[False] * len(order) + [True]).head(cfg["shortlist_size"])
    return " | ".join(top["cmid"])


def cmid_only(g, cfg):
    """Lower-bound baseline: accept a valid typed CMID without reading the narrative, send everything else to a person."""
    out = {"entry_id": g["entry_id"].iloc[0], "shortlist": shortlist(g, cfg)}
    hit = g[g["is_typed_cmid"] == 1]
    if g["cmid_valid"].iloc[0] and len(hit) == 1:
        return out | {"action": "AUTO_MATCH", "pred_cmid": hit["cmid"].iloc[0], "reason": "valid typed CMID"}
    return out | {"action": "NEEDS_REVIEW", "pred_cmid": None, "reason": "no valid typed CMID"}


def decide(g, cfg):
    """Route one entry. The first check that applies decides."""
    e = g.iloc[0]
    typed, typed_src = typed_vote(g, cfg)
    text, text_src = text_vote(g, cfg)
    out = {"entry_id": e["entry_id"], "typed_vote": typed, "typed_source": typed_src,
           "text_vote": text, "text_source": text_src, "shortlist": shortlist(g, cfg)}

    def result(action, cmid, reason):
        return out | {"action": action, "pred_cmid": cmid, "reason": reason}

    def accept(cmid, reason):
        if _row(g, cmid)["not_open"]:
            others = g[(g["cmid"] != cmid) & (g["not_open"] == 0)]
            return result("FLAG_MISMATCH", text_vote(others, cfg)[0], "matter not open on the work date")
        return result("AUTO_MATCH", cmid, reason)

    # 1. More than one matter
    named = sorted(g.loc[g["own_keyterm_hits"] > 0, "cmid"])
    if e["n_matters_named"] >= 2:
        return result("MULTI_MATTER", " | ".join(named), "two or more matters named")
    if cfg["multi_other_client"] and other_client_named(g, cfg):
        both = sorted({c for c in [typed or text, *named] if c})
        return result("MULTI_MATTER", " | ".join(both), "matter at another client named")

    # 2. Typed name not in Matter Master
    if typed_name_unknown(g, cfg):
        return result("NO_MATCH", None, "typed client/matter not in Matter Master")

    # 3. CMID that doesn't exist
    if e["cmid_status"] == "invalid":
        guess = closest_cmid(g)
        if cfg["bad_cmid_use_text"] and text:
            if text == guess:
                return accept(text, f"closest CMID agrees with {text_src}")
            if _row(g, text)["same_client_number"]:
                return accept(text, f"{text_src} picks a matter at the CMID's client")
        client_unknown = not e["has_client"] or g[cfg["client_col"]].max() < cfg["no_match_max"]
        if not e["cmid_client_known"] and client_unknown and e["n_clients_named"] == 0:
            return result("NO_MATCH", None, "CMID client number not in Matter Master")
        if cfg["bad_cmid_accept_closest"] and guess and not text and not text_disagrees(g, guess, cfg):
            return accept(guess, "closest CMID, nothing disagrees")
        if cfg["bad_cmid_fallback"] == "NO_MATCH":
            return result("NO_MATCH", None, "CMID does not exist, nothing else decides")
        return result("NEEDS_REVIEW", guess, "CMID does not exist (closest CMID suggested)")

    # 4. Compare the typed vote with the narrative vote
    if typed and text:
        if typed != text and text_src == "task cue" and cfg["cue_fit_agrees"] and _row(g, typed)["cue_hits"] > 0:
            return accept(typed, f"{typed_src}, its type also fits the task cues")
        if typed != text:
            return result("FLAG_MISMATCH", text, f"{typed_src} disagrees with {text_src}")
        return accept(typed, f"{typed_src} agrees with {text_src}")
    if typed:
        if text_disagrees(g, typed, cfg):
            return result("FLAG_MISMATCH", None, f"{typed_src} only, narrative points away from it")
        row = _row(g, typed)
        no_history = e["tk_known"] and not row["in_history"] and not row["opened_after_history"]
        if cfg["require_history"] and no_history:
            return result("NEEDS_REVIEW", typed, f"{typed_src} only, not in TK history")
        return accept(typed, f"{typed_src} only")
    if text:
        if text_src == "task cue" and not cfg["accept_cue_only"]:
            return result("NEEDS_REVIEW", text, "task cue only")
        return accept(text, f"{text_src} only")

    # 5. Nothing points to one matter
    return result("NEEDS_REVIEW", None, "no single candidate (shortlist only)")


def run_rules(pairs, cfg, decide_fn=decide):
    """One decision per entry."""
    cols = list(dict.fromkeys(RULE_COLS + [cfg["client_col"], cfg["matter_col"]]))
    return pd.DataFrame([decide_fn(g, cfg) for _, g in pairs[cols].groupby("entry_id")])


# The synthetic answer key calls entries that cover more than one matter "splits"
RENAMED = {"SPLIT_ENTRY": "MULTI_MATTER", "SPLIT_NAMED": "MULTI_NAMED", "SPLIT_UNNAMED": "MULTI_UNNAMED"}


def answer_key(pairs):
    """One row per entry: the data split (TRAIN/VAL/TEST) and the answer-key (key_) columns."""
    cols = ["entry_id", "split"] + [c for c in pairs.columns if c.startswith("key_")]
    answers = pairs[cols].drop_duplicates("entry_id").reset_index(drop=True)
    return answers.replace({"key_expected_action": RENAMED, "key_entry_category": RENAMED})


def _cmids(text):
    """"a | b" -> {"a", "b"}. Blank and "NONE" (the NO_MATCH answer) -> empty set."""
    if text is None or pd.isna(text):
        return set()
    return {c for c in str(text).split(" | ") if c and c != "NONE"}


def _outcome(row):
    """An auto-accept is right when all of the time lands on the right matter: the matter is right
    and the entry doesn't cover more than one matter (accepting just one bills part of the time wrong)."""
    if row["action"] == "AUTO_MATCH":
        safe = row["matter_right"] and row["key_expected_action"] != "MULTI_MATTER"
        return OUTCOMES[0] if safe else OUTCOMES[1]
    return {"FLAG_MISMATCH": OUTCOMES[2], "NEEDS_REVIEW": OUTCOMES[3], "MULTI_MATTER": OUTCOMES[4]}.get(row["action"], OUTCOMES[5])


def score(decisions, answers):
    """Attach the answer key and mark whether each action, matter, and shortlist is right.

    The true matters are True CMID plus Secondary CMID (multi-matter entries). When the narrative doesn't name the second
    matter can't be known, so the primary matter is enough.
    """
    df = decisions.merge(answers, on="entry_id", how="left", validate="one_to_one")
    right = []
    for pred, primary, second, category in zip(df["pred_cmid"], df["key_true_cmid"], df["key_secondary_cmid"],
                                               df["key_entry_category"]):
        pred, primary, both = _cmids(pred), _cmids(primary), _cmids(primary) | _cmids(second)
        needed = primary if category == "MULTI_UNNAMED" else both
        right.append(needed <= pred <= both)
    df["matter_right"] = right
    df["action_right"] = df["action"] == df["key_expected_action"]
    df["in_shortlist"] = [bool(_cmids(t)) and _cmids(t) <= _cmids(s) for t, s in zip(df["key_true_cmid"], df["shortlist"])]
    df["outcome"] = df.apply(_outcome, axis=1)
    return df


def pick_best(results, target):
    """The best row of a settings sweep (one row per setting, columns from summarize).

    Among rows with decided_precision >= target: highest straight_through, then highest action_accuracy.
    If none reach the target: highest decided_precision to the half point (smaller differences are noise with a
    few thousand entries), then highest straight_through.
    """
    ok = results[results["decided_precision"] >= target]
    if ok.empty:
        half_point = np.floor(results["decided_precision"] * 200) / 200
        return results.assign(half_point=half_point).sort_values(["half_point", "straight_through"], ascending=False).index[0]
    return ok.sort_values(["straight_through", "action_accuracy"], ascending=False).index[0]


def precision_ci(scored, n_boot=2000, seed=210):
    """95% bootstrap interval for decided_precision: resample the decided entries with replacement."""
    decided = scored[scored["action"].isin(["AUTO_MATCH", "NO_MATCH"])]
    right = np.where(decided["action"] == "AUTO_MATCH", decided["outcome"] == OUTCOMES[0],
                     decided["key_expected_action"] == "NO_MATCH")
    rng = np.random.default_rng(seed)
    boots = [rng.choice(right, len(right)).mean() for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def summarize(scored):
    """Headline numbers for one set of decisions.

    auto_precision     share of auto-accepts that put all of the time on the right matter
    decided_precision  share of AUTO_MATCH and NO_MATCH decisions that are right (both skip a reviewer)
    straight_through   share of entries decided with no person at all (AUTO_MATCH or NO_MATCH)
    silent_by_design   wrong auto-accepts where nothing in the entry could catch it (KEY: Silent Error = Y)
    multi_flag_right   share of MULTI_MATTER calls where the entry really covers more than one matter
    multi_caught       share of entries covering more than one matter that get a MULTI_MATTER call
    multi_matters_right  share of MULTI_MATTER calls where all suggested matters are right
    """
    auto = scored["action"] == "AUTO_MATCH"
    multi = scored["action"] == "MULTI_MATTER"
    no_match = scored["action"] == "NO_MATCH"
    review = scored["action"] == "NEEDS_REVIEW"
    flagged = scored["action"] == "FLAG_MISMATCH"
    auto_right = scored["outcome"] == OUTCOMES[0]
    auto_wrong = auto & ~auto_right
    no_match_right = no_match & (scored["key_expected_action"] == "NO_MATCH")
    is_multi = scored["key_expected_action"] == "MULTI_MATTER"
    by_design = scored["key_silent_error"] == "Y"

    def rate(hits, base):
        return float(hits[base].mean()) if base.any() else np.nan

    return {
        "entries": len(scored),
        "action_accuracy": float(scored["action_right"].mean()),
        "straight_through": float((auto | no_match).mean()),
        "auto_share": float(auto.mean()),
        "auto_precision": rate(auto_right, auto),
        "decided_precision": rate(auto_right | no_match_right, auto | no_match),
        "silent_errors": int(auto_wrong.sum()),
        "silent_by_design": int((auto_wrong & by_design).sum()),
        "silent_catchable": int((auto_wrong & ~by_design).sum()),
        "review_share": float(review.mean()),
        "flag_share": float(flagged.mean()),
        "flag_suggestion_right": rate(scored["matter_right"], flagged),
        "shortlist_hit": rate(scored["in_shortlist"], review),
        "multi_share": float(multi.mean()),
        "multi_flag_right": rate(is_multi, multi),
        "multi_caught": rate(multi, is_multi),
        "multi_matters_right": rate(scored["matter_right"], multi),
        "no_match_share": float(no_match.mean()),
        "no_match_precision": rate(no_match_right, no_match),
    }
