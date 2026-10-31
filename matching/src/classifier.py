"""Classifier: turn match probabilities into one decision per entry, with a reason for each.

Step 1 scores every (entry, candidate matter) pair in the pairs table from notebook 02.
- route: the one-step version, cutoffs on the top two probabilities
- entry_features, action_labels, route_actions: step 2, an action model on entry-level clues
- multi_text_model: a narrative score for wording that mentions other work
- two_step: step 2 on top of any step-1 model, for comparing models (notebook 07)
- explain, suggest_split: plain-language reasons and a suggested split for multi-matter entries
"""
import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
from sklearn.pipeline import make_pipeline

from rules import pick_robust, score

# Settings are explained in notebook 04, section 6
DEFAULTS = {"auto_min": 0.8, "multi_min": 0.5, "no_match_max": 0.1}


def top_candidates(pairs, probs, k=3):
    """One row per entry: the two best candidates, a top-k shortlist, and the typed CMID if it's valid."""
    df = pairs[["entry_id", "cmid", "is_typed_cmid", "cmid_valid", "not_open"]].copy()
    df["prob"] = probs
    df = df.sort_values(["entry_id", "prob"], ascending=[True, False])

    first = df.groupby("entry_id").nth(0).set_index("entry_id")
    second = df.groupby("entry_id").nth(1).set_index("entry_id")
    typed = df[(df["is_typed_cmid"] == 1) & (df["cmid_valid"] == 1)].set_index("entry_id")

    top = pd.DataFrame({
        "top_cmid": first["cmid"],
        "top_prob": first["prob"],
        "top_not_open": first["not_open"],
        "second_cmid": second["cmid"],
        "second_prob": second["prob"],
        "shortlist": df.groupby("entry_id")["cmid"].apply(lambda s: " | ".join(s.head(k))),
    })
    top["typed_cmid"] = typed["cmid"]
    top["typed_prob"] = typed["prob"]
    top["typed_not_open"] = typed["not_open"]
    return top.rename_axis("entry_id").reset_index()


def route(top, cfg):
    """One decision per entry. The first check that applies decides.

    1. Nothing scores above no_match_max         -> NO_MATCH
    2. The second candidate scores multi_min+     -> MULTI_MATTER (both matters)
    3. A valid typed CMID isn't the top pick,
       or the typed matter wasn't open             -> FLAG_MISMATCH (suggest the top pick)
    4. The top pick scores auto_min+ but wasn't
       open on the work date                       -> FLAG_MISMATCH
    5. The top pick scores auto_min+              -> AUTO_MATCH
    6. Otherwise                                  -> NEEDS_REVIEW with the shortlist
    """
    top_prob, second_prob = top["top_prob"], top["second_prob"]
    has_typed = top["typed_cmid"].notna()
    typed_disagrees = has_typed & ((top["top_cmid"] != top["typed_cmid"]) | (top["typed_not_open"] == 1))
    confident = top_prob >= cfg["auto_min"]

    checks = [top_prob < cfg["no_match_max"],
              second_prob >= cfg["multi_min"],
              typed_disagrees,
              confident & (top["top_not_open"] == 1),
              confident]
    actions = ["NO_MATCH", "MULTI_MATTER", "FLAG_MISMATCH", "FLAG_MISMATCH", "AUTO_MATCH"]
    reasons = ["no candidate scores high enough", "two candidates score high", "typed CMID disagrees with the model",
               "matter not open on the work date", "top candidate scores high"]

    out = top[["entry_id", "shortlist", "top_prob", "second_prob"]].copy()
    out["action"] = np.select(checks, actions, "NEEDS_REVIEW")
    out["reason"] = np.select(checks, reasons, "not confident enough (shortlist only)")
    both = top["top_cmid"] + " | " + top["second_cmid"]
    out["pred_cmid"] = np.where(out["action"] == "MULTI_MATTER", both, top["top_cmid"])
    out.loc[out["action"] == "NO_MATCH", "pred_cmid"] = None
    return out


def top_k_accuracy(pairs, probs, k=1):
    """Share of entries with a true matter whose true matter is in the model's top k."""
    df = pairs[["entry_id", "is_true_matter"]].copy()
    df["rank"] = pd.Series(probs, index=df.index).groupby(df["entry_id"]).rank(ascending=False, method="first")
    has_true = df.groupby("entry_id")["is_true_matter"].transform("max") == 1
    hits = df[has_true & (df["is_true_matter"] == 1)].groupby("entry_id")["rank"].min() <= k
    return hits.mean()


# ---------------- Step 2: deciding the action ----------------

ENTRY_FEATURES = ["top_prob", "second_prob", "third_prob", "prob_total", "top_share", "same_client_top2", "typed_is_top",
                  "typed_prob", "top_not_open", "top_in_history", "top_past_sim", "second_past_sim", "best_client_score",
                  "best_matter_score", "has_cmid", "cmid_valid", "cmid_client_known", "has_client", "has_matter",
                  "n_parts", "n_types_cued", "n_matters_named", "n_clients_named", "is_email", "n_other_sources"]
ACTION_FEATURES = ENTRY_FEATURES + ["multi_text_score"]

# Evidence in the narrative that points away from the top pick, in the order it's trusted to name the second matter.
# A client's name names the client directly, while a key term can also be a counterparty inside another client's
# matter name ("Credit Facility - Kestrel Foods" belongs to the bank, not to Kestrel Foods).
OTHER_EVIDENCE = {
    "client": "the narrative names its client",
    "person": "someone named in the narrative works with its client",
    "matter": "the narrative names it",
}


def other_evidence(df, first, min_score=90):
    """For each entry: how many kinds of evidence point to a different matter or client than the top pick
    (n_other_sources, 0 to 3), and the matter that evidence points to most (evidence_cmid, evidence_basis).

    client: another client named in the narrative. person: someone named who was billed with another client.
    matter: another matter named by one of its own key terms.
    df: pairs with prob, sorted by entry and prob (highest first). first: the top pick of each entry.
    """
    top_cmid, top_client = df["entry_id"].map(first["cmid"]), df["entry_id"].map(first["client"])
    other_client = df["client"] != top_client
    found = {"client": other_client & (df["client_in_text"] >= min_score),
             "person": other_client & (df["person_hits"] > 0),
             "matter": (df["cmid"] != top_cmid) & (df["own_keyterm_hits"] > 0)}
    out = pd.DataFrame({kind: hit.groupby(df["entry_id"]).any() for kind, hit in found.items()})
    out = pd.DataFrame({"n_other_sources": out.sum(axis=1)}, index=out.index)
    out["evidence_cmid"], out["evidence_basis"] = None, None
    for kind, hit in found.items():
        best = df[hit].drop_duplicates("entry_id").set_index("entry_id")["cmid"]
        fill = out["evidence_cmid"].isna() & out.index.isin(best.index)
        out.loc[fill, "evidence_cmid"] = best.reindex(out.index[fill]).to_numpy()
        out.loc[fill, "evidence_basis"] = kind
    return out


def entry_features(pairs, probs, client_col, matter_col, min_score=90):
    """One row per entry for the action model: how the top candidates score, plus clues about the whole entry.
    min_score: the phrase cutoff from notebook 02, for a client named in the narrative."""
    top = top_candidates(pairs, probs)
    df = pairs.assign(prob=probs).sort_values(["entry_id", "prob"], ascending=[True, False])
    g = df.groupby("entry_id")
    first, second = g.nth(0).set_index("entry_id"), g.nth(1).set_index("entry_id")

    e = pd.DataFrame({
        "third_prob": g.nth(2).set_index("entry_id")["prob"],
        "prob_total": g["prob"].sum(),
        "same_client_top2": (first["client"] == second["client"]).astype(int),
        "top_in_history": first["in_history"],
        "top_past_sim": first["past_narrative_sim"],
        "second_past_sim": second["past_narrative_sim"],
        "best_client_score": g[client_col].max(),
        "best_matter_score": g[matter_col].max(),
    })
    for col in ["has_cmid", "cmid_valid", "cmid_client_known", "has_client", "has_matter", "n_parts", "n_types_cued",
                "n_matters_named", "n_clients_named", "is_email"]:
        e[col] = first[col]

    e = e.join(other_evidence(df, first, min_score))

    top = top.merge(e.rename_axis("entry_id").reset_index(), on="entry_id")
    top["top_share"] = top["top_prob"] / top["prob_total"].clip(lower=1e-9)
    top["typed_is_top"] = (top["typed_cmid"] == top["top_cmid"]).astype(int)
    top["typed_prob"] = top["typed_prob"].fillna(0.0)
    return top


def action_labels(top, answers):
    """What the action model learns for each entry:
    multi (covers more than one matter), no_match (new client or matter), auto (the top pick is the right matter),
    or review (the top pick is wrong)."""
    a = top[["entry_id", "top_cmid"]].merge(answers[["entry_id", "key_true_cmid", "key_expected_action"]], on="entry_id")
    return np.select([a["key_expected_action"] == "MULTI_MATTER", a["key_expected_action"] == "NO_MATCH",
                      a["top_cmid"] == a["key_true_cmid"]], ["multi", "no_match", "auto"], "review")


def route_actions(top, action_probs, cfg):
    """One decision per entry from the action model's probabilities. The first check that applies decides.

    1. P(no_match) >= no_match_min                 -> NO_MATCH
    2. P(multi) >= multi_min                       -> MULTI_MATTER (the top pick, plus a second matter: the one the
                                                      narrative points to (other_evidence), else the second pick
                                                      if it's likely on its own)
    3. correct_typed_cmid is on, a valid typed CMID
       isn't the top pick, P(auto) >= auto_min,
       and the top pick was open                   -> AUTO_MATCH to the top pick (typed CMID corrected)
    4. A valid typed CMID isn't the top pick,
       or the typed matter wasn't open             -> FLAG_MISMATCH (suggest the top pick)
    5. P(auto) >= auto_min but the top pick
       wasn't open on the work date                -> FLAG_MISMATCH
    6. P(auto) >= auto_min                         -> AUTO_MATCH
    7. Otherwise                                   -> NEEDS_REVIEW with the shortlist
    action_probs: DataFrame with columns auto, review, multi, no_match (one row per entry, same order as top).
    cfg: the cutoffs, plus correct_typed_cmid (True or False, off if missing).
    """
    p = action_probs.set_axis(top.index)
    has_typed = top["typed_cmid"].notna()
    typed_disagrees = has_typed & ((top["top_cmid"] != top["typed_cmid"]) | (top["typed_not_open"] == 1))
    confident = p["auto"] >= cfg["auto_min"]
    correctable = (cfg.get("correct_typed_cmid", False) & has_typed & (top["top_cmid"] != top["typed_cmid"])
                   & confident & (top["top_not_open"] == 0))

    checks = [p["no_match"] >= cfg["no_match_min"],
              p["multi"] >= cfg["multi_min"],
              correctable,
              typed_disagrees,
              confident & (top["top_not_open"] == 1),
              confident]
    actions = ["NO_MATCH", "MULTI_MATTER", "AUTO_MATCH", "FLAG_MISMATCH", "FLAG_MISMATCH", "AUTO_MATCH"]
    reasons = ["likely a new client or matter", "likely covers more than one matter", "typed CMID corrected to the top pick",
               "typed CMID disagrees with the model", "matter not open on the work date", "top pick very likely right"]

    out = top[["entry_id", "shortlist", "top_prob", "second_prob"]].copy()
    out["action"] = np.select(checks, actions, "NEEDS_REVIEW")
    out["reason"] = np.select(checks, reasons, "not confident enough (shortlist only)")
    likely_second = (top["second_prob"] >= 0.5) & top["evidence_cmid"].isna()
    second = top["evidence_cmid"].where(top["evidence_cmid"].notna(), top["second_cmid"].where(likely_second))
    basis = top["evidence_basis"].where(top["evidence_basis"].notna(), np.where(likely_second, "score", None))
    is_multi = out["action"] == "MULTI_MATTER"
    out["pred_cmid"] = np.where(is_multi & second.notna(), top["top_cmid"] + " | " + second.fillna(""), top["top_cmid"])
    out["second_basis"] = basis.where(is_multi & second.notna())
    out.loc[out["action"] == "NO_MATCH", "pred_cmid"] = None
    return out


def multi_text_model(C=1.0):
    """Reads the narrative alone and scores how likely the entry covers more than one matter,
    from wording like "misc. calls for other client" or "plus time on other deal"."""
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=C, max_iter=2000))


def two_step(train_pairs, train_probs, val_pairs, val_probs, answers, narratives, text_model, client_col, matter_col,
             target, action_grid, cutoff_grid, multi_grid, multi_beta=2, correct_typed_cmid=True, seed=210,
             robust_share=0.9, refit_text=False):
    """Step 2 on top of any step-1 model, the same way notebook 04 builds it, so step-1 models can be compared end to end.

    train_probs: cross-validated step-1 probabilities for the TRAIN pairs. val_probs: from the step-1 model refit on TRAIN.
    text_model: the narrative model from notebook 04, already fit on TRAIN (it doesn't use step-1 probabilities).
    refit_text: refit it on these TRAIN entries first (when train_pairs is only part of TRAIN).
    The action model is tuned on the same entry folds, multi_min is picked by F-beta, and the other cutoffs by
    pick_robust at the target, all on TRAIN. Returns the scored VAL decisions, the cutoffs, and the action model's
    grid search.
    """
    entries_train = entry_features(train_pairs, train_probs, client_col, matter_col)
    labels = action_labels(entries_train, answers)
    is_multi = labels == "multi"
    folds = entries_train["entry_id"].map(train_pairs.drop_duplicates("entry_id").set_index("entry_id")["fold"])
    entry_cv = PredefinedSplit(folds)

    narrative_train = entries_train["entry_id"].map(narratives)
    if refit_text:
        text_model = clone(text_model).fit(narrative_train, is_multi)
    entries_train["multi_text_score"] = cross_val_predict(text_model, narrative_train, is_multi, cv=entry_cv,
                                                          method="predict_proba", n_jobs=-1)[:, 1]
    search = GridSearchCV(LGBMClassifier(learning_rate=0.05, n_jobs=1, random_state=seed, verbose=-1), action_grid,
                          cv=entry_cv, scoring="neg_log_loss", n_jobs=-1).fit(entries_train[ACTION_FEATURES], labels)
    action_model = search.best_estimator_
    action_cv = pd.DataFrame(cross_val_predict(action_model, entries_train[ACTION_FEATURES], labels, cv=entry_cv,
                                               method="predict_proba", n_jobs=-1), columns=action_model.classes_)

    multi_score = pd.Series({t: fbeta_score(is_multi, action_cv["multi"] >= t, beta=multi_beta) for t in multi_grid})
    same_for_all = {"multi_min": float(multi_score.idxmax()), "correct_typed_cmid": correct_typed_cmid}
    scored = {}
    for settings in itertools.product(*cutoff_grid.values()):
        cfg = dict(zip(cutoff_grid, settings)) | same_for_all
        scored[settings] = score(route_actions(entries_train, action_cv, cfg), answers)
    best = pick_robust(scored, target, robust_share, seed=seed)
    cfg = {k: float(v) for k, v in zip(cutoff_grid, best)} | same_for_all

    entries_val = entry_features(val_pairs, val_probs, client_col, matter_col)
    entries_val["multi_text_score"] = text_model.predict_proba(entries_val["entry_id"].map(narratives))[:, 1]
    action_val = pd.DataFrame(action_model.predict_proba(entries_val[ACTION_FEATURES]), columns=action_model.classes_)
    return score(route_actions(entries_val, action_val, cfg), answers), cfg, search


# ---------------- Reasons ----------------

# Clues about the top pick a reason can cite. Each one: (is it there?, short reason, full reason).
# A decision cites only the clues that are there, in the order they pushed that decision (clue_weights).
CLUES = [
    (lambda r: r["is_typed_cmid"] == 1 and r["cmid_valid"] == 1, "Typed CMID matches", "the typed CMID is this matter"),
    (lambda r: r["past_narrative_sim"] >= 0.6, "Reads like past entries", "the narrative reads like past entries on this matter"),
    (lambda r: r["in_history"] == 1, "You've billed this matter", "the timekeeper billed {history_hours:.1f} hours to it recently"),
    (lambda r: r["own_keyterm_hits"] > 0, "Narrative names the matter", "the narrative names the matter"),
    (lambda r: r["client_named"], "Narrative names the client", "the narrative names {client}"),
    (lambda r: r["client_score"] >= 80, "Typed client matches", "the typed client matches {client}"),
    (lambda r: r["matter_score"] >= 80, "Typed matter matches", "the typed matter matches"),
    (lambda r: r["cue_share"] >= 0.5, "Tasks fit the matter type", "the tasks fit a {matter_type} matter"),
    (lambda r: r["person_hits"] > 0, "Names someone at this client", "the narrative names someone billed with {client} before"),
]


def clue_features(features):
    """Which pair-model inputs each clue in CLUES stands for, so the model's push from them can be added up."""
    return {
        "Typed CMID matches": [f for f in features if f.startswith("cmid_")
                               or f in ("is_typed_cmid", "is_flipped_cmid", "same_client_number", "has_cmid")],
        "Reads like past entries": [f for f in features if f.startswith("past_narrative_sim")],
        "You've billed this matter": [f for f in features if f.startswith("history_") or f == "in_history"],
        "Narrative names the matter": [f for f in features if f.startswith(("keyterm_", "own_keyterm_"))],
        "Narrative names the client": [f for f in features if f.startswith("client_in_text")],
        "Typed client matches": [f for f in features if f.startswith("client_") and "ratio" in f],
        "Typed matter matches": [f for f in features if f.startswith("matter_") and "ratio" in f],
        "Tasks fit the matter type": [f for f in features if f.startswith("cue_")],
        "Names someone at this client": [f for f in features if f.startswith("person_")],
    }


def clue_weights(model, pick_rows, features):
    """How much each clue pushed each top pick toward "right matter", from the model's own per-decision contributions
    (LightGBM's pred_contrib, in log-odds; the inputs of one clue are added up). One row per top pick."""
    contrib = model.predict(pick_rows[features], pred_contrib=True)[:, :-1]
    contrib = pd.DataFrame(contrib, columns=features, index=pick_rows.index)
    return pd.DataFrame({clue: contrib[cols].sum(axis=1) for clue, cols in clue_features(features).items()})


SECOND_BASIS = OTHER_EVIDENCE | {"score": "it scores high on its own"}


def confidence(prob):
    if prob >= 0.9:
        return "high"
    return "medium" if prob >= 0.5 else "low"


def suggest_split(decisions, parts):
    """For MULTI_MATTER decisions: which matter each piece of work seems to belong to.

    With two suggested matters, a part goes to the one it names, else to the one that ranks higher among its
    most similar matters. With one suggested matter, a part that names a different matter goes to that matter;
    otherwise the part that reads most like other work (other_work_score) is marked "another matter" and the
    rest stay with the suggested matter. parts: match_parts output plus other_work_score.
    """
    multi = decisions[decisions["action"] == "MULTI_MATTER"].set_index("entry_id")["pred_cmid"].str.split(" | ", regex=False)
    out = {}
    for entry_id, entry_parts in parts[parts["entry_id"].isin(multi.index)].groupby("entry_id"):
        preds = multi[entry_id]
        if len(entry_parts) < 2:
            out[entry_id] = "the narrative reads as one piece of work, so the timekeeper needs to divide it"
            continue
        if len(preds) == 1:
            named_other = entry_parts["named_cmid"].notna() & (entry_parts["named_cmid"] != preds[0])
            picks = pd.Series(preds[0], index=entry_parts.index)
            if named_other.any():
                picks[named_other] = entry_parts.loc[named_other, "named_cmid"]
            else:
                picks[entry_parts["other_work_score"].idxmax()] = "another matter"
        else:
            ranked = entry_parts["top_cmids"].str.split(" | ", regex=False)
            rank = pd.DataFrame({c: [r.index(c) if c in r else 9 for r in ranked] for c in preds}, index=entry_parts.index)
            for c in preds:
                rank.loc[entry_parts["named_cmid"] == c, c] = -1
            picks = rank.idxmin(axis=1)
            if picks.nunique() == 1:
                other = next(c for c in preds if c != picks.iloc[0])
                picks[rank[other].idxmin()] = other
        out[entry_id] = "; ".join(f'"{t}" -> {c}' for t, c in zip(entry_parts["part_norm"], picks))
    return pd.Series(out, dtype=object)


def reasons_for(row, split=None):
    """(short, full) reason for one decision. row has the decision plus the top pick's clues.

    With clue weights (from the model), only clues that are there and pushed toward the pick are cited, strongest
    first. Without them, clues that are there are cited in the order of CLUES.
    """
    weights = row.get("clue_weights")
    present = [(short, full) for check, short, full in CLUES if check(row)]
    if isinstance(weights, dict):
        present = sorted([c for c in present if weights.get(c[0], 0) > 0], key=lambda c: -weights[c[0]])
    clues = [(short, full.format(**row)) for short, full in present]
    why = "; ".join(full for _, full in clues[:3])
    pick = f"{row['top_cmid']} ({row['client']} - {row['matter']}), {confidence(row['top_prob'])} confidence"
    action = row["action"]

    if action == "AUTO_MATCH":
        if isinstance(row["typed_cmid"], str) and row["typed_cmid"] != row["top_cmid"]:
            return "Typed CMID corrected", f"Auto-match to {pick}, correcting the typed CMID {row['typed_cmid']}: {why or 'strong overall match'}."
        short = clues[0][0] if clues else "Strong overall match"
        return short, f"Auto-match to {pick}: {why or 'strong overall match'}."

    if action == "FLAG_MISMATCH":
        if isinstance(row["typed_cmid"], str) and row["typed_cmid"] != row["top_cmid"]:
            short = "Typed CMID doesn't fit narrative"
        elif row["typed_not_open"] == 1:
            short = "Typed matter wasn't open"
        else:
            short = "Matter wasn't open on that date"
        return short, f"Flag: {short}. Suggested {pick}" + (f": {why}." if why else ".")

    if action == "NEEDS_REVIEW":
        one_client = len({c.split(".")[0] for c in row["shortlist"].split(" | ")}) == 1
        if one_client:
            short = "Client clear, matter unclear"
        elif row["top_prob"] < 0.3:
            short = "No clear match"
        elif row["second_prob"] >= row["top_prob"] / 2:
            short = "Close call between matters"
        else:
            short = "Not sure enough to accept"
        return short, f"Review: {short}. Shortlist: {row['shortlist']}." + (f" Top pick {row['top_cmid']}: {why}." if why else "")

    if action == "MULTI_MATTER":
        if row.get("multi_text_score", 0) >= 0.5:
            short = "Mentions other work"
        elif row["n_clients_named"] >= 2:
            short = "Names two clients"
        elif row["n_matters_named"] >= 2:
            short = "Names two matters"
        else:
            short = "May cover two matters"
        preds = row["pred_cmid"]
        matters = preds.replace(" | ", " and ") if " | " in preds else f"{preds} and one the narrative doesn't name"
        detail = "" if short == "May cover two matters" else f" ({short[0].lower() + short[1:]})"
        basis = row.get("second_basis")
        because = f", the second because {SECOND_BASIS[basis]}" if isinstance(basis, str) else ""
        full = f"May cover more than one matter{detail}: {matters}{because}."
        if split:
            full += f" Suggested split: {split}."
        return short, full

    short = "Likely a new client or matter"
    return short, f"No match: likely a new client or matter (closest is {row['top_cmid']} at {row['top_prob']:.0%})."


def explain(decisions, top, pairs, client_col, matter_col, phrase_min=90, parts=None, model=None, features=None):
    """Two plain-language reasons per decision, built from that entry's own clues.

    reason_short  a few words for the review list ("Typed CMID matches")
    reason_full   the decision, confidence, the top pick's strongest clues, and the shortlist or suggested split
    With the pair model and its inputs, clues are ordered by how much they pushed that decision (clue_weights).
    """
    cols = ["entry_id", "cmid", "client", "matter", "matter_type", "is_typed_cmid", "cmid_valid", "past_narrative_sim",
            "in_history", "history_hours", "own_keyterm_hits", "client_in_text", client_col, matter_col, "cue_share",
            "person_hits"]
    entry_cols = ["entry_id", "top_cmid", "typed_cmid", "typed_not_open", "n_clients_named", "n_matters_named"]
    keep = list(dict.fromkeys(cols + [f for f in features or [] if f not in entry_cols]))
    clues = pairs[keep].rename(columns={"cmid": "top_cmid", client_col: "client_score", matter_col: "matter_score"})
    rows = top[entry_cols + (["multi_text_score"] if "multi_text_score" in top else [])].merge(clues, on=["entry_id", "top_cmid"], how="left")
    rows["client_named"] = rows["client_in_text"] >= phrase_min
    rows["matter_type"] = rows["matter_type"].astype(str).str.lower()
    if model is not None:
        model_rows = rows.rename(columns={"top_cmid": "cmid", "client_score": client_col, "matter_score": matter_col})
        rows["clue_weights"] = clue_weights(model, model_rows, features).to_dict("records")
    rows = rows.drop(columns=[f for f in features or [] if f in rows and f not in cols + entry_cols])

    splits = suggest_split(decisions, parts) if parts is not None else {}
    reasons = [reasons_for(row, splits.get(row["entry_id"])) for row in decisions.merge(rows, on="entry_id", how="left").to_dict("records")]
    return decisions.assign(reason_short=[short for short, _ in reasons], reason_full=[full for _, full in reasons])
