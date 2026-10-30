"""Train the matching system from labeled billing history: clues, both models, and the cutoffs, in one place.

The same steps as notebooks 02 and 04, so a firm can retrain on its own entries and set its own targets, and an
app or a scheduled job can call it. The notebooks keep the step-by-step version with checks and charts, and check
that these functions give the same result.

- SETTINGS: what a firm sets (precision targets, the margin, how much a missed multi-matter entry costs, ...)
- GRIDS: the settings each model is tuned over
- build_training_pairs: the pairs table with every clue (notebook 02)
- fit_pair_model, fit_action_model: the two models (notebook 04, sections 4 and 7)
- pick_cutoffs: the cutoffs for each target, fast to rerun when a target changes (notebook 04, section 7)
- train, save: everything above, and the files the pipeline loads
"""
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict

from classifier import ACTION_FEATURES, action_labels, entry_features, multi_text_model, route_actions
from cues import cue_table
from features import (add_gaps, add_past_narrative_sim, add_person_features, build_pairs, client_mention_table,
                      feature_groups, find_people, keyterm_table, known_people, not_people, person_history)
from fuzzy import client_name_table, matter_name_table
from pipeline import HISTORY_COLS, MATTER_COLS
from rules import answer_key, pick_robust, score

# What a firm sets. Targets are the share of automatic decisions that have to be right; the cutoffs are re-picked
# to reach them on the firm's own history, so a firm never sets a raw cutoff.
SETTINGS = {
    "targets": {"strict": 0.98, "default": 0.97, "balanced": 0.93},   # balanced: about the firm's current accuracy
    "setting": "default",          # which target the pipeline uses
    "robust_share": 0.9,           # a target has to be reached in this share of timekeeper resamples
    "resamples": 200,
    "multi_beta": 2,               # a missed multi-matter entry counts about twice as much as a false alarm
    "correct_typed_cmid": True,    # auto-match past a typed CMID the model is confident is wrong (False = flag it)
    "phrase_min": 90,
    "past_k": 3,
    "folds": 5,
    "seed": 210,
    "embedding_model": "all-MiniLM-L6-v2",
}

GRIDS = {
    "pair": {"num_leaves": [15, 31], "min_child_samples": [20, 100, 200], "n_estimators": [100, 200, 400]},
    "text": {"logisticregression__C": [0.3, 1, 3, 10]},
    "action": {"n_estimators": [50, 100, 200], "num_leaves": [4, 7, 31], "min_child_samples": [20, 50, 100]},
    "cutoffs": {"auto_min": [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995],
                "no_match_min": [0.3, 0.5, 0.7, 0.9, 0.95]},
    "multi_min": [float(x) for x in np.round(np.arange(0.05, 0.95, 0.05), 2)],
}


def assign_folds(answers, n_folds, seed):
    """TRAIN entry_id -> fold, shuffled with the seed (notebook 02, section 10)."""
    ids = answers.loc[answers["split"] == "TRAIN", "entry_id"].sample(frac=1, random_state=seed)
    return dict(zip(ids, np.arange(len(ids)) % n_folds))


def scorer_columns(choice):
    """The fuzzy scorers and the typed client and matter columns, from notebook 01's choice (as in notebook 02)."""
    scorers = list(dict.fromkeys([choice["client_scorer"], choice["matter_scorer"], choice["client_runner_up"],
                                  choice["matter_runner_up"]]))
    return scorers, f"client_{choice['client_scorer'].lower()}", f"matter_{choice['matter_scorer'].lower()}"


def build_training_pairs(entries, answers, sheets, encoder, scorer_choice, acronym_skip, history_end, settings=SETTINGS):
    """The pairs table with every clue and the label, as notebook 02 saves it.

    entries: entry_id, tk_id, source_format, work_date, cmid, client, matter, narrative (emails parsed).
    answers: entry_id, split, and the key_ columns (key_true_cmid, key_secondary_cmid, ...). TRAIN entries are the
    billing history; a firm's labeled history is all TRAIN.
    sheets: the time-entry file's sheets (Matter Master, TK Matter History). scorer_choice: notebook 01's choice.
    Returns (pairs, features, info) with info: folds, people, person_hist, scorers, client_col, matter_col, gap_cols.
    """
    s = settings
    mm = sheets["Matter Master"]
    matters = mm[list(MATTER_COLS)].rename(columns=MATTER_COLS)
    history = sheets["TK Matter History"][list(HISTORY_COLS)].rename(columns=HISTORY_COLS)
    scorers, client_col, matter_col = scorer_columns(scorer_choice)
    gap_cols = [client_col, matter_col, "keyterm_score", "cue_share", "client_in_text", "history_share", "cmid_similarity"]
    pairs = build_pairs(entries, matters, history, client_name_table(mm, acronym_skip), matter_name_table(mm),
                        keyterm_table(mm), cue_table(), client_mention_table(mm, acronym_skip=acronym_skip), scorers,
                        s["phrase_min"], gap_cols, history_end)
    features = feature_groups(scorers, gap_cols)

    pairs = pairs.merge(answers, on="entry_id", how="left")
    pairs["is_true_matter"] = [int(c in (t, x)) for c, t, x in
                               zip(pairs["cmid"], pairs["key_true_cmid"], pairs["key_secondary_cmid"])]

    folds = assign_folds(answers, s["folds"], s["seed"])
    entry_text = pairs.drop_duplicates("entry_id")[["entry_id", "split", "key_true_cmid", "narrative_norm"]]
    entry_text = entry_text.reset_index(drop=True)
    vecs = encoder.encode(entry_text["narrative_norm"].tolist(), normalize_embeddings=True)
    pairs = add_gaps(add_past_narrative_sim(pairs, entry_text, vecs, folds, k=s["past_k"]), ["past_narrative_sim"])
    pairs["fold"] = pairs["entry_id"].map(folds).fillna(-1).astype(int)
    features["past_narratives"] = ["past_narrative_sim", "past_narrative_sim_gap"]

    people = known_people(entries.loc[entries["entry_id"].isin(list(folds)), "narrative"], not_people(mm))
    named = find_people(entry_text.set_index("entry_id")["narrative_norm"], people)
    person_hist = person_history(answers, named, matters, folds)
    pairs = add_person_features(pairs, named, person_hist, folds)
    features["people"] = ["person_hits", "person_conflict"]
    info = {"folds": folds, "people": people, "person_hist": person_hist, "scorers": scorers, "client_col": client_col,
            "matter_col": matter_col, "gap_cols": gap_cols}
    return pairs, features, info


def fit_pair_model(train_pairs, features, grid=GRIDS["pair"], seed=SETTINGS["seed"]):
    """Step 1: LightGBM tuned on the entry folds by log loss. Returns (model refit on all of TRAIN,
    cross-validated probabilities for every TRAIN pair, grid search)."""
    cv = PredefinedSplit(train_pairs["fold"])
    X, y = train_pairs[features], train_pairs["is_true_matter"]
    search = GridSearchCV(LGBMClassifier(learning_rate=0.05, random_state=seed, verbose=-1), grid, cv=cv,
                          scoring="neg_log_loss").fit(X, y)
    cv_probs = cross_val_predict(search.best_estimator_, X, y, cv=cv, method="predict_proba")[:, 1]
    return search.best_estimator_, cv_probs, search


def fit_action_model(train_pairs, cv_probs, answers, client_col, matter_col, text_grid=GRIDS["text"],
                     action_grid=GRIDS["action"], settings=SETTINGS):
    """Step 2: the narrative text model and the action model, both tuned on the entry folds by log loss.
    answers: rules.answer_key(pairs), the answer key with multi-matter entries named MULTI_MATTER.

    Returns a dict: entries (one row per TRAIN entry, with out-of-fold multi_text_score), labels, text_model,
    action_model, action_cv (out-of-fold action probabilities), and both grid searches.
    """
    entries = entry_features(train_pairs, cv_probs, client_col, matter_col, settings["phrase_min"])
    labels = action_labels(entries, answers)
    per_entry = train_pairs.drop_duplicates("entry_id").set_index("entry_id")
    entry_cv = PredefinedSplit(entries["entry_id"].map(per_entry["fold"]))

    narratives = entries["entry_id"].map(per_entry["narrative_norm"])
    is_multi = labels == "multi"
    text_search = GridSearchCV(multi_text_model(), text_grid, cv=entry_cv, scoring="neg_log_loss", n_jobs=-1)
    text_search.fit(narratives, is_multi)
    entries["multi_text_score"] = cross_val_predict(text_search.best_estimator_, narratives, is_multi, cv=entry_cv,
                                                    method="predict_proba")[:, 1]

    action_search = GridSearchCV(LGBMClassifier(learning_rate=0.05, n_jobs=1, random_state=settings["seed"], verbose=-1),
                                 action_grid, cv=entry_cv, scoring="neg_log_loss", n_jobs=-1)
    action_search.fit(entries[ACTION_FEATURES], labels)
    action_model = action_search.best_estimator_
    action_cv = pd.DataFrame(cross_val_predict(action_model, entries[ACTION_FEATURES], labels, cv=entry_cv,
                                               method="predict_proba"), columns=action_model.classes_)
    return {"entries": entries, "labels": labels, "text_model": text_search.best_estimator_, "action_model": action_model,
            "action_cv": action_cv, "text_search": text_search, "action_search": action_search}


def pick_cutoffs(entries, action_cv, answers, targets=None, settings=SETTINGS, grid=GRIDS["cutoffs"],
                 multi_grid=GRIDS["multi_min"]):
    """The cutoffs for each target, from the out-of-fold action probabilities. Nothing is refit, so changing a target
    (say 97% to 98%) takes seconds.

    multi_min: the cutoff on P(multi) with the best F-beta (multi_beta). The rest: for each target, the setting with
    the most entries decided with no person among those that reach the target in robust_share of timekeeper
    resamples (pick_robust). Returns {name: cutoffs}.
    """
    targets = targets or settings["targets"]
    is_multi = action_labels(entries, answers) == "multi"
    multi_score = pd.Series({t: fbeta_score(is_multi, action_cv["multi"] >= t, beta=settings["multi_beta"])
                             for t in multi_grid})
    same_for_all = {"multi_min": float(multi_score.idxmax()), "correct_typed_cmid": settings["correct_typed_cmid"]}
    scored = {}
    for values in itertools.product(*grid.values()):
        cfg = dict(zip(grid, values)) | same_for_all
        scored[values] = score(route_actions(entries, action_cv, cfg), answers)
    routing = {}
    for name, target in targets.items():
        best = pick_robust(scored, target, settings["robust_share"], settings["resamples"], settings["seed"])
        routing[name] = {k: float(v) for k, v in zip(grid, best)} | same_for_all
    return routing


def train(entries, answers, sheets, encoder, scorer_choice, acronym_skip, history_end, settings=SETTINGS, grids=GRIDS):
    """Every step above, in order. Returns everything save() writes, plus the TRAIN entries, their out-of-fold action
    probabilities, and the answer key, so pick_cutoffs can re-pick cutoffs for another target without refitting."""
    pairs, features, info = build_training_pairs(entries, answers, sheets, encoder, scorer_choice, acronym_skip,
                                                 history_end, settings)
    feature_cols = [c for cols in features.values() for c in cols]
    client_col, matter_col = info["client_col"], info["matter_col"]
    labels = answer_key(pairs)
    train_pairs = pairs[pairs["split"] == "TRAIN"].reset_index(drop=True)
    pair_model, cv_probs, pair_search = fit_pair_model(train_pairs, feature_cols, grids["pair"], settings["seed"])
    step2 = fit_action_model(train_pairs, cv_probs, labels, client_col, matter_col, grids["text"], grids["action"], settings)
    routing = pick_cutoffs(step2["entries"], step2["action_cv"], labels, settings["targets"], settings, grids["cutoffs"],
                           grids["multi_min"])
    feature_config = {"features": features, "scorers": info["scorers"], "client_col": client_col, "matter_col": matter_col,
                      "phrase_min": settings["phrase_min"], "gap_cols": info["gap_cols"],
                      "history_end": str(pd.Timestamp(history_end).date()), "folds": settings["folds"],
                      "seed": settings["seed"], "embedding_model": settings["embedding_model"], "past_k": settings["past_k"],
                      "acronym_skip": list(acronym_skip), "people": info["people"]}
    model_config = {"model": "gradient boosting", "model_params": pair_search.best_params_, "features": feature_cols,
                    "action_features": ACTION_FEATURES, "text_model_params": step2["text_search"].best_params_,
                    "action_model_params": step2["action_search"].best_params_, "routing": routing,
                    "multi_beta": settings["multi_beta"], "precision_target": settings["targets"]["strict"],
                    "default_target": settings["targets"]["default"], "robust_share": settings["robust_share"]}
    return {"pairs": pairs, "person_history": info["person_hist"], "feature_config": feature_config,
            "model_config": model_config,
            "models": {"pair_model": pair_model, "text_model": step2["text_model"], "action_model": step2["action_model"]},
            "entries": step2["entries"], "action_cv": step2["action_cv"], "answers": labels}


def save(trained, results_dir):
    """The files the pipeline loads (load_reference), under the same names as notebooks 02 and 04 write them."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    features = [c for cols in trained["feature_config"]["features"].values() for c in cols]
    keep = (["entry_id", "tk_id", "source_format", "work_date", "cmid", "client", "matter", "matter_type", "typed_cmid",
             "cmid_clean", "cmid_status", "typed_client", "typed_matter", "narrative_norm", "split", "fold"]
            + features + ["is_true_matter"] + [c for c in trained["pairs"].columns if c.startswith("key_")])
    trained["pairs"][keep].to_parquet(results_dir / "02_pairs.parquet", index=False)
    trained["person_history"].to_parquet(results_dir / "02_people.parquet", index=False)
    (results_dir / "02_config.json").write_text(json.dumps(trained["feature_config"], indent=2))
    joblib.dump(trained["models"], results_dir / "04_model.joblib")
    (results_dir / "04_config.json").write_text(json.dumps(trained["model_config"], indent=2, default=str))
