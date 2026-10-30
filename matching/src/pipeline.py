"""Run the saved matching pipeline on entries it hasn't seen: clues, both models, and the cutoffs.

This is the code the deployed service runs. In order:
1. load_reference: once at startup. Matter Master, timekeeper history, settings, models, past narratives, people.
2. decide: for each batch of new entries (emails already parsed). Clues, both models, cutoffs, and reasons.
3. add_confirmed: after a reviewer confirms entries, so tomorrow's clues use them (notebook 09).
Retraining is separate (src/train.py).

Notebooks: 05 scores changed copies of VAL (stress tests), 06 runs the final TEST and holdout sets, 09 replays feedback.
load_entries reads a labeled time-entry file the way notebooks 00 and 02 do; the service gets entries without labels.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from classifier import ACTION_FEATURES, entry_features, explain, route_actions
from clean import normalize
from cues import cue_table
from email_parser import parse_email_entries
from features import (add_gaps, add_person_features, build_pairs, client_mention_table, find_people, keyterm_table,
                      known_people, not_people, past_sim)
from fuzzy import client_name_table, matter_name_table
from rules import RENAMED

RAW_ENTRY_COLS = {"Time ID": "entry_id", "TK ID": "tk_id", "Source Format": "source_format", "Date": "work_date",
                  "CMID": "cmid", "Client": "client", "Matter": "matter", "Narrative": "narrative", "Hours": "hours"}
ENTRY_COLS = ["entry_id", "tk_id", "source_format", "work_date", "cmid", "client", "matter", "narrative"]
MATTER_COLS = {"CMID": "cmid", "Client": "client", "Matter": "matter", "Matter Type": "matter_type",
               "Status": "status", "Open Date": "open_date", "Close Date": "close_date"}
HISTORY_COLS = {"TK ID": "tk_id", "CMID": "cmid", "Hours (04/01-06/30/2026)": "hours"}


def load_reference(data, results_dir, encoder, setting="default"):
    """Everything that stays fixed between runs: Matter Master, history, phrase lists, settings, models,
    and the past narratives and people (TRAIN entries with a known matter, standing in for billing history).

    data: the time-entry workbook's path, or a dict of DataFrames with its "Matter Master" and "TK Matter History"
        sheets (same column names), for a service that reads them from the firm's database instead of a file.
    results_dir: the saved files the service needs: 02_config.json, 02_pairs.parquet (past narratives),
        02_people.parquet, 04_config.json, 04_model.joblib.
    encoder: the SentenceTransformer from notebook 02. setting: "strict", "default", or "balanced" cutoffs from notebook 04.
    """
    results_dir = Path(results_dir)
    sheets = data if isinstance(data, dict) else pd.read_excel(data, sheet_name=None, dtype=str)
    mm = sheets["Matter Master"]
    feature_config = json.loads((results_dir / "02_config.json").read_text())
    model_config = json.loads((results_dir / "04_config.json").read_text())

    past = pd.read_parquet(results_dir / "02_pairs.parquet", columns=["entry_id", "split", "narrative_norm", "key_true_cmid"])
    past = past.drop_duplicates("entry_id").query("split == 'TRAIN' and key_true_cmid != 'NONE'")

    return {
        "matters": mm[list(MATTER_COLS)].rename(columns=MATTER_COLS),
        "history": sheets["TK Matter History"][list(HISTORY_COLS)].rename(columns=HISTORY_COLS),
        "client_names": client_name_table(mm, feature_config["acronym_skip"]),
        "matter_names": matter_name_table(mm),
        "keyterms": keyterm_table(mm),
        "cues": cue_table(),
        "client_mentions": client_mention_table(mm, acronym_skip=feature_config["acronym_skip"]),
        "feature_config": feature_config,
        "features": model_config["features"],
        "cutoffs": model_config["routing"][setting],
        "models": joblib.load(results_dir / "04_model.joblib"),
        "encoder": encoder,
        "past_vecs": encoder.encode(past["narrative_norm"].tolist(), normalize_embeddings=True),
        "past_cmids": past["key_true_cmid"].to_numpy(),
        "people": feature_config["people"],
        "people_skip": not_people(mm),
        "person_history": pd.read_parquet(results_dir / "02_people.parquet"),
    }


def load_entries(data_path, ref, skip=("Firm",)):
    """A time-entry file read the way notebooks 00 and 02 read the main one: emails parsed, answer key split off.

    Returns (entries, answers). answers has the same columns as rules.answer_key (split, tk_id, key_ columns).
    skip: clients whose name is an ordinary word, as in notebook 00.
    """
    raw = pd.read_excel(data_path, sheet_name="Time Entries", dtype=str)
    entries = raw[list(RAW_ENTRY_COLS)].rename(columns=RAW_ENTRY_COLS)
    entries = parse_email_entries(entries, ref["client_names"], ref["matter_names"], skip)
    key = raw[[c for c in raw.columns if c.startswith("KEY: ")]]
    key = key.rename(columns=lambda c: c.replace("KEY: ", "key_").lower().replace(" ", "_")).rename(columns={"key_split": "split"})
    answers = pd.concat([entries[["entry_id", "tk_id"]], key], axis=1)
    return entries, answers.replace({"key_expected_action": RENAMED, "key_entry_category": RENAMED})


def mark_true_matters(pairs, answers):
    """is_true_matter for each pair, as in notebook 02: 1 for the right matter and for the second matter of a
    multi-matter entry. Only used for grading the pair model."""
    key = answers.set_index("entry_id")
    first, second = pairs["entry_id"].map(key["key_true_cmid"]), pairs["entry_id"].map(key["key_secondary_cmid"])
    return pairs.assign(is_true_matter=((pairs["cmid"] == first) | (pairs["cmid"] == second)).astype(int))


def add_past_sim(pairs, ref):
    """past_narrative_sim and its gap: each narrative compared with all past narratives, like VAL in notebook 02."""
    cfg = ref["feature_config"]
    text = pairs.drop_duplicates("entry_id")[["entry_id", "narrative_norm"]].reset_index(drop=True)
    vecs = ref["encoder"].encode(text["narrative_norm"].tolist(), normalize_embeddings=True)
    no_fold = np.full(len(vecs), np.nan)
    sims = past_sim(vecs, no_fold, ref["past_vecs"], np.zeros(len(ref["past_vecs"])), ref["past_cmids"],
                    pairs["cmid"].unique(), cfg["past_k"])
    long = sims.set_axis(text["entry_id"]).stack().rename("past_narrative_sim").rename_axis(["entry_id", "cmid"]).reset_index()
    return add_gaps(pairs.merge(long, on=["entry_id", "cmid"], how="left"), ["past_narrative_sim"])


def add_people(pairs, ref):
    """person_hits and person_conflict: people named in each narrative, checked against all of billing history,
    like VAL in notebook 02."""
    named = find_people(pairs.drop_duplicates("entry_id").set_index("entry_id")["narrative_norm"], ref["people"])
    return add_person_features(pairs, named, ref["person_history"], folds={})


def build_entry_pairs(entries, ref):
    """The pairs table for these entries, with every clue the pair model uses."""
    cfg = ref["feature_config"]
    pairs = build_pairs(entries[ENTRY_COLS], ref["matters"], ref["history"], ref["client_names"], ref["matter_names"],
                        ref["keyterms"], ref["cues"], ref["client_mentions"], cfg["scorers"], cfg["phrase_min"],
                        cfg["gap_cols"], pd.Timestamp(cfg["history_end"]))
    return add_past_sim(add_people(pairs, ref), ref)


def decide_pairs(pairs, ref):
    """Both models and the cutoffs on a finished pairs table. Returns (pairs with prob, entry clues with the action
    model's probabilities, decisions)."""
    cfg, models = ref["feature_config"], ref["models"]
    pairs = pairs.assign(prob=models["pair_model"].predict_proba(pairs[ref["features"]])[:, 1])
    top = entry_features(pairs, pairs["prob"].to_numpy(), cfg["client_col"], cfg["matter_col"], cfg["phrase_min"])
    narratives = pairs.drop_duplicates("entry_id").set_index("entry_id")["narrative_norm"]
    top["multi_text_score"] = models["text_model"].predict_proba(top["entry_id"].map(narratives))[:, 1]
    action_probs = pd.DataFrame(models["action_model"].predict_proba(top[ACTION_FEATURES]),
                                columns=models["action_model"].classes_)
    top = top.join(action_probs.add_prefix("p_"))   # p_auto, p_review, p_multi, p_no_match
    return pairs, top, route_actions(top, action_probs, ref["cutoffs"])


def score_entries(entries, ref):
    """Decisions for entries end to end. Returns (pairs with prob, entry clues, decisions)."""
    return decide_pairs(build_entry_pairs(entries, ref), ref)


def decide(entries, ref):
    """What the service calls for new entries: score_entries, then a short and full reason for each decision.

    entries: one row per entry with ENTRY_COLS (emails already parsed). Returns (pairs with prob, entry clues,
    decisions). decisions has one row per entry: action, pred_cmid ("top | second" for a multi-matter call
    that names the second matter), second_basis, shortlist, top_prob, second_prob, reason_short, and reason_full.
    """
    pairs, top, decisions = score_entries(entries, ref)
    cfg = ref["feature_config"]
    decisions = explain(decisions, top, pairs, cfg["client_col"], cfg["matter_col"], cfg["phrase_min"],
                        model=ref["models"]["pair_model"], features=ref["features"])
    return pairs, top, decisions


def add_confirmed(ref, confirmed):
    """Billing history with these confirmed entries added, so the history clues use them right away, no retraining:
    past narratives (embedded and added), people (new names learned, and who they were billed with), and each
    timekeeper's hours by matter. Returns an updated copy of ref; the models and cutoffs don't change.

    confirmed: entry_id, tk_id, narrative, hours, cmid (the one matter a reviewer confirmed, or that was billed
    without changes). Leave out multi-matter entries until they're split, like notebook 02 does.
    """
    ref = dict(ref)
    text = confirmed["narrative"].map(lambda t: normalize(t, drop_suffixes=False))
    vecs = ref["encoder"].encode(text.tolist(), normalize_embeddings=True)
    ref["past_vecs"] = np.vstack([ref["past_vecs"], vecs])
    ref["past_cmids"] = np.concatenate([ref["past_cmids"], confirmed["cmid"].to_numpy()])

    ref["people"] = sorted(set(ref["people"]) | set(known_people(confirmed["narrative"], ref["people_skip"])))
    named = find_people(text, ref["people"])
    client_of = ref["matters"].set_index("cmid")["client"]
    new_people = pd.DataFrame({"person": named.to_numpy(), "client": confirmed["cmid"].map(client_of).to_numpy()})
    new_people = new_people.explode("person").dropna().assign(fold=np.nan)
    ref["person_history"] = pd.concat([ref["person_history"], new_people], ignore_index=True).drop_duplicates()

    hours = pd.concat([ref["history"], confirmed[["tk_id", "cmid", "hours"]]], ignore_index=True)
    hours["hours"] = pd.to_numeric(hours["hours"], errors="coerce").fillna(0.0)
    ref["history"] = hours.groupby(["tk_id", "cmid"], as_index=False)["hours"].sum()
    return ref
