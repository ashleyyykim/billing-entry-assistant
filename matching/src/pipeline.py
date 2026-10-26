"""Run the saved matching pipeline on entries it hasn't seen: clues, both models, and the cutoffs.

Used by notebook 05 to score changed copies of VAL (stress tests), and by notebook 06 for the final TEST and holdout run.
load_entries reads a time-entry file the way notebooks 00 and 02 do; the rest takes entries with emails already parsed.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from classifier import ACTION_FEATURES, entry_features, route_actions
from cues import cue_table
from email_parser import parse_email_entries
from features import add_gaps, build_pairs, client_mention_table, keyterm_table, past_sim
from fuzzy import client_name_table, matter_name_table
from rules import RENAMED

RAW_ENTRY_COLS = {"Time ID": "entry_id", "TK ID": "tk_id", "Source Format": "source_format", "Date": "work_date",
                  "CMID": "cmid", "Client": "client", "Matter": "matter", "Narrative": "narrative"}
ENTRY_COLS = ["entry_id", "tk_id", "source_format", "work_date", "cmid", "client", "matter", "narrative"]
MATTER_COLS = {"CMID": "cmid", "Client": "client", "Matter": "matter", "Matter Type": "matter_type",
               "Status": "status", "Open Date": "open_date", "Close Date": "close_date"}
HISTORY_COLS = {"TK ID": "tk_id", "CMID": "cmid", "Hours (04/01-06/30/2026)": "hours"}


def load_reference(data_path, results_dir, encoder, setting="default"):
    """Everything that stays fixed between runs: Matter Master, history, phrase lists, settings, models,
    and the past narratives (TRAIN entries with a known matter, standing in for billing history).

    encoder: the SentenceTransformer from notebook 02. setting: "strict", "default", or "balanced" cutoffs from notebook 04.
    """
    results_dir = Path(results_dir)
    sheets = pd.read_excel(data_path, sheet_name=None, dtype=str)
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


def build_entry_pairs(entries, ref):
    """The pairs table for these entries, with every clue the pair model uses."""
    cfg = ref["feature_config"]
    pairs = build_pairs(entries[ENTRY_COLS], ref["matters"], ref["history"], ref["client_names"], ref["matter_names"],
                        ref["keyterms"], ref["cues"], ref["client_mentions"], cfg["scorers"], cfg["phrase_min"],
                        cfg["gap_cols"], pd.Timestamp(cfg["history_end"]))
    return add_past_sim(pairs, ref)


def decide_pairs(pairs, ref):
    """Both models and the cutoffs on a finished pairs table. Returns (pairs with prob, entry clues, decisions)."""
    cfg, models = ref["feature_config"], ref["models"]
    pairs = pairs.assign(prob=models["pair_model"].predict_proba(pairs[ref["features"]])[:, 1])
    top = entry_features(pairs, pairs["prob"].to_numpy(), cfg["client_col"], cfg["matter_col"])
    narratives = pairs.drop_duplicates("entry_id").set_index("entry_id")["narrative_norm"]
    top["multi_text_score"] = models["text_model"].predict_proba(top["entry_id"].map(narratives))[:, 1]
    action_probs = pd.DataFrame(models["action_model"].predict_proba(top[ACTION_FEATURES]),
                                columns=models["action_model"].classes_)
    return pairs, top, route_actions(top, action_probs, ref["cutoffs"])


def score_entries(entries, ref):
    """Decisions for entries end to end. Returns (pairs with prob, entry clues, decisions)."""
    return decide_pairs(build_entry_pairs(entries, ref), ref)
