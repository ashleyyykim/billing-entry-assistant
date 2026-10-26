"""Build the pairs table: one row per (entry, candidate matter), with the clues that point to the right matter."""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from clean import clean_cmid, clean_entries, cmid_similarity, cmid_status, flip_cmid, normalize
from cues import check_cue_types
from fuzzy import add_name_scores, client_acronyms

SOURCE_FORMATS = ["PDF", "HANDWRITTEN", "EMAIL"]   # TYPED is the baseline
# Where one piece of work ends and the next begins: semicolons, hours in parentheses "(1.1)", " + ",
# sentence ends ("...schedules. Call with..."), and linking words. Initials like "O. Grant" aren't sentence ends.
PART_SEPARATORS = re.compile(r";|\(\s*\d*\.?\d+\s*(?:(?i:hrs?))?\s*\)|\s\+\s|(?<=[a-z]{2})\.\s+(?=[A-Z])"
                             r"|(?i:\b(?:also|separately|plus|then)\b)")

# Numbers, roman numerals, and single letters ("fund ii", "series a") have to match exactly
ID_WORD = re.compile(r"^(\d+|[ivx]+|[a-z])$")

# Words in matter names that describe the kind of work or company, not which one ("Acquisition of Brightline Pumps")
NAME_SEPARATORS = re.compile(r"\s+-\s+|[()/]")
GENERIC_NAME_WORDS = {
    "of", "the", "and", "to", "no", "project", "general", "corporate", "matters", "admin", "business",
    "development", "recruiting", "acquisition", "add", "on", "divestiture", "sale", "portfolio", "co", "financing",
    "formation", "offering", "follow", "reporting", "board", "governance", "licensing", "license", "lease",
    "credit", "facility", "amendment", "commercial", "contracts", "investment", "finance", "administration",
    "program", "annual", "meeting", "vehicle", "invest", "extension", "equity", "plan", "exec", "comp", "tax",
    "structuring", "convertible", "notes", "pumps", "valves", "tools", "foods", "software", "holdings", "health",
    "clinics", "bakery", "photonics", "dental", "controls", "logistics", "sensors", "systems", "pharma", "bio",
    "network", "unit", "mfg", "hq", "st", "ave", "rd", "way",
}
TOO_VAGUE_ALONE = {"fund", "series"}


# ---------------- Pairs ----------------

def make_pairs(entries, matters):
    """Clean each entry's typed fields, then pair the entry with every matter in Matter Master.

    entries: entry_id, tk_id, source_format, work_date, cmid, client, matter, narrative
    matters: cmid, client, matter, matter_type, status, open_date, close_date
    """
    typed = clean_entries(entries)
    e = entries[["entry_id", "tk_id", "source_format"]].copy()
    e["work_date"] = pd.to_datetime(entries["work_date"])
    e["typed_cmid"] = typed["typed_cmid"]
    e["typed_client"] = typed["typed_client"].map(normalize)
    e["typed_matter"] = typed["typed_matter"].map(normalize)
    e["narrative"] = entries["narrative"]
    e["narrative_norm"] = entries["narrative"].map(lambda t: normalize(t, drop_suffixes=False))
    for field in ["cmid", "client", "matter"]:
        e[f"has_{field}"] = (typed[f"typed_{field}"].fillna("").astype(str).str.strip() != "").astype(int)
    for col in ["client_is_ditto", "client_had_matter", "cmid_from_text"]:
        e[col] = typed[col]
    for fmt in SOURCE_FORMATS:
        e[f"is_{fmt.lower()}"] = (entries["source_format"] == fmt).astype(int)

    m = matters.copy()
    m["open_date"] = pd.to_datetime(m["open_date"])
    m["close_date"] = pd.to_datetime(m["close_date"])
    return e.merge(m, how="cross")


# ---------------- CMID ----------------

def add_cmid_features(pairs, valid_cmids):
    """How the typed CMID relates to each candidate.

    cmid_valid            the cleaned CMID exists
    cmid_client_known     it doesn't exist, but its first 4 digits are a real client number
    is_typed_cmid         the cleaned CMID is this candidate
    is_flipped_cmid       it doesn't exist, but reading its zeros the other way gives this candidate
    cmid_similarity       0-100 digit similarity to this candidate (only when the CMID doesn't exist)
    same_client_number    the typed CMID's client number matches this candidate's
    """
    p = pairs
    p["cmid_clean"] = p["typed_cmid"].map(clean_cmid)
    p["cmid_status"] = [cmid_status(t, c, valid_cmids) for t, c in zip(p["typed_cmid"], p["cmid_clean"])]
    p["cmid_flipped"] = p["cmid_clean"].map(flip_cmid)
    p["cmid_valid"] = p["cmid_status"].isin(["valid", "reformatted"]).astype(int)

    invalid = p["cmid_status"] == "invalid"
    client_numbers = {c[:4] for c in valid_cmids}
    p["cmid_client_known"] = (invalid & p["cmid_clean"].str[:4].isin(client_numbers)).astype(int)
    p["is_typed_cmid"] = (p["cmid_clean"] == p["cmid"]).astype(int)
    p["is_flipped_cmid"] = (invalid & (p["cmid_flipped"] == p["cmid"])).astype(int)
    p["cmid_similarity"] = np.where(invalid, [cmid_similarity(a, b) for a, b in zip(p["cmid_clean"], p["cmid"])], 0.0)
    p["same_client_number"] = (p["cmid_clean"].str[:4] == p["cmid"].str[:4]).astype(int)
    return p


# ---------------- Narrative ----------------

def name_terms(name):
    """Distinctive phrases in a matter name or codename: each part of the name, plus the part with
    generic words trimmed from both ends. Parts made only of generic words ("General Corporate") are skipped.

    "Acquisition of Brightline Pumps" -> ["acquisition of brightline pumps", "brightline"]
    "Lease - 12 Mission Ave."          -> ["12 mission ave", "12 mission"]
    """
    terms = []
    for part in NAME_SEPARATORS.split(str(name)):
        words = normalize(part, drop_suffixes=False).split()
        if all(w in GENERIC_NAME_WORDS for w in words):
            continue
        terms.append(" ".join(words))
        while words[0] in GENERIC_NAME_WORDS:
            words = words[1:]
        while words[-1] in GENERIC_NAME_WORDS:
            words = words[:-1]
        trimmed = " ".join(words)
        if trimmed not in TOO_VAGUE_ALONE and any(len(w) >= 3 and not w.isdigit() for w in words):
            terms.append(trimmed)
    return list(dict.fromkeys(terms))


def keyterm_table(mm):
    """One row per key term: distinctive phrases from each matter's name and codename (name_terms).
    shared = the term belongs to more than one matter ("fund ii", "general corporate")."""
    rows = []
    for cmid, matter, codename in zip(mm["CMID"], mm["Matter"], mm["Codename"]):
        found = name_terms(matter) + (name_terms(codename) if pd.notna(codename) else [])
        rows += [(t, cmid) for t in found]
    table = pd.DataFrame(rows, columns=["term_norm", "cmid"]).drop_duplicates().reset_index(drop=True)
    table["shared"] = table.groupby("term_norm")["cmid"].transform("nunique") > 1
    return table


def client_mention_table(mm, skip=("Firm",), acronym_skip=()):
    """Client names to look for in narratives: name, name without spaces, listed aliases, and initials.
    skip: clients whose name is an ordinary word ("firm meeting" doesn't name a client).
    Initials (kind "acronym") only count when written exactly, in capitals (see acronym_mentions)."""
    rows = []
    for client, aliases in mm[["Client", "Known Aliases"]].drop_duplicates().itertuples(index=False):
        if client in skip:
            continue
        names = [client, client.replace(" ", "")]
        if pd.notna(aliases):
            names += [a.strip() for a in str(aliases).split(",") if a.strip()]
        rows += [(n, client, "name") for n in names]
    rows += [(a, c, "acronym") for a, c in client_acronyms(mm, acronym_skip).itertuples(index=False) if c not in skip]
    table = pd.DataFrame(rows, columns=["name", "client", "kind"])
    table["name_norm"] = table["name"].map(lambda t: normalize(t, drop_suffixes=False))
    return table.drop_duplicates(["name_norm", "client"]).reset_index(drop=True)


def acronym_mentions(pairs, acronyms):
    """Client initials written exactly, in capitals, in the original narrative ("call w/ NCP re closing").
    Lowercase or partial matches don't count, so "mts" in "mtgs" never names a client."""
    texts = pairs[["narrative", "narrative_norm"]].drop_duplicates()
    rows = []
    for acronym, client in zip(acronyms["name"], acronyms["client"]):
        found = texts["narrative"].fillna("").str.contains(rf"\b{re.escape(acronym)}\b", regex=True)
        rows += [(t, client, 100.0) for t in texts.loc[found, "narrative_norm"]]
    return pd.DataFrame(rows, columns=["narrative_norm", "client", "score"])


def window_score(phrase, text):
    """Best fuzz.ratio between phrase and any run of the same number of whole words in text.

    Whole words keep "tern" from matching inside "internal", and typos still score high:
    window_score("capital call", "prep capial call notices") -> 95.7
    """
    phrase_words, words = phrase.split(), text.split()
    n, best = len(phrase_words), 0.0
    for i in range(len(words) - n + 1):
        window = words[i:i + n]
        if all(p == w for p, w in zip(phrase_words, window) if ID_WORD.match(p)):
            best = max(best, fuzz.ratio(phrase, " ".join(window)))
    return best


def phrase_scores(phrases, texts):
    """window_score for every (text, phrase) pair, done in bulk so it runs in seconds.
    Returns a table with one row per unique text and one column per unique phrase."""
    texts = sorted(set(texts))
    phrases = list(dict.fromkeys(phrases))
    out = pd.DataFrame(0.0, index=texts, columns=phrases)
    text_words = [t.split() for t in texts]

    for n in sorted({len(p.split()) for p in phrases}):
        group = [p for p in phrases if len(p.split()) == n]
        windows = [(i, w[j:j + n]) for i, w in enumerate(text_words) for j in range(len(w) - n + 1)]
        if not windows:
            continue
        owner = np.array([i for i, _ in windows])
        scores = process.cdist(group, [" ".join(w) for _, w in windows], scorer=fuzz.ratio, workers=-1)
        for k, phrase in enumerate(group):
            ids = [(pos, word) for pos, word in enumerate(phrase.split()) if ID_WORD.match(word)]
            if ids:
                agree = np.array([all(w[pos] == word for pos, word in ids) for _, w in windows])
                scores[k, ~agree] = 0
        best = pd.DataFrame(scores.T, columns=group).groupby(owner).max()
        out.loc[[texts[i] for i in best.index], group] = best.to_numpy(dtype=float)
    return out


def split_parts(narrative):
    """The separate pieces of work in a narrative, normalized ("Draft SPA (1.1); call w/ client (.3)" -> 2 pieces).
    Pieces under two words ("and" left over from "and also") are dropped."""
    if narrative is None or pd.isna(narrative):
        return []
    pieces = [normalize(p, drop_suffixes=False) for p in PART_SEPARATORS.split(str(narrative))]
    pieces = [p for p in pieces if len(p.split()) >= 2]
    whole = normalize(narrative, drop_suffixes=False)
    return pieces or ([whole] if whole else [])


def count_parts(narrative):
    """How many separate pieces of work a narrative describes."""
    return len(split_parts(narrative))


def _long(scores, phrases):
    long = scores[list(dict.fromkeys(phrases))].stack().reset_index()
    long.columns = ["narrative_norm", "phrase", "score"]
    return long


def add_text_features(pairs, keyterms, cues, client_mentions, min_score):
    """Key terms, task cues, and client names found in the narrative.

    keyterm_score / keyterm_hits    best score / count of the candidate's key terms found
    own_keyterm_hits                same, counting only terms listed for this matter alone
    n_matters_named                 matters named by a term of their own (2+ can mean more than one matter)
    cue_hits / cue_share            cues pointing to the candidate's type / as a share of all cues found
    n_types_cued                    matter types with at least one cue
    n_parts                         separate pieces of work in the narrative
    client_in_text                  best score of the candidate's client name in the narrative
    n_clients_named                 clients named in the narrative
    """
    names = client_mentions[client_mentions["kind"] != "acronym"]
    scores = phrase_scores([*keyterms["term_norm"], *cues["cue_norm"], *names["name_norm"]], pairs["narrative_norm"])
    keys = ["narrative_norm"]

    terms = _long(scores, keyterms["term_norm"]).merge(keyterms, left_on="phrase", right_on="term_norm")
    terms["hit"] = terms["score"] >= min_score
    terms["own_hit"] = terms["hit"] & ~terms["shared"]
    terms = terms.groupby(keys + ["cmid"], as_index=False).agg(
        keyterm_score=("score", "max"), keyterm_hits=("hit", "sum"), own_keyterm_hits=("own_hit", "sum"))
    named = terms[terms["own_keyterm_hits"] > 0].groupby(keys).agg(n_matters_named=("cmid", "nunique"))

    found_cues = _long(scores, cues["cue_norm"]).query("score >= @min_score")
    found_cues = found_cues.merge(cues[["cue_norm", "matter_type"]], left_on="phrase", right_on="cue_norm")
    by_type = found_cues.groupby(keys + ["matter_type"], as_index=False).agg(cue_hits=("phrase", "nunique"))
    per_text = found_cues.groupby(keys).agg(cue_total=("phrase", "nunique"), n_types_cued=("matter_type", "nunique"))

    clients = _long(scores, names["name_norm"]).merge(names, left_on="phrase", right_on="name_norm")
    clients = pd.concat([clients[keys + ["client", "score"]],
                         acronym_mentions(pairs, client_mentions[client_mentions["kind"] == "acronym"])])
    clients = clients.groupby(keys + ["client"], as_index=False).agg(client_in_text=("score", "max"))
    n_clients = clients.query("client_in_text >= @min_score").groupby(keys).agg(n_clients_named=("client", "nunique"))

    p = pairs.merge(terms, on=keys + ["cmid"], how="left")
    p = p.merge(by_type, on=keys + ["matter_type"], how="left")
    p = p.merge(clients, on=keys + ["client"], how="left")
    for table in [per_text, named, n_clients]:
        p = p.merge(table, on=keys, how="left")

    counts = ["keyterm_score", "keyterm_hits", "own_keyterm_hits", "n_matters_named", "cue_hits", "cue_total",
              "n_types_cued", "client_in_text", "n_clients_named"]
    p[counts] = p[counts].fillna(0)
    p["cue_share"] = np.where(p["cue_total"] > 0, p["cue_hits"] / p["cue_total"].clip(lower=1), 0.0)
    p["n_parts"] = p["narrative"].map(count_parts)
    return p.drop(columns="cue_total")


# ---------------- Dates and history ----------------

def add_date_features(pairs, history_end):
    """not_open: the work date is before the matter opened or after it closed.
    opened_after_history: the matter opened after the history window, so nobody has history on it."""
    before = pairs["work_date"] < pairs["open_date"]
    after = pairs["close_date"].notna() & (pairs["work_date"] > pairs["close_date"])
    pairs["not_open"] = (before | after).astype(int)
    pairs["opened_after_history"] = (pairs["open_date"] > history_end).astype(int)
    return pairs


def add_history_features(pairs, history):
    """Hours each timekeeper billed to each matter in the history window.

    history: tk_id, cmid, hours
    """
    h = history[["tk_id", "cmid", "hours"]].copy()
    h["hours"] = pd.to_numeric(h["hours"], errors="coerce").fillna(0.0)
    totals = h.groupby("tk_id")["hours"].sum()

    p = pairs.merge(h.rename(columns={"hours": "history_hours"}), on=["tk_id", "cmid"], how="left")
    p["history_hours"] = p["history_hours"].fillna(0.0)
    p["tk_known"] = p["tk_id"].isin(totals.index).astype(int)
    p["in_history"] = (p["history_hours"] > 0).astype(int)
    p["history_share"] = (p["history_hours"] / p["tk_id"].map(totals)).fillna(0.0)
    return p


def past_sim(query_vecs, query_folds, past_vecs, past_folds, past_cmids, cmids, k=3):
    """Average cosine similarity of each query to the k most similar past narratives of each matter (0 if none).

    A query is never compared with past narratives from its own fold (folds are NaN outside TRAIN).
    Returns one row per query and one column per cmid.
    """
    sims = query_vecs @ past_vecs.T
    sims[query_folds[:, None] == past_folds[None, :]] = -1
    scores = {}
    for cmid in cmids:
        cols = sims[:, past_cmids == cmid]
        scores[cmid] = np.sort(cols, axis=1)[:, -k:].mean(axis=1).clip(min=0) if cols.shape[1] else 0.0
    return pd.DataFrame(scores, index=range(len(query_vecs)))


def _past_set(entries, vecs, folds):
    """Past narratives: TRAIN entries with a known matter, standing in for the firm's billing history."""
    is_past = ((entries["split"] == "TRAIN") & (entries["key_true_cmid"] != "NONE")).to_numpy()
    fold = entries["entry_id"].map(folds).to_numpy(dtype=float)
    return vecs[is_past], fold[is_past], entries.loc[is_past, "key_true_cmid"].to_numpy()


def add_past_narrative_sim(pairs, entries, vecs, folds, k=3):
    """past_narrative_sim: how closely each narrative matches past narratives billed to each candidate matter.

    It's the average cosine similarity to the k most similar past narratives of that matter (0 if none).

    entries: one row per entry (entry_id, split, key_true_cmid), in the same order as vecs (unit-length embeddings).
    folds: TRAIN entry_id -> fold. A TRAIN entry is only compared with other folds (never itself), the same way
    the model is cross-validated. VAL and TEST entries are compared with all of TRAIN.
    """
    fold = entries["entry_id"].map(folds).to_numpy(dtype=float)
    scores = past_sim(vecs, fold, *_past_set(entries, vecs, folds), pairs["cmid"].unique(), k)
    long = scores.set_axis(entries["entry_id"]).stack().rename("past_narrative_sim")
    return pairs.merge(long.rename_axis(["entry_id", "cmid"]).reset_index(), on=["entry_id", "cmid"], how="left")


# ---------------- Narrative parts ----------------

def narrative_parts(entries):
    """One row per piece of work in each narrative (split_parts), normalized like narrative_norm."""
    rows = [(entry_id, i, piece) for entry_id, text in zip(entries["entry_id"], entries["narrative"])
            for i, piece in enumerate(split_parts(text))]
    return pd.DataFrame(rows, columns=["entry_id", "part", "part_norm"])


def match_parts(parts, part_vecs, entries, vecs, folds, cmids, keyterms, min_score, k=3):
    """Each piece of work matched on its own, to suggest how to divide an entry that covers more than one matter.

    named_cmid    a matter the part names by one of its own key terms (blank if none)
    top_cmids     the part's three most similar matters by past narratives, with their similarities (top_sims)

    parts: narrative_parts output, in the same order as part_vecs.
    """
    part_folds = parts["entry_id"].map(folds).to_numpy(dtype=float)
    sims = past_sim(part_vecs, part_folds, *_past_set(entries, vecs, folds), cmids, k).to_numpy()
    order = np.argsort(-sims, axis=1)[:, :3]
    cmids = np.asarray(cmids)

    own_terms = keyterms[~keyterms["shared"]]
    found = _long(phrase_scores(own_terms["term_norm"], parts["part_norm"]), own_terms["term_norm"])
    found = found[found["score"] >= min_score].merge(own_terms, left_on="phrase", right_on="term_norm")
    named = found.sort_values("score", ascending=False).drop_duplicates("narrative_norm").set_index("narrative_norm")["cmid"]

    return parts.assign(named_cmid=parts["part_norm"].map(named),
                        top_cmids=[" | ".join(cmids[o]) for o in order],
                        top_sims=[" | ".join(f"{v:.2f}" for v in row[o]) for row, o in zip(sims, order)])


def add_gaps(pairs, cols):
    """Each candidate's value minus the best value in its entry (0 = tied for best)."""
    for col in cols:
        pairs[f"{col}_gap"] = pairs[col] - pairs.groupby("entry_id")[col].transform("max")
    return pairs


# ---------------- All together ----------------

def feature_groups(scorers, gap_cols):
    """The model's feature columns, grouped by kind of clue."""
    s = [x.lower() for x in scorers]
    return {
        "typed_fields": ["has_cmid", "has_client", "has_matter", "client_is_ditto", "client_had_matter",
                         "cmid_from_text", "is_pdf", "is_handwritten", "is_email"],
        "cmid": ["cmid_valid", "cmid_client_known", "is_typed_cmid", "is_flipped_cmid", "cmid_similarity",
                 "same_client_number"],
        "names": [f"client_{x}" for x in s] + [f"matter_{x}" for x in s],
        "narrative": ["keyterm_score", "keyterm_hits", "own_keyterm_hits", "n_matters_named", "cue_hits",
                      "cue_share", "n_types_cued", "n_parts", "client_in_text", "n_clients_named"],
        "dates": ["not_open", "opened_after_history"],
        "history": ["tk_known", "in_history", "history_hours", "history_share"],
        "gaps": [f"{c}_gap" for c in gap_cols],
    }


def build_pairs(entries, matters, history, client_names, matter_names, keyterms, cues, client_mentions,
                scorers, min_score, gap_cols, history_end):
    """Run every step above in order."""
    check_cue_types(cues, matters["matter_type"])
    pairs = make_pairs(entries, matters)
    pairs = add_cmid_features(pairs, set(matters["cmid"]))
    pairs = add_name_scores(pairs, client_names, matter_names, scorers)
    pairs = add_text_features(pairs, keyterms, cues, client_mentions, min_score)
    pairs = add_date_features(pairs, history_end)
    pairs = add_history_features(pairs, history)
    pairs = add_gaps(pairs, gap_cols)
    return pairs.sort_values(["entry_id", "cmid"]).reset_index(drop=True)
