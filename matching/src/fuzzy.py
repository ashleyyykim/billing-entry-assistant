"""Fuzzy matching of typed client and matter names against Matter Master."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from clean import normalize

# name -> (scorer, multiplier that puts it on a 0-100 scale)
SCORERS = {
    "ratio":            (fuzz.ratio, 1),
    "partial_ratio":    (fuzz.partial_ratio, 1),
    "token_sort_ratio": (fuzz.token_sort_ratio, 1),
    "token_set_ratio":  (fuzz.token_set_ratio, 1),
    "WRatio":           (fuzz.WRatio, 1),
    "JaroWinkler":      (JaroWinkler.normalized_similarity, 100),
}


def _names(cell):
    return [] if pd.isna(cell) else [n.strip() for n in str(cell).split(",") if n.strip()]


def client_name_table(mm):
    """One row per known client name: the client's own name plus its listed aliases."""
    rows = []
    for client, aliases in zip(mm["Client"], mm["Known Aliases"]):
        rows += [(client, client, "canonical")] + [(a, client, "alias") for a in _names(aliases)]
    table = pd.DataFrame(rows, columns=["name", "id", "kind"]).drop_duplicates(["name", "id"])
    table["name_norm"] = table["name"].map(normalize)
    return table.reset_index(drop=True)


def matter_name_table(mm):
    """One row per known matter name: the matter name plus its codename, keyed by CMID."""
    rows = [(m, cmid, "matter") for m, cmid in zip(mm["Matter"], mm["CMID"])]
    rows += [(c, cmid, "codename") for c, cmid in zip(mm["Codename"], mm["CMID"]) if pd.notna(c)]
    table = pd.DataFrame(rows, columns=["name", "id", "kind"]).drop_duplicates(["name", "id"])
    table["name_norm"] = table["name"].map(normalize)
    return table.reset_index(drop=True)


def score_table(typed, names, scorer):
    """Score every typed value against every name. Rows = typed values, columns = ids (best name per id)."""
    fn, mult = SCORERS[scorer]
    scores = process.cdist(list(typed), names["name_norm"].tolist(), scorer=fn, workers=-1) * mult
    return pd.DataFrame(scores, columns=names["id"].values).T.groupby(level=0).max().T


def match(text, names, scorer="ratio", k=3, allowed=None):
    """Top-k (id, score) pairs for one typed value. allowed limits the search to some ids."""
    query = normalize(text)
    if allowed is not None:
        names = names[names["id"].isin(allowed)]
    if not query or names.empty:
        return []
    best = score_table([query], names, scorer).iloc[0].sort_values(ascending=False)
    return [(i, round(float(s), 1)) for i, s in best.head(k).items()]


def rank(df, names, scorer, allowed=None):
    """Add the top pick, its score, the gap to second place, and the true id's rank to each row.

    df needs columns typed_norm and true_id. allowed: one set of ids per row (empty = any id).
    """
    scores = score_table(df["typed_norm"].fillna(""), names, scorer)
    arr, ids = scores.to_numpy().copy(), scores.columns.to_numpy()
    if allowed is not None:
        mask = np.array([np.isin(ids, list(a)) if a else np.ones(len(ids), bool) for a in allowed])
        arr = np.where(mask, arr, -1.0)

    order = np.argsort(-arr, axis=1, kind="stable")
    rows = np.arange(len(arr))
    ranked = ids[order]
    is_true = ranked == df["true_id"].to_numpy()[:, None]

    out = df.copy()
    out["pred_id"] = ranked[:, 0]
    out["top_score"] = arr[rows, order[:, 0]]
    out["runner_up_id"] = ranked[:, 1]
    out["gap"] = out["top_score"] - arr[rows, order[:, 1]].clip(min=0)
    out["true_rank"] = np.where(is_true.any(axis=1), is_true.argmax(axis=1) + 1, np.nan)
    out["correct"] = out["pred_id"] == out["true_id"]
    return out


def compare_scorers(df, names, allowed=None, k=3):
    """Run every scorer. Returns (one row of results per scorer, {scorer: ranked rows})."""
    results, ranked = [], {}
    for scorer in SCORERS:
        r = rank(df, names, scorer, allowed)
        ranked[scorer] = r
        results.append({"scorer": scorer, "top1_acc": r["correct"].mean(),
                        f"recall@{k}": (r["true_rank"] <= k).mean(),
                        "median_gap": r.loc[r["correct"], "gap"].median(),
                        "min_gap": r.loc[r["correct"], "gap"].min()})
    return pd.DataFrame(results).set_index("scorer"), ranked


def exact_match_acc(df, names):
    """Share of rows whose typed text exactly matches a name of the true id (and no other id)."""
    lookup = names.groupby("name_norm")["id"].agg(set)
    hits = df["typed_norm"].map(lookup)
    return np.mean([isinstance(h, set) and h == {t} for h, t in zip(hits, df["true_id"])])


def add_name_scores(pairs, client_names, matter_names, scorers):
    """Score the typed client against each candidate's client and the typed matter against its
    matter name or codename, one column per scorer (client_<scorer>, matter_<scorer>). 0 if blank."""
    for scorer in scorers:
        for field, names, key in [("client", client_names, "client"), ("matter", matter_names, "cmid")]:
            typed = sorted(set(pairs[f"typed_{field}"]) - {""})
            col = f"{field}_{scorer.lower()}"
            long = score_table(typed, names, scorer).set_axis(typed).stack().astype(float)
            long = long.rename(col).rename_axis([f"typed_{field}", key]).reset_index()
            pairs = pairs.merge(long, on=[f"typed_{field}", key], how="left")
            pairs[col] = pairs[col].fillna(0.0)
    return pairs
