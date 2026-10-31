#!/usr/bin/env python3
"""
Evaluation of synthetic_time_entries_v3(.1).xlsx (TRAIN / VAL / TEST).

  Stage 1  Integrity and label audit (v3 key schema)
  Stage 2  Split hygiene (3-way split, leakage, drift, effective size)
  Stage 3  Shortcut and difficulty audit
  Stage 4  Fidelity vs the real fee-application file, per Source Format      (needs --real)
  Stage 6  Privacy / provenance scan                                          (needs --real for the full scan)

Stage 5 (utility on labeled real data) is not run: the real file has no quality labels or matter IDs.
The sealed HOLDOUT workbook is never read by this script.

Usage:
  python synth_eval_v3.py --synthetic synthetic_time_entries_v3.xlsx [--real time_entries_clean_balanced.csv] [--out outdir]
Requires synth_eval_stages_4_6.py next to this file (real-data cleaning and style features are reused from it).
"""
import argparse
import hashlib
import os
import re
import sys
import warnings
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_eval_stages_4_6 as M  # noqa: E402  (clean_real, style_frame, stage4, real_name_terms)

# ---------------------------------------------------------------- gates (starting points, not standards)
MIN_ROWS_PER_ACTION = 35        # README's own promise for VAL / TEST / HOLDOUT
MIN_ROWS_PER_CATEGORY = 30
NN_GAP_MIN = 0.15               # TRAIN->TRAIN minus TEST->TRAIN median similarity, long narratives only
LONG_WORDS = 7                  # narratives with at least this many words count as "specific"
DRIFT_P_FAIL, DRIFT_P_WARN = 0.01, 0.10
HOURS_WASS_REL = 0.10
SHORTCUT_MARGIN = 0.10
AUC_WARN, AUC_FAIL = 0.70, 0.80
KEYTERM_SURVIVE_MIN = 0.95
NN_COPY_SIM = 0.95
RES = []


def add(stage, check, value, gate, status, note=""):
    RES.append(dict(stage=stage, check=check, value=value, gate=gate, status=status, note=note))


def st(ok, warn=False):
    return "PASS" if ok else ("WARN" if warn else "FAIL")


# ---------------------------------------------------------------- loading and helpers
CAT_TO_ACTION = {"CLEAN": "AUTO_MATCH", "INCOMPLETE": "AUTO_MATCH", "MESSY": "AUTO_MATCH",
                 "WRONG_CMID_SILENT": "AUTO_MATCH", "AMBIGUOUS": "NEEDS_REVIEW", "WRONG_CMID_TEXT": "FLAG_MISMATCH",
                 "CLOSED_OR_NOT_OPEN": "FLAG_MISMATCH", "NO_MATCH": "NO_MATCH",
                 "SPLIT_NAMED": "SPLIT_ENTRY", "SPLIT_UNNAMED": "SPLIT_ENTRY"}
PROBLEM = {"AMBIGUOUS", "WRONG_CMID_TEXT", "WRONG_CMID_SILENT", "CLOSED_OR_NOT_OPEN", "NO_MATCH", "SPLIT_NAMED", "SPLIT_UNNAMED"}


def load(path):
    te = pd.read_excel(path, "Time Entries", dtype=str, keep_default_na=False)
    te.columns = [c.replace("KEY: ", "K_") for c in te.columns]
    te["Hours"] = te["Hours"].astype(float)
    te["d"] = pd.to_datetime(te["Date"], format="%m/%d/%Y")
    te["ed"] = pd.to_datetime(te["Entry Date"], format="%m/%d/%Y")
    mm = pd.read_excel(path, "Matter Master", dtype=str, keep_default_na=False)
    mm["od"] = pd.to_datetime(mm["Open Date"], format="%m/%d/%Y", errors="coerce")
    mm["cd"] = pd.to_datetime(mm["Close Date"], format="%m/%d/%Y", errors="coerce")
    tk = pd.read_excel(path, "Timekeepers", dtype=str, keep_default_na=False)
    hist = pd.read_excel(path, "TK Matter History", dtype=str, keep_default_na=False)
    hcol = [c for c in hist.columns if c.startswith("Hours")][0]
    hist["h"] = hist[hcol].astype(float)
    hist["fb"] = pd.to_datetime(hist["First Billed"], format="%m/%d/%Y", errors="coerce")
    hist["lb"] = pd.to_datetime(hist["Last Billed"], format="%m/%d/%Y", errors="coerce")
    try:
        summ = pd.read_excel(path, "Summary", header=None)
    except Exception:
        summ = None
    return te, mm, tk, hist, summ


def norm_cmid(s):
    s = str(s).strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d{4})\s*[.\-/ ]\s*\.?\s*(\d{1,3})", s)
    if m:
        return f"{m.group(1)}.{int(m.group(2)):03d}"
    if re.fullmatch(r"\d{7}", s):
        return f"{s[:4]}.{s[4:]}"
    return None


def open_on(mm_idx, cmid, d):
    if cmid not in mm_idx.index:
        return False
    r = mm_idx.loc[cmid]
    return (pd.isna(r.od) or r.od <= d) and (pd.isna(r.cd) or r.cd >= d)


def norm_text(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def slices(n):
    return [float(x) for x in re.findall(r"\((\d*\.?\d+)\)", str(n)) if float(x) <= 24]


# ================================================================ stage 1
def stage1(te, mm, tk, hist, summ):
    S = "1 Integrity"
    mi = mm.set_index("CMID")
    valid = set(mi.index)
    act, cat = te["K_Expected Action"], te["K_Entry Category"]
    n = len(te)
    add(S, "Rows / unique Time IDs", f"{n} / {te['Time ID'].nunique()}", "equal", st(n == te["Time ID"].nunique()))
    dup = te.duplicated(["TK ID", "Date", "Hours", "Narrative"]).sum()
    add(S, "Exact duplicate entries (TK, date, hours, narrative)", str(dup), "0", st(dup == 0, warn=True),
        "Real timekeepers can repeat short entries; check they are not copy artifacts.")

    nm = act == "NO_MATCH"
    bad = ((te["K_True CMID"] == "NONE") != nm).sum()
    add(S, "True CMID is NONE exactly for NO_MATCH", f"{bad} violations", "0", st(bad == 0))
    bad = (~te.loc[~nm, "K_True CMID"].isin(valid)).sum()
    add(S, "True CMID exists in Matter Master", f"{bad} violations", "0", st(bad == 0))
    j = te[~nm].join(mi[["Client", "Matter"]].add_prefix("M_"), on="K_True CMID")
    bad = ((j["K_True Client"] != j["M_Client"]) | (j["K_True Matter"] != j["M_Matter"])).sum()
    add(S, "True Client/Matter agree with master", f"{bad} violations", "0", st(bad == 0))

    bad = (cat.map(CAT_TO_ACTION) != act).sum()
    add(S, "Entry Category -> Expected Action mapping", f"{bad} violations", "0", st(bad == 0))
    bad = (te["K_Problem Entry"].eq("Y") != cat.isin(PROBLEM)).sum()
    add(S, "Problem Entry flag matches category", f"{bad} violations", "0", st(bad == 0))
    bad = (te["K_Silent Error"].eq("Y") != cat.eq("WRONG_CMID_SILENT")).sum()
    add(S, "Silent Error flag matches category", f"{bad} violations", "0", st(bad == 0))
    triv = (te["CMID"].str.strip() == te["K_True CMID"]) & act.eq("AUTO_MATCH")
    bad = (te["K_Trivial"].eq("Y") != triv).sum()
    add(S, "Trivial = typed CMID exactly equals True CMID", f"{bad} violations", "0", st(bad == 0))
    bad = ((te["K_History Hint"] != "") != cat.eq("WRONG_CMID_SILENT")).sum()
    add(S, "History Hint filled only on silent errors", f"{bad} violations", "0", st(bad == 0))

    sp = act == "SPLIT_ENTRY"
    bad = ((te["K_Secondary CMID"] != "") != sp).sum()
    add(S, "Secondary CMID filled exactly for SPLIT_ENTRY", f"{bad} violations", "0", st(bad == 0))
    s2 = te.loc[sp, "K_Secondary CMID"]
    bad = (~s2.isin(valid)).sum() + (s2 == te.loc[sp, "K_True CMID"]).sum()
    add(S, "Secondary CMID valid and different from primary", f"{bad} violations", "0", st(bad == 0))

    # typed-CMID relationships
    te = te.copy()
    te["ncm"] = te["CMID"].map(norm_cmid)
    te["typed_valid"] = te.ncm.isin(valid)
    te["typed_open"] = [open_on(mi, c, d) if c else False for c, d in zip(te.ncm, te.d)]
    silent = cat == "WRONG_CMID_SILENT"
    bad = (~(te.typed_valid & te.typed_open & (te.ncm != te["K_True CMID"]))[silent]).sum()
    unparsed = te.loc[silent, "ncm"].isna().sum()
    add(S, "Silent error: typed CMID is a real open matter, differs from truth", f"{bad} violations ({unparsed} typed forms not normalizable)",
        "0", st(bad == unparsed or bad == 0, warn=True),
        "Unparseable (OCR-garbled) typed forms cannot be checked.")
    wt = cat == "WRONG_CMID_TEXT"
    bad = (~(te.typed_valid & te.typed_open & (te.ncm != te["K_True CMID"]))[wt]).sum()
    add(S, "WRONG_CMID_TEXT: typed CMID is a real open matter, differs from truth", f"{bad} violations", "0", st(bad == 0, warn=True))
    cl = cat == "CLOSED_OR_NOT_OPEN"
    not_in = (~te.typed_valid & te.ncm.notna())[cl].sum()
    closed = (te.typed_valid & ~te.typed_open)[cl].sum()
    bad = (te.typed_valid & te.typed_open)[cl].sum()
    add(S, "CLOSED_OR_NOT_OPEN: typed CMID is closed/not yet open/absent on the work date",
        f"{bad} violations (closed or not-yet-open {closed}, absent from master {not_in}, unparsed {cl.sum() - closed - not_in - bad})",
        "0", st(bad == 0))
    ok = te.loc[~nm, ["K_True CMID", "d"]].apply(lambda r: open_on(mi, r["K_True CMID"], r["d"]), axis=1)
    add(S, "True matter open on the work date", f"{(~ok).sum()} violations", "0", st((~ok).sum() == 0))

    # canonical-client check for NO_MATCH
    canon = {x.strip().lower() for x in mm["Client"]} | {a.strip().lower() for s in mm["Known Aliases"] for a in s.split(",") if a.strip()}
    nmc = te[nm & (te.Client != "")]
    hit = nmc.Client.str.lower().isin(canon).sum()
    add(S, "NO_MATCH split: listed client with unopened matter (new-matter intake) vs unlisted client (new-client intake)", f"{hit} / {len(nmc) - hit}", "report", "INFO",
        "By design per Resolution Basis; both routes share one NO_MATCH label.")

    # dates and history leakage
    lag = (te.ed - te.d).dt.days
    add(S, "Entry-date lag in days", f"min {lag.min()}, median {int(lag.median())}, max {lag.max()}", "0-30", st(lag.min() >= 0 and lag.max() <= 30))
    add(S, "Work dates start after history window", f"first work date {te.d.min():%m/%d/%Y}; history last billed {hist.lb.max():%m/%d/%Y}",
        "history end < first entry", st(hist.lb.max() < te.d.min()))
    add(S, "History 'First Billed' within window", f"{hist.fb.min():%m/%d/%Y} to {hist.fb.max():%m/%d/%Y}", "04/01-06/30/2026",
        st(hist.fb.min() >= pd.Timestamp("2026-04-01") and hist.lb.max() <= pd.Timestamp("2026-06-30")))
    tkm = tk.set_index("TK ID").Timekeeper
    bad = (te["TK ID"].map(tkm) != te.Timekeeper).sum()
    add(S, "TK ID/name consistent with Timekeepers sheet", f"{bad} violations", "0", st(bad == 0))
    em = te["Source Format"] == "EMAIL"
    bad = ((te.loc[em, ["CMID", "Client", "Matter"]] != "").any(axis=1)).sum()
    add(S, "EMAIL rows have blank CMID/Client/Matter fields", f"{bad} violations of {em.sum()}", "0", st(bad == 0))

    # split entries: second matter named / unnamed by key terms
    def terms(c):
        r = mi.loc[c] if c in mi.index else None
        if r is None:
            return []
        t = [x.strip() for x in str(r["Key Terms"]).split(",") if x.strip()] + ([r["Codename"]] if r["Codename"] else [])
        return [x for x in t if len(x) >= 4]
    for c, want in (("SPLIT_NAMED", True), ("SPLIT_UNNAMED", False)):
        d = te[cat == c]
        hit = sum(any(t.lower() in nar.lower() for t in terms(s2_)) for nar, s2_ in zip(d.Narrative, d["K_Secondary CMID"]))
        share = hit / max(1, len(d))
        add(S, f"{c}: secondary matter's key term/codename appears in narrative", f"{hit} of {len(d)} ({share:.0%})",
            ">= 70%" if want else "<= 10%", st(share >= 0.7 if want else share <= 0.10, warn=True),
            "Key-term lookup only; contacts and other cues are not counted.")

    # label support: key term quoted in the resolution basis must be in the narrative
    kt = te[te["K_Resolution Basis"].str.contains("key term", case=False)]
    q = kt["K_Resolution Basis"].str.extract(r"key term '([^']+)'")[0]
    okk = [(isinstance(t, str) and t.lower() in nar.lower()) for t, nar in zip(q, kt.Narrative)]
    kt = kt.assign(ok=okk, parsed=q.notna())
    kt = kt[kt.parsed]
    if len(kt):
        tot = kt.ok.mean()
        by = kt.groupby("Source Format").ok.agg(["mean", "size"]).round(3)
        add(S, "Key term cited in Resolution Basis is present in the Narrative", f"{tot:.1%} of {len(kt)} rows; by format "
            + ", ".join(f"{i} {r['mean']:.0%} (n={int(r['size'])})" for i, r in by.iterrows()),
            f">= {KEYTERM_SURVIVE_MIN:.0%}", st(tot >= KEYTERM_SURVIVE_MIN, warn=True),
            "README: oracle assumes key terms stay recognizable after OCR.")
    # slice sums vs hours (finalized-register formats)
    fin = te[te["Source Format"].isin(["TYPED", "PDF"])]
    sl = fin.Narrative.map(slices)
    multi = fin[sl.map(len) >= 2]
    ms = sl[multi.index]
    match = np.mean([abs(sum(s) - h) < 0.051 for s, h in zip(ms, multi.Hours)]) if len(multi) else float("nan")
    add(S, "TYPED/PDF narratives with >=2 time slices: slice sum equals Hours", f"{match:.1%} of {len(multi)}", ">= 95%",
        st(match >= 0.95, warn=True), "Real fee data showed 787 of 796 repaired rows matching; the generator should hold at least that.")

    # summary sheet
    if summ is not None:
        titles = {"Trivial (typed CMID already correct)": "K_Trivial", "Entry Category": "K_Entry Category",
                  "Expected Action": "K_Expected Action", "Problem Entry": "K_Problem Entry",
                  "Source Format": "Source Format", "Narrative Quality": "K_Narrative Quality"}
        bad = chk = 0
        for i in summ.index[summ[0].isin(list(titles))]:
            col = titles[summ.iloc[i, 0]]
            ct = pd.crosstab(te[col], te["K_Split"])
            k = i + 1
            while k < len(summ) and pd.notna(summ.iloc[k, 0]):
                lab = str(summ.iloc[k, 0])
                for j, sp_ in ((1, "TRAIN"), (2, "VAL"), (3, "TEST")):
                    chk += 1
                    try:
                        bad += int(ct.loc[lab, sp_]) != int(summ.iloc[k, j])
                    except Exception:
                        bad += 1
                k += 1
        add(S, "Summary sheet counts match recomputed", f"{bad} of {chk} cells differ", "0", st(bad == 0, warn=True))
    return te


# ================================================================ stage 2
def cramers_v(ct):
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0.0


def stage2(te, mm):
    S = "2 Split hygiene"
    sz = te.K_Split.value_counts()
    add(S, "Split sizes", ", ".join(f"{k} {sz[k]} ({sz[k] / len(te):.1%})" for k in ["TRAIN", "VAL", "TEST"]), "70 / 15 / 15", "INFO")
    for sp_ in ["VAL", "TEST"]:
        ct = te[te.K_Split == sp_]["K_Expected Action"].value_counts()
        for c, k in ct.items():
            if c == "AUTO_MATCH":
                continue
            add(S, f"{sp_} rows in action {c}", str(k), f">= {MIN_ROWS_PER_ACTION}", st(k >= MIN_ROWS_PER_ACTION))
    tt = te[te.K_Split == "TEST"]["K_Entry Category"].value_counts()
    small = {c: int(k) for c, k in tt.items() if k < MIN_ROWS_PER_CATEGORY}
    add(S, "TEST rows per Entry Category below 30", str(small) if small else "none", f">= {MIN_ROWS_PER_CATEGORY} each",
        st(not small, warn=True), "Category-level results (e.g., SPLIT_UNNAMED, silent errors) are noisy at these sizes.")
    nt = te[(te.K_Split == "TEST") & (te.K_Trivial == "N")]
    add(S, "TEST rows that are non-trivial / silent errors", f"{len(nt)} / {(te[te.K_Split == 'TEST']['K_Silent Error'] == 'Y').sum()}", "report", "INFO")

    te = te.copy()
    te["n"] = te["K_Clean Narrative"].map(norm_text)
    te["w"] = te.n.str.split().str.len()
    tr, ts, va = (te[te.K_Split == s] for s in ["TRAIN", "TEST", "VAL"])
    for lab, other in (("TEST", ts), ("VAL", va)):
        ov = set(tr.n) & set(other.n)
        long_ov = [t for t in ov if len(t.split()) >= LONG_WORDS]
        rows = other.n.isin(ov).mean()
        add(S, f"Clean narratives shared {lab} and TRAIN",
            f"{len(ov)} distinct ({len(long_ov)} with >= {LONG_WORDS} words); {rows:.1%} of {lab} rows",
            f"0 long (README: only stock phrases may repeat)", st(len(long_ov) == 0, warn=True),
            "Examples of long overlaps: " + "; ".join(sorted(long_ov)[:3]) if long_ov else "")
    # memorization baseline
    maj = tr.groupby("n")["K_Expected Action"].agg(lambda s: s.value_counts().index[0])
    seen = ts[ts.n.isin(maj.index)]
    if len(seen):
        acc = (seen.n.map(maj) == seen["K_Expected Action"]).mean()
        base = (seen["K_Expected Action"] == "AUTO_MATCH").mean()
        add(S, "Memorization: TEST rows with a seen narrative, TRAIN-majority action accuracy",
            f"{acc:.2f} on {len(seen)} rows (always-AUTO_MATCH would score {base:.2f})", "report", "INFO",
            "If this is far above the always-AUTO_MATCH rate, repeated phrases leak the label.")

    # nearest-neighbour similarity (TF-IDF char cosine) on distinct long narratives
    trL = tr[tr.w >= LONG_WORDS].n.unique()
    tsL = ts[ts.w >= LONG_WORDS].n.unique()
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(np.r_[trL, tsL])
    A, B = tv.transform(trL), tv.transform(tsL)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(A)
    d_tt = nn.kneighbors(A)[0][:, 1]        # nearest other TRAIN narrative
    d_st = nn.kneighbors(B, n_neighbors=1)[0][:, 0]
    s_tt, s_st = 1 - np.median(d_tt), 1 - np.median(d_st)
    add(S, "Nearest-neighbour similarity, long narratives (median, TF-IDF cosine)",
        f"TEST->TRAIN {s_st:.2f} vs TRAIN->TRAIN {s_tt:.2f}", f"gap >= {NN_GAP_MIN}", st(s_tt - s_st >= NN_GAP_MIN, warn=True),
        "A small gap means TEST phrasings are about as close to TRAIN as TRAIN is to itself.")
    add(S, "Share of long TEST narratives with a TRAIN neighbour at cosine >= 0.9", f"{(1 - d_st >= 0.9).mean():.1%} of {len(tsL)}", "report", "INFO")

    # drift
    for col in ["Source Format", "K_Entry Category", "K_Narrative Quality", "K_TK Style", "TK ID", "K_True CMID", "K_CMID Form",
                "K_Client Surface", "K_Matter Surface"]:
        for lab, o in (("TEST", ts), ("VAL", va)):
            ct = pd.crosstab(te.loc[te.K_Split.isin(["TRAIN", lab]), col], te.loc[te.K_Split.isin(["TRAIN", lab]), "K_Split"])
            p = chi2_contingency(ct)[1]
            a, b = ct["TRAIN"] / ct["TRAIN"].sum(), ct[lab] / ct[lab].sum()
            status = "FAIL" if p < DRIFT_P_FAIL else ("WARN" if p < DRIFT_P_WARN else "PASS")
            add(S, f"Drift TRAIN vs {lab}: {col.replace('K_', '')}", f"JSD {jensenshannon(a.values, b.values, base=2):.3f}, chi2 p {p:.3f}",
                f"p >= {DRIFT_P_WARN}", status)
    for lab, o in (("TEST", ts), ("VAL", va)):
        w = wasserstein_distance(tr.Hours, o.Hours)
        add(S, f"Drift TRAIN vs {lab}: Hours", f"Wasserstein {w:.3f} ({w / te.Hours.mean():.1%} of mean), KS p {ks_2samp(tr.Hours, o.Hours).pvalue:.2f}",
            f"< {HOURS_WASS_REL:.0%} of mean", st(w / te.Hours.mean() < HOURS_WASS_REL))
    # surface-form reuse
    var = te[te["K_Client Surface"].isin(["MISSPELLED", "UNLISTED_ACRONYM", "LOOKALIKE", "TRUNCATED", "SHORT"])]
    trv = set(var[var.K_Split == "TRAIN"].Client)
    tsr = var[var.K_Split == "TEST"]
    add(S, "TEST rows whose misspelled/acronym client string also appears in TRAIN", f"{tsr.Client.isin(trv).mean():.0%} of {len(tsr)}",
        "report (v3 reserves unseen variants only for HOLDOUT)", "INFO")
    # temporal structure
    add(S, "Date range per split", "; ".join(f"{k} {g.d.min():%m/%d}-{g.d.max():%m/%d}" for k, g in te.groupby("K_Split")),
        "report", "INFO", "Splits are not temporal; every split spans the same weeks.")


# ================================================================ stage 3
def boot_f1(y, p, labels, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    v = [f1_score(y[i], p[i], labels=labels, average="macro", zero_division=0)
         for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    return np.percentile(v, [2.5, 97.5])


def stage3(te, mm, hist):
    S = "3 Shortcuts"
    mi = mm.set_index("CMID")
    valid = set(mi.index)
    te = te.copy()
    te["ncm"] = te["CMID"].map(norm_cmid)
    y = te["K_Expected Action"]
    tr = (te.K_Split == "TRAIN").values
    ts = (te.K_Split == "TEST").values
    nt_mask = (te.K_Trivial == "N").values
    labels = sorted(y.unique())

    F = pd.DataFrame(index=te.index)
    F["hours"] = te.Hours
    F["len"] = te.Narrative.str.len()
    F["words"] = te.Narrative.str.split().str.len()
    F["lag"] = (te.ed - te.d).dt.days
    for s in ["TYPED", "PDF", "HANDWRITTEN", "EMAIL"]:
        F[f"src_{s}"] = (te["Source Format"] == s).astype(int)
    G = pd.DataFrame(index=te.index)
    G["has_cmid"] = (te.CMID != "").astype(int)
    G["has_client"] = (te.Client != "").astype(int)
    G["has_matter"] = (te.Matter != "").astype(int)
    G["cmid_valid"] = te.ncm.isin(valid).astype(int)
    G["cmid_open"] = [int(open_on(mi, c, d)) if c else 0 for c, d in zip(te.ncm, te.d)]
    shortcut_cols = list(F.columns)

    def row(name, p, gate_shortcut=False, note=""):
        mf = f1_score(y[ts], p, labels=labels, average="macro", zero_division=0)
        lo, hi = boot_f1(y[ts], p, labels)
        sub = nt_mask[ts]
        mfn = f1_score(y[ts][sub], p[sub], labels=labels, average="macro", zero_division=0)
        add(S, f"Baseline {name}", f"TEST acc {accuracy_score(y[ts], p):.3f}, macro-F1 {mf:.3f} [{lo:.2f}-{hi:.2f}]; non-trivial macro-F1 {mfn:.3f}",
            f"<= majority + {SHORTCUT_MARGIN}" if gate_shortcut else "report", "INFO", note)
        return mf

    maj_pred = np.repeat(y[tr].mode()[0], ts.sum())
    mf_maj = row("majority class", maj_pred)
    gb = lambda X: GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr]).predict(X[ts])
    p = gb(F[shortcut_cols].values)
    mf = f1_score(y[ts], p, labels=labels, average="macro", zero_division=0)
    row("hours + length + entry lag + source format (no narrative content)", p, gate_shortcut=True)
    RES[-1]["status"] = st(mf <= mf_maj + SHORTCUT_MARGIN)
    add(S, "Shortcut gate", f"{mf:.3f} vs majority {mf_maj:.3f}", f"<= majority + {SHORTCUT_MARGIN}", st(mf <= mf_maj + SHORTCUT_MARGIN),
        "Features available without reading the narrative.")
    row("entered-field presence and validity only (legitimate signal)", gb(G.values))
    tvz = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(te.Narrative[tr])
    Xn = tvz.transform(te.Narrative)
    p_txt = LogisticRegression(max_iter=3000, C=5, class_weight="balanced").fit(Xn[tr], y[tr]).predict(Xn[ts])
    row("narrative text only", p_txt)
    D = StandardScaler(with_mean=False).fit(pd.concat([F, G], axis=1)[tr] if False else pd.concat([F, G], axis=1).values[tr])
    Xall = hstack([Xn, csr_matrix(D.transform(pd.concat([F, G], axis=1).values))]).tocsr()
    lr = LogisticRegression(max_iter=4000, C=2, class_weight="balanced").fit(Xall[tr], y[tr])
    p_all = lr.predict(Xall[ts])
    row("text + entered fields + shortcut features", p_all)
    for c in labels:
        m_ = (y[ts] == c).values
        rec = (p_all[m_] == c).mean()
        prec = (y[ts].values[p_all == c] == c).mean() if (p_all == c).any() else float("nan")
        add(S, f"Best baseline per-class: {c}", f"recall {rec:.2f}, precision {prec:.2f} (n={m_.sum()})", "report", "INFO")
    sil = (te.K_Silent_Error if "K_Silent_Error" in te else te["K_Silent Error"]).eq("Y").values & ts
    add(S, "Best baseline: predicted action on silent errors (labelled AUTO_MATCH)",
        f"{(lr.predict(Xall[sil]) == 'AUTO_MATCH').mean():.0%} predicted AUTO_MATCH (n={sil.sum()})", "report", "INFO",
        "Action is correct by construction; the matter is wrong. See the matter-resolution baseline below.")

    # matter-resolution baseline: typed CMID, else history top-hours (client-restricted if the client resolves)
    alias = {}
    for _, r in mm.iterrows():
        for a in [r.Client] + [x.strip() for x in str(r["Known Aliases"]).split(",") if x.strip()]:
            alias[a.lower()] = r.Client
    th = {tk: g.sort_values("h", ascending=False) for tk, g in hist.groupby("TK ID")}
    open_by_client = mm.groupby("Client").CMID.apply(list).to_dict()

    def resolve(r):
        if r.ncm in valid:
            return r.ncm
        cl = alias.get(str(r.Client).lower().strip())
        g = th.get(r["TK ID"])
        if g is not None:
            g2 = g[g.Client == cl] if cl else g
            if len(g2):
                return g2.iloc[0].CMID
        return open_by_client.get(cl, [None])[0] if cl else None
    am = te[(y == "AUTO_MATCH") & ts]
    pred = am.apply(resolve, axis=1)
    ok = pred == am["K_True CMID"]
    add(S, "Matter-resolution baseline (typed CMID, else history top-hours) on TEST AUTO_MATCH",
        f"overall {ok.mean():.1%} (n={len(am)}); non-trivial {ok[am.K_Trivial == 'N'].mean():.1%} (n={(am.K_Trivial == 'N').sum()}); "
        f"silent errors {ok[am['K_Silent Error'] == 'Y'].mean():.1%} (n={(am['K_Silent Error'] == 'Y').sum()})", "report", "INFO",
        "Silent errors are wrong for any method that trusts the typed CMID; they test history and content use.")
    by = ok.groupby(am["K_Entry Category"]).mean().round(3).to_dict()
    add(S, "Matter-resolution baseline by category (TEST AUTO_MATCH)", str(by), "report", "INFO")

    # history ceiling on silent errors
    sil_all = te[te["K_Silent Error"] == "Y"]
    inh = sil_all.apply(lambda r: ((hist["TK ID"] == r["TK ID"]) & (hist.CMID == r["K_True CMID"])).any(), axis=1)
    top = sil_all.apply(lambda r: th.get(r["TK ID"], pd.DataFrame({"CMID": [None]})).iloc[0].CMID == r["K_True CMID"], axis=1)
    for hh in ["Y", "N"]:
        m_ = sil_all["K_History Hint"] == hh
        add(S, f"Silent errors with History Hint = {hh}: true matter in TK history / is TK's top matter",
            f"{inh[m_].mean():.0%} / {top[m_].mean():.0%} (n={m_.sum()})", "report", "INFO",
            "Hint=N rows have no history clue; the true matter cannot be derived from the entry, the typed CMID or history."
            if hh == "N" else "")
    n_unlearnable = (sil_all["K_History Hint"] == "N").sum()
    add(S, "Silent errors with no history clue (ceiling on matter accuracy)", f"{n_unlearnable} rows = {n_unlearnable / (y == 'AUTO_MATCH').sum():.1%} of AUTO_MATCH",
        "report", "WARN" if n_unlearnable / (y == "AUTO_MATCH").sum() > 0.02 else "INFO",
        "Score these separately: a resolver that correctly trusts the typed CMID is marked wrong on them.")

    # within-stratum single-feature AUC
    qual = te["K_Narrative Quality"].values

    def strat_auc(yb, x):
        num = den = 0.0
        for q in set(qual):
            i = qual == q
            pos, neg = yb[i].sum(), (1 - yb[i]).sum()
            if pos >= 5 and neg >= 5:
                num += pos * neg * roc_auc_score(yb[i], x[i])
                den += pos * neg
        return num / den if den else float("nan")
    flagged = 0
    tested = 0
    for c in labels:
        yb = (y == c).astype(int).values
        for f in ["hours", "len", "words", "lag"]:
            a = strat_auc(yb, F[f].values)
            if np.isnan(a):
                continue
            tested += 1
            dev = max(a, 1 - a)
            status = "FAIL" if dev >= AUC_FAIL else ("WARN" if dev >= AUC_WARN else "PASS")
            if status != "PASS":
                flagged += 1
                add(S, f"Within-quality AUC: {f} vs {c}", f"{a:.2f}", f"within {1 - AUC_WARN:.2f}-{AUC_WARN:.2f}", status,
                    "Identifiable from this input even among entries of the same narrative quality.")
    if not flagged:
        add(S, "Within-quality single-feature AUCs (hours, length, words, entry lag vs each action)", f"all {tested} within gate", "0.30-0.70", "PASS")
    # categorical predictors
    for col in ["Source Format", "TK ID"]:
        ct = pd.crosstab(te[col], y)
        add(S, f"Association with action: {col}", f"Cramér's V {cramers_v(ct):.2f}", "report", "INFO")

    # class-exclusive tokens
    toks = te.Narrative.str.lower().str.findall(r"[a-z0-9&/+']+")
    doc, byc = Counter(), {c: Counter() for c in labels}
    for t, c in zip(toks, y):
        for w in set(t):
            doc[w] += 1
            byc[c][w] += 1
    majority = y.value_counts().idxmax()
    found = False
    for c in labels:
        if c == majority:
            continue
        tot = int((y == c).sum())
        hits = sorted([(w, byc[c][w], n) for w, n in doc.items() if n >= 8 and byc[c][w] / n >= 0.90 and byc[c][w] / tot >= 0.10],
                      key=lambda t: -t[1])
        if hits:
            found = True
            add(S, f"Class-exclusive tokens for {c}", ", ".join(f"'{w}' {k}/{n}" for w, k, n in hits[:6]),
                "none (>=90% precision, >=10% recall, df>=8)", "WARN", "Lexical marker that identifies the class without reading content.")
    if not found:
        add(S, "Class-exclusive tokens (>=90% precision, >=10% recall, df>=8)", "none", "none", "PASS")
    # trivial-rule stats
    add(S, "Trivial rows (typed CMID already exact)", f"{(te.K_Trivial == 'Y').sum()} = {(te.K_Trivial == 'Y').mean():.1%} of all; TEST {(te.K_Trivial[ts] == 'Y').sum()}",
        "report; also score non-trivial", "INFO")


# ================================================================ stage 4 / 6 (real data)
def stage4_6(te, mm, tk, real, workbook, out):
    s4, s6 = "4 Fidelity", "6 Privacy/provenance"
    if real is None:
        add(s4, "All Stage 4 checks", "NOT RUN", "needs --real", "NOT RUN")
        add(s6, "Name scan / near-duplicate vs real", "NOT RUN", "needs --real", "NOT RUN")
        return
    groups = {"TYPED+PDF (finalized register)": te["Source Format"].isin(["TYPED", "PDF"]),
              "HANDWRITTEN+EMAIL (raw register)": te["Source Format"].isin(["HANDWRITTEN", "EMAIL"])}
    for lab, mask in groups.items():
        M.RES.clear()
        d = te[mask][["Narrative", "Hours"]].reset_index(drop=True)
        os.makedirs(os.path.join(out, "stage4_tmp"), exist_ok=True)
        M.stage4(d, real, os.path.join(out, "stage4_tmp"))
        for r in M.RES:
            if r["check"].startswith("Blind reviewer"):
                continue
            RES.append({**r, "stage": f"{s4} [{lab}]"})
    # blind sheet across the whole v3 sample
    rr = real.sample(min(40, len(real)), random_state=1)
    ss = te.sample(40, random_state=1)
    a = pd.DataFrame({"narrative": rr.narrative.values, "hours": rr.hours.values, "source": "real", "synthetic_format": ""})
    b = pd.DataFrame({"narrative": ss.Narrative.values, "hours": ss.Hours.values, "source": "synthetic", "synthetic_format": ss["Source Format"].values})
    m_ = pd.concat([a, b]).sample(frac=1, random_state=1).reset_index(drop=True)
    m_.insert(0, "id", range(1, len(m_) + 1))
    m_[["id", "narrative", "hours"]].assign(rating_looks_real_1to5="").to_csv(os.path.join(out, "blind_review_sheet_v3.csv"), index=False)
    m_[["id", "source", "synthetic_format"]].to_csv(os.path.join(out, "blind_review_key_v3.csv"), index=False)
    add(s4, "Blind reviewer sheet written (v3)", "80 items", "2-3 reviewers; detection accuracy < 65%", "TODO")

    # Stage 6: names (incl. contacts and timekeepers) and near-duplicates
    J = lambda s: " ".join(s.fillna("").astype(str))
    corpus = " \n ".join([J(te.Narrative), J(te.Client), J(te.Matter), J(te.Timekeeper), J(tk.Timekeeper)] +
                         [J(mm[c]) for c in ["Client", "Matter", "Codename", "Known Aliases", "Key Terms", "Client Contacts", "Matter Contacts"] if c in mm])
    terms = M.real_name_terms(real)
    found = [t for t in terms if len(t) >= 4 and re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", corpus, re.I)]
    add(s6, "Real firm / case / person names found in synthetic text and reference tables", f"{len(found)} of {len(terms)} terms {found[:10]}", "0",
        st(not found), "Includes client/matter contacts and timekeeper names. Review any hit: surnames can coincide with ordinary words.")
    rn = sorted({norm_text(t) for t in real.narrative.astype(str) if len(norm_text(t)) >= 3})
    sn = sorted({norm_text(t) for t in te.Narrative.astype(str) if len(norm_text(t)) >= 3})
    add(s6, "Exact narrative matches synthetic == real", f"{len(set(rn) & set(sn))} total; {len([t for t in set(rn) & set(sn) if len(t.split()) >= 6])} with >= 6 words",
        "0 with >= 6 words", st(len([t for t in set(rn) & set(sn) if len(t.split()) >= 6]) == 0))
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(rn + sn)
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(tv.transform(rn))
    idx = nn.kneighbors(tv.transform(sn), return_distance=False)
    best = []
    for s_, ii in zip(sn, idx):
        best.append(max(SequenceMatcher(None, s_, rn[i], autojunk=False).ratio() for i in ii))
    best = np.array(best)
    longhit = [(b, s_) for b, s_ in zip(best, sn) if b >= NN_COPY_SIM and len(s_.split()) >= 6]
    add(s6, "Synthetic->real nearest-neighbour char similarity", f"median {np.median(best):.2f}, 95th pct {np.percentile(best, 95):.2f}, max {best.max():.2f}",
        f"no >= {NN_COPY_SIM} with >= 6 words", st(not longhit), "Examples: " + "; ".join(s_ for _, s_ in longhit[:3]) if longhit else "")
    add(s6, "Synthetic client names vs real companies", "manual check", "no unintended real entities", "TODO")
    add(s6, "Workbook fingerprint (sha256, first 16)", hashlib.sha256(open(workbook, "rb").read()).hexdigest()[:16], "record", "INFO")
    add(s6, "Generator provenance scan", "NOT RUN", "generate_synthetic_v3.py not provided", "NOT RUN",
        "README says handwritten samples were used as a style guide only; without the generator this cannot be verified.")


# ================================================================ report
def write_report(out, note):
    df = pd.DataFrame(RES)
    df.to_csv(os.path.join(out, "scorecard_v3.csv"), index=False)
    L = ["# Synthetic data evaluation: v3", "", note, "", "Status counts: " + ", ".join(f"{k} {v}" for k, v in df.status.value_counts().items()), ""]
    for stage, g in df.groupby("stage", sort=True):
        L += [f"## Stage {stage}", "", "| Status | Check | Result | Gate |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| {r.status} | {r.check} | {r.value} | {r.gate} |")
        L.append("")
    L += ["## Action items", ""]
    for _, r in df[df.status.isin(["FAIL", "WARN", "TODO", "NOT RUN"])].iterrows():
        L.append(f"- **{r.status}** {r.check}: {r.value}. {r.note}".rstrip())
    open(os.path.join(out, "scorecard_v3.md"), "w").write("\n".join(L) + "\n")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--real")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    te, mm, tk, hist, summ = load(a.synthetic)
    te2 = stage1(te, mm, tk, hist, summ)
    stage2(te, mm)
    stage3(te, mm, hist)
    real = None
    if a.real:
        raw = M.read_any(a.real)
        real, rep = M.clean_real(raw)
        add("0 Real-data cleaning", "Rows in / kept", f"{len(raw)} / {len(real)}", "report", "INFO")
    stage4_6(te, mm, tk, real, a.synthetic, a.out)
    df = write_report(a.out, f"Synthetic: `{os.path.basename(a.synthetic)}`" + (f" | Real: `{os.path.basename(a.real)}`" if a.real else ""))
    pd.set_option("display.width", 250, "display.max_colwidth", 140, "display.max_rows", 400)
    print(df[["status", "stage", "check", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
