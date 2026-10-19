"""Task cues: phrases that hint at the type of work, keyed by Matter Master "Matter Type".

Rules for this list:
  1. Cues come only from public sources, and each one records its source (CUE_SOURCES).
  2. Cues are never added because an entry in the data was missed.
  3. "FIRM" cues are common law-firm wording that still needs review by someone who has billed time.
"""
import pandas as pd

from clean import normalize

CUE_SOURCES = {
    "UTBMS_MA": "ABA/LOC Mergers & Acquisitions UTBMS code set (2016): phase and task descriptions",
    "SEC":      "SEC form and filing names (EDGAR)",
    "NVCA":     "NVCA model venture capital financing documents",
    "DOCS":     "Standard transaction document names",
    "NALP":     "NALP (National Association for Law Placement) recruiting terms",
    "FIRM":     "Common law-firm internal work and wording (to be vetted)",
}

TASK_CUES = {
    "ADMIN": {
        "FIRM": ["CLE", "continuing legal education", "firm meeting", "practice group meeting",
                 "time entry", "timekeeping", "training"],
    },
    "BD": {
        "FIRM": ["pitch", "RFP", "request for proposal", "business development", "client development",
                 "marketing", "client alert"],
    },
    "BOARD": {
        "UTBMS_MA": ["board presentation", "board minutes", "board resolutions", "fiduciary duties"],
        "DOCS":     ["board meeting", "board consent", "unanimous written consent", "committee charter",
                     "D&O questionnaire", "minutes"],
    },
    "CAPMKT": {
        "SEC":  ["S-1", "S-3", "F-1", "registration statement", "prospectus", "424(b)"],
        "DOCS": ["underwriting agreement", "underwriters", "comfort letter", "lock-up agreement",
                 "roadshow", "bring-down", "offering"],
    },
    "DEBT": {
        "UTBMS_MA": ["security documents", "guarantee"],
        "DOCS":     ["credit agreement", "loan agreement", "security agreement", "guaranty",
                     "promissory note", "intercreditor agreement", "UCC financing statement",
                     "perfection certificate", "borrower", "payoff letter"],
    },
    "EXECCOMP": {
        "UTBMS_MA": ["executive agreements", "change in control payments", "equity incentive"],
        "DOCS":     ["employment agreement", "offer letter", "severance", "equity plan", "option grant",
                     "RSU", "409A", "280G"],
    },
    "FUND_FORM": {
        "DOCS": ["limited partnership agreement", "LPA", "private placement memorandum", "PPM",
                 "subscription agreement", "subscription documents", "investor questionnaire",
                 "fund formation", "MFN election", "general partner"],
        "SEC":  ["Form D", "Form ADV"],
    },
    "FUND_GEN": {
        "DOCS": ["capital call", "capital call notice", "distribution notice", "LP inquiry",
                 "LPAC", "investor reporting", "fund administration"],
    },
    "GENCORP": {
        "DOCS": ["certificate of incorporation", "bylaws", "stockholder consent", "NDA",
                 "confidentiality agreement", "good standing", "annual report", "corporate housekeeping",
                 "founder", "option grant"],
    },
    "LICENSING": {
        "UTBMS_MA": ["intellectual property", "licensing"],
        "DOCS":     ["license agreement", "license", "royalty", "exclusive license", "IP assignment",
                     "distribution agreement", "technology transfer"],
    },
    "MA_BUY": {
        "UTBMS_MA": ["acquisition financing", "integration", "post-closing restructuring",
                     "transfer of assets"],
        "FIRM":     ["target", "seller", "diligence request list", "diligence memo", "buy side"],
    },
    "MA_SELL": {
        "UTBMS_MA": ["bidding procedures", "auction", "service providers"],
        "FIRM":     ["buyer", "bidder", "confidential information memorandum", "CIM", "sell side",
                     "management presentation"],
    },
    "RE_ACQ": {
        "UTBMS_MA": ["real property", "land use", "property due diligence"],
        "DOCS":     ["purchase and sale agreement", "PSA", "title commitment", "title insurance",
                     "ALTA survey", "survey", "zoning", "deed", "environmental report"],
    },
    "RE_LEASE": {
        "UTBMS_MA": ["lease assignment", "landlord consent"],
        "DOCS":     ["lease", "lease amendment", "landlord", "tenant", "SNDA", "estoppel certificate",
                     "sublease", "TI allowance"],
    },
    "RECRUIT": {
        "NALP": ["on-campus interview", "OCI", "callback interview", "summer associate", "summer program",
                 "lateral hire", "recruiting"],
    },
    "SEC": {
        "SEC":      ["10-K", "10-Q", "8-K", "Form 4", "Form 3", "Section 16", "proxy statement",
                     "Schedule 13D", "MD&A", "annual meeting", "periodic report"],
        "UTBMS_MA": ["proxy filings", "securities regulatory"],
    },
    "TAX": {
        "UTBMS_MA": ["tax structuring", "tax due diligence", "tax allocation", "tax indemnity"],
        "DOCS":     ["tax opinion", "contribution agreement", "reorganization", "step plan", "tax"],
    },
    "VC": {
        "NVCA": ["term sheet", "stock purchase agreement", "certificate of incorporation",
                 "investors rights agreement", "voting agreement", "right of first refusal",
                 "co-sale agreement", "ROFR", "management rights letter", "preferred stock",
                 "cap table", "capitalization table", "lead investor", "pro rata"],
    },
}

SHARED_CUES = {
    ("MA_BUY", "MA_SELL"): {
        "UTBMS_MA": ["purchase agreement", "merger agreement", "due diligence", "disclosure schedules",
                     "data room", "ancillary documents", "third party consents", "working capital",
                     "purchase price adjustment", "earn-out", "earnout", "indemnification claims",
                     "fairness opinion", "antitrust"],
        "DOCS":     ["HSR filing", "LOI", "letter of intent", "escrow agreement", "transition services agreement"],
    },
    ("MA_BUY", "MA_SELL", "DEBT", "VC", "CAPMKT", "RE_ACQ"): {
        "UTBMS_MA": ["closing documents", "closing"],
        "DOCS":     ["closing checklist", "signature pages", "funds flow", "closing set"],
    },
    ("VC", "FUND_FORM"): {
        "DOCS": ["side letter"],
    },
}


def cue_table():
    """One row per (cue, matter type) with its source. A cue can point to several types."""
    rows = []
    for mtype, by_source in TASK_CUES.items():
        rows += [(cue, mtype, src) for src, cues in by_source.items() for cue in cues]
    for mtypes, by_source in SHARED_CUES.items():
        rows += [(cue, mtype, src) for src, cues in by_source.items() for cue in cues for mtype in mtypes]
    table = pd.DataFrame(rows, columns=["cue", "matter_type", "source"]).drop_duplicates(["cue", "matter_type"])
    table["cue_norm"] = table["cue"].map(lambda c: normalize(c, drop_suffixes=False))
    return table.reset_index(drop=True)


def check_cue_types(cues, matter_types):
    """Stop if a matter type has no cues, or a cue points to a type that isn't in Matter Master."""
    listed, used = set(matter_types), set(cues["matter_type"])
    if listed - used:
        raise ValueError(f"Matter types with no task cues: {sorted(listed - used)}")
    if used - listed:
        raise ValueError(f"Cue types not in Matter Master: {sorted(used - listed)}")
