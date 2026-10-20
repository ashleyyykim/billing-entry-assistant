# Matching

Matches each time entry to the right client and matter. Entries come in as typed spreadsheets, PDFs, handwritten sheets, and emails, often with missing or wrong fields. Each one gets one of five decisions:

| Decision | What happens |
|---|---|
| `AUTO_MATCH` | The matter is filled in with no review |
| `FLAG_MISMATCH` | The typed fields and the narrative disagree, so a person checks the suggested matter |
| `NEEDS_REVIEW` | A person picks from a shortlist of likely matters |
| `MULTI_MATTER` | The entry seems to cover more than one matter, with a suggested split |
| `NO_MATCH` | Likely a new client or matter, sent to intake |

Every decision comes with a short reason for the review list ("Typed CMID matches") and a full one with the clues behind it.

## Approach

| Notebook | What it does |
|---|---|
| `00_email_parsing` | Pulls the CMID, client, matter, and hours out of email bodies so emails have the same fields as other formats |
| `01_fuzzy_scorer_comparison` | Picks the fuzzy matching scorers for typed client and matter names |
| `02_feature_building` | Pairs every entry with every matter and describes each pair with clues: the typed fields, the narrative, the work date, the timekeeper's billing history, and how closely the narrative reads like past entries on that matter (sentence embeddings). Also splits each narrative into its pieces of work. |
| `03_baseline_rules` | Hand-written rules that combine the clues, compared with a baseline that trusts any valid typed CMID |
| `04_classifier` | A gradient boosting model scores every pair, and a second model decides the action for the whole entry. Cutoffs are learned on TRAIN and compared with the rules on the same scorecard. |

## Contents

| Folder | What it is |
|---|---|
| `notebooks/` | The notebooks above, run in order |
| `src/` | Shared code: `email_parser.py` (email entries), `clean.py` (typed fields), `fuzzy.py` (name matching), `cues.py` (task cues), `features.py` (the pairs table), `rules.py` (rule baseline and scoring), `classifier.py` (decisions and reasons) |
| `data/` | Synthetic time entries with Matter Master and timekeeper history, a holdout file with new phrasings and a timekeeper with no history, and example emails in common layouts for checking the email parser |
| `results/` | What each notebook saves for the next one (pairs table, settings, decisions, trained models) |

## Running it

1. Set up and activate the repo environment (see the main README).
2. Run the notebooks in order. Each one reads what the previous one saved in `results/`.