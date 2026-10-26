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
| `05_evaluation` | Grades the system stage by stage on VAL: slices, error analysis, checks on every decision, stress tests, and gates the final run has to pass |
| `06_final_run` | Runs the frozen system once on TEST and the holdout file, with the same metrics, checks, stress tests, and gates as notebook 05. Nothing changes after it runs. |

## Contents

| Folder | What it is |
|---|---|
| `notebooks/` | The notebooks above, run in order |
| `src/` | Shared code: `email_parser.py` (email entries), `clean.py` (typed fields), `fuzzy.py` (name matching), `cues.py` (task cues), `features.py` (the pairs table), `rules.py` (rule baseline and scoring), `classifier.py` (decisions and reasons), `pipeline.py` (reads a time-entry file and runs the saved models on it), `evaluate.py` (metrics, checks, stress tests, and the gates) |
| `data/` | Synthetic time entries with Matter Master and timekeeper history, a holdout file with new phrasings and a timekeeper with no history, example emails in common layouts for checking the email parser, and a billing shorthand list (`shorthand_bank.csv`, so shorthand like "SPA" is never read as client initials) |
| `results/` | What each notebook saves for the next one (pairs table, settings, decisions, trained models) |

## Running it

1. Set up and activate the repo environment (see the main README).
2. Run the notebooks in order. Each one reads what the previous one saved in `results/`.

## Model card

**Intended use:** suggest the client and matter for each time entry and decide which entries a person needs to see. It supports billing staff and doesn't replace their review of flagged, multi-matter, and uncertain entries.

**Data:** synthetic time entries for a corporate and transactional practice, since firms can't share real entries for privacy reasons. No real clients or people. TRAIN tunes everything, VAL compares versions, and TEST and the holdout file are scored once at the end.

**How decisions are made:** a gradient boosting model scores every entry and matter pair, and a second model decides the action. The cutoffs are chosen on TRAIN from a precision target the firm sets (default: at least 97% of automatic decisions right). With `CORRECT_TYPED_CMID` on, a typed CMID the model is confident is wrong is corrected and the reason says so; off, it's flagged instead.

**How it's checked:** stage metrics, slices by format and entry type, checks on every decision, stress tests, and fixed gates (`05_evaluation`), with the numbers saved in `results/05_eval_report.json`.

**Known limits:**
- The models learn the synthetic data's patterns, so a firm's own entries would likely score lower until the models and cutoffs are refit on that firm's history.
- Silent errors (wrong auto-matches) are rare but possible, mostly wrong CMIDs that nothing in the entry contradicts. A firm should spot-check a random sample of auto-matches.
- Client initials are only recognized when they belong to one client and are at least three letters. Nicknames need the firm's alias list.
- Every entry is compared with every matter, which is fine for about a hundred matters. A firm with thousands would narrow the candidates first (by client and timekeeper history).
