# Matching

Stage 1 of the billing entry assistant. It matches each time entry to the right client and matter, and decides which entries a person needs to check.

Entries come in as typed spreadsheets, PDFs, handwritten sheets, and emails, often with missing or wrong fields. Each one gets a decision and a short reason:

| Decision | What happens |
|---|---|
| `AUTO_MATCH` | The matter is filled in with no review |
| `FLAG_MISMATCH` | The typed fields and the narrative disagree, so a person checks the suggested matter |
| `NEEDS_REVIEW` | A person picks from a shortlist of likely matters |
| `MULTI_MATTER` | The entry seems to cover more than one matter, with a suggested split |
| `NO_MATCH` | Likely a new client or matter, sent to intake |

## How it works

1. **Clues.** Every entry is paired with every matter, and each pair is described by clues: the typed CMID, client, and matter, the narrative, the work date, the timekeeper's billing history, and how much the narrative reads like past entries on that matter.
2. **Step 1: score every pair.** A gradient boosting model (LightGBM) gives each pair the probability that it's the right matter.
3. **Step 2: decide the action.** A second model looks at the whole entry and decides whether the top pick is safe to accept, needs review, covers two matters, or is a new client.
4. **Cutoffs.** The firm sets a precision target (by default, at least 97% of automatic decisions right), and the cutoffs are picked to meet it.

## Notebooks

Run in order. Each one reads what the previous one saved in `results/`.

| Notebook | What it does |
|---|---|
| `00_email_parsing` | Pulls the CMID, client, matter, and hours out of email bodies |
| `01_fuzzy_scorer_comparison` | Picks the fuzzy matching scorers for typed names |
| `02_feature_building` | Builds the clues for every entry and matter pair |
| `03_baseline_rules` | Hand-written rules, the baseline to beat |
| `04_classifier` | Trains both models and picks the cutoffs |
| `05_evaluation` | Grades the system on VAL and sets the pass/fail gates |
| `06_final_run` | Scores the frozen system once on TEST and the holdout file |
| `07_model_comparison` | Compares other model types, and checks which clues matter |
| `08_review_diagnosis` | Looks at why some right answers still go to review |
| `09_feedback_loop` | Shows how reviewer confirmations improve results without retraining |
| `10_robustness` | Checks real attorney writing, writing styles, and retraining |

Notebooks 07 to 10 are extra analyses. The system itself only needs 00 to 06.

## Folders

| Folder | What it is |
|---|---|
| `notebooks/` | The notebooks above |
| `src/` | Shared code used by the notebooks and the app |
| `data/` | Synthetic time entries, Matter Master, timekeeper history, a holdout file, and a billing shorthand list |
| `results/` | Files each notebook saves for the next one, including the trained models |

Key files in `src/`:
- `pipeline.py`: runs the trained system on new entries (what the app uses)
- `train.py`: retrains the whole system on a firm's own history
- `features.py`, `classifier.py`, `rules.py`, `evaluate.py`: clues, models, scoring, and evaluation

## Running it

1. Set up and activate the repo environment (see the main README).
2. Run the notebooks in order.

The first run of notebook 02 downloads the sentence embedding model (about 90 MB).

## Using it in the app

The app calls three functions from `src/pipeline.py`:

| When | Function | What it does |
|---|---|---|
| Startup | `load_reference` | Loads Matter Master, timekeeper history, and the trained models |
| Each batch of entries | `decide` | Returns a decision, suggested matter, shortlist, and reasons for each entry |
| After review | `add_confirmed` | Adds confirmed entries to billing history, so they help right away |

Retraining runs separately with `train` and `save` in `src/train.py`. The app needs five files from `results/`: `02_config.json`, `02_pairs.parquet`, `02_people.parquet`, `04_config.json`, and `04_model.joblib`.

## Limitations

- The data is synthetic, since firms can't share real time entries. A firm's own entries would need the models retrained on its history.
- Every entry is compared with every matter, which works for about a hundred matters. A firm with thousands would narrow the candidates first.
- A typed CMID that's wrong but fits the narrative can't be caught. A firm should spot-check a sample of automatic matches.
