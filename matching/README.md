# Matching

Stage 1 of the billing entry assistant, after extraction. It matches each time entry to the right client and matter, and decides which entries a person needs to check.

| Decision | What happens |
|---|---|
| `AUTO_MATCH` | The matter is filled in with no review |
| `FLAG_MISMATCH` | The typed CMID and the narrative disagree, so a person checks the suggested matter |
| `NEEDS_REVIEW` | A person picks from a shortlist of likely matters |
| `MULTI_MATTER` | The entry seems to cover more than one matter |
| `NO_MATCH` | Likely a new client or matter, sent to intake |

## How it works

1. **Clues.** Every entry is paired with every matter, and each pair gets clues: the typed CMID, client, and matter, the narrative, the work date, the timekeeper's billing history, and how much the narrative reads like past entries on that matter.
2. **Score every pair.** A LightGBM model gives each pair the probability that it's the right matter.
3. **Decide.** A second model looks at the whole entry and gives the chance the top pick is safe to accept, needs a person, covers two matters, or is a new client. The entry's hours and narrative length help it spot entries that cover two matters, since those tend to be long entries with several tasks.
4. **Cutoffs.** The firm picks a precision target, and the cutoffs are set to meet it.

## Presets

The firm sets one preset when the service starts (the `SETTING` environment variable, `default` if unset). The page can't change it, and `/health` shows which one is running.

| Preset | Target | Matched with no person | Wrong and unseen, per 100 entries |
|---|---|---|---|
| `strict` | 98% of automatic calls right | about 55% | about 0.5 |
| `default` | 97% | about 65% | about 1.5 |
| `balanced` | as accurate as the hand-written rules (86%) | about 81% | about 6 |

VAL and TEST together, from the November 20, 2026 build. `02_evaluation` has every set, including the holdout file.

Correcting a typed CMID has its own, higher bar, since it overrules what someone typed. Below it, the entry is flagged instead.

To set a different target without retraining:

```python
from train import repick_cutoffs
repick_cutoffs("results", {"careful": 0.985})   # then start the service with SETTING=careful
```

## Notebooks

| Notebook | What it does |
|---|---|
| `00_build` | Builds everything the app loads: email fields, clues, both models, and each preset's cutoffs |
| `01_llm_answers` | Asks an LLM (Bedrock) to match every VAL entry and saves its answers, for comparison only (needs AWS) |
| `02_evaluation` | Grades the system on VAL, TEST, and the holdout file, and compares it with the rules and the LLM |

Run `00_build`, then `02_evaluation`. Only `01_llm_answers` pays for calls, and it saves every answer, so the others run without AWS. The first run downloads the sentence embedding model (about 90 MB).

## Folders

| Folder | What it is |
|---|---|
| `notebooks/` | The notebooks above |
| `src/` | Shared code for the notebooks and the app |
| `data/` | Synthetic time entries, Matter Master, timekeeper history, a holdout file, a billing shorthand list, and the LLM's saved answers (`llm_answers/`) |
| `results/` | What `00_build` saves for the app, and `02_evaluation`'s report |

Key files in `src/`:
- `pipeline.py`: runs the saved system on new entries (what the app uses)
- `train.py`: builds and trains everything, so a firm can retrain on its own history
- `features.py`, `classifier.py`, `rules.py`, `evaluate.py`: clues, models, the rules baseline, and grading

## Using it in the app

| When | Function in `pipeline.py` | What it does |
|---|---|---|
| Startup | `load_reference` | Loads Matter Master, billing history, models, and cutoffs |
| Each batch | `decide` | A decision, suggested matter, shortlist, reasons, and `p_auto` (how sure) for each entry |
| After review | `add_confirmed` | Adds confirmed entries to billing history, so they help right away |

The app loads `02_config.json`, `02_pairs.parquet`, `02_people.parquet`, `04_config.json`, `04_model.joblib`, and `00_entries.parquet` from `results/`.

The same service also runs the narrative check (stage 2, `POST /review`), from `../phase2_model/` (see its README). Without those models, matching still works and `/review` returns 503. The Docker files are in `../docker/`.

## Sample files

The page's "Try a sample" files (`data/samples/`) are one demo firm's notes, built from synthetic VAL and TEST entries whose decisions are known:

| File | Whose | What it shows |
|---|---|---|
| `notebook.jpg` | C. Dunn | Handwritten page; one entry's hours written without the decimal |
| `sticky-note.jpg` | B. Quinn | A typed CMID for a matter this attorney doesn't bill |
| `napkin.jpg` | F. Castillo | A day jotted at a coffee meeting, including the coffee, which goes to the firm's business development matter |
| `typed-note.txt` | B. Quinn | A client that isn't on the firm's list |
| `excel-sheet.xlsx` | Several | A billing export: reviews to accept in bulk, a close call, a closed matter, two matters in one entry, a new client |
| `csv-export.csv` | C. Dunn | A timer app's export in minutes |
| `word-log.docx` | I. Novak | A two-week log with one review |
| `pdf-report.pdf` | B. Quinn | A billing system's report with a typed CMID Tenth corrects |
| `forwarded-email.eml` | I. Novak | An email with a timesheet attached, and one entry sent in both |
| `voice-memo.m4a` | H. Sato | A partner's voice memo: one entry says "same client", and one is a new client. It shows only where `TENTH_S3_BUCKET` is set, since it needs Amazon Transcribe |

81 of the 96 entries (84%) are matched with no person, more than the 65% in `02_evaluation`, since the samples are a demo. Vague narratives ("flag open issues"), which the page's narrative review (`phase2_model`) asks about, were swapped out of the typed files, except two in the email and where a sample shows that check on purpose. Block entries (several tasks under one time) keep their narratives as the data has them: the page asks about them in one bulk card. `data/sample_answer_key.csv` lists each entry and what Tenth should do with it, and `02_evaluation`'s appendix checks matching on them.

## Limitations

- The data is synthetic, since firms can't share real time entries. A firm would retrain on its own history.
- Every entry is compared with every matter, which works for about a hundred matters, not thousands.
- A typed CMID that's wrong but fits the narrative can't be caught, so a firm should spot-check some automatic matches.
- Multi-matter entries are the weakest area. When the second matter is named only by a word that's in neither its matter name nor much of its billing history, the entry is usually matched to its first matter alone. Hours and narrative length catch more of them here, but the synthetic multi-matter entries are long by design, so a real firm's long single-matter entries would mean more false alarms.
- People are matched in lowercase, so "call with banks" counts as a person named Banks.
- A CMID written in a narrative counts if it starts with a client number, so "invoice 3120-445" can be read as one.
