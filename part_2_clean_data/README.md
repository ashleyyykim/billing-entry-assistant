# Part 2: Clean Data

Cleans real attorney time entries from CourtListener and builds a hand-labeled benchmark for the compliance stage.

## Contents

| File | What it is |
|---|---|
| `01_label_sample.ipynb` | Cleans `time_entries_all.csv.gz` and draws the label sample |
| `data/final/label_sample.csv` | 240 entries (20 per case, spread across firms) to hand-label |

## Running it

1. Download `time_entries_all.csv.gz` from the team Drive into `courtlistener_scrape/data/raw/`.
2. Open `01_label_sample.ipynb` and run all cells (about a minute).

## What the cleaning does

- Keeps attorney entries the parser was confident about (236k of 506k rows).
- Strips scrape artifacts that aren't attorney writing: matter-name prefixes, PG&E matter codes (`WILD`, `NONB`, ...), the Sears page header, task codes, page footers.
- Drops expenses, narratives cut off mid-sentence, and exact duplicates within a case.
- Keeps short and ALL CAPS entries, since that's real attorney writing.

## Labeling

Fill in two columns in `label_sample.csv`:

- `quality_label`: `GOOD`, `SPECIFIC`, `VAGUE`, `MISSING_SUBJECT`, `DENSE_SHORTHAND`, `CLERICAL`, or `JUNK` (parse error; excluded from the benchmark). Same labels as the synthetic v3 data.
- `block_billed`: `Y` or `N`. This is a separate flag, since block billing is allowed or not depending on the firm and client.

## Why a hand-labeled sample

Fee examiner reports only list individual flagged entries for one case (General Motors, about 210 entries). Every other case reports totals, so the examiner text can't label these entries directly.
