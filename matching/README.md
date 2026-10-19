# Matching

Matches each time entry to the right client and matter. Entries come in as typed spreadsheets, PDFs, handwritten sheets, and emails, often with missing or wrong fields. Each one is either matched automatically, flagged for a mismatch, split across two matters, sent to intake as a new client or matter, or sent to a person with a shortlist of likely matters.

## Approach

1. **Fuzzy matching:** compares typed client and matter names against Matter Master to handle typos, abbreviations, and formatting differences.
2. **Feature building:** pairs every entry with every matter and describes each pair with clues from the typed fields, the narrative, the work date, and the timekeeper's billing history.
3. **Rule baseline:** combines those clues to make a decision for each entry, compared against how billing staff handle entries today.

## Contents

| Folder | What it is |
|---|---|
| `notebooks/` | One notebook per step above, run in order |
| `src/` | Shared code used by the notebooks |
| `data/` | Synthetic time entries, Matter Master, and timekeeper history |
| `results/` | Outputs each notebook saves for the next one |

## Data

Synthetic time entries modeled on corporate and transactional practice, with the correct matter for each entry. Methods are tuned on a training split, checked on a validation split, and compared once on a held-out test set.

## Running it

1. Activate the repo environment: `conda activate tenth`
2. Run the notebooks in order.
