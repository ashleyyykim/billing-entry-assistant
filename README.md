# Tenth

Lawyers write down their time anywhere: handwritten notes, sticky notes, emails, PDFs, spreadsheets, voice memos. Tenth turns those notes into billing entries on the right client and matter, with narratives that hold up when a client or fee examiner reads them. It's named for how lawyers bill, in tenths of an hour.

Built as a capstone project for UC Berkeley's Master of Information and Data Science (MIDS) program.

## How it works

1. **Extraction:** reads any input type into one table of entries (timekeeper, date, client, CMID, matter, narrative, hours), and flags entries whose reading looks wrong.
2. **Matching:** finds the client and matter each entry belongs to. Confident matches go through on their own, and everything else goes to a person with a shortlist and a short reason.
3. **Narrative check:** splits block-billed entries into tasks, flags vague or incomplete ones, and suggests a rewrite, so a person always makes the final call.

## Try it

```
docker compose -f docker/compose.yaml up --build
```

Then open http://localhost:8080. See `docker/README.md` for what has to exist before the build, and the AWS setup extraction needs.

## Data

- **Extraction:** made-up samples of every input type, plus a few real handwritten notes.
- **Matching:** synthetic time entries modeled on a corporate and transactional practice, with the correct matter for each entry. Real client time records are confidential, so this stage uses synthetic data. `time_entries/` has the generator and its evaluation.
- **Narrative check:** real time entries and fee examiner reports from public bankruptcy fee filings on [CourtListener](https://www.courtlistener.com/).

## Repo layout

| Folder | What it is |
|---|---|
| `extraction/` | Stage 0, reading inputs into entries. See its README. |
| `matching/` | Stage 1, matching entries to matters. Also the web app (`app.py`, `frontend/`). See its README. |
| `phase2_model/` | Stage 2, the narrative check. See its README. |
| `docker/` | Builds and runs the whole app in one container. See its README. |
| `time_entries/` | The synthetic data generator and its evaluation |
| `courtlistener_scrape/` | Real time entries and fee examiner reports from public bankruptcy filings. See its README. |
| `part_2_clean_data/` | Cleans the real entries and draws a hand-labeled sample for the narrative check. See its README. |
| `nlp_model/`, `ocr/`, `shorthand_bank/` | Early experiments |

## Setup

```
conda env create -f environment.yml
conda activate tenth
python -m ipykernel install --user --name tenth
```

Then select the `tenth` kernel in Jupyter or VS Code. After adding a package to `environment.yml`, run `conda env update -f environment.yml --prune`.

## Team

Ashley Kim, Serina Li, Jennifer Nishimura
