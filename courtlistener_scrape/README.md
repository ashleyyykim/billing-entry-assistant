# CourtListener Reference Data

Builds reference data from public bankruptcy fee filings on CourtListener:

- **Time entries:** real attorney billing narratives, used to ground synthetic data generation.
- **Fee examiner reports:** what examiners flag and why, used to define compliance rules.

## Contents

| File | What it is |
|---|---|
| `courtlistener_reference_data.ipynb` | Scrapes CourtListener and builds the files in `data/final/` |
| `courtlistener_eda.ipynb` | Checks data quality and what the data means for rules and synthetic data |
| `data/final/time_entries_clean_balanced.csv` | Clean attorney time entries, capped per case |
| `data/final/examiner_paragraphs.csv` | Examiner report paragraphs tagged by violation category |

## Running it

1. Activate the repo environment: `conda activate tenth`
2. Add `COURTLISTENER_TOKEN=your_token_here` to a `.env` file at the repo root.
3. Set `RUN_SCRAPE = True` in the reference data notebook and run all cells.

A full scrape takes a while because of API rate limits. If you already have `data/raw/raw_scrape.json.gz`, skip steps 2 and 3 and run all cells with the default `RUN_SCRAPE = False`.

## Sources

- [CourtListener RECAP Archive](https://www.courtlistener.com/recap/), Free Law Project
- [U.S. Trustee Appendix B Guidelines](https://www.federalregister.gov/documents/2013/06/17/2013-14323/appendix-b-guidelines-for-reviewing-applications-for-compensation-and-reimbursement-of-expenses) (2013)