# CourtListener Reference Data

Builds reference data from public bankruptcy fee filings on CourtListener:

- **Time entries:** real attorney billing narratives, used to ground synthetic data generation.
- **Fee examiner reports:** what examiners flag and why, used to define compliance rules.

All source documents are public court records.

## Outputs

The two CSVs in `data/final/` are ready to use:

| File | Contents |
|---|---|
| `time_entries_clean_balanced.csv` | Clean attorney time entries from bankruptcy cases listed in `cases.csv`, capped per case. `SOURCE:` columns link each entry back to its filing. |
| `examiner_paragraphs.csv` | Fee examiner report paragraphs tagged by violation category, such as vague entries, block billing, and non-working travel. |

The notebook also produces a full audit file of every parsed entry and the full text of each examiner report. Those are too large for GitHub, so they aren't included here, but the notebook rebuilds them.

## Running it

1. Install dependencies: `pip install "pandas>=2.0" numpy requests python-dotenv`
2. Add a CourtListener API token to a `.env` file at the repo root: `COURTLISTENER_TOKEN=your_token_here`
3. In `courtlistener_reference_data.ipynb`, set `RUN_SCRAPE = True` and run all cells.

A full scrape takes a while because of API rate limits. If you already have `data/raw/raw_scrape.json.gz`, skip steps 2 and 3 and run all cells with the default `RUN_SCRAPE = False`.

## How it works

1. Searches each case's docket for fee-related filings and downloads their text.
2. Finds fee examiner reports, rebuilds their paragraphs, and tags each one by violation category.
3. Finds itemized time entries in fee statements, parses them, and keeps entries billed by attorneys.
4. Cleans and balances the entries so no single case dominates.

## Sources

- [CourtListener RECAP Archive](https://www.courtlistener.com/recap/), Free Law Project
- [U.S. Trustee Appendix B Guidelines](https://www.federalregister.gov/documents/2013/06/17/2013-14323/appendix-b-guidelines-for-reviewing-applications-for-compensation-and-reimbursement-of-expenses) (2013)