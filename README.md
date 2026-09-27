# Billing Entry Assistant

A two-stage pipeline that automates legal time-entry review: matching raw, multi-format time entries to the correct client/matter, then flagging narratives that fail billing guideline standards (vagueness, block billing) before invoice submission.

Capstone project for UC Berkeley MIDS DATASCI 210.

## Team

Serina Li, Ashley Kim, Jennifer Nishimura

## Pipeline

1. **Extraction & Matching** — ingest raw time entries (Excel, PDF, email, handwritten notes) and identify the correct client/matter
2. **Compliance Detection** — flag narratives against billing guideline standards, with explanations, for human review before submission

## Data

- **Compliance stage**: real, itemized time entries and fee examiner findings pulled from CourtListener/RECAP bankruptcy fee filings, evaluated against U.S. Trustee Fee Guidelines
- **Extraction/matching stage**: synthetic time entries (starting with Excel format) built to mirror corporate/transactional practice, with known ground-truth matter assignments

## Repo structure

```
courtlistener_scrape/
  courtlistener_fee_scraper.ipynb  # scraper notebook
  cases.csv                        # case list
  data/
    raw/                           # raw scraped filings
    intermediate/                  # parsed, pre-cleaning
    final/                         # cleaned time entries + fee examiner reports
```
