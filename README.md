# Billing Entry Assistant

Legal time entries show up in messy formats: spreadsheets, PDFs, emails, even handwritten notes. Billing Entry Assistant matches each entry to the right client and matter, then flags narratives that won't hold up to billing guidelines before they reach a client.

Built as a capstone project for UC Berkeley's Master of Information and Data Science (MIDS) program.

## How it works

1. **Extraction and matching:** reads raw time entries in different formats and identifies the client and matter each one belongs to.
2. **Compliance detection:** checks each narrative against billing guidelines and explains anything it flags, so a person always makes the final call.

## Data

- **Compliance:** real time entries and fee examiner reports from public bankruptcy fee filings on [CourtListener](https://www.courtlistener.com/), with violation categories based on the U.S. Trustee fee guidelines.
- **Matching:** synthetic time entries modeled on corporate and transactional practice, with known correct matters for evaluation. Real client time records are confidential, so this stage uses synthetic data.

## Repo layout

- `courtlistener_scrape/`: builds reference data from public bankruptcy filings. See its README for details.

## Team

Serina Li, Ashley Kim, Jennifer Nishimura