# Billing Entry Assistant

Legal time entries show up in messy formats: spreadsheets, PDFs, emails, even handwritten notes. Billing Entry Assistant matches each entry to the right client and matter, then flags narratives that won't hold up to billing guidelines before they reach a client.

Built as a capstone project for UC Berkeley's Master of Information and Data Science (MIDS) program.

## How it works

1. **Matching:** finds the client and matter each entry belongs to. Confident matches go through on their own, and everything else goes to a person with a shortlist and a short reason. An entry that covers more than one matter is split into one entry per matter before the next step.
2. **Compliance:** checks each narrative against its client's billing guidelines (vague wording, block billing, shorthand) and explains anything it flags, so a person always makes the final call.

## Data

- **Matching:** synthetic time entries modeled on corporate and transactional practice, with the correct matter for each entry. Real client time records are confidential, so this stage uses synthetic data.
- **Compliance:** real time entries and fee examiner reports from public bankruptcy fee filings on [CourtListener](https://www.courtlistener.com/), with violation categories based on the U.S. Trustee fee guidelines.

## Repo layout

| Folder | What it is |
|---|---|
| `matching/` | Matching stage: fuzzy matching, feature building, a rule baseline, and a classifier. See its README. |
| `courtlistener_scrape/` | Reference data for the compliance stage, built from public bankruptcy filings. See its README. |

## Setup

```
conda env create -f environment.yml
conda activate tenth
python -m ipykernel install --user --name tenth
```

Then select the `tenth` kernel in Jupyter or VS Code. After adding a package to `environment.yml`, run `conda env update -f environment.yml --prune`.

## Team

Serina Li, Ashley Kim, Jennifer Nishimura