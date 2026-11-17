# Docker

One image runs the whole app: extraction (`extraction/src`), matching (`matching/`), and the narrative check (`phase2_model/`), behind one web page.

| File | What it is |
|---|---|
| `Dockerfile` | Builds the image. The build context is the repo root. |
| `Dockerfile.dockerignore` | What may go into the image. Everything else, including the holdout file, stays out. |
| `compose.yaml` | Builds and runs it with one command. |

## Run it

From the repo root:

```
docker compose -f docker/compose.yaml up --build
```

Then open http://localhost:8080. The first start takes a minute or two while matching loads its history.

Without Compose:

```
docker build -f docker/Dockerfile -t tenth .
docker run -p 8080:8080 tenth
```

Set `TENTH_PASSWORD` to put a sign-in prompt in front of the page and the API. Leave it empty only on your own machine.

## Before you build

The image copies the trained models, so they must exist first:

- **Matching:** `matching/results/` (from the matching notebooks; already in git).
- **Narrative check:** `phase2_model/models/`. Run `phase2_model/training/train_classifier.py` (seconds). `train_rewriter.py` is optional (about 45 minutes on CPU); without it, rewrites come from rules. See `phase2_model/README.md`.

`GET /health` shows which narrative-check rewriter loaded: `flan-t5 (fine-tuned)`, `rules`, or `off` with the reason.

## Extraction and AWS

`POST /extract` reads one upload (a photo, PDF, Word or text file, spreadsheet, `.eml` email, or `text` typed in) and returns its entries with their review reasons, plus matching's decisions for them. The page's **Upload a photo or file** button calls it.

It uses Textract for photos and scans, and Nova Pro with Llama 4 Maverick on Bedrock (us-west-2). Credentials come from your shell when the container starts and are never copied into the image:

```
eval "$(aws configure export-credentials --format env)"
docker compose -f docker/compose.yaml up --build
```

Without credentials, `/extract` still works with `use_aws=false`: the rule readers read typed inputs and spreadsheets for free, and every entry is flagged `read_by_rules`. Photos and scans need AWS. Each upload is read in a temporary folder that is deleted afterwards, so nothing from it is kept.

`GET /health` shows whether extraction loaded.
