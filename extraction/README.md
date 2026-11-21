# Extraction

The first step of the billing entry assistant, before matching. It turns time notes in any format into one table, with a row per entry: timekeeper, work date, client, CMID, matter, narrative, and hours.

Lawyers record time anywhere. Each input type is turned into text first:

| Input | How the text is read |
|---|---|
| Note photos, scanned time sheets, screenshots | Textract, and the models also see the image |
| Emails, typed notes, texts | The text itself; each attachment is read as its own input |
| PDFs and Word files | The document's text, with table cells kept apart |
| Spreadsheets | The models say what each column holds; code reads the rows |
| Voice memos | Transcribe |

## How it works

1. **Photo check.** A blurry, dark, or small photo gets a retake prompt before anything is read.
2. **Two models read it.** Nova Pro and Qwen3 VL on Bedrock each fill the same entry fields and cite the lines each entry came from. Long inputs are read in pieces.
3. **Clean-up.** Code works out dates like "Tues" or "yesterday", fixes CMID formats, keeps matter names apart from CMIDs, and converts units.
4. **Checks.** An entry is flagged when its hours, client, CMID, date, or description look wrong or missing, or the two models read it differently.
5. **Review or matching.** Flagged entries go to a person, and the rest go to matching. If a model call fails, the rule readers read the upload instead and a person checks it.

## Notebooks

| Notebook | What it does |
|---|---|
| `00_textract` | Runs Textract on the samples and saves its answers (needs AWS) |
| `01_evaluation` | Reads every sample set with the flow and grades it against the answer keys, from saved answers |

## Folders

| Folder | What it is |
|---|---|
| `notebooks/` | The notebooks above |
| `src/` | The flow, model calls, checks, rule readers, and grading |
| `data/` | Made-up samples, answer keys, and saved Textract, Transcribe, and model answers |
| `tests/cases/` | Typed notes and stress cases the evaluation reads |

Key files in `src/`:
- `one_flow.py`: runs the flow on one upload (what the app uses)
- `prompts_v6.py`, `prompts.py`, `model_readers.py`, `open_models.py`, `column_map.py`: what the models are told, and the calls to Bedrock
- `intake.py`, `review.py`, `upload_check.py`: the entry table, the checks, and the photo check
- `extract.py`, `free_text.py`, `tables.py`, `documents.py`, `spreadsheets.py`: the rule readers, used when a model call fails
- `textract.py`, `messages.py`, `voice.py`: Textract, emails with attachments, and Transcribe

## Running it

1. Set up and activate the repo environment (see the main README).
2. Run `01_evaluation`. Every model answer is saved, so it makes no AWS calls. Section 8 downloads matching's sentence embedding model on its first run.

Photos and voice memos aren't in the repository, since photos carry location data and memos are people's voices. The notebook skips the sets that need them.

## Using it in the app

| When | Function | What it does |
|---|---|---|
| Startup | `ModelReader(**firm_lookup(matter_master), ...)` | Sets up the models with the firm's clients and CMIDs |
| A photo arrives | `check_photo` | Says whether to ask for a retake |
| Each upload | `read` | Returns the entries, each with its check reasons |
| An email with attachments | `message_entries` | Reads the body and each attachment, credited to the writer |
| A voice memo | `transcribe`, `recorded_on` | The transcript, and the day it was recorded |
| Before matching | `add_review` | Marks which entries go to a person |

## Limitations

- The samples are made up and the flow was built on them, so real entries will do worse.
- Long documents, a real firm's client list, real audio, and natural handwriting aren't tested.
- Timekeepers aren't graded.
