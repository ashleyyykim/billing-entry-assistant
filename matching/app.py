"""Tenth matching service.

Wraps src/pipeline.py's decide() behind a small HTTP API, the same call
notebook 06 uses. Everything heavy is loaded once at startup: the embedding
model and the reference data (Matter Master, billing history, models, cutoffs)
via load_reference().

Stage 2, narrative review (/review), runs review.Reviewer from phase2_model/pipeline with the models in
phase2_model/models (in the image: /app/phase2, see docker/Dockerfile). It is optional: if they're missing,
matching still works and /review returns 503.

Stage 1, extraction (/extract), reads one upload (a photo, PDF, Word file, spreadsheet, email, or typed note) into
entries with extraction/src/one_flow.read, then sends them through the same decide() as /decide. It calls Textract
and Bedrock with AWS credentials from the environment at run time, never from the image. It is optional too: if
extraction/src is missing, matching still works and /extract returns 503.
"""
import base64
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import secrets
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from pipeline import decide, load_entries, load_reference  # /app/src is on PYTHONPATH

# Stage 2, narrative review: code from phase2_model/pipeline, models from phase2_model/models.
# In the image, docker/Dockerfile puts both in /app/phase2 (PHASE2_DIR). Run from matching/ without Docker,
# it reads them where they live in the repo.
_REPO_PHASE2 = Path(__file__).resolve().parent.parent / "phase2_model"
PHASE2_DIR = Path(os.getenv("PHASE2_DIR", _REPO_PHASE2 / "pipeline"))
PHASE2_MODELS = Path(os.getenv("PHASE2_MODELS", PHASE2_DIR / "models" if (PHASE2_DIR / "models").is_dir() else _REPO_PHASE2 / "models"))
sys.path.insert(0, str(PHASE2_DIR))

# Stage 1, extraction: code from extraction/src, and the firm's timekeeper list (made up, like the rest of the data).
# In the image, docker/Dockerfile puts the code in /app/extraction (EXTRACTION_DIR) and the list in data/.
# Run from matching/ without Docker, it reads them where they live in the repo.
_REPO_EXTRACTION = Path(__file__).resolve().parent.parent / "extraction"
EXTRACTION_DIR = Path(os.getenv("EXTRACTION_DIR", _REPO_EXTRACTION / "src"))
TIMEKEEPERS_PATH = Path(os.getenv("TIMEKEEPERS_PATH", _REPO_EXTRACTION / "data" / "timekeepers.csv"))
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")   # Textract and Bedrock
# Testing only: a folder where each upload's Textract and model answers are kept, keyed by the file's contents, so
# uploading the same file again costs nothing. Unset (the default), every upload is read in a temporary folder that is
# deleted after, and nothing from it is kept.
DEV_CACHE = os.getenv("TENTH_DEV_CACHE", "")
sys.path.append(str(EXTRACTION_DIR))   # last, so phase2's review.py keeps its name (see extraction_module)
log = logging.getLogger("tenth")

DATA_PATH = Path(os.getenv("DATA_PATH", "data/synthetic_time_entries_v3.xlsx"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
SETTING = os.getenv("SETTING", "default")
SKIP_CLIENTS = [c for c in os.getenv("SKIP_CLIENTS", "Firm").split(",") if c]
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "frontend"))
# Set TENTH_PASSWORD (App Runner: from Secrets Manager) to put a sign-in prompt in front of the page and the API.
# The browser asks once and remembers it. Left empty, the service is open: fine on your laptop, not on a public URL.
PASSWORD = os.getenv("TENTH_PASSWORD", "")

state = {}


def sample_entries() -> pd.DataFrame:
    """Entries from the baked-in data file. Gives /schema its column list and /selftest its input."""
    if "sample" not in state:
        entries, _ = load_entries(DATA_PATH, state["ref"], SKIP_CLIENTS)
        state["sample"] = entries
    return state["sample"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    feature_config = json.loads((RESULTS_DIR / "02_config.json").read_text())
    encoder = SentenceTransformer(feature_config["embedding_model"])
    state["ref"] = load_reference(DATA_PATH, RESULTS_DIR, encoder, SETTING)
    sample_entries()
    load_reviewer()
    load_extractor()
    yield
    state.clear()


def load_reviewer():
    """Narrative review is optional: if phase2/ is missing or broken, matching still runs and /review returns 503."""
    state["reviewer"], state["reviewer_error"] = None, ""
    try:
        from review import Reviewer
        state["reviewer"] = Reviewer(str(PHASE2_MODELS))
    except Exception as err:  # noqa: BLE001 - report any load failure on /health instead of crashing the service
        state["reviewer_error"] = f"{type(err).__name__}: {err}"
        log.warning("narrative review not loaded: %s", state["reviewer_error"])
    # Client and matter names by matter number, so the page can title an entry by its matter.
    m = state["ref"]["matters"]
    state["matter_names"] = {str(c): (cl, mt) for c, cl, mt in zip(m["cmid"], m["client"], m["matter"])}
    # Billing history by matter, for placeholder candidates: TRAIN entries only, as load_reference uses.
    try:
        past = pd.read_parquet(RESULTS_DIR / "00_entries.parquet", columns=["entry_id", "tk_id", "work_date", "cmid", "narrative"])
        train_ids = pd.read_parquet(RESULTS_DIR / "02_pairs.parquet", columns=["entry_id", "split"]).query("split == 'TRAIN'").entry_id
        past = past[past.entry_id.isin(set(train_ids)) & past.narrative.notna()]
        past["work_date"] = pd.to_datetime(past.work_date, errors="coerce")
        state["history"] = {c: g.sort_values("work_date", ascending=False) for c, g in past.groupby("cmid")}
    except Exception as err:  # noqa: BLE001
        state["history"] = {}
        log.warning("no billing history for narrative review: %s", err)


def extraction_module(name):
    """An extraction module loaded from its file under another name, for a name phase2 also uses (review.py)."""
    spec = importlib.util.spec_from_file_location(f"extraction_{name}", EXTRACTION_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_extractor():
    """Extraction is optional too: if extraction/src or the timekeeper list is missing, matching still runs and
    /extract returns 503. The Bedrock client needs no credentials until the first model call."""
    state["extractor"], state["extractor_error"] = None, ""
    try:
        import boto3
        from botocore.config import Config
        from intake import timekeeper_table
        from one_flow import firm_lookup
        matter_master = pd.read_excel(DATA_PATH, sheet_name="Matter Master", dtype=str)
        # A page read with its image can take over a minute, so a call may wait 5 minutes (as in extraction notebook 01)
        runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(
            connect_timeout=20, read_timeout=300, retries={"max_attempts": 4, "mode": "adaptive"}))
        state["extractor"] = {"firm": firm_lookup(matter_master), "runtime": runtime,
                              "timekeepers": timekeeper_table(pd.read_csv(TIMEKEEPERS_PATH)),
                              "review": extraction_module("review")}
    except Exception as err:  # noqa: BLE001 - report any load failure on /health instead of crashing the service
        state["extractor_error"] = f"{type(err).__name__}: {err}"
        log.warning("extraction not loaded: %s", state["extractor_error"])


app = FastAPI(title="Tenth", lifespan=lifespan)


@app.middleware("http")
async def sign_in(request: Request, call_next):
    """HTTP Basic sign-in for the page and the API. /health stays open so App Runner can check it."""
    if not PASSWORD or request.url.path == "/health":
        return await call_next(request)
    header = request.headers.get("authorization", "")
    ok = False
    if header.lower().startswith("basic "):
        try:
            _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            ok = secrets.compare_digest(given.encode(), PASSWORD.encode())
        except ValueError:
            ok = False
    if not ok:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Tenth", charset="UTF-8"'})
    return await call_next(request)


@app.middleware("http")
async def fresh_page(request: Request, call_next):
    """The page is one file that changes with each release, so browsers check for a new copy instead of showing a stale one."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Batch(BaseModel):
    entries: list[dict]


def run(entries: pd.DataFrame) -> list[dict]:
    """Decide, then join Matter Master fields (client, matter name, ...) onto the chosen CMID."""
    ref = state["ref"]
    _, _, decisions = decide(entries, ref)
    decisions = decisions.drop(columns=[c for c in decisions.columns if c.startswith("key_")])
    matters = ref["matters"].add_prefix("matter_")
    decisions = decisions.merge(matters, left_on="pred_cmid", right_on="matter_cmid", how="left", suffixes=("", "_master"))
    # Echo the entry's own fields, so the page can show the narrative and send it on to /review
    echo = [c for c in ("narrative", "tk_id", "work_date", "hours", "parsed_hours") if c in entries and c not in decisions]
    if echo and "entry_id" in entries and "entry_id" in decisions:
        decisions = decisions.merge(entries[["entry_id", *echo]].drop_duplicates("entry_id"), on="entry_id", how="left")
    return json.loads(decisions.to_json(orient="records", date_format="iso"))


@app.get("/health")
def health():
    r = state.get("reviewer")
    return {"status": "ok", "matters": len(state["ref"]["matters"]),
            "narrative_review": r.rewriter_name if r else f"off ({state.get('reviewer_error') or 'not loaded'})",
            "extraction": "Nova Pro, Llama 4 Maverick" if state.get("extractor")
                          else f"off ({state.get('extractor_error') or 'not loaded'})"}


@app.get("/schema")
def schema():
    """The fields an entry needs, taken from what load_entries() produces. Map OCR rows onto these."""
    sample = sample_entries()
    return {"columns": {c: str(t) for c, t in sample.dtypes.items()},
            "example": json.loads(sample.head(1).to_json(orient="records", date_format="iso"))[0]}


def as_schema(entries: pd.DataFrame) -> pd.DataFrame:
    """Dates and numbers in the types load_entries() gives them, so decide() reads them the same way."""
    for col, dtype in sample_entries().dtypes.items():
        if col not in entries:
            continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            entries[col] = pd.to_datetime(entries[col])
        elif pd.api.types.is_numeric_dtype(dtype):
            entries[col] = pd.to_numeric(entries[col])
    return entries


@app.post("/decide")
def decide_entries(batch: Batch):
    if not batch.entries:
        raise HTTPException(422, "no entries")
    entries = as_schema(pd.DataFrame(batch.entries))
    try:
        return {"decisions": run(entries)}
    except KeyError as err:
        raise HTTPException(422, f"missing or misnamed field {err}; see /schema")


@app.get("/timekeepers")
def timekeepers():
    """The firm's timekeepers, for the page's "Whose time?" picker on time note uploads."""
    x = state.get("extractor")
    if not x:
        raise HTTPException(503, f"extraction is not loaded: {state.get('extractor_error') or 'unknown'}")
    t = x["timekeepers"]
    return {"timekeepers": [{"tk_id": i, "name": n} for i, n in zip(t["tk_id"], t["name"])]}


# What Tenth noticed about an entry (shown under Why?) and how to fix it where the page has a way (shown in amber).
# intake.CHECK_REASONS says what the system did; these say it for the person reviewing. Reasons not listed fall back.
FIX_HOURS = "You can fix the hours in the next step."
FIX_MATTER = "Use Change matter if it's wrong."
IN_WORDS = {
    "model_unsure": ("Part of the note was unclear.", ""),
    "readers_disagree": ("Two readings of the note didn't agree.", ""),
    "low_conf": ("Part of the note was hard to read.", ""),
    "poor_photo": ("The photo was blurry, dark, or small.", ""),
    "read_by_rules": ("This was read without AI.", ""),
    "second_reader_only": ("Only one of two readings found this entry.", ""),
    "multiple_entries": ("This may be more than one entry.", ""),
    "hours_10x": ("The hours may be missing a decimal point.", FIX_HOURS),
    "hours_mismatch": ("The hours may be misread.", FIX_HOURS),
    "not_on_page": ("The hours weren't where expected on the note.", FIX_HOURS),
    "no_hours": ("No hours were found.", FIX_HOURS),
    "day_over_max": ("This adds up to over 14 hours in one day.", FIX_HOURS),
    "no_date": ("No date was found.", ""),
    "date_gap": ("This entry has no date, though the ones before it do.", ""),
    "date_weekday": ("The weekday doesn't match the date.", ""),
    "client_unknown": ("The client isn't one of the firm's clients.", FIX_MATTER),
    "client_ambiguous": ("The client name could mean more than one client.", FIX_MATTER),
    "client_cmid_mismatch": ("The client and the matter number don't go together.", FIX_MATTER),
    "client_gap": ("No client is written; it may belong to the client above it.", FIX_MATTER),
    "cmid_unknown": ("The matter number written isn't one of the firm's.", FIX_MATTER),
    "cmid_shape": ("The matter number has the wrong number of digits.", FIX_MATTER),
    "unknown_tk": ("No timekeeper was found.", ""),   # the page asks once for the whole upload (missing_tk), not per entry
    "unused_hours": ("The note has hours that no entry uses.", ""),
    "rows_unread": ("The spreadsheet has rows that no entry uses.", ""),
    "answer_cut_off": ("The note was too long to read in one go.", ""),
    "possible_duplicate": ("This may repeat another entry.", ""),
    "not_billable": ("It's marked not billable or no charge.", ""),
}
# What part of the entry each reason is about, for the amber "Check ..." line
SOME_PART = "the hours, date, and client against the note"
CHECK_WHAT = {
    **dict.fromkeys(["hours_10x", "hours_mismatch", "not_on_page", "no_hours", "day_over_max"], "the hours"),
    **dict.fromkeys(["no_date", "date_gap", "date_weekday"], "the date"),
    **dict.fromkeys(["client_unknown", "client_ambiguous", "client_cmid_mismatch", "client_gap", "cmid_unknown", "cmid_shape"],
                    "the matter"),
    **dict.fromkeys(["unused_hours", "rows_unread", "answer_cut_off"], "whether an entry is missing"),
    "unknown_tk": "the timekeeper",
    "second_reader_only": "that this entry is on the note",
    "possible_duplicate": "whether it's a duplicate",
    "not_billable": "whether it's billable",
}


def shown(field, value):
    """A value one reader read, as a person would write it."""
    if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
        return "nothing"
    if field == "hours":
        return f"{float(value):g}"
    if field == "date":
        return pd.Timestamp(value).strftime("%b %-d")
    return str(value)


def differences(a, b):
    """The fields two readings of one entry disagree on, as (field, first reading, second reading), compared the way
    model_readers.cross_check compares them."""
    from grade import same_cmid, similar
    out = []
    if not (pd.notna(a["hours"]) and pd.notna(b["hours"]) and round(a["hours"], 2) == round(b["hours"], 2)):
        out.append(("hours", a["hours"], b["hours"]))
    if not (similar(a["client"], b["client"]) or similar(b["client"], a["client"])):
        out.append(("client", a["client"], b["client"]))
    if not (pd.isna(a["cmid"]) or pd.isna(b["cmid"]) or same_cmid(a["cmid"], b["cmid"])):
        out.append(("matter number", a["cmid"], b["cmid"]))
    if pd.notna(a["work_date"]) and pd.notna(b["work_date"]) and pd.Timestamp(a["work_date"]) != pd.Timestamp(b["work_date"]):
        out.append(("date", a["work_date"], b["work_date"]))
    return out


def what_to_do(reasons, diffs=()):
    """The amber line: what to check ("Check the hours (4 or 0.4) and the timekeeper."), then how to fix it where the
    page has a way. Where the two readings disagreed, it names the fields and both readings."""
    if not reasons:
        return ""
    if "read_by_rules" in reasons:
        return "Check every field against the note."
    disagree = {f: f"the {f} ({shown(f, a)} or {shown(f, b)})" for f, a, b in diffs} if "readers_disagree" in reasons else {}
    parts = list(dict.fromkeys(CHECK_WHAT.get(r, SOME_PART) for r in reasons))
    parts = list(disagree.values()) + [x for x in parts if x.removeprefix("the ") not in disagree]   # the specific ones first
    if len(parts) > 1 and SOME_PART in parts:
        parts.remove(SOME_PART)
    check = "Check " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
    hints = [IN_WORDS[r][1] for r in reasons if r in IN_WORDS and IN_WORDS[r][1]]
    if "hours" in disagree:
        hints.insert(0, FIX_HOURS)
    return " ".join([check] + list(dict.fromkeys(hints))[:2])


def what_was_noticed(reasons, fallback, diffs=()):
    """The Why? text: what Tenth noticed about the entry, one sentence per reason, each said once."""
    said = []
    for r in reasons:
        if r == "readers_disagree" and diffs:
            said.append("Two readings of the note gave different " + " and ".join(
                f"{f} ({shown(f, a)} and {shown(f, b)})" for f, a, b in diffs) + ".")
        else:
            said.append(IN_WORDS.get(r, (fallback.get(r, r) + ".", ""))[0])
    return " ".join(dict.fromkeys(said))


def new_readers(folder: Path, use_aws: bool):
    """Nova Pro and Llama 4 Maverick (the second reader), as extraction notebook 01 sets them up but each on its own, so
    the app can run them side by side (one_flow.read runs the second only after the first) and see where they disagree.
    Their answers and page images go in this request's folder, which is deleted after, so nothing from an upload is kept.
    use_aws=False: no Bedrock calls, so the rule readers read the upload (every entry gets read_by_rules)."""
    from one_flow import MAVERICK, ModelReader
    x = state["extractor"]
    common = dict(**x["firm"], runtime=x["runtime"] if use_aws else None, catch_errors=True)
    first = ModelReader(**common, answers_dir=folder / "first", images_dir=folder / "pages_first")
    second = ModelReader(**common, answers_dir=folder / "second", images_dir=folder / "pages_second", model_id=MAVERICK,
                         force_tool=False, open_note_prompt=True)
    return first, second


def read_both(content, kind, first, second, disagreements, image=None, **kw):
    """one_flow.read with both readers at once, then one_flow.combine the way read does. disagreements gets, for each
    entry the readers read differently, the fields they differ on (differences)."""
    from grade import align
    from one_flow import combine, read
    x = state["extractor"]
    run = lambda model: read(content, kind, state["ref"], x["timekeepers"], model, image=image, **kw)
    with ThreadPoolExecutor(max_workers=2) as pool:
        ones, twos = (job.result() for job in [pool.submit(run, first), pool.submit(run, second)])
    if len(ones) and ones["check_reasons"].str.contains("read_by_rules").all():
        return ones   # read stops here when the first reader has no answer, without asking the second
    fixes = set(first.fixes or ())
    if len(ones) and len(twos):
        for i, j in align(ones, twos):
            if i is not None and j is not None:
                found = differences(ones.iloc[i], twos.iloc[j])
                if found:
                    disagreements[ones.iloc[i]["entry_id"]] = found
    return combine(ones, twos, dates="date_agree" in fixes, note_missing="one_reader" in fixes)


def read_upload(path: Path, input_type: str, reader, folder: Path, common: dict, use_aws: bool, send_anyway: bool,
                timing: dict):
    """(entries, photo check, email parts) for one uploaded file, read the way extraction/README.md lays out: Textract
    for photos and scans, then one_flow.read for every input type. entries is None when a photo should be retaken."""
    from documents import is_scanned
    from messages import NOT_READ, message_entries, read_eml
    from spreadsheets import read_spreadsheet
    from textract import PHOTO_TYPES, load_pages, run_textract
    from upload_check import check_photo

    pages_dir = folder / "textract"

    def textract(mode):
        if not use_aws:
            raise HTTPException(422, "photos and scans need Textract: send them with use_aws=true")
        start = time.perf_counter()
        run_textract(path, pages_dir, mode, AWS_REGION)
        timing["textract_s"] = round(time.perf_counter() - start, 1)
        return load_pages(pages_dir, path, mode)

    suffix = path.suffix.lower()
    if suffix in PHOTO_TYPES:
        kind = input_type or ("SCREENSHOT" if suffix == ".png" else "NOTE")   # phone photos are JPEGs, screenshots PNGs
        check = json.loads(json.dumps(check_photo(path), default=float)) if kind in ("NOTE", "SHEET") else None
        if check and not check["ok"] and not send_anyway:
            return None, check, None
        return reader(textract("sheet" if kind == "SHEET" else "note"), kind, image=path, **common), check, None
    if suffix == ".pdf":
        if is_scanned(path):
            return reader(textract("sheet"), "SHEET", image=path, **common), None, None
        return reader(path, "PDF", **common), None, None
    if suffix in (".docx", ".txt"):
        return reader(path, "DOC", **common), None, None
    if suffix in (".xlsx", ".csv"):
        return reader(read_spreadsheet(path), "SPREADSHEET", **common), None, None
    if suffix == ".eml":
        x = state["extractor"]
        entries, parts = message_entries(read_eml(path, folder / "attachments"), state["ref"], x["timekeepers"],
                                         pages_dir, tk_id=common["tk_id"], run_ocr=use_aws, read=reader)
        return entries, None, parts.to_dict(orient="records")
    raise HTTPException(422, f"{path.name}: {NOT_READ.get(suffix, f'{suffix or path.name} files are not read yet')}")


@app.post("/extract")
def extract_upload(file: UploadFile | None = File(None), text: str = Form(""), input_type: str = Form(""),
                   tk_id: str = Form(""), upload_date: str = Form(""), use_aws: bool = Form(True),
                   send_anyway: bool = Form(False)):
    """Stage 1 extraction for one upload, then matching on its entries (the same run() as /decide).

    file: a photo (.jpg, .png), PDF, Word or text file, spreadsheet (.xlsx, .csv), or email (.eml). Or text: a typed
    note or text message. input_type: NOTE, SHEET (a time sheet form), or SCREENSHOT for a photo; left empty, a .png
    is a SCREENSHOT and a .jpg a NOTE. tk_id: whose time it is, when the assistant knows. upload_date: the day it was
    sent (default today), for "yesterday" and entries with no date. use_aws=false: no Textract or Bedrock calls, so the
    rule readers read it (free, for testing; photos and scans need AWS). send_anyway: read a photo that failed the
    photo check instead of asking for a retake.

    Returns photo_check (note photos only), entries (each with review, notes, and check_reasons), parts (emails only:
    how each part was read), and decisions, each with extraction_review and extraction_reasons in words."""
    x = state.get("extractor")
    if not x:
        raise HTTPException(503, f"extraction is not loaded: {state.get('extractor_error') or 'unknown'}")
    if file is None and not text.strip():
        raise HTTPException(422, "send a file or text")
    from intake import for_matching
    common = dict(tk_id=tk_id or None, upload_date=upload_date or date.today().isoformat())
    name = Path(file.filename or "upload").name if file is not None else "typed"
    timing, start = {}, time.perf_counter()
    data = file.file.read() if file is not None else text.encode()
    keep = Path(DEV_CACHE) / hashlib.sha256(data).hexdigest()[:16] if DEV_CACHE else None
    with contextlib.nullcontext(str(keep)) if keep else tempfile.TemporaryDirectory(prefix="tenth-") as tmp:
        folder = Path(tmp)
        folder.mkdir(parents=True, exist_ok=True)
        first, second = new_readers(folder, use_aws)
        disagreements = {}
        reader = lambda content, kind, image=None, **kw: read_both(content, kind, first, second, disagreements,
                                                                   image=image, **kw)
        try:
            if file is None:
                entries, check, parts = reader(text, input_type or "TEXT", source=name, **common), None, None
            else:
                path = folder / name
                path.write_bytes(data)
                entries, check, parts = read_upload(path, input_type, reader, folder, {**common, "source": name},
                                                    use_aws, send_anyway, timing)
        except HTTPException:
            raise
        except ValueError as err:   # written for people: an encrypted email, a photo type Textract doesn't take
            raise HTTPException(422, f"{name}: {err}")
        except OSError as err:      # a file that isn't what its name says
            log.warning("couldn't open %s: %s", name, err)
            raise HTTPException(422, f"{name} couldn't be opened. It may be damaged, or not the kind of file its name says.")
        except Exception:  # noqa: BLE001 - the details go to the log; the page gets a message a person can act on
            log.exception("couldn't read %s", name)
            raise HTTPException(502, f"Tenth couldn't read {name} right now. Try again in a minute, and if it keeps "
                                     "happening, let the team know.")
    timing["reading_s"] = round(time.perf_counter() - start - timing.get("textract_s", 0), 1)   # both models and the checks
    if entries is None:
        return {"photo_check": check, "entries": [], "parts": parts, "decisions": [], "timing": timing}
    review = x["review"]
    entries = review.add_review(entries)
    said = {e: review.split_reasons(c)[0] for e, c in zip(entries["entry_id"], entries["check_reasons"])}
    start = time.perf_counter()
    decisions = run(as_schema(for_matching(entries))) if len(entries) else []
    timing["matching_s"] = round(time.perf_counter() - start, 1)
    for d in decisions:
        reasons = said.get(d.get("entry_id"), [])
        d["missing_tk"] = "unknown_tk" in reasons            # asked once for the whole upload, so not on each card
        reasons = [r for r in reasons if r != "unknown_tk"]
        d["extraction_review"] = bool(reasons)
        diffs = disagreements.get(d.get("entry_id"), [])
        d["extraction_check"] = what_to_do(reasons, diffs)
        d["extraction_reasons"] = what_was_noticed(reasons, review.REVIEW_REASONS, diffs)
    print(f"/extract {name}: " + ", ".join(f"{k[:-2]} {v}s" for k, v in timing.items()), flush=True)   # shows in the container log
    return {"photo_check": check, "entries": json.loads(entries.to_json(orient="records", date_format="iso")),
            "parts": parts, "decisions": decisions, "timing": timing}


class ReviewBatch(BaseModel):
    entries: list[dict]   # {entry_id?, narrative, hours?, cmid?, tk_id?, work_date?, history?: [str]}
    block: bool = True    # False = "no block billing": one entry per task


def matter_history(cmid, tk_id=None, before=None, k=40):
    """Earlier narratives on the same matter, the timekeeper's own first. Stands in for the account history."""
    g = state.get("history", {}).get(str(cmid)) if cmid else None
    if g is None:
        return []
    if before is not None and not pd.isna(before):
        g = g[g.work_date < before]
    if tk_id:
        g = pd.concat([g[g.tk_id == tk_id], g[g.tk_id != tk_id]])
    return g.narrative.head(k).tolist()


@app.get("/review/info")
def review_info():
    r = state.get("reviewer")
    if not r:
        raise HTTPException(503, f"narrative review is not loaded: {state.get('reviewer_error') or 'unknown'}")
    return {"rewriter": r.rewriter_name, "flags": r.flags, "models": r.meta,
            "matters_with_history": len(state.get("history", {}))}


@app.post("/review")
def review_entries(batch: ReviewBatch):
    """Split, expand shorthand, flag each line, suggest a rewrite with placeholders, and list candidates."""
    r = state.get("reviewer")
    if not r:
        raise HTTPException(503, f"narrative review is not loaded: {state.get('reviewer_error') or 'unknown'}")
    if not batch.entries:
        raise HTTPException(422, "no entries")
    if len(batch.entries) > 500:
        raise HTTPException(422, "send at most 500 entries per request")
    prepared = []
    for e in batch.entries:
        narrative = e.get("narrative") or e.get("narrative_raw") or ""
        hours = pd.to_numeric(e.get("hours", e.get("parsed_hours")), errors="coerce")
        history = e.get("history")
        if history is None:
            history = matter_history(e.get("cmid") or e.get("pred_cmid"), e.get("tk_id"),
                                     pd.to_datetime(e.get("work_date"), errors="coerce"))
        prepared.append(dict(narrative=str(narrative), hours=None if pd.isna(hours) else float(hours), history=history))
    results = r.review_many(prepared, block=batch.block)
    for e, res, p in zip(batch.entries, results, prepared):
        res["entry_id"] = e.get("entry_id")
        res["history_used"] = len(p["history"])
        res["history_preview"] = p["history"][:3]   # earlier entries on the matter, the timekeeper's own first, for context
        cmid = e.get("cmid") or e.get("pred_cmid")
        res["cmid"] = cmid
        res["client"], res["matter_name"] = state.get("matter_names", {}).get(str(cmid), (None, None)) if cmid else (None, None)
    return {"rewriter": r.rewriter_name, "reviews": results}


@app.post("/selftest")
def selftest(n: int = 20):
    """Run the first n baked-in entries. Compare with results/06_decisions.csv to confirm the container matches the notebook."""
    return {"decisions": run(sample_entries().head(n).copy())}


# The page itself. Mounted last so the API routes above take priority. Same address as the API, so the page needs no setup.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")