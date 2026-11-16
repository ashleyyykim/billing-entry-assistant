"""Tenth matching service.

Wraps src/pipeline.py's decide() behind a small HTTP API, the same call
notebook 06 uses. Everything heavy is loaded once at startup: the embedding
model and the reference data (Matter Master, billing history, models, cutoffs)
via load_reference().

Stage 2, narrative review (/review), runs review.Reviewer from phase2_model/pipeline with the models in
phase2_model/models (in the image: /app/phase2, see docker/Dockerfile). It is optional: if they're missing,
matching still works and /review returns 503.
"""
import base64
import json
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
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
            "narrative_review": r.rewriter_name if r else f"off ({state.get('reviewer_error') or 'not loaded'})"}


@app.get("/schema")
def schema():
    """The fields an entry needs, taken from what load_entries() produces. Map OCR rows onto these."""
    sample = sample_entries()
    return {"columns": {c: str(t) for c, t in sample.dtypes.items()},
            "example": json.loads(sample.head(1).to_json(orient="records", date_format="iso"))[0]}


@app.post("/decide")
def decide_entries(batch: Batch):
    if not batch.entries:
        raise HTTPException(422, "no entries")
    entries = pd.DataFrame(batch.entries)
    for col, dtype in sample_entries().dtypes.items():
        if col not in entries:
            continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            entries[col] = pd.to_datetime(entries[col])
        elif pd.api.types.is_numeric_dtype(dtype):
            entries[col] = pd.to_numeric(entries[col])
    try:
        return {"decisions": run(entries)}
    except KeyError as err:
        raise HTTPException(422, f"missing or misnamed field {err}; see /schema")


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