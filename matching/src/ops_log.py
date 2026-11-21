"""The alpha's operations log: one JSON line for each model call and one for each upload.

It records how the reading went (which model and prompt, tokens, seconds, retries, failures, fallbacks, review
reasons, matching actions) and nothing an upload says: no file names, client names, CMIDs, narratives, hours, or
error messages (only the error's type). Lines are printed, so the container's log (CloudWatch on ECS) collects
them, and, when TENTH_OPS_LOG names a file, written to that file as well. Each line starts with "tenth-ops " so a log
query can pick them out.
"""
import json
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

PATH = os.getenv("TENTH_OPS_LOG", "")
WRITE_LOCK = threading.Lock()


def new_upload_id():
    return uuid.uuid4().hex[:12]


def write(line):
    """Sends one line (a dict) to the log, as JSON."""
    text = json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **line}, default=str)
    print("tenth-ops " + text, flush=True)
    if PATH:
        with WRITE_LOCK, Path(PATH).open("a") as f:
            f.write(text + "\n")


def error_type(problem):
    """The type of an error ("ReadTimeoutError"), from text like "ReadTimeoutError: Read timeout on ...", without the
    message, which can repeat what was sent."""
    return str(problem).split(":", 1)[0].strip()[:60] if problem else None


def call_line(upload, reader, piece, number, out, seconds, retries):
    """The line for one model call. piece: what the reader was asked (one_flow.pieces or a column map). out: its
    answer, or None when there was no answer to give (no saved answer and no AWS)."""
    out = out if isinstance(out, dict) else {}
    usage = out.get("usage") or {}
    return {"event": "model_call", "upload": upload, "reader": reader, "model": out.get("model"),
            "input_type": piece.get("input_type"), "part": "columns" if piece.get("columns") else number,
            "lines": len(piece.get("lines") or []), "photo": piece.get("photo") is not None,
            "prompt_hash": out.get("prompt_hash"), "tokens_in": usage.get("inputTokens"),
            "cache_read": usage.get("cacheReadInputTokens", 0), "cache_write": usage.get("cacheWriteInputTokens", 0),
            "tokens_out": usage.get("outputTokens"), "model_seconds": round(out.get("latency_ms", 0) / 1000, 2),
            "seconds": round(seconds, 2), "stop_reason": out.get("stop_reason"), "retries": retries,
            "failed": not out or "error" in out, "error": error_type(out.get("error"))}


def upload_line(upload, suffix, use_aws, status, seconds, result=None):
    """The line for one upload. result: what /extract returned (None when it failed with status)."""
    result = result or {}
    entries, decisions = result.get("entries") or [], result.get("decisions") or []
    reasons = Counter(r for e in entries for r in str(e.get("check_reasons") or "").split("; ") if r)
    check = result.get("photo_check")
    return {"event": "upload", "upload": upload, "suffix": suffix, "use_aws": use_aws, "status": status,
            "seconds": round(seconds, 1), "timing": result.get("timing"),
            "input_type": sorted({e.get("input_type") for e in entries if e.get("input_type")}),
            "models": sorted({e.get("model_id") for e in entries if e.get("model_id")}),
            "prompt_hashes": sorted({e.get("prompt_hash") for e in entries if e.get("prompt_hash")}),
            "entries": len(entries), "to_review": sum(bool(e.get("review")) for e in entries),
            "read_by_rules": reasons.get("read_by_rules", 0), "one_reader": reasons.get("one_reader", 0),
            "reasons": dict(reasons), "actions": dict(Counter(d.get("action") for d in decisions if d.get("action"))),
            "read_errors": len(result.get("read_errors") or []),
            "photo_check_failed": bool(check) and not check.get("ok", True)}
