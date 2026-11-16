"""Asking a model on Bedrock to read the lines of an upload into entries, and turning its answer into entry rows.

Two kinds of reader:
- Nova on Bedrock reads Textract's lines for a page, each with an id, where it sits, and Textract's confidence, and
  can also see the photo. It answers with a tool call, so every entry has the same fields. Each entry names the lines
  it came from and the hours exactly as written, so the answer can be checked against the page.
- Bedrock Data Automation reads the photo itself with a blueprint of the same fields. It only takes document
  blueprints asynchronously, so each photo is uploaded to S3 and deleted after.

Both get the same knowledge of how notes are written that the rules have (NOTE_HABITS, READER_MISTAKES). Nova also
gets three worked examples. The examples are made up with clients that appear on none of the photos, so no graded
photo is part of a prompt.

Every answer goes through model_rows, the same kind of checks the rules get (hours that look wrong, a CMID with the
wrong shape or not in Matter Master, hard-to-read lines), plus checks only a model needs: hours that aren't on the
lines it pointed to (not_on_page), and hours that don't match how they're written (hours_mismatch). Answers are saved
as JSON, so each photo or case is paid for once.
"""
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from free_text import CMID_SHAPE, fix_misreads, page_amounts, page_order, spoken_hours, unused_hours
from grade import align, pick_cutoff, same_cmid, similar
from intake import add_reason
from text_parser import find_hours
from textract import page_lines

NOVA_MODELS = {"Nova Pro": "us.amazon.nova-pro-v1:0", "Nova 2 Lite": "us.amazon.nova-2-lite-v1:0"}
MAX_HOURS = 12
LOW_CONF = 80                       # the cutoff the rules use for Textract's confidence
NOVA_PHOTO = (2048, 3_750_000)      # longest side and bytes for a photo sent to Nova
BDA_PHOTO = (7680, 5_000_000)       # Data Automation's limits for an image
BLANKS = {"", "null", "none", "n/a", "na", "-"}
BARE_NUMBER = re.compile(r"^\D*?(\d{1,2})\D*$")

FIELDS = {
    "client": "The client name as written, with only clear reading mistakes fixed. \"Firm\" for firm time (admin, "
              "business development, recruiting). Empty if none.",
    "cmid": "The client-matter number (4 digits, a dot, 3 digits, like 5750.002) as written, or from the heading or "
            "\"same\" line the entry falls under. Empty if none.",
    "date": "The date that applies to this entry, as written (like 11/9). Empty if none.",
    "hours_written": "The hours exactly as they appear in the text, mistakes included (like \", 8\", \"4\", or "
                     "\"two hours\"). Empty if none.",
    "hours": "What the hours mean as a decimal number of hours (\".4\" is 0.4). Left out if no hours are written.",
    "narrative": "The work done, as written, with wrapped lines joined and without the client, CMID, hours, or date.",
    "unsure": "What was guessed or fixed, and why. Empty if the reading is clear.",
}
ENTRY_SCHEMA = {
    "type": "object",
    "properties": {"entries": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "lines": {"type": "array", "items": {"type": "integer"},
                      "description": "The ids of every line this entry uses, including its heading, date, and hours."},
            **{name: {"type": "number" if name == "hours" else "string", "description": text}
               for name, text in FIELDS.items()}},
        "required": ["lines"] + [name for name in FIELDS if name != "hours"]}}},
    "required": ["entries"],
}

NOTE_HABITS = """How the notes are written:
- A time entry is one piece of work: hours, usually a client name or a client-matter number (CMID: 4 digits, a dot, 3 digits, like 5750.002), and a few words on what was done.
- Hours are decimal hours, usually in tenths: ".4" is 0.4 and "1.5" is 1.5. Hours can be written in words ("two hours" is 2, "half an hour" is 0.5) or as a fraction ("1/2 hr" is 0.5).
- A description often wraps onto the next line or two. A line with no hours, client, or CMID of its own continues the entry above it.
- Hours can sit apart from their entry, on their own line just above, below, or to the right of it, sometimes with an arrow.
- A line starting with "same", a ditto mark, or an arrow is a new entry with the same client and CMID as the entry above.
- A CMID or client alone on a line is a heading for the entry right below it.
- A date on its own line ("Tues 10/27") or at the start of an entry applies to that entry and the ones below it, until another date.
- Several entries can be written as one sentence, joined by "then", "and", or commas. Each amount of hours is its own entry.
- Firm time (admin, business development, recruiting) has no client: write "Firm" as the client.
- Headings, totals, and crossed-out words are not entries."""

READER_MISTAKES = """How the text reader makes mistakes on handwriting:
- A decimal point can be read as a comma or a space (", 8" is .8), dropped (".4" read as "4"), or read as a leading zero ("05" is .5).
- A date's slash can be read as a 1 ("10127" is 10/27).
- Letters and digits get swapped in CMIDs and names ("5S20.OO4" is 5520.004), and words can run together.
- A whole number from 5 to 12 for one short task is almost always a lost decimal point ("8" for .8). Smaller whole numbers can be real (2 hours) or a lost point (4 for .4).
Fix a reading mistake only when the note clearly means something else, and say what was fixed in "unsure". When hours are a whole number, give the most likely reading and say in "unsure" that the decimal point may be missing."""

OUTPUT_RULES = """What to record:
- Every time entry, in the order written, and nothing that isn't one.
- "lines": the ids of every line the entry uses.
- "hours_written": the hours exactly as the text shows them; "hours": what they mean. Leave "hours" out if none are written.
- Use "" for any other field that isn't written. Never fill in a value that isn't on the note.
- "client" as written (fix only clear reading mistakes, using the firm's client list). Don't swap in a different client.
- "unsure": anything guessed or fixed. Leave it "" when the reading is clear."""

EXAMPLES = [
    {"lines": [
        {"id": 1, "across": 20, "down": 10, "conf": 97, "text": "Mon 11/9"},
        {"id": 2, "across": 18, "down": 18, "conf": 95, "text": "Saffron .6 call w/"},
        {"id": 3, "across": 18, "down": 24, "conf": 96, "text": "founders re SAFE"},
        {"id": 4, "across": 19, "down": 33, "conf": 88, "text": "Firm admin , 3"},
        {"id": 5, "across": 19, "down": 39, "conf": 91, "text": "expense reports"}],
     "entries": [
        {"lines": [1, 2, 3], "client": "Saffron", "cmid": "", "date": "11/9", "hours_written": ".6", "hours": 0.6,
         "narrative": "call w/ founders re SAFE", "unsure": ""},
        {"lines": [1, 4, 5], "client": "Firm", "cmid": "", "date": "11/9", "hours_written": ", 3", "hours": 0.3,
         "narrative": "admin expense reports", "unsure": "\", 3\" read as .3"}]},
    {"lines": [
        {"id": 1, "across": 10, "down": 8, "conf": 99, "text": "6888.002"},
        {"id": 2, "across": 8, "down": 14, "conf": 93, "text": "Kestrel - APA comments"},
        {"id": 3, "across": 70, "down": 13, "conf": 90, "text": "1.3"},
        {"id": 4, "across": 8, "down": 22, "conf": 94, "text": "same - call w/ seller counsel"},
        {"id": 5, "across": 62, "down": 20, "conf": 72, "text": "8 ->"},
        {"id": 6, "across": 8, "down": 31, "conf": 95, "text": "Juniper Peak 2.0 LPA markup"}],
     "entries": [
        {"lines": [1, 2, 3], "client": "Kestrel", "cmid": "6888.002", "date": "", "hours_written": "1.3",
         "hours": 1.3, "narrative": "APA comments", "unsure": ""},
        {"lines": [4, 5], "client": "Kestrel", "cmid": "6888.002", "date": "", "hours_written": "8", "hours": 0.8,
         "narrative": "call w/ seller counsel", "unsure": "8 hours for a call is likely .8 with the point missing"},
        {"lines": [6], "client": "Juniper Peak", "cmid": "", "date": "", "hours_written": "2.0", "hours": 2.0,
         "narrative": "LPA markup", "unsure": ""}]},
    {"lines": [
        {"id": 1, "across": 5, "down": 20, "conf": 96, "text": "11/12 Copperline two hours board deck, then"},
        {"id": 2, "across": 5, "down": 26, "conf": 94, "text": "Pelican Row half an hour lease redlines and"},
        {"id": 3, "across": 5, "down": 32, "conf": 97, "text": "7633.003 .7 option grant memo"}],
     "entries": [
        {"lines": [1], "client": "Copperline", "cmid": "", "date": "11/12", "hours_written": "two hours",
         "hours": 2.0, "narrative": "board deck", "unsure": ""},
        {"lines": [1, 2], "client": "Pelican Row", "cmid": "", "date": "11/12", "hours_written": "half an hour",
         "hours": 0.5, "narrative": "lease redlines", "unsure": ""},
        {"lines": [1, 3], "client": "", "cmid": "7633.003", "date": "11/12", "hours_written": ".7", "hours": 0.7,
         "narrative": "option grant memo", "unsure": ""}]},
]
EXAMPLE_CLIENTS = ["Saffron", "Kestrel", "Juniper Peak", "Copperline", "Pelican Row"]

TIMEKEEPER_FIELD = ("The timekeeper (the lawyer or paralegal whose time it is) when the text names one for this entry, "
                    "like a report's Timekeeper heading or a name column, as written. Empty if none is named.")
TIMEKEEPER_RULE = ('- "timekeeper": who the time belongs to, only when the text names them for this entry (a '
                   '"Timekeeper:" heading, a name or initials column). Leave it "" otherwise; never use the sender, '
                   'the assistant, or a name in a signature.')


def entry_schema(with_timekeeper=False):
    """ENTRY_SCHEMA, with an optional timekeeper field (one_flow uses it for documents, scans, and spreadsheets)."""
    if not with_timekeeper:
        return ENTRY_SCHEMA
    schema = json.loads(json.dumps(ENTRY_SCHEMA))
    item = schema["properties"]["entries"]["items"]
    item["properties"]["timekeeper"] = {"type": "string", "description": TIMEKEEPER_FIELD}
    item["required"] = item["required"] + ["timekeeper"]
    return schema

NOTE_INTRO = ("Lines read from a photo of a handwritten time note. Each line shows its id, where it starts on the page "
              "(percent across and down), the reader's confidence (0 to 100), and its text. The position and "
              "confidence are not part of the note.")
TYPED_INTRO = "Lines of a typed time note, email, or dictation, each with its id."


def line_table(lines):
    """Lines as the model sees them: "L3 | 14% across, 29% down | conf 98 | text", or "L3 | text" for typed text."""
    out = []
    for line in lines:
        if line.get("across") is None:
            out.append(f"L{line['id']} | {line['text']}")
        else:
            out.append(f"L{line['id']} | {line['across']}% across, {line['down']}% down | conf {line['conf']} | "
                       f"{line['text']}")
    return "\n".join(out)


def system_prompt(with_timekeeper=False):
    """The system prompt (prompt v1). with_timekeeper adds the timekeeper rule; the examples then show it empty."""
    shown = [{**e, "timekeeper": ""} if with_timekeeper else e for ex in EXAMPLES for e in ex["entries"]]
    examples, start = [], 0
    for n, ex in enumerate(EXAMPLES, 1):
        entries, start = shown[start:start + len(ex["entries"])], start + len(ex["entries"])
        examples.append(f"Example {n}:\n{NOTE_INTRO}\n{line_table(ex['lines'])}\n\nAnswer:\n"
                        f"{json.dumps({'entries': entries})}")
    rules = OUTPUT_RULES + ("\n" + TIMEKEEPER_RULE if with_timekeeper else "")
    return ("You read lawyers' time notes for a law firm's billing team. A text reader has already read the note; "
            "you decide which lines make up each time entry and what each field says, and record them with "
            "record_entries.\n\n"
            f"{NOTE_HABITS}\n\n{READER_MISTAKES}\n\n{rules}\n\n" + "\n\n".join(examples))


def page_text_lines(page):
    """A page's handwritten lines in reading order, as dicts with an id, position (percent), confidence (of the
    least sure word, and of the least sure word with a digit), and text."""
    lines = page_lines(page, drop_repeats=True)
    if lines["handwritten"].any():
        lines = lines[lines["handwritten"]]
    return [{"id": n, "across": round(r.left * 100), "down": round(r.top * 100), "conf": round(r.conf),
             "number_conf": None if pd.isna(r.number_conf) else round(r.number_conf), "text": r.text}
            for n, r in enumerate(page_order(lines).itertuples(), 1)]


def typed_lines(text):
    """Typed text as lines with ids and no position."""
    return [{"id": n, "text": line.strip()} for n, line in enumerate(
        (line for line in text.splitlines() if line.strip()), 1)]


def photo_bytes(path, longest, max_bytes):
    """The photo upright as JPEG bytes, at most `longest` pixels on a side and `max_bytes` long."""
    from preprocess import load_image
    image = load_image(path).convert("RGB")
    image.thumbnail((longest, longest))
    quality = 92
    while True:
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=quality)
        if out.tell() <= max_bytes:
            return out.getvalue()
        if quality > 70:
            quality -= 8
        else:
            image.thumbnail((int(max(image.size) * 0.85),) * 2)


def saved(path, ask):
    """The saved answer at path, or ask() for a new one and save it. A failed answer isn't saved, so a rerun asks
    again."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    answer = ask()
    if "error" not in answer:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer, indent=1))
    return answer


def run_all(jobs, workers=4):
    """Run each job (a function with no arguments) a few at a time, in order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: job(), jobs))


def ask_nova(runtime, lines, clients, model_id, photo=None, typed=False, tries=3, intro=None, with_timekeeper=False,
             system=None, schema=None, force_tool=True):
    """Nova's entries for one note, forced into ENTRY_SCHEMA by a tool call. Saves the lines it was shown, how long it
    took, and the tokens used, so the answer can be checked and costed from the saved file alone.

    intro: what the lines are, in place of NOTE_INTRO or TYPED_INTRO (one_flow says what kind of input it is).
    with_timekeeper: also ask who each entry belongs to (entry_schema).
    system: a system prompt in place of prompt v1's (one_flow uses prompts.system_prompt for inputs other than photos).
    schema: the tool's input schema in place of entry_schema(with_timekeeper) (prompt v4 adds matter and billable).
    force_tool: False for models that can't be forced to call a tool (Llama 4 Maverick): the tool is offered, and a
    reply with JSON in its text instead is read with open_models.parse_entries.
    The model sometimes returns a broken tool call (ModelErrorException). It's asked again up to `tries` times; if
    it still fails, the answer has no entries and an "error", so the note falls back to the rules."""
    intro = intro or (TYPED_INTRO if typed else NOTE_INTRO)
    text = f"The firm's clients: {', '.join(clients)}\n\n{intro}\n{line_table(lines)}"
    content = [{"text": text}]
    if photo is not None:
        content.append({"image": {"format": "jpeg", "source": {"bytes": photo_bytes(photo, *NOVA_PHOTO)}}})
    for attempt in range(tries):
        try:
            response = runtime.converse(
                modelId=model_id,
                system=[{"text": system or system_prompt(with_timekeeper)}],
                messages=[{"role": "user", "content": content}],
                toolConfig={"tools": [{"toolSpec": {"name": "record_entries", "description": "Record the time entries.",
                                                    "inputSchema": {"json": schema or entry_schema(with_timekeeper)}}}],
                            **({"toolChoice": {"tool": {"name": "record_entries"}}} if force_tool else {})},
                inferenceConfig={"temperature": 0, "maxTokens": 4000},
            )
            break
        except Exception as error:
            if "ModelErrorException" not in type(error).__name__ + str(error):
                raise
            if attempt == tries - 1:
                return {"model": model_id, "entries": [], "lines": lines, "stop_reason": "model_error",
                        "error": str(error), "tries": tries, "usage": {"inputTokens": 0, "outputTokens": 0},
                        "latency_ms": 0}
    blocks = response["output"]["message"]["content"]
    entries = [block["toolUse"]["input"].get("entries", []) for block in blocks if "toolUse" in block]
    if not force_tool:          # an unforced tool call can come back as text, or with the entries as a string
        from open_models import parse_entries
        if entries and not isinstance(entries[0], list):
            entries[0] = parse_entries(entries[0] if isinstance(entries[0], str) else json.dumps(entries[0])) or []
        if not entries:
            found = parse_entries(" ".join(block.get("text", "") for block in blocks))
            entries = [found] if found is not None else []
    return {"model": model_id, "entries": entries[0] if entries else [], "lines": lines,
            "stop_reason": response["stopReason"], "tries": attempt + 1, "usage": response["usage"],
            "latency_ms": response["metrics"]["latencyMs"]}


def blank(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip().lower() in BLANKS


def to_number(value):
    try:
        return None if blank(value) else float(value)
    except (TypeError, ValueError):
        return None


def read_hours(written):
    """The hours in text as written, read the way the rules read them (", 8" is .8, "two hours" is 2), or None
    when there isn't exactly one amount."""
    found = find_hours(spoken_hours(fix_misreads(written)[0]))
    return found[0][0] if len(found) == 1 else None


def hours_checks(written, hours):
    """Reasons to check a model's hours: over MAX_HOURS, not matching how they're written, or a bare whole number
    the model changed or that looks like a lost decimal point (5 to 12)."""
    reasons = []
    if hours is not None and not 0 < hours <= MAX_HOURS:
        reasons.append("hours_10x")
    if blank(written) or hours is None:
        return reasons
    read = read_hours(written)
    bare = BARE_NUMBER.match(written)
    if read is not None and abs(read - hours) > 0.01:
        reasons.append("hours_mismatch")
    elif read is None and bare and (abs(int(bare.group(1)) - hours) > 0.01 or 5 <= hours <= MAX_HOURS):
        reasons.append("hours_10x")
    return list(dict.fromkeys(reasons))


def squash(text):
    return re.sub(r"\s+", "", str(text).lower())


def model_rows(entries, source, upload_date, cmids, lines=None, input_type="NOTE", conf_of=None, cutoff=None,
               confidence="line", amount_lines=None):
    """A model's entries as rows for intake.make_entries, with check reasons.

    lines: the lines the model was shown (Nova). Each entry's hours must appear in the lines it names, and the
    lowest Textract confidence among them is checked like the rules check it: of every word ("line"), or only of
    words with a digit ("numbers", as free_text.note_rows does).
    conf_of, cutoff: for Data Automation, each entry's confidence in its hours (0 to 1) and the cutoff below which
    it's flagged.
    amount_lines: the texts to look for hours no entry uses (unused_hours). None: every line, for notes only (as
    model first does); an empty list: no such check.
    """
    by_id = {line["id"]: line for line in lines or []}
    rows = []
    for n, entry in enumerate(entries):
        e = {name: (None if blank(value) else value) for name, value in entry.items()}
        hours = to_number(e.get("hours"))
        written = e.get("hours_written") or ""
        reasons = hours_checks(written, hours)
        cmid = str(e["cmid"]).strip() if e.get("cmid") else None
        if cmid and not CMID_SHAPE.fullmatch(cmid):
            reasons.append("cmid_shape")
        elif cmid and cmid not in cmids:
            reasons.append("cmid_unknown")
        min_conf = None
        if lines is not None:
            cited = [by_id[i] for i in e.get("lines") or [] if i in by_id]
            if not cited or (written and squash(written) not in squash(" ".join(c["text"] for c in cited))):
                reasons.append("not_on_page")
            field = "number_conf" if confidence == "numbers" else "conf"
            confs = [c[field] for c in cited if c.get(field) is not None]
            min_conf = min(confs) if confs else None
            if min_conf is not None and min_conf < LOW_CONF:
                reasons.append("low_conf")
        if conf_of is not None and conf_of[n] is not None:
            min_conf = round(conf_of[n] * 100)
            if conf_of[n] < cutoff:
                reasons.append("low_conf")
        if e.get("unsure"):
            reasons.append("model_unsure")
        rows.append({"input_type": input_type, "source": source, "rows": n + 1,
                     "work_date": e.get("date") or upload_date,
                     "client": e.get("client"), "cmid": cmid, "hours": hours, "hours_read": written or None,
                     "narrative": e.get("narrative") or "", "min_conf": min_conf,
                     "check_reasons": "; ".join(dict.fromkeys(reasons))})
    if amount_lines is None:
        amount_lines = [line["text"] for line in lines] if lines is not None and input_type == "NOTE" else []
    if amount_lines and unused_hours(page_amounts(amount_lines), [row["hours"] for row in rows]):
        for row in rows:
            row["check_reasons"] = "; ".join(filter(None, [row["check_reasons"], "unused_hours"]))
    return rows


def nova_rows(answer, source, upload_date, cmids, input_type="NOTE", confidence="line", lines=None,
              amount_lines=None):
    """Rows from a saved Nova answer. A first-try answer has no lines, so only the other checks apply.

    lines: the page's lines to check against instead of the saved ones (for confidence values saved answers lack).
    amount_lines: as in model_rows.
    """
    entries = answer["entries"]
    if answer.get("stop_reason") == "max_tokens":
        entries = [{**e, "unsure": "the answer was cut off"} for e in entries]
    return model_rows(entries, source, upload_date, cmids, lines or answer.get("lines"), input_type,
                      confidence=confidence, amount_lines=amount_lines)


def cross_check(first, second, dates=False):
    """The first reader's entries for one note, with the ones the second reader disagrees with flagged
    (readers_disagree), and the second reader's entries the first didn't find added in place (second_reader_only).

    dates: also count two different work dates as a disagreement."""
    first, second = first.reset_index(drop=True), second.reset_index(drop=True)
    pieces = []
    for i, j in align(first, second):
        if i is None:
            pieces.append(add_reason(second.iloc[[j]], [True], "second_reader_only"))
            continue
        row = first.iloc[[i]]
        agree = False
        if j is not None:
            a, b = first.iloc[i], second.iloc[j]
            agree = (pd.notna(a["hours"]) and pd.notna(b["hours"]) and round(a["hours"], 2) == round(b["hours"], 2)
                     and (similar(a["client"], b["client"]) or similar(b["client"], a["client"]))
                     and (pd.isna(a["cmid"]) or pd.isna(b["cmid"]) or same_cmid(a["cmid"], b["cmid"]))
                     and not (dates and pd.notna(a["work_date"]) and pd.notna(b["work_date"])
                              and pd.Timestamp(a["work_date"]) != pd.Timestamp(b["work_date"])))
        pieces.append(row if agree else add_reason(row, [True], "readers_disagree"))
    return pd.concat(pieces, ignore_index=True) if pieces else first


BLUEPRINT_ENTRIES = (
    "Every time entry on the note, in the order written, and nothing else (not headings, totals, or crossed-out "
    "words). A description can wrap onto the next lines. Hours can sit apart from their entry, sometimes with an "
    "arrow. A line starting with \"same\", a ditto mark, or an arrow is a new entry with the client and CMID of the "
    "entry above. A CMID or client alone on a line heads the entry below it. A date applies to the entries below it "
    "until another date.")
BLUEPRINT = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "description": "A lawyer's handwritten time note (sticky note or notepad page) with one or more time entries.",
    "class": "Time note",
    "type": "object",
    "definitions": {"Entry": {"type": "object", "properties": {
        name: {"type": "number" if name == "hours" else "string", "inferenceType": "explicit", "instruction": text}
        for name, text in FIELDS.items() if name != "unsure"}}},
    "properties": {"entries": {"type": "array", "instruction": BLUEPRINT_ENTRIES,
                               "items": {"$ref": "#/definitions/Entry"}}},
}


def listed(call, key, **filters):
    """Every item from a Data Automation list call, page by page. Filters that the service rejects are dropped."""
    try:
        page = call(**filters)
    except Exception as error:
        if "ValidationException" not in str(type(error)) + str(error):
            raise
        filters, page = {}, call()
    items = page.get(key, [])
    while page.get("nextToken"):
        page = call(nextToken=page["nextToken"], **filters)
        items += page.get(key, [])
    return items


def find_or_create_bda(bda, name, blueprint):
    """The ARNs of the blueprint and project called `name`, made the first time. Data Automation only takes document
    blueprints in an asynchronous project, so photos go through S3."""
    blueprints = listed(bda.list_blueprints, "blueprints", resourceOwner="ACCOUNT")
    found = [b["blueprintArn"] for b in blueprints if b["blueprintName"] == name]
    blueprint_arn = found[0] if found else bda.create_blueprint(
        blueprintName=name, type="DOCUMENT", blueprintStage="LIVE", schema=json.dumps(blueprint))["blueprint"][
        "blueprintArn"]
    projects = listed(bda.list_data_automation_projects, "projects", resourceOwner="ACCOUNT")
    found = [p["projectArn"] for p in projects if p["projectName"] == name]
    project_arn = found[0] if found else bda.create_data_automation_project(
        projectName=name, projectStage="LIVE", projectType="ASYNC",
        standardOutputConfiguration={"document": {
            "extraction": {"granularity": {"types": ["DOCUMENT"]}, "boundingBox": {"state": "DISABLED"}},
            "generativeField": {"state": "DISABLED"},
            "outputFormat": {"textFormat": {"types": ["PLAIN_TEXT"]}, "additionalFileFormat": {"state": "DISABLED"}}}},
        customOutputConfiguration={"blueprints": [{"blueprintArn": blueprint_arn, "blueprintStage": "LIVE"}]},
        overrideConfiguration={"modalityRouting": {"jpeg": "DOCUMENT", "png": "DOCUMENT"}},
    )["projectArn"]
    while bda.get_data_automation_project(projectArn=project_arn)["project"]["status"] == "IN_PROGRESS":
        time.sleep(2)
    return {"blueprint": blueprint_arn, "project": project_arn}


def start_bda(runtime, s3, bucket, photo, project_arn, profile_arn, longest=BDA_PHOTO[0], max_bytes=BDA_PHOTO[1]):
    """Upload one photo and start Data Automation on it. Returns what finish_bda needs."""
    stem = Path(photo).stem
    s3.put_object(Bucket=bucket, Key=f"input/{stem}.jpg", Body=photo_bytes(photo, longest, max_bytes))
    response = runtime.invoke_data_automation_async(
        inputConfiguration={"s3Uri": f"s3://{bucket}/input/{stem}.jpg"},
        outputConfiguration={"s3Uri": f"s3://{bucket}/output/{stem}/"},
        dataAutomationConfiguration={"dataAutomationProjectArn": project_arn, "stage": "LIVE"},
        dataAutomationProfileArn=profile_arn,
    )
    return {"arn": response["invocationArn"], "stem": stem, "started": time.time()}


def read_s3_json(s3, uri):
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def custom_paths(metadata):
    """Every custom_output_path in a job's metadata, wherever it sits."""
    if isinstance(metadata, dict):
        own = [v for k, v in metadata.items() if k == "custom_output_path" and isinstance(v, str)]
        return own + [p for v in metadata.values() for p in custom_paths(v)]
    if isinstance(metadata, list):
        return [p for v in metadata for p in custom_paths(v)]
    return []


def finish_bda(runtime, s3, bucket, job, wait=5):
    """Wait for one job, read its custom output, and delete the photo and output from S3."""
    while True:
        status = runtime.get_data_automation_status(invocationArn=job["arn"])
        if status["status"] not in ("Created", "InProgress"):
            break
        time.sleep(wait)
    try:
        if status["status"] != "Success":
            raise RuntimeError(f"{job['stem']}: {status['status']} {status.get('errorMessage', '')}")
        metadata = read_s3_json(s3, status["outputConfiguration"]["s3Uri"])
        return {"custom": [read_s3_json(s3, path) for path in custom_paths(metadata)],
                "seconds": round(time.time() - job.get("started", time.time()))}
    finally:
        s3.delete_object(Bucket=bucket, Key=f"input/{job['stem']}.jpg")
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=f"output/{job['stem']}/").get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])


def bda_entries(answer):
    """The entries in a saved Data Automation answer, each with its confidence in the hours (0 to 1)."""
    entries = []
    for custom in answer["custom"]:
        found = (custom.get("inference_result") or {}).get("entries") or []
        explained = ((custom.get("explainability_info") or [{}])[0] or {}).get("entries") or []
        for n, entry in enumerate(found):
            info = explained[n] if n < len(explained) and isinstance(explained[n], dict) else {}
            entries.append({**entry, "hours_conf": (info.get("hours") or {}).get("confidence")})
    return entries


def bda_rows(answer, source, upload_date, cmids, cutoff=0.0):
    """Rows from a saved Data Automation answer, with hours flagged when its confidence is under `cutoff`."""
    entries = bda_entries(answer)
    confs = [e.pop("hours_conf") for e in entries]
    return model_rows(entries, source, upload_date, cmids, conf_of=confs, cutoff=cutoff)
