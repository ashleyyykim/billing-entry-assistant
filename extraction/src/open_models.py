"""Open-weight vision models on Bedrock (Llama 4 Maverick is one flow's second reader) reading note photos.

Every model gets the same input as Nova Pro in the model-first flow: Textract's lines (id, position, confidence, text),
the photo, and the firm's client list. Answers come back in the same fields (model_readers.ENTRY_SCHEMA), so
model_first.read_photo checks and grades them exactly like Nova's.

Prompt v2 (system_prompt_v2) keeps everything in Nova's prompt (v1, model_readers.system_prompt) and adds:
- how to use the photo, which v1 never mentions: check each line against it, record an entry the reader skipped, and
  say what came from the photo;
- minutes and spoken numbers as hours;
- a checklist to run before answering (one entry per amount of hours, totals left out, dates and clients carried down);
- two more worked examples (a ditto mark, a split decimal, and a total line; and a dictated note);
- for models that can't be made to call the tool, the exact JSON to reply with.
The new text and examples use invented clients and CMIDs, none from our samples, the held-out cards, or the firm's
list, so the prompt can't hand a model the answers.

Not every model on Bedrock can be forced to call a tool. ask_model tries, in order: a forced tool call, an optional
tool call, and plain JSON in the reply, and remembers what worked for each model (the answer records it as "mode").
A model that doesn't take a system prompt gets it at the top of the message instead.
"""
import json
import re
import time

from model_readers import (EXAMPLES, FIELDS, ENTRY_SCHEMA, NOTE_HABITS, NOTE_INTRO, NOVA_PHOTO, OUTPUT_RULES,
                           READER_MISTAKES, TYPED_INTRO, line_table, photo_bytes, system_prompt)

PROMPTS = ["v1", "v2"]
MODES = ["tool", "tool_auto", "json"]
WANTED = {   # name in the notebook: words that must all appear in Bedrock's model name (case ignored)
    "Qwen3 VL": ["qwen3", "vl"],
    "Llama 4 Maverick": ["llama 4", "maverick"],
    "Mistral Large 3": ["mistral large 3"],
    "Gemma 3 27B": ["gemma 3", "27b"],
}

INTRO = ("You read lawyers' time notes for a law firm's billing team. A text reader (OCR) has already read the note "
         "into numbered lines; you decide which lines make up each time entry and what each field says.")

MORE_HABITS = """More about hours and CMIDs:
- Minutes are hours divided by 60: "24 minutes" is 0.4, "48 min" is 0.8, "an hour and six minutes" is 1.1.
- Spoken decimals: "point seven" is 0.7, "three point one" is 3.1, "four tenths" is 0.4, "one point oh" is 1.0.
- A CMID can be spoken in pieces: "thirty-one seventy point oh oh two", "five two eight eight dot zero zero four", and "ninety-four sixty-one, oh oh one" are 3170.002, 5288.004, and 9461.001.
- When someone corrects themselves ("two point four, no, two point one"), the last amount is the one that counts."""

PHOTO_GUIDE = """You also get the photo the lines were read from. Use it to check the reader:
- Read each line yourself in the photo. Where the photo clearly shows something different from the reader's text (a decimal point, a digit, a letter in a name or CMID), go with the photo and say what you changed in "unsure", for example: "photo shows .8; the reader has 8".
- The reader sometimes skips a line, most often a short one, one written at an angle, or one in the margin. If the photo shows an entry that isn't in the lines, record it anyway: cite the nearest line ids, put the hours as the photo shows them in "hours_written", and start "unsure" with "not in the reader's lines".
- Crossed-out writing, doodles, and printed form text are not entries.
- Don't guess past the photo: if a word is unclear in the photo too, keep the reader's text and say so in "unsure"."""

CHECKLIST = """Before you answer, check:
1. Every separate amount of hours on the page is in exactly one entry. A total ("Total 2.8", "= 4.1") is not an entry.
2. Each entry's "hours_written" appears in the lines it cites, unless "unsure" says it came from the photo.
3. A date that applies to several entries is repeated on each of them. A day said in words ("Monday", "yesterday") goes in "date" as said.
4. "same", a ditto mark, "again", or "another" carries the client and CMID down from the entry above.
5. Firm time (admin, business development, recruiting) has "Firm" as the client.
6. Nothing is filled in that isn't on the note: no client, CMID, or date from another entry unless the note carries it down."""

EXTRA_EXAMPLES = [
    {"lines": [
        {"id": 1, "across": 12, "down": 6, "conf": 98, "text": "Thurs 11/12"},
        {"id": 2, "across": 10, "down": 15, "conf": 95, "text": "Copperline - board consent"},
        {"id": 3, "across": 74, "down": 15, "conf": 93, "text": "1.1"},
        {"id": 4, "across": 10, "down": 23, "conf": 81, "text": "\" - call w/ auditors .5"},
        {"id": 5, "across": 10, "down": 31, "conf": 96, "text": "4412.001 diligence"},
        {"id": 6, "across": 10, "down": 37, "conf": 84, "text": "tracker updates 1 2"},
        {"id": 7, "across": 60, "down": 48, "conf": 97, "text": "Total 2.8"}],
     "entries": [
        {"lines": [1, 2, 3], "client": "Copperline", "cmid": "", "date": "11/12", "hours_written": "1.1",
         "hours": 1.1, "narrative": "board consent", "unsure": ""},
        {"lines": [1, 4], "client": "Copperline", "cmid": "", "date": "11/12", "hours_written": ".5", "hours": 0.5,
         "narrative": "call w/ auditors", "unsure": "ditto mark: same client as the entry above"},
        {"lines": [1, 5, 6], "client": "", "cmid": "4412.001", "date": "11/12", "hours_written": "1 2",
         "hours": 1.2, "narrative": "diligence tracker updates", "unsure": "\"1 2\" read as 1.2"}]},
]
TYPED_EXAMPLE = {
    "lines": [
        {"id": 1, "text": "Yesterday, Saffron, point six on the call with the founders about the SAFE, then same "
                          "client, two point three on the board deck."},
        {"id": 2, "text": "Firm time, twenty-four minutes, expense reports."}],
    "entries": [
        {"lines": [1], "client": "Saffron", "cmid": "", "date": "yesterday", "hours_written": "point six",
         "hours": 0.6, "narrative": "call with the founders about the SAFE", "unsure": ""},
        {"lines": [1], "client": "Saffron", "cmid": "", "date": "yesterday", "hours_written": "two point three",
         "hours": 2.3, "narrative": "board deck", "unsure": ""},
        {"lines": [2], "client": "Firm", "cmid": "", "date": "", "hours_written": "twenty-four minutes", "hours": 0.4,
         "narrative": "expense reports", "unsure": ""}]}

JSON_CONTRACT = ("Reply with only one JSON object and nothing else: no words before or after it, no code fences. "
                 "It has one key, \"entries\", a list with one object per time entry, in the order written. Each object "
                 "has exactly these keys: \"lines\" (a list of line ids, as numbers), "
                 + ", ".join(f"\"{name}\"" for name in FIELDS if name != "hours")
                 + " (strings, \"\" when not written), and \"hours\" (a number, left out when no hours are written). "
                 "If the note has no time entries, reply {\"entries\": []}.")


def example_block(n, example, typed):
    intro = TYPED_INTRO if typed else NOTE_INTRO
    return f"Example {n}:\n{intro}\n{line_table(example['lines'])}\n\nAnswer:\n{json.dumps({'entries': example['entries']})}"


def system_prompt_v2(photo=True, typed=False, mode="tool"):
    """Prompt v2 for a note (photo=True adds how to use the photo) or typed text (typed=True), and how to answer:
    with the record_entries tool, or as plain JSON (mode="json")."""
    examples = [TYPED_EXAMPLE] if typed else EXAMPLES + EXTRA_EXAMPLES
    answer = (JSON_CONTRACT if mode == "json" else "Record the entries by calling record_entries once, with every "
              "entry on the note.")
    parts = [INTRO, NOTE_HABITS, MORE_HABITS, READER_MISTAKES if not typed else "",
             PHOTO_GUIDE if photo and not typed else "", OUTPUT_RULES, CHECKLIST, answer,
             "\n\n".join(example_block(n, ex, typed) for n, ex in enumerate(examples, 1))]
    return "\n\n".join(p for p in parts if p)


def note_message(lines, clients, typed=False):
    return f"The firm's clients: {', '.join(clients)}\n\n{TYPED_INTRO if typed else NOTE_INTRO}\n{line_table(lines)}"


def parse_entries(text):
    """The entries in a reply that should be one JSON object: code fences and words around it are tolerated, a bare
    list is taken as the entries, and line ids are made numbers. None if no JSON can be read."""
    text = re.sub(r"```(?:json)?", "", text or "")
    found = None
    spans = [(text.find("{"), text.rfind("}")), (text.find("["), text.rfind("]"))]
    for start, end in sorted(spans, key=lambda span: span[0] if span[0] >= 0 else len(text)):
        if 0 <= start < end:
            try:
                found = json.loads(text[start:end + 1])
                break
            except json.JSONDecodeError:
                continue
    if isinstance(found, dict):
        found = found.get("entries")
    if not isinstance(found, list):
        return None
    entries = []
    for entry in found:
        if not isinstance(entry, dict):
            continue
        ids = entry.get("lines") or []
        ids = [str(i).strip().lstrip("Ll") for i in (ids if isinstance(ids, list) else [ids])]
        entry["lines"] = [int(i) for i in ids if i.isdigit()]
        entries.append(entry)
    return entries


def pick_models(models, profiles, wanted=WANTED):
    """{name: model id to call} for each wanted model Bedrock offers with image input and text output. A model that
    can only be called through an inference profile gets the profile's id (a US one if there is one)."""
    picked = {}
    for name, words in wanted.items():
        matches = [m for m in models if all(w in m["modelName"].lower() for w in words)
                   and "IMAGE" in m.get("inputModalities", []) and "TEXT" in m.get("outputModalities", [])
                   and m.get("modelLifecycle", {}).get("status", "ACTIVE") == "ACTIVE"]
        for model in sorted(matches, key=lambda m: m["modelId"], reverse=True):
            if "ON_DEMAND" in model.get("inferenceTypesSupported", []):
                picked[name] = model["modelId"]
                break
            ids = [p["inferenceProfileId"] for p in profiles
                   if any(model["modelId"] in m.get("modelArn", "") for m in p.get("models", []))]
            if ids:
                picked[name] = sorted(ids, key=lambda i: (not i.startswith("us."), i))[0]
                break
    return picked


WORKS = {}   # model id -> (mode, system prompt allowed) that worked, so later calls skip the ones that didn't


def ask_model(runtime, lines, clients, model_id, photo=None, typed=False, prompt="v2", tries=3, max_tokens=4000,
              image_first=False):
    """A model's entries for one note, like model_readers.ask_nova, with prompt v1 or v2 and whichever answer mode the
    model supports. A model that fails every mode, or keeps returning broken answers, gets an answer with no entries
    and an "error", so the note falls back to the rules.

    image_first: put the photo before the text in the message, as Amazon's Nova prompting guide recommends (Nova's
    reader so far, model_readers.ask_nova, puts the text first). With prompt v1 and image_first=False the request is
    the same as ask_nova's."""
    content = [{"text": note_message(lines, clients, typed)}]
    if photo is not None:
        image = {"image": {"format": "jpeg", "source": {"bytes": photo_bytes(photo, *NOVA_PHOTO)}}}
        content = [image] + content if image_first else content + [image]
    tool = {"tools": [{"toolSpec": {"name": "record_entries", "description": "Record the time entries.",
                                    "inputSchema": {"json": ENTRY_SCHEMA}}}]}
    tried = []
    plans = [(m, s) for m in MODES for s in (True, False)]
    if model_id in WORKS:                    # what worked last time first, the rest only if it stops working
        plans = [WORKS[model_id]] + [p for p in plans if p != WORKS[model_id]]
    for mode, with_system in plans:
        system = system_prompt() if prompt == "v1" and mode != "json" else (
            system_prompt() + "\n\n" + JSON_CONTRACT if prompt == "v1" else
            system_prompt_v2(photo is not None, typed, mode))
        request = {"modelId": model_id, "inferenceConfig": {"temperature": 0, "maxTokens": max_tokens}}
        if with_system:
            message = content
        else:                                        # the system prompt goes at the top of the text part
            message = [{"text": system + "\n\n" + c["text"]} if "text" in c else c for c in content]
        request["messages"] = [{"role": "user", "content": message}]
        if with_system:
            request["system"] = [{"text": system}]
        if mode != "json":
            request["toolConfig"] = {**tool, **({"toolChoice": {"tool": {"name": "record_entries"}}}
                                                if mode == "tool" else {})}
        for attempt in range(tries):
            started = time.time()
            try:
                response = runtime.converse(**request)
            except Exception as error:
                name = type(error).__name__ + " " + str(error)
                if "ValidationException" in name:
                    tried.append(f"{mode}{'' if with_system else ', no system'}: {str(error)[:160]}")
                    break                                    # this mode isn't supported: try the next one
                if "ModelErrorException" in name or "ThrottlingException" in name:
                    time.sleep(2 ** attempt)
                    continue
                raise
            blocks = response["output"]["message"]["content"]
            used = [b["toolUse"]["input"] for b in blocks if "toolUse" in b]
            text = " ".join(b.get("text", "") for b in blocks)
            if used:
                entries = used[0].get("entries")
                entries = entries if isinstance(entries, list) else parse_entries(json.dumps(used[0]))
            else:
                entries = parse_entries(text)
            if entries is None:
                tried.append(f"{mode}: no entries in the reply: {text[:160]}")
                continue
            WORKS[model_id] = (mode, with_system)
            return {"model": model_id, "prompt": prompt, "mode": mode, "system_prompt": with_system,
                    "image_first": image_first and photo is not None,
                    "entries": entries, "lines": lines, "stop_reason": response["stopReason"], "tries": attempt + 1,
                    "usage": response["usage"], "latency_ms": response.get("metrics", {}).get(
                        "latencyMs", round(1000 * (time.time() - started)))}
    return {"model": model_id, "prompt": prompt, "entries": [], "lines": lines, "stop_reason": "model_error",
            "error": " | ".join(tried) or "no answer", "usage": {"inputTokens": 0, "outputTokens": 0},
            "latency_ms": 0}
