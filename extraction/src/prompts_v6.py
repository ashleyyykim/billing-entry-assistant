"""Prompt v6: prompt v4 (prompts.py) with its schema made to agree with it, and clearer rules for narratives,
matters, and timekeeper initials. Note photos keep their own prompt, and spreadsheets with a header row are read from
a column map (column_map.py), so this changes emails, typed notes, voice memos, typed documents, and scanned sheets.

What changes from v4, and why:
- The tool's field descriptions say what the system prompt says. v4 reused the photo prompt's descriptions, so the
  model read two instructions that disagree ("unsure" as "what was guessed or fixed" against "only real guesses"; a
  date "like 11/9" against "weekday included").
- "narrative": all of the entry's own words, across wrapped lines, and none of another entry's. On the samples Nova
  put "board call; follow-up email to GC re minutes" in the first of two entries on one line, unflagged.
- "matter": words saying what was done are the narrative, even right after the client. Both models read "Bluefin
  Robotics - equity plan grant tracker update 1.2" as a client and a matter with no narrative
  (one_flow's matter_narrative repairs it after the fact).
- Initials in a timekeeper column, or after the date on each row of a report, are the timekeeper. v4 said initials
  are never enough and its own pre-bill example left "MLE" out, so a report with several timekeepers went to whoever
  uploaded it, with nothing shown. Initials the firm can't place give unknown_tk.
- "A task with no amount anywhere else is not an entry" read as if it were about tables; reworded.
These are v5's changes (drafted Nov 15, measured Nov 16 on Nova alone: a tie, so v4 stayed) plus the narrative and
matter rules. Everything else is v4's, word for word, including the made-up example clients and CMIDs.
"""
import copy
import json

import prompts as v4
from model_readers import line_table
from prompts import intro

FIELDS = {
    "client": "The client as written, fixing only clear typos. Don't expand an abbreviation or swap in a different "
              "client. \"Firm\" for firm time. Empty if none is written.",
    "cmid": "Only the client-matter number (4 digits, a dot, 3 digits), as written or from the matter heading the "
            "entry falls under. Empty if no number is written.",
    "matter": "The name of a deal, case, or engagement (\"Series A\", \"Fund IV Formation\"), apart from the client "
              "and the CMID. Words saying what was done are the narrative, not the matter. Empty if none.",
    "date": "The date or day exactly as written, weekday included (\"Thu 11/19\", \"Mon\", \"yesterday\"), or from "
            "the date heading the entry falls under. Empty if none applies.",
    "hours_written": "The amount of time exactly as it appears in the lines (\"1 hr 15 min\", \"2-3:30pm\", \"45\", "
                     "\"1,3\"). Empty if none.",
    "hours": "What the amount means in decimal hours. Left out if no amount is written.",
    "narrative": "Every word of this entry's own work, joined across wrapped lines, and nothing from another entry; "
                 "without the client, CMID, matter name, hours, date, or task codes.",
    "unsure": "A short reason, only when you had to guess something that changes the entry. Empty when the text is "
              "clear, and empty for routine conversions and clear misreads you fixed.",
    "timekeeper": "Whose time it is, when the text names them for this entry, as written (a name, an email address, "
                  "or initials in a timekeeper column or after each row's date). Empty otherwise.",
}

CHANGES = {
    "A task with no amount anywhere else is not an entry.":
        "Outside a table, work mentioned with no amount of time is not an entry.",
    '- "matter": the matter\'s name when one is written ("Series A", "Fund IV Formation", "Lease renewal"), apart from '
    'the client and the CMID. A matter name is never a CMID. Empty if none.':
        '- "matter": the name of a deal, case, or engagement when one is written ("Series A", "Fund IV Formation", '
        '"Lease renewal"), apart from the client and the CMID. A matter name is never a CMID. Words that say what was '
        'done are the narrative, even right after the client: "Tamsin Robotics - option ledger cleanup 1.2" has the '
        'narrative "option ledger cleanup" and no matter. Empty if none.',
    '- "narrative": the work done, without the client, CMID, matter name, hours, date, or task codes.':
        '- "narrative": every word of this entry\'s own work as written, without the client, CMID, matter name, '
        'hours, date, or task codes.\n'
        '  - A description that wraps onto the next lines belongs to the entry above it until a line with its own '
        'amount, client, or CMID starts the next entry. Join the lines.\n'
        '  - Nothing from another entry. When one line or bullet holds two entries, each narrative is only the words '
        'that go with its own amount: "2-3:30pm board call; email to GC re minutes .2" gives "board call" (1.5) and '
        '"email to GC re minutes" (0.2).',
}
OLD_INITIALS = 'Initials alone ("RP") are not enough.'
NEW_INITIALS = ('Initials count when they sit in a timekeeper column or right after the date on each row of a report '
                '("11/16 MLE Revise purchase agreement 2.4"): write them as they appear ("MLE"). Initials in a '
                'signature or a greeting don\'t.')

CORE = v4.CORE
for old, new in CHANGES.items():
    assert old in CORE, old
    CORE = CORE.replace(old, new)
assert OLD_INITIALS in v4.TIMEKEEPER
TIMEKEEPER = v4.TIMEKEEPER.replace(OLD_INITIALS, NEW_INITIALS)

EXAMPLES = copy.deepcopy(v4.EXAMPLES)
for _entry in EXAMPLES["PDF"][1][1]:     # the pre-bill whose rows carry "MLE"
    _entry["timekeeper"] = "MLE"
_lines, _entries = EXAMPLES["PDF"][2]    # the log written as text: add a clock range and a second task on one line
_lines[0]["text"] += " 9-10:15am Corvane Biologics diligence call; notes to file .2"
_entries += [v4.entry([1], "9-10:15am", 1.25, "diligence call", client="Corvane Biologics", date="Thu 11/19",
                      timekeeper=""),
             v4.entry([1], ".2", 0.2, "notes to file", client="Corvane Biologics", date="Thu 11/19", timekeeper="")]
EXAMPLES["DOC"] = EXAMPLES["PDF"]
_lines, _entries = EXAMPLES["TEXT"][0]   # a "Client - description" line
_lines.append({"id": len(_lines) + 1, "text": "Tamsin Robotics - option ledger cleanup 1.2"})
_entries.append(v4.entry([len(_lines)], "1.2", 1.2, "option ledger cleanup", client="Tamsin Robotics"))


def entry_schema(with_timekeeper):
    """v4's schema (the same fields and types) with descriptions that match the system prompt."""
    schema = json.loads(json.dumps(v4.entry_schema(with_timekeeper)))
    for name, field in schema["properties"]["entries"]["items"]["properties"].items():
        if name in FIELDS:
            field["description"] = FIELDS[name]
    return schema


def system_prompt(input_type, with_timekeeper):
    """Prompt v6's system prompt for one input type, put together the way prompts.system_prompt does."""
    core = CORE + ("\n" + TIMEKEEPER if with_timekeeper else "")
    shown = []
    for n, (lines, entries) in enumerate(EXAMPLES[input_type], 1):
        entries = [{k: v for k, v in e.items() if k != "timekeeper" or with_timekeeper} for e in entries]
        if with_timekeeper:
            entries = [{**e, "timekeeper": e.get("timekeeper", "")} for e in entries]
        shown.append(f"## Example {n}\n{intro(input_type)}\n{line_table(lines)}\n\nAnswer:\n"
                     f"{json.dumps({'entries': entries})}")
    return "\n\n".join([v4.TASK, core, v4.GUIDES[input_type], "# Examples\n"
                        "Each shows the lines as you'll see them and the answer to record. The clients and numbers "
                        "are made up.", *shown])
