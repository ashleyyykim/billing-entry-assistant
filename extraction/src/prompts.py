"""What the model is told for every input except note photos (one_flow.py).

One pipeline and one schema, with instructions and worked examples chosen by input type. The input type is known from
the upload, so there's no classification step. The system prompt is markdown in four parts:
1. The task and the fields (the same for every input).
2. What counts as an entry, what never does, and how to read hours (minutes, clock ranges, words, tenths).
3. The habits of this input type.
4. Two or three worked examples of this input type.

Note photos use the photo prompt in model_readers.system_prompt. Every example uses made-up clients and CMIDs that
appear in no sample or answer key (tests/test_one_flow.py checks), so no graded input is part of a prompt.
"""
import json

from model_readers import entry_schema as base_schema, line_table

EXAMPLE_CLIENTS = ["Ostrander Labs", "Pemberton Holdings", "Quillfeather", "Tamsin Robotics", "Elmstead Partners",
                   "Corvane Biologics"]
EXAMPLE_CMIDS = ["3170.002", "4412.001", "5288.004", "6631.003", "2847.001", "9461.001", "1904.002"]

TASK = """# Task
You read what lawyers send about their time and turn it into billing entries for a law firm. The text has already been taken out of the upload and split into numbered lines. Decide which lines make up each time entry and what each field says, and record every entry with record_entries."""

CORE = """# What a time entry is
- One piece of work: an amount of time, usually a client name or a client-matter number (CMID: 4 digits, a dot, 3 digits, like 3170.002), and a few words on what was done.
- Each amount of time is its own entry. Two tasks with two amounts, on one line, in one sentence, or in one bullet, are two entries.
- A table row with an empty hours cell is still an entry (record it with no hours, so a person can fill it in). A task with no amount anywhere else is not an entry.
- Firm time (admin, business development, recruiting, training, CLE) is an entry with "Firm" as the client. Firm matters have CMIDs too: when one is written, keep it.

# Never an entry
Greetings, sign-offs, signatures, names and titles, phone numbers, extensions, addresses, quoted or forwarded earlier messages (lines starting with ">", "On ... wrote:", "Original Message"), titles and headings, column headers, page numbers, totals and subtotals, rates, dollar amounts, task or activity codes (like L120 or A103), expenses, and anything crossed out or taken back.

# Hours
- "hours_written": the amount exactly as it appears in the lines ("1 hr 15 min", "2-3:30pm", "point four", "45", "1.25", "1,3").
- "hours": what it means in decimal hours.
  - Minutes divide by 60: "45 min" is 0.75, "1 hr 15 min" is 1.25.
  - A clock range is how long it lasted: "2-3:30pm" is 1.5, "9:00-9:45am" is 0.75.
  - Words become numbers: "half an hour" is 0.5, "an hour and a half" is 1.5, "point four" is 0.4, "six tenths" is 0.6.
  - Tenths of an hour divide by 10 (see the notes for each input on telling units apart).
  - Keep the precision written: 1.25 stays 1.25.
  - When the writer corrects an amount ("1.5, sorry, 1.2", "~~.5~~ .6"), use the correction.
- Leave "hours" out when no amount is written for the entry.

# The other fields
- "client": the client as written, fixing only clear typos. Don't expand an abbreviation or swap in a different client ("NGB" stays "NGB"). Empty if none is written.
- "cmid": only the number (4 digits, a dot, 3 digits), as written or from the matter heading the entry falls under. When one cell or phrase holds both a client and a CMID ("Acme Corp / 3170.002"), put each in its own field. Empty if no number is written.
- "matter": the matter's name when one is written ("Series A", "Fund IV Formation", "Lease renewal"), apart from the client and the CMID. A matter name is never a CMID. Empty if none.
- "date": the date or day exactly as written, weekday included ("Thu 11/19", "11/9", "Mon", "yesterday", "20261104"), or from the date heading the entry falls under. Empty if none applies.
- "narrative": the work done, without the client, CMID, matter name, hours, date, or task codes.
- "billable": false only when the text marks this entry as not billable or no charge (a "Billable" or "Chargeable" column saying No or N, "N/C", "no charge", "non-billable", "don't bill"). Otherwise true.
- "lines": the ids of every line the entry uses, including its heading, its date, and the line with its hours. "hours_written" must appear in those lines.
- "unsure": a short reason only when you had to guess something that changes the entry: an amount that could be read two ways, a unit the header doesn't settle, a client or CMID you couldn't make out, which heading an entry falls under, a value you took from the page image instead of the lines. Leave it "" when the text is clear.
  - Not guesses, so leave "unsure" empty: converting minutes, words, a clock range, or a tenths column; fixing a clear misread (a comma for a decimal point, an O for a 0 in a CMID); carrying a ditto mark down.
- Never fill in a value that isn't in the text or the page image."""

TIMEKEEPER = """- "timekeeper": whose time it is, only when the text names them for this entry: a "Timekeeper:", "Attorney:", or "Name:" field, or a name or email column. Write it as it appears ("R. Patel", "Patel, R.", "rpatel@firm.com"). Initials alone ("RP") are not enough. Leave it "" otherwise; never use the sender, the assistant, or a name in a signature."""

GUIDES = {
    "EMAIL": """# This input: an email
- The sender is writing to their assistant about their own time. The entries are in the body.
- Not entries: the greeting, sign-off, signature block (name, title, phone, extension, suite, address), disclaimers, and any quoted or forwarded earlier message, including amounts mentioned in it.
- When the email forwards an attorney's own note ("---------- Forwarded message ---------", "From: ..."), the note's entries count; the forwarder's own lines ("see below", "FYI") don't.
- Days are written as said ("today", "yesterday", "Monday"); the email's date settles them later.""",
    "TEXT": """# This input: a typed note or text messages
- Short and informal, often lowercase, with abbreviations: "w/" (with), "re" (about), "rev" (review), "tc" (telephone call), "mtg" (meeting), "agmt" (agreement).
- One message can hold several entries. In a list, each line with an amount is an entry.
- A later message can correct an earlier one ("actually make the nda .6"): record the entry once, with the corrected amount.
- Messages that aren't about time worked ("running late", "call me", "thx") are not entries, even when they have a number or a clock time in them.""",
    "VOICE": """# This input: a transcript of a voice memo
- The transcript is automatic, so numbers and names can come out wrong. Read what the speaker meant:
  - Spoken numbers: "point four" is 0.4, "two point three" is 2.3, "six tenths" is 0.6, "an hour" is 1.0.
  - Hours can come out with stray marks: "1.4%" or "1. 4" is 1.4, "2.one" is 2.1.
  - A CMID always has exactly 3 digits after the dot. When more digits follow straight after it, they are the hours: "3170.0021.4" is CMID 3170.002 and 1.4 hours (say so in "unsure").
  - A CMID can be split or spelled out: "31 70.002" or "3170 dot 002" is 3170.002.
- When the speaker corrects themselves ("1.5, sorry, 1.2"), use the correction.
- "Same client", "for them", and "another .3 on that" refer to the entry before.
- Fillers and asides ("um", "okay, my time for today", "that's it") are not entries.
- Days are written as said ("Monday", "yesterday"); work with no day said leaves "date" empty.""",
    "PDF": """# This input: a typed document (a time report, pre-bill, or time log)
- Table rows show their cells separated by " | ", under a header line naming the columns (Date, Matter, Description, Hours). Each data row is one entry; a line that only continues the description above is part of that entry. A row with an empty hours cell is still an entry, with no hours.
- A matter heading (a CMID, a client, or a matter name on its own line, as on a pre-bill) applies to the entries under it until the next heading. A title naming the client applies to every entry.
- A "Client / Matter" cell can hold a client and a matter name ("Ostrander Labs / Series B"): the client goes in "client" and the name in "matter".
- A "Timekeeper:" or "Attorney:" line, or a name column, says whose time each entry is.
- Logs written as text: each line or bullet with an amount is an entry; a bullet with two amounts is two entries, each with its own task.
- A clock range is the time worked ("10-11:30am board meeting" is 1.5 hours).
- Not entries: the title, run date, column headers, page numbers, totals and subtotals, rates, amounts billed, and expenses. Task or activity codes (L120, A103) are not hours and stay out of the narrative.""",
    "SPREADSHEET": """# This input: a spreadsheet of time entries
- The header row names the columns; each data row below it is one entry. Rows above the header (a title, a run date) and total rows are not entries.
- Use the header to tell what each column holds. Hours: "Hours", "Hrs", "Time", "Qty", "Duration". Minutes: "Minutes", "Mins". Timekeeper: "Timekeeper", "TK", "Atty", "User", "Name", or an email address. Client and matter: "Client", "Matter", "Client/Matter", "Matter No", "CMID". Billable: "Billable", "Chargeable", "Bill?", "N/C".
- A "Minutes" column divides by 60.
- A "Units" column:
  - In a LEDES e-billing file (column names like LINE_ITEM_NUMBER_OF_UNITS), units are hours: "2" is 2.0 hours, "1.5" is 1.5.
  - Elsewhere, units are tenths of an hour when the values show it: whole numbers with some of 10 or more (13 is 1.3, 5 is 0.5), or a header or note that says tenths.
  - When every value is a small whole number (1 to 9) and nothing says which, read them as tenths and say in "unsure" that they could be hours.
- A cell with a client and a CMID ("Acme Corp / 3170.002") gives both. A cell with a client and a matter name ("Acme Corp / Series A") gives the client and the matter.
- Columns that aren't part of the entry (rate, amount, status) are left out. A row marked not billable is still an entry, with "billable" false.""",
    "SHEET": """# This input: a scanned time sheet
- A printed form filled in by hand or typed. The printed column headings (Date, Client / Matter, Description, Hours) say what each column holds; they and other printed labels are not entries.
- Each filled-in row is an entry. A row with only words continues the description of the row above. A ditto mark (") or "same" repeats the value above it.
- A "Name:" or "Timekeeper:" field at the top says whose time it is.
- The handwriting was read by a machine, and you also see the page:
  - A decimal point can come out as a comma (",4" or "1,3"), and letters can swap into CMIDs ("3l70.0O2"). Fix these clear misreads and leave "unsure" empty.
  - A small decimal point can be dropped: the lines say "4" where the page shows ".4". Give the hours the page shows, keep "hours_written" as the lines have it, and say in "unsure" that the decimal point came from the page.
- Leave a client as written, even abbreviated or misspelled.""",
    "SCREENSHOT": """# This input: a screenshot of an email or message
- The lines include the app's own text: buttons and labels ("Reply", "Forward", "Inbox"), the sender line, and times sent. These are not entries.
- Otherwise read it like an email: greetings, signatures, and quoted earlier messages (often indented or after "On ... wrote:") are not entries, even when they mention hours.""",
}
GUIDES["DOC"] = GUIDES["PDF"]

TYPED_LINES = "Lines of {kind}, each with its id."
POSITIONED_LINES = ("Lines read from {kind}. Each line shows its id, where it starts (percent across and down), the "
                    "reader's confidence (0 to 100), and its text.")
KINDS = {"EMAIL": "an email about time worked", "TEXT": "a typed note or text messages about time worked",
         "VOICE": "an automatic transcript of a voice memo about time worked",
         "PDF": "a typed document (a time report, pre-bill, or time log)",
         "DOC": "a typed document (a time log or report)",
         "SPREADSHEET": "a spreadsheet of time entries, cells separated by \" | \", title and header rows first",
         "SHEET": "a scanned time sheet", "SCREENSHOT": "a screenshot of an email or message"}

MATTER_FIELD = ("The matter's name when one is written (\"Series A\", \"Fund IV Formation\"), apart from the client "
                "and the CMID. Never a CMID. Empty if none.")
BILLABLE_FIELD = ("false only when the text marks this entry not billable or no charge (\"No\" in a Billable column, "
                  "\"N/C\", \"no charge\"). Otherwise true.")


def entry_schema(with_timekeeper):
    """Prompt v4's schema: model_readers' fields (and timekeeper, when asked) plus matter and billable."""
    schema = json.loads(json.dumps(base_schema(with_timekeeper)))
    item = schema["properties"]["entries"]["items"]
    item["properties"]["matter"] = {"type": "string", "description": MATTER_FIELD}
    item["properties"]["billable"] = {"type": "boolean", "description": BILLABLE_FIELD}
    item["required"] = item["required"] + ["matter", "billable"]
    return schema


def intro(input_type):
    """The line before the lines in the user message, as in model_readers.NOTE_INTRO."""
    template = POSITIONED_LINES if input_type in ("SHEET", "SCREENSHOT") else TYPED_LINES
    return template.format(kind=KINDS[input_type])


def at(n, across, down, conf, text):
    return {"id": n, "across": across, "down": down, "conf": conf, "text": text}


def typed(*texts):
    return [{"id": n, "text": t} for n, t in enumerate(texts, 1)]


def entry(lines, hours_written, hours, narrative, client="", cmid="", date="", unsure="", timekeeper=None, matter="",
          billable=True):
    out = {"lines": lines, "client": client, "cmid": cmid, "matter": matter, "date": date,
           "hours_written": hours_written}
    if hours is not None:
        out["hours"] = hours
    out.update({"narrative": narrative, "billable": billable, "unsure": unsure})
    if timekeeper is not None:
        out["timekeeper"] = timekeeper
    return out


EXAMPLES = {
    "EMAIL": [
        (typed("Dana -", "Thursday's time below.",
               "Ostrander Labs: markup of the supply agreement (1 hour 45 minutes)",
               "3170.002: call w/ opposing counsel .4, then drafted the follow-up email .2",
               "Firm admin 1904.002: expense reports .3", "Best,", "Marcus Lee | Associate",
               "T: (415) 555-0182 x214 | 100 Main St, Suite 1200"),
         [entry([2, 3], "1 hour 45 minutes", 1.75, "markup of the supply agreement", client="Ostrander Labs",
                date="Thursday"),
          entry([2, 4], ".4", 0.4, "call w/ opposing counsel", cmid="3170.002", date="Thursday"),
          entry([2, 4], ".2", 0.2, "drafted the follow-up email", cmid="3170.002", date="Thursday"),
          entry([2, 5], ".3", 0.3, "expense reports", client="Firm", cmid="1904.002", date="Thursday")]),
        (typed("One more I forgot to send:", "Pemberton Holdings 1-2:30pm lender call, then notes to file .3",
               "Corvane Biologics - no charge - .5 quick question from their GC",
               "On Wed, Nov 4, 2026 at 8:02 AM Dana Ruiz wrote:",
               "> Please send anything missing from last week. I have 6.5 hours for you."),
         [entry([2], "1-2:30pm", 1.5, "lender call", client="Pemberton Holdings"),
          entry([2], ".3", 0.3, "notes to file", client="Pemberton Holdings"),
          entry([3], ".5", 0.5, "quick question from their GC", client="Corvane Biologics", billable=False)]),
    ],
    "TEXT": [
        (typed("quillfeather .5 rev nda", "also 5288.004 tc w/ cfo .3 and 1.1 on the disclosure sched",
               "omw, can you call the clerk at 3?", "thurs: elmstead fund docs 2",
               "actually make the nda .6 not .5"),
         [entry([1, 5], ".6", 0.6, "rev nda", client="quillfeather"),
          entry([2], ".3", 0.3, "tc w/ cfo", cmid="5288.004"),
          entry([2], "1.1", 1.1, "disclosure sched", cmid="5288.004"),
          entry([4], "2", 2.0, "fund docs", client="elmstead", date="thurs")]),
    ],
    "VOICE": [
        (typed("Okay, my time. Ostrander Labs, point six, auditors call. Then 3170.0021.4 reviewing the purchase "
               "agreement. And for them, .4 on the board consent. Tamsin Robotics, two point one, sorry, 1.8 on "
               "the option grant memo. Done."),
         [entry([1], "point six", 0.6, "auditors call", client="Ostrander Labs"),
          entry([1], "1.4", 1.4, "reviewing the purchase agreement", cmid="3170.002",
                unsure="the CMID and hours ran together in the transcript (3170.0021.4): read as 3170.002 and 1.4"),
          entry([1], ".4", 0.4, "board consent", cmid="3170.002"),
          entry([1], "1.8", 1.8, "option grant memo", client="Tamsin Robotics")]),
        (typed("Yesterday, two hours on Elmstead Partners fund documents, and 45 minutes on 9461.001 for the lease "
               "review.", "Pemberton, half an hour."),
         [entry([1], "two hours", 2.0, "fund documents", client="Elmstead Partners", date="Yesterday"),
          entry([1], "45 minutes", 0.75, "lease review", cmid="9461.001", date="Yesterday"),
          entry([2], "half an hour", 0.5, "", client="Pemberton")]),
    ],
    "PDF": [
        (typed("Professional Services Detail", "Attorney: J. Ortega",
               "Work Date | Client-Matter | Narrative | Hrs | Rate | Value",
               "11/16/2026 | 3170.002 | Review lender comments on credit | 1.30 | 650.00 | 845.00",
               "agreement; call w/ client",
               "11/17/2026 | 4412.001 | L120 Draft board minutes | 0.50 | 650.00 | 325.00",
               "11/17/2026 | 4412.001 | Call w/ auditors re year-end |  | 650.00 | ",
               "Total | | | 1.80 | | 1,170.00", "Page 1 of 1"),
         [entry([2, 3, 4, 5], "1.30", 1.3, "Review lender comments on credit agreement; call w/ client",
                cmid="3170.002", date="11/16/2026", timekeeper="J. Ortega"),
          entry([2, 3, 6], "0.50", 0.5, "Draft board minutes", cmid="4412.001", date="11/17/2026",
                timekeeper="J. Ortega"),
          entry([2, 3, 7], "", None, "Call w/ auditors re year-end", cmid="4412.001", date="11/17/2026",
                timekeeper="J. Ortega")]),
        (typed("Pre-Bill: Pemberton Holdings", "6631.003 Acquisition of Harlow Mills",
               "11/16 MLE Revise purchase agreement 2.4", "11/17 MLE Diligence call 9:00-9:45am",
               "2847.001 General Corporate", "11/17 MLE Annual consents .3"),
         [entry([1, 2, 3], "2.4", 2.4, "Revise purchase agreement", client="Pemberton Holdings", cmid="6631.003",
                matter="Acquisition of Harlow Mills", date="11/16", timekeeper=""),
          entry([1, 2, 4], "9:00-9:45am", 0.75, "Diligence call", client="Pemberton Holdings", cmid="6631.003",
                matter="Acquisition of Harlow Mills", date="11/17", timekeeper=""),
          entry([1, 5, 6], ".3", 0.3, "Annual consents", client="Pemberton Holdings", cmid="2847.001",
                matter="General Corporate", date="11/17", timekeeper="")]),
        (typed("Thu 11/19: Tamsin Robotics option plan amendment 1.1; follow-up call w/ GC .2. Quillfeather / "
               "Series B board meeting 10-11:30am. Elmstead side letter comments (A103) 30 min."),
         [entry([1], "1.1", 1.1, "option plan amendment", client="Tamsin Robotics", date="Thu 11/19",
                timekeeper=""),
          entry([1], ".2", 0.2, "follow-up call w/ GC", client="Tamsin Robotics", date="Thu 11/19", timekeeper=""),
          entry([1], "10-11:30am", 1.5, "board meeting", client="Quillfeather", matter="Series B",
                date="Thu 11/19", timekeeper=""),
          entry([1], "30 min", 0.5, "side letter comments", client="Elmstead", date="Thu 11/19", timekeeper="")]),
    ],
    "SPREADSHEET": [
        (typed("Fee Earner | Date Worked | Matter # | Units | Work Performed",
               "Ortega, Jamie | 11/16/2026 | 3170.002 | 13 | Review lender comments",
               "Ortega, Jamie | 11/16/2026 | 4412.001 | 5 | Board minutes", "Grand total |  |  | 18 | "),
         [entry([1, 2], "13", 1.3, "Review lender comments", cmid="3170.002", date="11/16/2026",
                timekeeper="Ortega, Jamie"),
          entry([1, 3], "5", 0.5, "Board minutes", cmid="4412.001", date="11/16/2026", timekeeper="Ortega, Jamie")]),
        (typed("Weekly time log | Run 11/20/2026 |  |  | ",
               "Staff | Day | Account | Duration (min) | Comment | Chargeable",
               "jortega@example.com | 2026-11-17 | Ostrander Labs - 3170.002 | 45 | Call w/ CFO re audit | Yes",
               "jortega@example.com | 2026-11-17 | Corvane Biologics / Series A | 15 | Intro call w/ founders | No",
               "jortega@example.com | 2026-11-17 | Firm 1904.002 | 30 | Associate mentoring | No"),
         [entry([2, 3], "45", 0.75, "Call w/ CFO re audit", client="Ostrander Labs", cmid="3170.002",
                date="2026-11-17", timekeeper="jortega@example.com"),
          entry([2, 4], "15", 0.25, "Intro call w/ founders", client="Corvane Biologics", matter="Series A",
                date="2026-11-17", timekeeper="jortega@example.com", billable=False),
          entry([2, 5], "30", 0.5, "Associate mentoring", client="Firm", cmid="1904.002", date="2026-11-17",
                timekeeper="jortega@example.com", billable=False)]),
        (typed("INVOICE_DATE | LAW_FIRM_MATTER_ID | LINE_ITEM_NUMBER_OF_UNITS | LINE_ITEM_DATE | TIMEKEEPER_NAME | "
               "LINE_ITEM_DESCRIPTION", "20261120 | 9461.001 | 1.5 | 20261118 | Ortega, Jamie | Lease review",
               "20261120 | 9461.001 | 2 | 20261119 | Ortega, Jamie | Lease negotiation call"),
         [entry([1, 2], "1.5", 1.5, "Lease review", cmid="9461.001", date="20261118", timekeeper="Ortega, Jamie"),
          entry([1, 3], "2", 2.0, "Lease negotiation call", cmid="9461.001", date="20261119",
                timekeeper="Ortega, Jamie")]),
    ],
    "SHEET": [
        ([at(1, 10, 5, 99, "WEEKLY TIME SHEET"), at(2, 10, 9, 98, "Name: Jamie Ortega"), at(3, 8, 15, 99, "Date"),
          at(4, 22, 15, 99, "Client / Matter"), at(5, 50, 15, 99, "Description"), at(6, 88, 15, 99, "Hours"),
          at(7, 8, 21, 91, "11/16"), at(8, 22, 21, 88, "Ostrander Labs"), at(9, 50, 21, 90, "supply agmt markup"),
          at(10, 88, 21, 84, "1,3"), at(11, 8, 27, 93, '"'), at(12, 22, 27, 90, "3170.002"),
          at(13, 50, 27, 92, "call w/ lender cnsl"), at(14, 88, 27, 87, ".4")],
         [entry([2, 7, 8, 9, 10], "1,3", 1.3, "supply agmt markup", client="Ostrander Labs", date="11/16",
                timekeeper="Jamie Ortega"),
          entry([2, 11, 12, 13, 14], ".4", 0.4, "call w/ lender cnsl", cmid="3170.002", date="11/16",
                timekeeper="Jamie Ortega")]),
        ([at(1, 10, 6, 99, "Timekeeper: J. Ortega   Week of 11/16"), at(2, 8, 14, 99, "Day"),
          at(3, 22, 14, 99, "Client"), at(4, 50, 14, 99, "Work done"), at(5, 88, 14, 99, "Time"),
          at(6, 8, 20, 92, "Tue"), at(7, 22, 20, 86, "Tamsin Robotics"), at(8, 50, 20, 90, "option grant memo"),
          at(9, 88, 20, 71, "6"), at(10, 22, 26, 95, '"'), at(11, 50, 26, 89, "revise per GC"),
          at(12, 50, 30, 88, "comments"), at(13, 88, 26, 90, "1.2")],
         [entry([1, 6, 7, 8, 9], "6", 0.6, "option grant memo", client="Tamsin Robotics", date="Tue",
                unsure="the lines have 6; the page shows a small decimal point before it (.6)",
                timekeeper="J. Ortega"),
          entry([1, 6, 10, 11, 12, 13], "1.2", 1.2, "revise per GC comments", client="Tamsin Robotics", date="Tue",
                timekeeper="J. Ortega")]),
    ],
    "SCREENSHOT": [
        ([at(1, 4, 3, 99, "Inbox"), at(2, 70, 3, 99, "10:42 AM"), at(3, 4, 8, 98, "Marcus Lee"),
          at(4, 4, 12, 97, "To: Dana Ruiz"), at(5, 4, 20, 96, "Quillfeather .7 call re licensing terms"),
          at(6, 4, 25, 96, "Firm recruiting lunch 1.0"), at(7, 4, 32, 95, "Sent from my iPhone"),
          at(8, 4, 90, 99, "Reply   Forward")],
         [entry([5], ".7", 0.7, "call re licensing terms", client="Quillfeather"),
          entry([6], "1.0", 1.0, "recruiting lunch", client="Firm")]),
        ([at(1, 4, 3, 99, "< Messages"), at(2, 40, 3, 99, "Dana Ruiz"), at(3, 6, 12, 95, "Tues 11/17"),
          at(4, 6, 17, 94, "Elmstead 1.6 closing checklist"), at(5, 6, 26, 93, "On Mon, Dana Ruiz wrote:"),
          at(6, 8, 30, 92, "Did the 2.5 on Pemberton go in?"), at(7, 6, 88, 99, "Delivered")],
         [entry([3, 4], "1.6", 1.6, "closing checklist", client="Elmstead", date="Tues 11/17")]),
    ],
}
EXAMPLES["DOC"] = EXAMPLES["PDF"]


def system_prompt(input_type, with_timekeeper):
    """Prompt v4's system prompt for one input type (not NOTE, which keeps prompt v1)."""
    core = CORE + ("\n" + TIMEKEEPER if with_timekeeper else "")
    shown = []
    for n, (lines, entries) in enumerate(EXAMPLES[input_type], 1):
        entries = [{k: v for k, v in e.items() if k != "timekeeper" or with_timekeeper} for e in entries]
        if with_timekeeper:
            entries = [{**e, "timekeeper": e.get("timekeeper", "")} for e in entries]
        shown.append(f"## Example {n}\n{intro(input_type)}\n{line_table(lines)}\n\nAnswer:\n"
                     f"{json.dumps({'entries': entries})}")
    return "\n\n".join([TASK, core, GUIDES[input_type], "# Examples\n"
                        "Each shows the lines as you'll see them and the answer to record. The clients and numbers "
                        "are made up.", *shown])
