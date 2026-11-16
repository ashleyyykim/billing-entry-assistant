"""Step 5 of the pipeline: put task lines back together (placeholder, moved out of split_block.py).

Works on any record shaped like split_entry()'s output: {heading, lines: [{text, hours, hours_source}]}.
Once rewriting exists, pass lines whose `text` is the rewritten narrative.
"""


def _fmt(h):
    return f"{h:g}" if h is not None else "?"


def assemble_block(r, with_times=True):
    """Keep one block-billed entry (used when 'use block billing' is selected)."""
    parts = []
    for ln in r["lines"]:
        t = ln["text"].rstrip(".")
        parts.append(f"{t} ({_fmt(ln['hours'])})" if with_times and ln["hours"] is not None and len(r["lines"]) > 1 else t)
    head = (r["heading"] + " ") if r["heading"] else ""
    return head + "; ".join(parts) + "."


def assemble_separate(r):
    """One entry per task, each with its own hours (hours None where the timekeeper must supply them)."""
    return [dict(narrative=ln["text"], hours=ln["hours"], hours_source=ln["hours_source"]) for ln in r["lines"]]
