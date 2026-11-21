"""Which lines of an upload each entry came from, so the page can mark them on the original.

one_flow.read asks the model which lines each entry uses (its "lines") and checks them, then leaves them out of the
entry table. The model's answers are saved while the request runs, so locate() reads them back and gives each entry
an `evidence` (its source and the lines it used), plus a `layout` of the lines the model was shown, so the page can
draw the file. For a photo each line also gets its box on the image, from Textract's saved answer.

Everything here is optional: if anything goes wrong the page just shows the file without marks.
"""
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("tenth")
PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
SIDEWAYS = {5, 6, 7, 8}     # EXIF orientations that turn the photo a quarter turn


def saved_answers(reader, source, at_least=0):
    """The model answers saved for one source, one per piece (a page, or a part of a long file), in order, with None
    for a piece this reader has no answer for (the other reader's answer was used). at_least: how many pieces to
    look for even when some are missing."""
    made = getattr(reader, "made", {}).get(source)
    if made:
        return made
    one = reader.path(source, None)
    if one.exists():
        return [json.loads(one.read_text())]
    out = []
    while len(out) < at_least or reader.path(source, len(out)).exists():
        path = reader.path(source, len(out))
        out.append(json.loads(path.read_text()) if path.exists() else None)
    return out


def repeats(answers, n):
    """How many lines at the top of piece n repeat piece 0's (a long typed file gives each later piece the same first
    lines for context). Photos and scans have positions and never repeat."""
    lines, first = (answers[n] or {}).get("lines") or [], (answers[0] or {}).get("lines") or []
    if not n or not lines or "across" in lines[0]:
        return 0
    k = 0
    while k < min(len(lines), len(first)) and lines[k]["text"] == first[k]["text"]:
        k += 1
    return k


def to_upright(orientation, box):
    """A box (left, top, width, height as shares of the image) from the stored pixels to the photo as it's shown, for
    an EXIF orientation that turns it a quarter turn."""
    left, top, width, height = box
    if orientation == 5:
        return top, left, height, width
    if orientation == 6:
        return 1 - (top + height), left, height, width
    if orientation == 7:
        return 1 - (top + height), 1 - (left + width), height, width
    if orientation == 8:
        return top, 1 - (left + width), height, width
    return box


def stored_sideways(boxes, size):
    """True when the boxes are in the photo as stored, not as shown. Text lines run wider than they are tall, so in a
    photo stored sideways they come out tall when Textract worked on the stored pixels, and wide when it turned the
    photo upright first. size: the stored photo's (width, height) in pixels."""
    aspects = sorted((w * size[0]) / (h * size[1]) for _, _, w, h in boxes if h > 0)
    return bool(aspects) and aspects[len(aspects) // 2] < 1


def cover(table, text, across, down):
    """The row numbers in `table` (Textract's lines) that one model line is made of. The model reads a row of a notebook
    as one line ("Mon. 1238.001 Attend to call with"), but Textract finds each stretch of handwriting on its own, so a
    model line is usually several Textract lines joined by spaces. across, down: where the model says the line starts
    (percent of the photo). [] when the pieces don't add up to the model's text."""
    want = " ".join(str(text).split())
    if not want:
        return []
    near = table[(table["left"] * 100 >= across - 2) & (table["top"] * 100 >= down - 2) & (table["top"] * 100 <= down + 14)]
    parts = [(i, " ".join(str(s).split())) for i, s in near["text"].items()]

    def walk(pos, used):
        if pos >= len(want):
            return used
        for i, part in parts:
            if i in used or not part or not want.startswith(part, pos):
                continue
            end = pos + len(part)
            if end < len(want) and want[end] != " ":
                continue
            found = walk(end + 1, used + [i])
            if found:
                return found
        return None
    return walk(0, []) or []


def line_boxes(path, pages_dir, mode, answers):
    """({(piece, id): [left, top, width, height]}, {(piece, id): [[word, left, top, width, height], ...]}) for the lines
    of a photo, as shares of the photo as it is shown. The words let the page mark just the word a question is about."""
    from PIL import Image
    from textract import _children, load_pages, page_lines
    pages = load_pages(pages_dir, path, mode) or load_pages(pages_dir, path, "sheet" if mode == "note" else "note")
    found, words = {}, {}
    for n, answer in enumerate(answers):
        if n >= len(pages):
            break
        table = page_lines(pages[n], drop_repeats=True)
        blocks = {b["Id"]: b for b in pages[n]["Blocks"]}
        line_blocks = [b for b in pages[n]["Blocks"] if b["BlockType"] == "LINE"]   # table's rows, in the same order
        for line in answer.get("lines") or []:
            if line.get("across") is None:
                continue
            rows = cover(table, line["text"], line["across"], line["down"])
            if not rows:   # a line that is one Textract line, in the same place
                near = table[(table["text"] == line["text"]) & ((table["left"] * 100 - line["across"]).abs() <= 1.01)
                             & ((table["top"] * 100 - line["down"]).abs() <= 1.01)]
                rows = [near.index[0]] if len(near) else []
            if rows:
                part = table.loc[rows]
                left, top = float(part.left.min()), float(part.top.min())
                found[(n, line["id"])] = [left, top, float((part.left + part.width).max()) - left, float((part.top + part.height).max()) - top]
                box = lambda w: [w["Geometry"]["BoundingBox"][k] for k in ("Left", "Top", "Width", "Height")]
                words[(n, line["id"])] = [[w["Text"], *box(w)] for i in rows for w in _children(line_blocks[i], blocks)
                                          if w["BlockType"] == "WORD"]
    if found and len(answers) == 1:
        with Image.open(path) as image:
            orientation, size = image.getexif().get(0x0112, 1), image.size
        if orientation in SIDEWAYS and stored_sideways(list(found.values()), size):
            found = {k: list(to_upright(orientation, v)) for k, v in found.items()}
            words = {k: [[w[0], *to_upright(orientation, w[1:])] for w in v] for k, v in words.items()}
    clip = lambda v: [round(min(max(x, 0), 1), 4) for x in v]
    return {k: clip(v) for k, v in found.items()}, {k: [[w[0], *clip(w[1:])] for w in v] for k, v in words.items()}


def locate(entries, name, first, second, path=None, pages_dir=None, kind=None, spreadsheet=None):
    """(entries with an `evidence` column, layout) for one upload.

    evidence: {"source": ..., "hits": [[piece, line id], ...]} for each entry the model read, else None.
    layout: {source: {"input_type", "header", "pieces": [{"lines": [{"id", "text", "box"?, "words"?}]}]}} for each source.
    first, second: the one_flow.ModelReaders that read it. path, pages_dir, kind: for a photo's line boxes.
    spreadsheet: the raw cells, to tell the header rows from the data rows."""
    from one_flow import line_ids
    out = entries.copy()
    out["evidence"] = None
    layout, answers_of = {}, {}
    for source, group in out.groupby("source", dropna=False):
        if pd.isna(source):
            continue
        source = str(source)
        count = int(group["page"].max()) if group["page"].notna().any() else 0
        ones, twos = saved_answers(first, source, count), saved_answers(second, source, count)
        count = max(len(ones), len(twos))
        ones, twos = saved_answers(first, source, count), saved_answers(second, source, count)
        answers_of[source] = {"first": ones, "second": twos}
        answers = [one or two for one, two in zip(ones, twos)] if ones and twos else ones or twos
        if not answers:
            continue
        pieces = []
        for n, answer in enumerate(answers):
            skip = repeats(answers, n)
            pieces.append({"lines": [{"id": line["id"], "text": str(line["text"])} for line in ((answer or {}).get("lines") or [])[skip:]],
                           "repeat": skip})
        item = {"input_type": str(group["input_type"].iloc[0]), "header": None, "pieces": pieces}
        if spreadsheet is not None and Path(str(source)).name == Path(str(name)).name:
            from one_flow import spreadsheet_rows
            item["header"] = len(spreadsheet_rows(spreadsheet)[0])
        if path is not None and Path(str(path)).suffix.lower() in PHOTO_TYPES and Path(str(source)).name == Path(str(path)).name:
            try:
                boxes, words = line_boxes(path, pages_dir, "note", answers)
                for n, piece in enumerate(pieces):
                    for line in piece["lines"]:
                        if (n, line["id"]) in boxes:
                            line["box"] = boxes[(n, line["id"])]
                            line["words"] = words.get((n, line["id"]), [])
            except Exception as err:  # noqa: BLE001 - the marks are optional
                log.warning("no line boxes for %s: %s: %s", source, type(err).__name__, err)
        layout[source] = item
    for i, row in out.iterrows():
        source = str(row["source"])
        reasons = str(row["check_reasons"])
        if source not in answers_of or "read_by_rules" in reasons:
            continue
        n = 0 if pd.isna(row["page"]) else int(row["page"]) - 1
        ones = answers_of[source]["first"]
        mine = ones[n] if n < len(ones) else None
        found = answers_of[source]["second" if "second_reader_only" in reasons or mine is None else "first"]
        try:
            entry = found[n]["entries"][int(float(row["rows"])) - 1]
        except (IndexError, ValueError, TypeError, KeyError):
            continue
        pieces = layout.get(source, {}).get("pieces", [])
        hits = []
        for line_id in line_ids(entry):
            if n < len(pieces) and n and line_id <= pieces[n]["repeat"]:
                hits.append([0, line_id])
            else:
                hits.append([n, line_id])
        if hits:
            out.at[i, "evidence"] = {"source": source, "hits": hits}
    return out, layout
