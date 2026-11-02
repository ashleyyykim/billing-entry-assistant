"""Send pages to Amazon Textract, save its full answer, and read the saved answer back as simple tables.

Two modes:
- "sheet": AnalyzeDocument with TABLES and FORMS, for anything with columns (time sheets, printed reports).
  Returns every word on the page, the table cells, and labeled fields like "Timekeeper: R. Patel".
- "note": DetectDocumentText, every line of text, for loose notes (sticky notes, napkins, notepads).

Each page's answer is saved as <file>_<mode>_p<page>.json, and pages already saved are skipped, so each file is only
paid for once. Everything after that (page_lines, page_cells, page_fields) reads the saved JSON and needs no AWS.
AWS credentials come from the machine (aws configure locally, the server's own permissions on AWS), never the code.
"""
import json
from pathlib import Path

import boto3
import pandas as pd
import pymupdf

PHOTO_TYPES = {".jpg", ".jpeg", ".png"}
MAX_BYTES = 9_500_000   # Textract's limit is 10 MB per image


def page_images(path, dpi=200):
    """Each page as image bytes: PDF pages rendered as PNG, a photo as it is (shrunk if it's over the size limit)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return [page.get_pixmap(dpi=dpi).tobytes("png") for page in pymupdf.open(path)]
    if suffix not in PHOTO_TYPES:
        raise ValueError(f"{path.name}: Textract reads PDF, JPEG, and PNG. Convert HEIC photos to JPEG first.")
    image = path.read_bytes()
    pix = pymupdf.Pixmap(str(path))
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    while len(image) > MAX_BYTES:
        pix.shrink(1)
        image = pix.tobytes("jpeg")
    return [image]


def read_page(image, mode, client):
    """Textract's full answer for one page image."""
    if mode == "sheet":
        response = client.analyze_document(Document={"Bytes": image}, FeatureTypes=["TABLES", "FORMS"])
    elif mode == "note":
        response = client.detect_document_text(Document={"Bytes": image})
    else:
        raise ValueError('mode must be "sheet" or "note"')
    response.pop("ResponseMetadata", None)
    return response


def json_path(out_dir, path, mode, page):
    return Path(out_dir) / f"{Path(path).stem}_{mode}_p{page}.json"


def run_textract(path, out_dir, mode="sheet", region="us-west-2", dpi=200):
    """Textract on every page of a PDF or photo, saved to out_dir. Pages already saved are skipped.
    Returns the saved JSON paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    client, saved = None, []
    for n, image in enumerate(page_images(path, dpi), 1):
        out = json_path(out_dir, path, mode, n)
        if not out.exists():
            client = client or boto3.client("textract", region_name=region)
            out.write_text(json.dumps(read_page(image, mode, client), indent=1))
        saved.append(out)
    return saved


def load_pages(out_dir, path, mode="sheet"):
    """The saved Textract answers for a file, in page order."""
    files = Path(out_dir).glob(f"{Path(path).stem}_{mode}_p*.json")
    files = sorted(files, key=lambda f: int(f.stem.rsplit("_p", 1)[1]))
    return [json.loads(f.read_text()) for f in files]


# ---------------- Reading a saved page ----------------

def _children(block, blocks, kind="CHILD"):
    return [blocks[i] for rel in block.get("Relationships", []) if rel["Type"] == kind for i in rel["Ids"]]


def _words(block, blocks):
    words = [b for b in _children(block, blocks) if b["BlockType"] == "WORD"]
    text = " ".join(w["Text"] for w in words)
    conf = min((w["Confidence"] for w in words), default=None)
    handwritten = any(w.get("TextType") == "HANDWRITING" for w in words)
    return text, conf, handwritten


def _box(block):
    box = block["Geometry"]["BoundingBox"]
    return {"left": box["Left"], "top": box["Top"], "width": box["Width"], "height": box["Height"]}


def page_lines(page):
    """Every line of text on the page, in Textract's reading order: text, conf (0 to 100), handwritten, and its box
    (left, top, width, height as shares of the page)."""
    blocks = {b["Id"]: b for b in page["Blocks"]}
    rows = []
    for b in page["Blocks"]:
        if b["BlockType"] == "LINE":
            _, conf, handwritten = _words(b, blocks)
            rows.append({"text": b["Text"], "conf": conf if conf is not None else b["Confidence"],
                         "handwritten": handwritten, **_box(b)})
    return pd.DataFrame(rows, columns=["text", "conf", "handwritten", "left", "top", "width", "height"])


def page_cells(page):
    """Every table cell on the page: table number, row, col, text, conf (lowest word confidence), and its box."""
    blocks = {b["Id"]: b for b in page["Blocks"]}
    tables = [b for b in page["Blocks"] if b["BlockType"] == "TABLE"]
    rows = []
    for t, table in enumerate(tables, 1):
        for cell in _children(table, blocks):
            if cell["BlockType"] != "CELL":
                continue
            text, conf, handwritten = _words(cell, blocks)
            rows.append({"table": t, "row": cell["RowIndex"], "col": cell["ColumnIndex"], "text": text,
                         "conf": conf, "handwritten": handwritten, **_box(cell)})
    cols = ["table", "row", "col", "text", "conf", "handwritten", "left", "top", "width", "height"]
    return pd.DataFrame(rows, columns=cols).sort_values(["table", "row", "col"]).reset_index(drop=True)


def page_fields(page):
    """Labeled fields Textract found on the page ("Timekeeper:" -> "R. Patel"): key, value, conf (the lower of the two)."""
    blocks = {b["Id"]: b for b in page["Blocks"]}
    rows = []
    for b in page["Blocks"]:
        if b["BlockType"] == "KEY_VALUE_SET" and "KEY" in b.get("EntityTypes", []):
            key, key_conf, _ = _words(b, blocks)
            values = _children(b, blocks, "VALUE")
            value, value_conf, _ = _words(values[0], blocks) if values else ("", None, False)
            confs = [c for c in (key_conf, value_conf) if c is not None]
            rows.append({"key": key.rstrip(": "), "value": value, "conf": min(confs, default=None)})
    return pd.DataFrame(rows, columns=["key", "value", "conf"])


def outside_table(lines, cells):
    """Lines whose center falls outside the page's tables (header, margins, notes in the white space)."""
    if cells.empty:
        return lines
    left, top = cells["left"].min(), cells["top"].min()
    right, bottom = (cells["left"] + cells["width"]).max(), (cells["top"] + cells["height"]).max()
    x = lines["left"] + lines["width"] / 2
    y = lines["top"] + lines["height"] / 2
    return lines[(x < left) | (x > right) | (y < top) | (y > bottom)]


def table_grid(cells, table=1):
    """One table's cells laid out like the page, for a quick look."""
    grid = cells[cells["table"] == table].pivot(index="row", columns="col", values="text")
    return grid.fillna("")
