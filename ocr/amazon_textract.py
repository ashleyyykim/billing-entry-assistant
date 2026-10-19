import boto3, fitz, csv, sys

tx = boto3.client("textract", region_name="us-west-2")
doc = fitz.open(sys.argv[1])

def cell_text(cell, blocks):
    words = []
    for rel in cell.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for cid in rel["Ids"]:
                b = blocks[cid]
                if b["BlockType"] == "WORD":
                    words.append(b["Text"])
    return " ".join(words)

rows_out = []
for n, page in enumerate(doc, 1):
    png = page.get_pixmap(dpi=200).tobytes("png")
    resp = tx.analyze_document(Document={"Bytes": png}, FeatureTypes=["TABLES"])
    blocks = {b["Id"]: b for b in resp["Blocks"]}
    for t in (b for b in resp["Blocks"] if b["BlockType"] == "TABLE"):
        grid = {}
        for rel in t.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    c = blocks[cid]
                    if c["BlockType"] == "CELL":
                        grid[(c["RowIndex"], c["ColumnIndex"])] = cell_text(c, blocks)
        if not grid:
            continue
        for r in range(1, max(k[0] for k in grid) + 1):
            row = [grid.get((r, c), "") for c in range(1, max(k[1] for k in grid) + 1)]
            if any(row):
                rows_out.append([n] + row)
    print(f"page {n} done")

csv.writer(open("output.csv", "w", newline="")).writerows(rows_out)
print("saved output.csv")