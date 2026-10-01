"""Read the Coach B workbook without Excel or optional spreadsheet dependencies.

Category flags are retained for audit, never interpreted as runtime diagnoses.
Only manifest membership + CoachA + explicit Good Form qualifies a baseline clip.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "rule-based/Deviation_Category_Mapping.xlsx"
MANIFEST = ROOT / "data/training/keypoints/manifest.csv"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_annotations(path=WORKBOOK):
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(el.itertext()) for el in
                       ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//m:sheetData/m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                column = "".join(c for c in cell.get("r") if c.isalpha())
                value, inline = cell.find("m:v", NS), cell.find("m:is", NS)
                text = value.text if value is not None else (
                    "".join(inline.itertext()) if inline is not None else "")
                values[column] = strings[int(text)] if cell.get("t") == "s" else text
            rows.append(values)
    headers = rows[0]
    result = [{headers[k]: v.strip() for k, v in row.items() if k in headers}
              for row in rows[1:] if row.get("A")]
    ids = [r["Filename"] for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation filenames")
    return result


def training_ids(manifest=MANIFEST):
    with open(manifest, newline="", encoding="utf-8-sig") as handle:
        return {Path(r["clip_path"].replace("\\", "/")).stem for r in csv.DictReader(handle)}


def baseline_ids():
    train = training_ids()
    return sorted(Path(r["Filename"]).stem for r in read_annotations()
                  if r["Participant"] == "CoachA" and r["Form Judgment"] == "Good Form"
                  and Path(r["Filename"]).stem in train)


def provenance():
    return {"annotation_source": WORKBOOK.relative_to(ROOT).as_posix(),
            "annotation_sha256": hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(),
            "baseline_selection": "CoachA, explicit Good Form, training manifest only"}


def main():
    rows = read_annotations()
    train = training_ids()
    categories = [key for key in rows[0] if key.startswith(tuple(f"C{i}_" for i in range(1, 7)))]
    report = {**provenance(), "clips": len(rows),
              "forms": dict(Counter(r["Form Judgment"] for r in rows)),
              "category_counts": {c: sum(r[c] == "1" for r in rows) for c in categories},
              "bad_without_category": [r["Filename"] for r in rows
                                       if r["Form Judgment"] == "Bad Form"
                                       and not any(r[c] == "1" for c in categories)],
              "baseline_ids": baseline_ids(),
              "unmatched_training_annotations": [r["Filename"] for r in rows
                  if Path(r["Filename"]).stem not in train]}
    out = ROOT / "rule-based/annotation_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))
    print(f"Eligible baseline clips: {len(report['baseline_ids'])}; audit: {out}")


if __name__ == "__main__":
    main()
