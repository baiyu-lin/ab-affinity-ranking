#!/usr/bin/env python
"""Post-process report.docx: restyle all tables as academic
three-line tables (booktabs style: thick top/bottom rules, thin rule
under the header row, no vertical or inner horizontal borders).

Run after pandoc:
  pandoc report.md -o report.docx \
      --reference-doc=assets/reference.docx --resource-path=.
  code/.venv/bin/python docs/style_tables.py
"""
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX = Path(__file__).resolve().parent / "report.docx"


def _border(tag, val, sz):
    e = OxmlElement(f"w:{tag}")
    e.set(qn("w:val"), val)
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:color"), "000000")
    return e


def three_line(table):
    tblPr = table._tbl.tblPr
    for el in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(el)
    borders = OxmlElement("w:tblBorders")
    borders.append(_border("top", "single", 12))     # 1.5 pt
    borders.append(_border("bottom", "single", 12))  # 1.5 pt
    for tag in ("left", "right", "insideH", "insideV"):
        borders.append(_border(tag, "none", 0))
    tblPr.append(borders)
    # thin rule under the header row (0.75 pt)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        for el in tcPr.findall(qn("w:tcBorders")):
            tcPr.remove(el)
        tcB = OxmlElement("w:tcBorders")
        tcB.append(_border("bottom", "single", 6))
        for tag in ("top", "left", "right"):
            tcB.append(_border(tag, "none", 0))
        tcPr.append(tcB)


doc = Document(DOCX)
n = 0
for t in doc.tables:
    three_line(t)
    n += 1
doc.save(DOCX)
print(f"restyled {n} tables -> {DOCX}")
