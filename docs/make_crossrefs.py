#!/usr/bin/env python
"""Convert pandoc internal hyperlinks in report.docx into Word
cross-reference fields.

  - w:hyperlink w:anchor="ref_N" / "fig_N" / "tab_N"
      -> w:fldSimple w:instr=" REF <anchor> \\h "  (a real Word 交叉引用域,
         displays the bookmarked label, clickable, updatable via F9)
  - w:hyperlink w:anchor="sec*" stays a plain internal hyperlink (a REF
    field would render the whole heading text), but the Hyperlink character
    style is stripped so it reads as normal body text.

Run after pandoc and style_tables.py:
  code/.venv/bin/python docs/make_crossrefs.py
"""
import shutil
import zipfile
from pathlib import Path

from lxml import etree

DOCS = Path(__file__).resolve().parent
DOCX = DOCS / "report.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def qn(tag: str) -> str:
    return f"{{{W}}}{tag}"


def strip_hyperlink_style(run) -> None:
    rpr = run.find(qn("rPr"))
    if rpr is None:
        return
    for rstyle in rpr.findall(qn("rStyle")):
        if rstyle.get(qn("val")) == "Hyperlink":
            rpr.remove(rstyle)


def main() -> None:
    tmp = DOCX.with_suffix(".crossrefs.tmp")
    with zipfile.ZipFile(DOCX) as zin:
        items = {name: zin.read(name) for name in zin.namelist()}

    root = etree.fromstring(items["word/document.xml"])
    bookmarks = {b.get(qn("name")) for b in root.iter(qn("bookmarkStart"))}

    n_field = n_link = 0
    missing: set[str] = set()
    for hl in list(root.iter(qn("hyperlink"))):
        anchor = hl.get(qn("anchor"))
        if not anchor:
            continue
        if anchor not in bookmarks:
            missing.add(anchor)
        runs = list(hl.findall(qn("r")))
        for r in runs:
            strip_hyperlink_style(r)
        if anchor.split("_")[0] in ("ref", "fig", "tab"):
            fld = etree.Element(qn("fldSimple"))
            fld.set(qn("instr"), f" REF {anchor} \\h ")
            for r in runs:
                fld.append(r)
            hl.getparent().replace(hl, fld)
            n_field += 1
        else:
            n_link += 1

    items["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True)

    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in items.items():
            zout.writestr(name, data)
    shutil.move(tmp, DOCX)

    print(f"REF fields inserted: {n_field}; plain internal links kept: {n_link}")
    if missing:
        print("WARNING anchors without bookmarks:", sorted(missing))


if __name__ == "__main__":
    main()
