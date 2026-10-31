"""Put XML elements in schema order (table borders, updateFields) so Word opens the file without a repair prompt."""
import sys
import zipfile

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TBLPR = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
         "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]
PPR = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr", "suppressLineNumbers",
       "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
       "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
       "suppressOverlap", "jc", "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle",
       "rPr", "sectPr", "pPrChange"]


def reorder(el, order):
    rank = {n: i for i, n in enumerate(order)}
    kids = sorted(list(el), key=lambda e: rank.get(etree.QName(e).localname, len(order)))
    for e in kids:
        el.remove(e)
    for e in kids:
        el.append(e)


src, out, update_fields = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        root = etree.fromstring(data)
        for t in root.iter(f"{{{W}}}tblPr"):
            reorder(t, TBLPR)
        for pp in root.iter(f"{{{W}}}pPr"):
            reorder(pp, PPR)
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    elif item.filename == "word/settings.xml" and update_fields:
        root = etree.fromstring(data)
        uf = etree.Element(f"{{{W}}}updateFields")
        uf.set(f"{{{W}}}val", "true")
        after = next((e for e in root if etree.QName(e).localname in ("hdrShapeDefaults", "footnotePr", "endnotePr", "compat")), None)
        after.addprevious(uf) if after is not None else root.append(uf)
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    zout.writestr(item, data)
zout.close()
print("wrote", out)
