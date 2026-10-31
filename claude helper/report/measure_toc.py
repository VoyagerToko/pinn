"""Find each heading's real page in a PDF of the report and compare it with toc_entries.json; write the measured pages
to toc_entries.json when asked (python measure_toc.py <pdf> [--write])."""
import json
import re
import sys

import fitz

norm = lambda s: re.sub(r"\s+", " ", s.replace("\u2019", "'")).strip()
entries = json.load(open("toc_entries.json"))
texts = [norm(p.get_text()) for p in fitz.open(sys.argv[1])]
toc_first = next(i for i, t in enumerate(texts) if t.startswith("Table of Contents"))
# the contents pages end where the first numbered section starts
body0 = next(i for i in range(toc_first + 1, len(texts)) if texts[i].startswith("1. Introduction and Research Context"))
measured, cursor, bad = [], body0, 0
for lvl, h, page in entries:
    key = norm(h)
    if h == "Abstract":
        pg = next(i for i, t in enumerate(texts) if t.startswith("Abstract"))
    elif h == "Table of Contents":
        pg = toc_first
    elif lvl == 1:
        pg = next(i for i in range(cursor, len(texts)) if texts[i].startswith(key))
    else:
        pg = next(i for i in range(cursor, len(texts)) if key in texts[i])
    if h not in ("Abstract", "Table of Contents"):
        cursor = pg
    measured.append([lvl, h, pg + 1])
    if pg + 1 != page:
        bad += 1
        print(f"  differs: {h}: contents says {page}, real page {pg + 1}")
print(f"{len(entries) - bad}/{len(entries)} entries match; {len(texts)} pages; contents on pages {toc_first + 1}-{body0}")
if "--write" in sys.argv:
    json.dump(measured, open("toc_entries.json", "w"))
    print("toc_entries.json updated")
