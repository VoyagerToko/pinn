"""Rebuild toc_entries.json from the draft's Heading 1/2 paragraphs, keeping known page numbers (new headings get the
previous entry's page until measure_toc.py measures them)."""
import json

from docx import Document

old = {h: p for _, h, p in json.load(open("toc_entries.json"))}
entries, last = [], 1
for i, par in enumerate(Document("report_draft.docx").paragraphs):
    if i < 3 or par.style.name not in ("Heading 1", "Heading 2") or not par.text.strip():
        continue
    lvl = 1 if par.style.name == "Heading 1" else 2
    last = old.get(par.text, last)
    entries.append([lvl, par.text, last])
added = [h for _, h, _ in entries if h not in old]
dropped = [h for h in old if h not in {e[1] for e in entries}]
json.dump(entries, open("toc_entries.json", "w"))
print(len(entries), "entries; added:", added, "; dropped:", dropped)
