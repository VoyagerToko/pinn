"""Plagiarism self-check for the paper: report every run of N consecutive words (default 8) that the paper
shares with the handbook or with any external/*/README.md.

    python scripts/check_overlap.py                 # paper/sections/*.tex vs docs/PINN_Implementation_Handbook.md + external/*/README.md
    python scripts/check_overlap.py --n 6           # stricter

Text is compared after removing LaTeX commands, math, citations, markdown/code markup and punctuation, and
lower-casing, so formatting differences do not hide a copied sentence. Exit status 1 if anything matches.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean_tex(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", " ", s)  # comments
    s = re.sub(r"\\begin\{(equation|align|table|table\*|figure|figure\*|tabular)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", s, flags=re.S)  # math
    s = re.sub(r"\\(cite|ref|label|eqref|includegraphics|input|url|pending)\*?(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)  # remaining commands (their text arguments are kept)
    return s


def clean_md(s: str) -> str:
    s = re.sub(r"```.*?```", " ", s, flags=re.S)  # code blocks
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    return s


def words(s: str):
    s = s.lower().replace("’", "'")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s)


def shingles(ws, n):
    return {tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--paper", nargs="*", default=None)
    a = ap.parse_args()
    paper_files = [Path(p) for p in a.paper] if a.paper else sorted((ROOT / "paper").rglob("*.tex"))
    sources = [ROOT / "docs" / "PINN_Implementation_Handbook.md"] + sorted((ROOT / "external").glob("*/README.md"))
    src = {}
    for f in sources:
        src[f] = shingles(words(clean_md(f.read_text(errors="ignore"))), a.n)
    hits = 0
    for pf in paper_files:
        ws = words(clean_tex(pf.read_text()))
        for i in range(len(ws) - a.n + 1):
            g = tuple(ws[i : i + a.n])
            for f, sh in src.items():
                if g in sh:
                    hits += 1
                    print(f"{pf}: '{' '.join(g)}'  <- {f.relative_to(ROOT)}")
    print(f"{hits} shared {a.n}-word sequences across {len(paper_files)} paper files and {len(sources)} sources")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
