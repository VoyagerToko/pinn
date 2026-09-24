"""Digitise the reference dissipation curve of the 3D Taylor-Green vortex (Re = 1600) from
DeBonis (2013), NASA/TM-2013-217850, Figure 4(a) (page 13): the black "ref. soln." line, which is
the spectral DNS the HiOCFD workshops use as reference.

    python scripts/digitize_tgv_reference.py          # -> data/tgv3d_re1600/debonis2013_fig4a_ref_dissipation.csv

The page is rendered with poppler's ``pdftoppm`` (conda env ``tools``). Inside the axes box of
panel (a) every near-black pixel that is not part of the frame, the ticks or the legend belongs to
the reference curve (it is drawn last, on top of the coloured grid-study curves). Pixel columns are
mapped to t* with the axis limits 0..20 and rows to epsilon with 0..0.016, both read from the tick
labels of the figure. The result is a digitised curve, not the original data: expect about +-1 %
of full scale (1.6e-4) in epsilon and +-0.05 in t*.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "tgv3d_re1600" / "DeBonis2013_NASA_TGV.pdf"
OUT = ROOT / "data" / "tgv3d_re1600" / "debonis2013_fig4a_ref_dissipation.csv"


def render(page: int, dpi: int, pdftoppm: str) -> np.ndarray:
    import matplotlib.image as mpimg

    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "p"
        subprocess.run([pdftoppm, "-f", str(page), "-l", str(page), "-r", str(dpi), "-png", str(PDF), str(base)], check=True)
        png = sorted(Path(d).glob("p*.png"))[0]
        return mpimg.imread(str(png))[..., :3]


def _longest_run(mask_1d: np.ndarray):
    best, start, s = (0, 0, 0), None, 0
    for i, v in enumerate(np.append(mask_1d, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i - 1)
            start = None
    return best[1], best[2]


def axes_box(img: np.ndarray, region):
    """Locate the y axis (longest dark vertical line) and x axis (longest dark horizontal line) inside
    ``region`` = (r0, r1, c0, c1). Returns (top, bottom, left, right) pixel positions of the axis
    extents, i.e. of the tick marks at eps = 0.016 / 0 and t* = 0 / 20."""
    r0, r1, c0, c1 = region
    dark = img[r0:r1, c0:c1].max(axis=2) < 0.35
    col = int(np.argmax(dark.sum(axis=0)))  # y axis
    row = int(np.argmax(dark.sum(axis=1)))  # x axis
    top, _ = _longest_run(dark[:, col])
    left, right = _longest_run(dark[row, :])
    return r0 + top, r0 + row, c0 + col, c0 + right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--pdftoppm", default=str(Path.home() / "miniforge3/envs/tools/bin/pdftoppm"))
    ap.add_argument("--check", default=None, help="write an overlay PNG of the digitised curve on the figure")
    args = ap.parse_args()
    img = render(13, args.dpi, args.pdftoppm)
    H, W, _ = img.shape
    # panel (a) sits in the upper-left quarter of the page
    region = (int(0.18 * H), int(0.45 * H), int(0.10 * W), int(0.50 * W))
    top, bottom, left, right = axes_box(img, region)
    print(f"axes box rows {top}-{bottom} cols {left}-{right} (image {H}x{W})")
    t_lim, e_lim = (0.0, 20.0), (0.0, 0.016)
    # near-black pixels strictly inside the frame
    inner = img[top + 4 : bottom - 4, left + 4 : right - 4]
    black = (inner.max(axis=2) < 0.25) & (np.ptp(inner, axis=2) < 0.08)
    # drop the legend box (upper right) - it holds black text and the black line sample
    lh, lw = black.shape
    black[: int(0.35 * lh), int(0.60 * lw) :] = False
    # tick marks are black too: follow the curve column by column, taking the run of black pixels
    # closest to the previous column's position, starting from the (unambiguous) middle of the plot
    def runs(col):
        r = np.where(black[:, col])[0]
        if r.size == 0:
            return []
        splits = np.where(np.diff(r) > 1)[0] + 1
        return [seg.mean() for seg in np.split(r, splits)]

    start = lw // 2
    while not runs(start):
        start += 1
    pos = {start: min(runs(start))}
    for direction in (1, -1):
        prev = pos[start]
        j = start + direction
        while 0 <= j < lw:
            cand = runs(j)
            if cand:
                c = min(cand, key=lambda v: abs(v - prev))
                if abs(c - prev) < 25:  # a real curve moves only a few pixels per column
                    pos[j] = c
                    prev = c
            j += direction
    cols_sorted = np.array(sorted(pos))
    ts = t_lim[0] + (cols_sorted + 4) / (right - left) * (t_lim[1] - t_lim[0])
    es = e_lim[0] + (bottom - (top + 4 + np.array([pos[c] for c in cols_sorted]))) / (bottom - top) * (e_lim[1] - e_lim[0])
    # resample onto a uniform t grid
    grid = np.round(np.arange(0.0, 20.0001, 0.05), 4)
    eg = np.interp(grid, ts, es)
    i = int(np.argmax(eg))
    print(f"digitised {ts.size} columns; peak eps = {eg[i]:.5f} at t* = {grid[i]:.2f}")
    np.savetxt(OUT, np.stack([grid, eg], 1), delimiter=",", header="t,eps_ref  # digitised from DeBonis 2013 NASA/TM-2013-217850 Fig. 4(a), black ref. soln. curve", comments="")
    print("->", OUT)
    if args.check:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img[top - 40 : bottom + 40, left - 40 : right + 40], extent=(
            t_lim[0] - 40 / (right - left) * 20, t_lim[1] + 40 / (right - left) * 20,
            e_lim[0] - 40 / (bottom - top) * 0.016, e_lim[1] + 40 / (bottom - top) * 0.016), aspect="auto")
        ax.plot(grid, eg, "m--", lw=1.0, label="digitised")
        ax.legend()
        fig.savefig(args.check, dpi=150)
        print("check plot ->", args.check)


if __name__ == "__main__":
    main()
