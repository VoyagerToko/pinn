"""Find the catheter-location gel markers in each case's 3D GRE scan and give their head-feet positions (CPU, read-only).

In the 3D GRE anatomy scan (1 mm voxels) the phantom sits in a bright gel block. The vessels are open tunnels through
the block; the catheter-location markers are small sealed pockets in the gel beside them: one beside the ascending
aorta (the fixed AAo catheter location) and five stacked beside the descending aorta (DAo locations 1-5; "location 3"
is the one the publication and the prepared catheter drop use). Sealed pockets are exactly the holes of the gel mask
(binary_fill_holes(gel) minus gel); air bubbles are far smaller. This lists the marker-sized pockets with their patient
(LPS) coordinates, groups them by right-left position, and writes a PNG with the pockets drawn on a projection.

    python "claude helper/diagnostics/gre_find_markers.py" P1_baseline out.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
case, png = sys.argv[1], sys.argv[2]
f = [p for p in sorted((ROOT / "data/stanford_coa/Imaging" / case / "3D_GRE_highRes").iterdir())
     if "ORIGINAL" in list(pydicom.dcmread(p, stop_before_pixels=True).ImageType)][0]
ds = pydicom.dcmread(f)
v = ds.pixel_array.astype(np.float32)  # (slices, rows, cols)
pf = ds.PerFrameFunctionalGroupsSequence
sh = ds.SharedFunctionalGroupsSequence[0]
pos = np.array([np.array(p.PlanePositionSequence[0].ImagePositionPatient, float) for p in pf])
ori = np.array((pf[0].get("PlaneOrientationSequence") or sh.PlaneOrientationSequence)[0].ImageOrientationPatient, float)
sp = np.array((pf[0].get("PixelMeasuresSequence") or sh.PixelMeasuresSequence)[0].PixelSpacing, float)
rdir, cdir = ori[:3], ori[3:]  # increasing column index, increasing row index


def lps(k, r, c):
    k0 = int(np.clip(np.floor(k), 0, len(pos) - 2))
    p = pos[k0] + (k - k0) * (pos[k0 + 1] - pos[k0])
    return p + c * sp[1] * rdir + r * sp[0] * cdir


gel = ndimage.binary_opening(ndimage.uniform_filter(v, 3) > 90, iterations=1)
holes = ndimage.binary_fill_holes(gel) & ~gel
lab, n = ndimage.label(holes)
sizes = ndimage.sum(np.ones_like(v), lab, range(1, n + 1))
pockets = []
for L in np.nonzero((sizes >= 60) & (sizes <= 5000))[0] + 1:
    kk, rr, cc = np.nonzero(lab == L)
    pt = lps(kk.mean(), rr.mean(), cc.mean())
    pockets.append({"slice": float(kk.mean()), "row": float(rr.mean()), "col": float(cc.mean()), "voxels": int(len(kk)),
                    "extent_krc": [int(np.ptp(kk) + 1), int(np.ptp(rr) + 1), int(np.ptp(cc) + 1)], "lps": pt.round(1).tolist()})
pockets.sort(key=lambda p: p["col"])
groups, cur = [], []
for p in pockets:
    if cur and abs(p["col"] - np.mean([q["col"] for q in cur])) > 15:
        groups.append(cur)
        cur = []
    cur.append(p)
if cur:
    groups.append(cur)
print(f"{case}: GRE {v.shape}; {len(pockets)} sealed pockets of 60-5000 voxels")
for g in groups:
    g.sort(key=lambda p: p["row"])
    print(f"  group at col {np.mean([p['col'] for p in g]):5.0f}: {len(g)} pockets; LPS z (top to bottom) "
          + ", ".join(f"{p['lps'][2]:.1f}" for p in g) + "; sizes " + ", ".join(str(p["voxels"]) for p in g)
          + "; extent k/r/c " + ", ".join("/".join(map(str, p["extent_krc"])) for p in g))
out = Path(png).with_suffix(".json")
out.write_text(json.dumps({"case": case, "pockets": pockets}, indent=1))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
ax.imshow(np.where(gel, v, 0).max(0), cmap="gray", vmin=0, vmax=170)
ax.imshow(np.ma.masked_where(~holes.any(0), holes.any(0)), cmap="autumn", alpha=0.5)
for p in pockets:
    ax.annotate(f"{p['lps'][2]:.0f}", (p["col"], p["row"]), color="cyan", fontsize=8)
ax.set_title(f"{case}: sealed pockets in the gel (orange), label = LPS z mm")
fig.tight_layout()
fig.savefig(png)
