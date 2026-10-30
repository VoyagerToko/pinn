"""Look for the catheter-location gel markers in a case's 3D GRE anatomy scan (CPU, read-only).

The publication located the catheter slices "by gel markers visible on MRI". This loads the 3D spoiled gradient echo
series, writes maximum-intensity projections along the three patient axes to a PNG, and marks where the scan's
catheter planes (prepared aao_mm / dao_mm, the 2D-PC inlet and outlet planes) sit, so the markers can be compared with
the planes by eye.

    python "claude helper/diagnostics/gre_markers.py" P1_baseline out.png
"""
import sys
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

case, out = sys.argv[1], sys.argv[2]
files = sorted(p for p in (ROOT / "data/stanford_coa/Imaging" / case / "3D_GRE_highRes").rglob("*") if p.is_file())
files = [f for f in files if "ORIGINAL" in list(pydicom.dcmread(f, stop_before_pixels=True).get("ImageType", []))]  # not the MPR copy
slices = []
for f in files:
    d = pydicom.dcmread(f)
    if getattr(d, "NumberOfFrames", 1) and int(getattr(d, "NumberOfFrames", 1)) > 1:  # enhanced multi-frame: one file
        arr = d.pixel_array.astype(np.float32)
        pf = d.PerFrameFunctionalGroupsSequence
        pos = np.array([np.array(p.PlanePositionSequence[0].ImagePositionPatient, float) for p in pf])
        sh = d.SharedFunctionalGroupsSequence[0]
        ori = np.array((pf[0].get("PlaneOrientationSequence") or sh.PlaneOrientationSequence)[0].ImageOrientationPatient, float)
        sp = np.array((pf[0].get("PixelMeasuresSequence") or sh.PixelMeasuresSequence)[0].PixelSpacing, float)
        for k in range(arr.shape[0]):
            slices.append((pos[k], arr[k]))
    else:
        ori = np.array(d.ImageOrientationPatient, float)
        sp = np.array(d.PixelSpacing, float)
        slices.append((np.array(d.ImagePositionPatient, float), d.pixel_array.astype(np.float32)))
n = np.cross(ori[:3], ori[3:])
slices.sort(key=lambda s: float(s[0] @ n))
vol = np.stack([s[1] for s in slices])  # (k, rows, cols)
pos0 = slices[0][0]
dk = float((slices[1][0] - slices[0][0]) @ n) if len(slices) > 1 else 1.0
print(f"{case}: 3D GRE {vol.shape} (slices, rows, cols), pixel {sp} mm, slice step {dk:.2f} mm, normal {np.round(n, 3)}, "
      f"row dir {np.round(ori[:3], 3)}, col dir {np.round(ori[3:], 3)}")
print(f"  intensity p50 {np.percentile(vol, 50):.0f}, p99 {np.percentile(vol, 99):.0f}, p99.9 {np.percentile(vol, 99.9):.0f}, max {vol.max():.0f}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(1, 3, figsize=(18, 7), dpi=110)
for ax, a, name in zip(axs, (0, 1, 2), ("along slices", "along rows", "along columns")):
    mip = vol.max(axis=a)
    ax.imshow(mip, cmap="gray", vmax=np.percentile(vol, 99.9), aspect="auto")
    ax.set_title(f"{case} 3D GRE MIP {name} {mip.shape}")
fig.tight_layout()
fig.savefig(out)
print("saved", out)
