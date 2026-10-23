"""Is the 4D-FlowP acceleration on the scale its readme states (m/s^2)? (CPU, read-only)

In a straight vessel segment the cross-section mean of the axial acceleration is, to a good approximation, the time
derivative of the cross-section mean axial velocity (the convective part averages out over a straight section). Both
come from the same file. This compares, for head-feet slices through the descending and the ascending aorta, the measured
mean a_FH per frame against d/dt of the measured mean v_FH (central differences over the 60 ms frames, periodic), and
fits the ratio measured / derived. A ratio near 1 says the units are right; a constant factor says they are not.

    python "claude helper/diagnostics/flowp_accel_scale.py" data/stanford_coa/Imaging/P2_baseline/4DFlowP/data.npz
"""
import sys

import numpy as np
from scipy import ndimage

z = np.load(sys.argv[1], allow_pickle=False)
mask = z["approximate_vessel_mask"].astype(bool)
vx = z["velocity"][0].astype(np.float32) / 100.0  # FH component, m/s
ax = z["acceleration"][0].astype(np.float32)  # FH component, m/s^2
T = vx.shape[-1]
DT = 0.060
area = mask.sum((1, 2))
k_arch = int(np.argmax(area))
side = 1 if area[k_arch + 5:].sum() > area[:k_arch - 5].sum() else -1
inner = ndimage.binary_erosion(mask, iterations=1)
print(f"arch at FH slice {k_arch}; limbs towards {'+' if side > 0 else '-'}FH")
ratios = []
for depth_mm in (40, 60, 80, 100, 120, 140):
    k = k_arch + side * int(round(depth_mm / 1.7))
    if not (0 <= k < mask.shape[0]):
        continue
    lab, n = ndimage.label(mask[k])
    if n < 1:
        continue
    for L in range(1, n + 1):
        sec = (lab == L) & inner[k]
        if sec.sum() < 30:
            continue
        v = vx[k][sec].mean(0)  # (T,)
        a = ax[k][sec].mean(0)
        dvdt = (np.roll(v, -1) - np.roll(v, 1)) / (2 * DT)
        r = float(np.dot(dvdt, a) / np.dot(dvdt, dvdt))
        c = float(np.corrcoef(dvdt, a)[0, 1])
        limb = "ascending" if np.sign(v[np.argmax(np.abs(v))]) == -side else "descending"
        ratios.append(r)
        print(f"  {depth_mm:3d} mm below the arch, {limb:10s} section {int(sec.sum()):4d} voxels: peak |mean v| {np.abs(v).max():.2f} m/s, "
              f"peak |d<v>/dt| {np.abs(dvdt).max():5.1f}, peak |<a>| {np.abs(a).max():5.1f} m/s^2; corr {c:+.2f}, measured/derived {r:+.2f}")
print(f"median measured/derived ratio over the sections: {np.median(ratios):+.2f}")
