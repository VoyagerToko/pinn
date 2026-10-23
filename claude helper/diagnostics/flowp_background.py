"""Is there a background offset in the 4D-FlowP acceleration (as eddy currents put on the velocity)? (CPU, read-only)

Static tissue should have zero velocity and zero acceleration. This takes the voxels with signal (time-averaged image
magnitude above a threshold) outside the vessel mask dilated by 3 voxels, i.e. the static gel around the phantom, and
reports the mean and spread of the measured velocity and acceleration there per frame and component, plus a
2nd-order polynomial fit over the volume (the model used for the velocity's background correction, prep v3).

    python "claude helper/diagnostics/flowp_background.py" data/stanford_coa/Imaging/P2_baseline/4DFlowP/data.npz
"""
import sys

import numpy as np
from scipy import ndimage

z = np.load(sys.argv[1], allow_pickle=False)
mask = z["approximate_vessel_mask"].astype(bool)
mag = np.abs(z["time_averaged_data"][0])
far = ~ndimage.binary_dilation(mask, iterations=3)
thr = np.percentile(mag[mask], 20)  # brighter than the dimmest fifth of the lumen: real signal, not air
static = far & (mag > thr)
print(f"static voxels: {int(static.sum())} (vessel {int(mask.sum())}); magnitude threshold {thr:.3g}")

v = z["velocity"].astype(np.float32) / 100.0  # m/s
a = z["acceleration"].astype(np.float32)
T = v.shape[-1]
print("frame   velocity mean in static gel (m/s) x/y/z       acceleration mean (m/s^2) x/y/z      acc spread (std)")
for t in range(T):
    vm = [float(v[i, ..., t][static].mean()) for i in range(3)]
    am = [float(a[i, ..., t][static].mean()) for i in range(3)]
    asd = [float(a[i, ..., t][static].std()) for i in range(3)]
    print(f"{t:4d}    {vm[0]:+.4f} {vm[1]:+.4f} {vm[2]:+.4f}            {am[0]:+6.2f} {am[1]:+6.2f} {am[2]:+6.2f}        {asd[0]:5.1f} {asd[1]:5.1f} {asd[2]:5.1f}")

# time-averaged acceleration in the gel against a quadratic in space: a smooth offset field, as eddy currents make
X, Y, Zc = np.nonzero(static)
c = np.stack([X, Y, Zc], 1).astype(float)
c = (c - c.mean(0)) / c.std(0)
A = np.concatenate([np.ones((len(c), 1)), c, c**2, c[:, [0]] * c[:, [1]], c[:, [0]] * c[:, [2]], c[:, [1]] * c[:, [2]]], 1)
for i, name in enumerate("xyz"):
    y = a[i].mean(-1)[static]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ coef
    print(f"acc {name} (time mean in the gel): mean {y.mean():+.3f}, quadratic fit explains {1 - np.var(y - fit) / np.var(y):.2f} of its variance, "
          f"fitted field range {fit.min():+.2f} .. {fit.max():+.2f} m/s^2")
