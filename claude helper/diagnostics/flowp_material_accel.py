"""Is the 4D-FlowP acceleration the material (Lagrangian) acceleration Du/Dt or only the local dv/dt? (CPU, read-only)

Velocity and measured acceleration are smoothed alike (Gaussian, sigma in voxels) to suppress the noise that finite
differences amplify; then the measured field is regressed on (i) dv/dt and (ii) dv/dt + (v . grad) v in the systolic jet.
"""
import sys

import numpy as np
from scipy import ndimage

DX = np.array([1.7, 1.8, 1.8]) * 1e-3
DT = 0.060
z = np.load(sys.argv[1], allow_pickle=False)
v = (z["velocity"] / 100.0).astype(np.float32)
a = z["acceleration"].astype(np.float32)
mask = z["approximate_vessel_mask"].astype(bool)
T = v.shape[-1]
inner = ndimage.binary_erosion(mask, iterations=2)
speed = np.sqrt((v**2).sum(0))

for sigma in (1.0, 1.5):
    def sm(f):
        w = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
        out = np.empty_like(f)
        for i in range(3):
            for t in range(T):
                out[i, ..., t] = ndimage.gaussian_filter(f[i, ..., t] * mask, sigma) / np.maximum(w, 1e-3)
        return out

    vs, as_ = sm(v), sm(a)
    dvdt = np.gradient(vs, DT, axis=-1)
    conv = np.zeros_like(vs)
    for i in range(3):
        gi = [np.gradient(vs[i], DX[j], axis=j) for j in range(3)]
        conv[i] = sum(vs[j] * gi[j] for j in range(3))
    sel = np.zeros(mask.shape + (T,), bool)
    for t in (2, 3, 4, 5, 6):
        sel[..., t] = inner & (speed[..., t] > 0.4)
    print(f"sigma {sigma} voxels, {int(sel.sum())} systolic jet voxel-frames")
    for name, d in (("local dv/dt", dvdt), ("convective (v.grad)v", conv), ("material dv/dt + (v.grad)v", dvdt + conv)):
        m = np.concatenate([as_[i][sel] for i in range(3)])
        x = np.concatenate([d[i][sel] for i in range(3)])
        slope = float(np.dot(x, m) / np.dot(x, x))
        r = float(np.corrcoef(m, x)[0, 1])
        print(f"  measured vs {name:28s}: corr {r:+.3f}  slope {slope:+.3f}  RMS model {np.sqrt(np.mean(x**2)):6.1f}  measured {np.sqrt(np.mean(m**2)):6.1f} m/s^2")
