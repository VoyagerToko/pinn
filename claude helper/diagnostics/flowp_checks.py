"""4D-FlowP checks on one case (CPU, read-only).

1. Convention: velocity component signs by the smallest divergence at peak flow; measured acceleration against the
   acceleration derived from the same scan's velocity, a_v = dv/dt + (v . grad) v, per component in the systolic jet.
2. Classical estimate: p(upstream) - p(downstream) = rho * integral of a . dl along a lumen path from the ascending to the
   descending aorta at matched heights below the arch, with the measured a and with a_v.
"""
import heapq
import sys

import numpy as np
from scipy import ndimage

RHO = 1100.0  # kg/m^3, the phantom fluid (STATUS: 1.1 g/cm^3)
MMHG = 133.322
DX = np.array([1.7, 1.8, 1.8]) * 1e-3  # m, readme: resolution (x, y, z) = (FH, RL, AP)
DT = 0.060  # s, readme: time resolution 60 ms

z = np.load(sys.argv[1], allow_pickle=False)
v = (z["velocity"] / 100.0).astype(np.float32)  # cm/s -> m/s, (3, X, Y, Z, T)
a = z["acceleration"].astype(np.float32)  # m/s^2
mask = z["approximate_vessel_mask"].astype(bool)
T = v.shape[-1]
speed = np.sqrt((v**2).sum(0))  # (X, Y, Z, T)
peak = int(np.argmax([np.percentile(speed[..., t][mask], 99) for t in range(T)]))
print(f"grid {mask.shape}, {T} frames, vessel voxels {mask.sum()}, peak-flow frame {peak}")

# ---- 1a. velocity sign convention: divergence with each sign combination at the peak frame
inner = ndimage.binary_erosion(mask, iterations=2)
best = []
for sx in (1, -1):
    for sy in (1, -1):
        for sz in (1, -1):
            s = np.array([sx, sy, sz])
            div = sum(np.gradient(s[i] * v[i, ..., peak], DX[i], axis=i) for i in range(3))
            best.append((float(np.sqrt(np.mean(div[inner] ** 2))), (sx, sy, sz)))
best.sort()
print("velocity divergence RMS (1/s) at peak, best sign combinations:", [(round(r, 1), s) for r, s in best[:3]], "worst", round(best[-1][0], 1))

# ---- 1b. measured vs velocity-derived acceleration
dvdt = np.gradient(v, DT, axis=-1)
conv = np.zeros_like(v)
for i in range(3):
    gi = [np.gradient(v[i], DX[j], axis=j) for j in range(3)]  # d v_i / d x_j, (X, Y, Z, T)
    conv[i] = sum(v[j] * gi[j] for j in range(3))
a_v = dvdt + conv
sys_frames = [t for t in range(T) if np.percentile(speed[..., t][mask], 99) > 0.5 * np.percentile(speed[..., peak][mask], 99)]
sel = np.zeros(mask.shape + (T,), bool)
for t in sys_frames:
    sel[..., t] = inner & (speed[..., t] > 0.4)
print(f"systolic frames {sys_frames}; jet sample: {int(sel.sum())} voxel-frames with |v| > 0.4 m/s inside the eroded mask")
for i, name in enumerate(("x (FH)", "y (RL)", "z (AP)")):
    m, d = a[i][sel], a_v[i][sel]
    slope = float(np.dot(d, m) / np.dot(d, d))
    corr = float(np.corrcoef(m, d)[0, 1])
    print(f"  a_{name}: corr(measured, derived) {corr:+.3f}, slope measured/derived {slope:+.3f}, RMS measured {np.sqrt(np.mean(m**2)):.1f}, derived {np.sqrt(np.mean(d**2)):.1f} m/s^2")
noise = a[:, ..., [t for t in range(T) if t not in sys_frames]][:, inner[..., None].repeat(T - len(sys_frames), -1)]
print(f"  diastolic measured acceleration, per-component RMS (mostly noise): {np.sqrt(np.mean(noise**2)):.1f} m/s^2")

# ---- 2. path integral from the ascending to the descending aorta
# the arch is where the two limbs join; find the FH axis direction and the limbs by components per FH slice
lab_counts = [ndimage.label(mask[k])[1] for k in range(mask.shape[0])]
two = [k for k, n in enumerate(lab_counts) if n >= 2]
print(f"FH slices with >= 2 vessel sections: {len(two)} of {mask.shape[0]} (first {two[:1]}, last {two[-1:]})")
area = mask.sum((1, 2))
k_arch = int(np.argmax(area))  # the widest FH slice ~ the arch, where the limbs join horizontally
print(f"widest FH slice (arch): {k_arch}")
dist = ndimage.distance_transform_edt(mask)
idx = {tuple(p): n for n, p in enumerate(np.argwhere(mask))}
pts = np.argwhere(mask)
offs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


def path(src, dst):
    s, d = idx[tuple(src)], idx[tuple(dst)]
    best_, prev, pq = {s: 0.0}, {}, [(0.0, s)]
    while pq:
        c, n = heapq.heappop(pq)
        if n == d:
            break
        if c > best_.get(n, np.inf):
            continue
        for o in offs:
            q = tuple(pts[n] + o)
            m = idx.get(q)
            if m is None:
                continue
            w = np.linalg.norm(o * DX) / (dist[q] ** 2)
            if c + w < best_.get(m, np.inf):
                best_[m], prev[m] = c + w, n
                heapq.heappush(pq, (c + w, m))
    out = [d]
    while out[-1] != s:
        out.append(prev[out[-1]])
    return pts[out[::-1]]


def smooth(f):  # 3x3x3 mean inside the mask, per frame and component
    w = ndimage.uniform_filter(mask.astype(np.float32), 3)
    return np.where(w[None, ..., None] > 0, ndimage.uniform_filter(f * mask[None, ..., None], size=(1, 3, 3, 3, 1)) / np.maximum(w, 1e-6)[None, ..., None], 0)


a_s, av_s = smooth(a), smooth(a_v)
sign_up = None
for depth_mm in (30, 50, 70):
    side = 1 if (k_arch + 1 < mask.shape[0] and area[k_arch + 5:].sum() > area[:k_arch - 5].sum()) else -1  # limbs extend this way
    k = k_arch + side * int(round(depth_mm / 1.7))
    if not (0 <= k < mask.shape[0]):
        continue
    lab, n = ndimage.label(mask[k])
    if n < 2:
        print(f"depth {depth_mm} mm: only {n} section at FH slice {k}")
        continue
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    two_big = np.argsort(sizes)[::-1][:2] + 1
    cents = []
    for L in two_big:
        cc = np.argwhere(lab == L)
        c = cc[np.argmax(dist[k][lab == L])]  # the most central voxel of that section
        flow = float(v[0, k][lab == L][:, peak].mean())  # FH velocity at peak, sign tells the direction
        cents.append((np.array([k, c[0], c[1]]), flow, int(sizes[L - 1])))
    # ascending = flow towards the arch: the FH velocity sign points from slice k towards k_arch
    towards = np.sign(k_arch - k)
    cents.sort(key=lambda c: -c[1] * towards)
    up, down = cents[0], cents[1]
    p = path(up[0], down[0])
    seg = np.diff(p, axis=0) * DX
    mids = p[:-1]
    L_mm = float(np.linalg.norm(seg, axis=1).sum() * 1e3)
    dpm = np.array([RHO * sum(a_s[:, m[0], m[1], m[2], t] @ s for m, s in zip(mids, seg)) for t in range(T)]) / MMHG
    dpv = np.array([RHO * sum(av_s[:, m[0], m[1], m[2], t] @ s for m, s in zip(mids, seg)) for t in range(T)]) / MMHG
    print(f"\ndepth {depth_mm} mm below the arch (FH slice {k}): ascending section {up[2]} voxels (v_FH {up[1]:+.2f} m/s), "
          f"descending {down[2]} (v_FH {down[1]:+.2f}); path {len(p)} voxels, {L_mm:.0f} mm")
    print("  frame:            " + " ".join(f"{t:5d}" for t in range(T)))
    print("  measured a  dp:   " + " ".join(f"{x:+5.1f}" for x in dpm))
    print("  derived a_v dp:   " + " ".join(f"{x:+5.1f}" for x in dpv))
    print(f"  peak / mean: measured {dpm.max():+.2f} / {dpm.mean():+.2f} mmHg; velocity-derived {dpv.max():+.2f} / {dpv.mean():+.2f} mmHg"
          "  (catheter P2 baseline: 13.1 / 3.68)")
