"""Register the 4D-FlowP scan of a coarctation phantom case onto the conventional 4D flow scan (CPU, read-only inputs).

The 4D-FlowP export (Imaging/<case>/4DFlowP/data.npz) has velocity AND acceleration but no position or orientation:
its array axes are (FH, RL, AP) with unknown direction and origin, and its vector components (Vx, Vy, Vz) along those
axes have unknown signs. The conventional scan (the DICOM the network trains on, pinnflow.coa_phantom) has full
patient (LPS) geometry. Both image the same phantom in the same session. This script finds the rigid mapping between
the two from the data and samples the 4D-FlowP acceleration and velocity at the voxels of the prepared training file,
in its local frame and frame order.

    1. Spatial mapping: axis permutation fixed (FH -> LPS z, RL -> LPS x, AP -> LPS y); for each of the 8 axis sign
       choices, the translation that maximises the overlap of the two lumen masks (FFT cross-correlation on a
       patient-aligned grid with the 4D-FlowP spacing), refined to sub-voxel by a parabolic fit of the overlap peak.
       Scored by Dice of the masks and the correlation of the time-mean speed; all 8 are printed.
    2. Vector component signs and the frame offset (-3..+3, cyclic): the 4D-FlowP velocity sampled at the conventional
       lumen voxels against the conventional velocity in LPS, all frames and voxels; signs from the correlation signs,
       offset by the largest summed correlation. The 3 x 3 component correlation matrix checks the permutation. The
       translation is then refined (sub-voxel, Nelder-Mead) to maximise the systolic velocity correlation, and the
       offset is re-checked. A fractional frame offset (linear interpolation in time) is estimated for information.
    3. Trilinear sampling of acceleration and velocity at every prepared voxel and frame, conversion to the prepared
       local frame (row_dir, col_dir, normal), and an `inside` flag (in the 4D-FlowP field of view and in its vessel
       mask dilated by one voxel, 6-connected).
    4. Output: data/stanford_coa/prepared/<case>_flowp.npz
    5. Checks: flux of the sampled 4D-FlowP velocity through the prepared upstream plane against the prepared file's,
       per-frame p99 |a|, a noise estimate, and the sampled acceleration against the conventional scan's own
       dv/dt + (v . grad) v.

    python "claude helper/diagnostics/flowp_register.py"                        # P2_baseline
    python "claude helper/diagnostics/flowp_register.py" --case P2_severe

Output keys (N prepared voxels, in the prepared file's order; T = 17 frames in the prepared file's frame order):
    acc (T, N, 3) m/s^2 and vel_flowp (T, N, 3) m/s, local frame (row_dir, col_dir, normal), 4D-FlowP frame
        (t + frame_offset) mod T at prepared frame t;
    acc_frac, vel_flowp_frac: the same, linearly interpolated in time at the fractional offset frame_offset_frac;
    inside (N,) bool; frame_offset; frame_offset_frac; axis_signs (s_FH, s_RL, s_AP); component_signs (c_FH, c_RL, c_AP);
    translation_mm (LPS x, y, z of 4D-FlowP voxel (0, 0, 0)); translation_mask_mm (the mask-only estimate);
    affine_lps (4 x 4, 4D-FlowP index (ix, iy, iz, 1) -> LPS mm); dice, speed_corr (and runner_up_*);
    vel_corr, vel_slope (3,) per local component in systole; acc_noise_ms2, acc_noise_comp_ms2; aao_flux_flowp_mLs.
Mapping: LPS (mm) of 4D-FlowP voxel (ix, iy, iz) = translation_mm + (s_RL * 1.8 * iy, s_AP * 1.8 * iz, s_FH * 1.7 * ix);
the 4D-FlowP vector (Vx, Vy, Vz) in LPS (x, y, z) = (c_RL * Vy, c_AP * Vz, c_FH * Vx).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pinnflow imports jax; never touch the GPU from here

import numpy as np  # noqa: E402
from scipy import ndimage, optimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinnflow.coa_phantom import COA_DIR, load_4dflow, segment_lumen  # noqa: E402

FLOWP_SP = np.array([1.7, 1.8, 1.8])  # mm along the 4D-FlowP array axes (FH, RL, AP), readme
LPS_OF_AXIS = np.array([2, 0, 1])  # 4D-FlowP axis / component (FH, RL, AP) -> LPS coordinate index (z, x, y)
MAX_OFFSET = 3
LOCAL = ("a (row_dir)", "b (col_dir)", "c (normal)")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------------------------
def lps_to_flowp_index(p_lps: np.ndarray, signs: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(..., 3) LPS mm -> (3, ...) fractional 4D-FlowP indices (ix, iy, iz) under the mapping (signs, translation)."""
    q = p_lps - trans
    return np.stack([q[..., LPS_OF_AXIS[a]] / (signs[a] * FLOWP_SP[a]) for a in range(3)])


def affine(signs, trans) -> np.ndarray:
    A = np.eye(4)
    A[:3, :3] = 0.0
    for a in range(3):
        A[LPS_OF_AXIS[a], a] = signs[a] * FLOWP_SP[a]
    A[:3, 3] = trans
    return A


def sample(vol: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Trilinear sample of a (X, Y, Z) volume at fractional indices (3, ...); zero outside the field of view."""
    return ndimage.map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)


def reference_grid(d, mask, smean):
    """The conventional mask and time-mean speed resampled onto a grid aligned with the patient axes, ordered like the
    4D-FlowP array (axis 0 = LPS z, 1 = LPS x, 2 = LPS y; all increasing) with the 4D-FlowP spacing.
    Returns the grid origin in (z, x, y), the LPS points (nz, nx, ny, 3), the mask and the speed."""
    xyz = d["xyz_mm"]
    lo, hi = xyz.reshape(-1, 3).min(0), xyz.reshape(-1, 3).max(0)
    g0 = lo[LPS_OF_AXIS]
    n = (np.floor((hi - lo)[LPS_OF_AXIS] / FLOWP_SP) + 1).astype(int)
    Gz, Gx, Gy = np.meshgrid(*[g0[a] + FLOWP_SP[a] * np.arange(n[a]) for a in range(3)], indexing="ij")
    p = np.stack([Gx, Gy, Gz], -1)
    q = p - xyz[0, 0, 0]
    sp_i, sp_j, dk = d["voxel_mm"]
    kij = np.stack([q @ d["normal"] / dk, q @ d["col_dir"] / sp_i, q @ d["row_dir"] / sp_j])  # conventional (k, i, j)
    gm = ndimage.map_coordinates(mask.astype(np.float32), kij, order=1, cval=0.0) > 0.5
    gs = ndimage.map_coordinates(smean.astype(np.float32), kij, order=1, cval=0.0)
    return g0, p, gm, gs


def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c[d] = sum_i a[i] b[i + d], zero padded (linear); index d modulo the padded shape."""
    P = tuple(int(x) for x in np.array(a.shape) + np.array(b.shape))
    ax = (0, 1, 2)
    return np.fft.irfftn(np.conj(np.fft.rfftn(a.astype(np.float32), P, axes=ax)) * np.fft.rfftn(b.astype(np.float32), P, axes=ax), P, axes=ax)


def parabolic(c: np.ndarray, idx) -> np.ndarray:
    """Sub-voxel offset of a correlation peak: a parabola through the peak and its two neighbours, per axis."""
    out = np.zeros(3)
    for ax in range(3):
        e = np.zeros(3, int)
        e[ax] = 1
        cm, c0, cp = c[tuple((np.array(idx) - e) % c.shape)], c[tuple(idx)], c[tuple((np.array(idx) + e) % c.shape)]
        den = cm - 2 * c0 + cp
        out[ax] = 0.5 * (cm - cp) / den if den < 0 else 0.0
    return out


def mask_scores(gp, gm, gs, fmask, fsmean, signs, trans):
    """Dice of the two lumen masks and Pearson correlation of the time-mean speed over their union, on the reference
    grid within the 4D-FlowP field of view, with the 4D-FlowP fields sampled through the (sub-voxel) mapping."""
    c = lps_to_flowp_index(gp, signs, trans)
    cov = np.all((c >= 0) & (c <= (np.array(fmask.shape) - 1)[:, None, None, None]), axis=0)
    fm = sample(fmask.astype(np.float32), c) > 0.5
    fs = sample(fsmean, c)
    A = gm & cov
    dice = 2.0 * (A & fm).sum() / max(1, A.sum() + fm.sum())
    u = (A | fm) & cov
    return float(dice), float(np.corrcoef(gs[u], fs[u])[0, 1])


def corr_slope(f, c):
    """Pearson correlation and least-squares slope through the origin of f regressed on c."""
    return float(np.corrcoef(f, c)[0, 1]), float(np.dot(f, c) / np.dot(c, c))


def sample_frames(vf, coords, frames=None):
    """4D-FlowP vector field (3, X, Y, Z, T) at fractional indices (3, n), components rearranged to LPS (x, y, z)
    order (signs not applied): (len(frames), n, 3). Fractional frames interpolate linearly in time (cyclic)."""
    T = vf.shape[-1]
    frames = np.arange(T) if frames is None else np.asarray(frames, float)
    out = np.zeros((len(frames), coords.shape[1], 3), np.float32)
    for n, t in enumerate(frames):
        t0 = int(np.floor(t))
        w = float(t - t0)
        for c in range(3):
            v = sample(vf[c, ..., t0 % T], coords)
            if w > 1e-6:
                v = (1 - w) * v + w * sample(vf[c, ..., (t0 + 1) % T], coords)
            out[n, :, LPS_OF_AXIS[c]] = v
    return out


def offset_scan(vf, coords, v_conv, offsets):
    """Correlation per LPS component between the 4D-FlowP velocity at prepared frame t + offset and the conventional
    velocity at frame t, over all frames and the given voxels, for each offset (fractional ones interpolate in time)."""
    T = v_conv.shape[0]
    rows = []
    for o in offsets:
        f = sample_frames(vf, coords, np.arange(T) + o)
        rows.append([np.corrcoef(f[..., c].ravel(), v_conv[..., c].ravel())[0, 1] for c in range(3)])
    return np.array(rows)


# ------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="P2_baseline")
    ap.add_argument("--out", default=None, help="output npz (default data/stanford_coa/prepared/<case>_flowp.npz)")
    ap.add_argument("--fit-voxels", type=int, default=8000, help="random inside voxels used to refine the translation")
    args = ap.parse_args()
    case = args.case
    out_path = Path(args.out) if args.out else COA_DIR / "prepared" / f"{case}_flowp.npz"
    t_start = time.time()
    rng = np.random.default_rng(0)

    # ---- the conventional scan and the prepared training file
    d = load_4dflow(case)
    mask = segment_lumen(d)
    prep = dict(np.load(COA_DIR / "prepared" / f"{case}.npz"))
    sp_i, sp_j, dk = d["voxel_mm"]
    K, I, J = np.nonzero(mask)
    pos_mm = np.stack([J * sp_j, I * sp_i, K * dk], -1)
    E = np.stack([d["row_dir"], d["col_dir"], d["normal"]])  # rows: local axes (a, b, c) in LPS
    if len(K) == len(prep["pos_mm"]) and np.abs(pos_mm - prep["pos_mm"]).max() < 1e-3:
        log(f"conventional mask reproduces the prepared voxels: N = {len(K)}")
    else:
        log("WARNING: segment_lumen does not reproduce the prepared voxels; using the prepared pos_mm with the scan axes")
        K, I, J = (np.round(prep["pos_mm"][:, 2] / dk).astype(int), np.round(prep["pos_mm"][:, 1] / sp_i).astype(int),
                   np.round(prep["pos_mm"][:, 0] / sp_j).astype(int))
    p_lps = d["xyz_mm"][0, 0, 0] + prep["pos_mm"].astype(np.float64) @ E  # (N, 3) LPS of the prepared voxels
    N = len(p_lps)
    vel_conv = prep["vel"].astype(np.float32)  # (T, N, 3) local frame
    v_conv = np.einsum("tna,ac->tnc", vel_conv, E).astype(np.float32)  # LPS
    T = v_conv.shape[0]
    p99 = np.array([np.percentile(np.linalg.norm(vel_conv[t], axis=-1), 99) for t in range(T)])
    sysf = np.where(p99 > 0.5 * p99.max())[0]
    g0, gp, gm, gs = reference_grid(d, mask, np.linalg.norm(d["velocity"], axis=-1).mean(0))
    del d
    log(f"reference grid (z, x, y) {gm.shape}, {int(gm.sum())} lumen voxels, origin LPS (z, x, y) {np.round(g0, 1)} mm")

    # ---- the 4D-FlowP mask and velocity
    zf = np.load(COA_DIR / "Imaging" / case / "4DFlowP" / "data.npz", allow_pickle=False)
    fmask = zf["approximate_vessel_mask"].astype(bool)
    vf = (zf["velocity"] / 100.0).astype(np.float32)  # (3, X, Y, Z, T) cm/s -> m/s
    if vf.shape[-1] != T:
        raise ValueError(f"4D-FlowP has {vf.shape[-1]} frames, the conventional scan {T}")
    fsmean = np.sqrt((vf**2).sum(0)).mean(-1)
    shape = np.array(fmask.shape)
    log(f"4D-FlowP grid {tuple(int(s) for s in shape)}, {int(fmask.sum())} vessel voxels; FOV {np.round(shape * FLOWP_SP)} mm")

    # ---- 1. spatial mapping: 8 axis sign choices, translation by FFT cross-correlation of the masks
    cands = []
    for signs in product((1, -1), repeat=3):
        signs = np.array(signs)
        flip = tuple(a for a in range(3) if signs[a] < 0)
        c = xcorr(np.flip(fmask, flip) if flip else fmask, gm)
        idx = np.unravel_index(int(np.argmax(c)), c.shape)
        dint = np.array([i - s if i >= gm.shape[a] else i for a, (i, s) in enumerate(zip(idx, c.shape))])  # wrap negatives
        dtot = dint + parabolic(c, idx)
        # flipped index i' = n - 1 - i; LPS (z, x, y) of flipped index i' = g0 + sp * (i' + d)
        t_zxy = g0 + FLOWP_SP * (dtot + np.where(signs < 0, shape - 1, 0))
        trans = np.empty(3)
        trans[LPS_OF_AXIS] = t_zxy  # LPS (x, y, z) of 4D-FlowP voxel (0, 0, 0)
        dice, scorr = mask_scores(gp, gm, gs, fmask, fsmean, signs, trans)
        cands.append({"signs": signs, "trans": trans, "dice": dice, "speed_corr": scorr, "overlap": float(c[idx])})
    cands.sort(key=lambda r: -r["dice"])
    log("\n1. axis signs (FH, RL, AP)   Dice   speed corr   overlap   translation LPS (x, y, z) mm")
    for r in cands:
        log(f"   {str(tuple(int(s) for s in r['signs'])):24s} {r['dice']:.3f}   {r['speed_corr']:+.3f}     {r['overlap']:8.0f}   {np.round(r['trans'], 2)}")
    best, runner = cands[0], cands[1]
    signs, trans_mask = best["signs"], best["trans"]
    log(f"   best {tuple(int(s) for s in signs)}: Dice {best['dice']:.3f}, speed corr {best['speed_corr']:+.3f};"
        f" runner-up {tuple(int(s) for s in runner['signs'])}: Dice {runner['dice']:.3f}, speed corr {runner['speed_corr']:+.3f}")

    def inside_of(trans):
        coords = lps_to_flowp_index(p_lps, signs, trans)
        fov = np.all((coords >= 0) & (coords <= (shape - 1)[:, None]), axis=0)
        near = np.clip(np.round(coords).astype(int), 0, (shape - 1)[:, None])
        return coords, fov, fov & ndimage.binary_dilation(fmask)[near[0], near[1], near[2]]

    # ---- 2. component signs and frame offset; then the translation refined on the velocity itself
    coords, fov, inside = inside_of(trans_mask)
    offsets = np.arange(-MAX_OFFSET, MAX_OFFSET + 1)
    cs = offset_scan(vf, coords[:, inside], v_conv[:, inside], offsets)
    log("\n2. integer frame offset scan (mask translation), corr per LPS component (x <- Vy, y <- Vz, z <- Vx), sum |corr|:")
    for o, r in zip(offsets, cs):
        log(f"   {o:+d}   {np.round(r, 3)}   {np.abs(r).sum():.3f}")
    k = int(np.argmax(np.abs(cs).sum(1)))
    offset, csign_lps = int(offsets[k]), np.sign(cs[k])
    comp_signs = csign_lps[LPS_OF_AXIS]  # per 4D-FlowP component (FH, RL, AP)
    log(f"   offset {offset:+d}, component signs (FH, RL, AP) {tuple(int(s) for s in comp_signs)}")

    fit = rng.choice(np.where(inside)[0], min(args.fit_voxels, int(inside.sum())), replace=False)

    def sys_corr(trans, idx, off):
        c = lps_to_flowp_index(p_lps[idx], signs, trans)
        f = sample_frames(vf, c, sysf + off) * csign_lps
        return np.array([np.corrcoef(f[..., j].ravel(), v_conv[sysf][:, idx, j].ravel())[0, 1] for j in range(3)])

    res = optimize.minimize(lambda t: -sys_corr(t, fit, offset).sum(), trans_mask, method="Nelder-Mead",
                            options={"xatol": 0.05, "fatol": 1e-4, "initial_simplex": trans_mask + np.vstack([np.zeros(3), np.eye(3)])})
    trans = res.x
    held = np.setdiff1d(np.where(inside)[0], fit)
    log(f"   translation refined on the systolic velocity ({len(fit)} voxels): change {np.round(trans - trans_mask, 2)} mm;"
        f" held-out systolic corr (x, y, z) {np.round(sys_corr(trans_mask, held, offset), 3)} -> {np.round(sys_corr(trans, held, offset), 3)}")
    dice, scorr = mask_scores(gp, gm, gs, fmask, fsmean, signs, trans)
    log(f"   at the refined translation: Dice {dice:.3f}, speed corr {scorr:+.3f}")
    coords, fov, inside = inside_of(trans)
    cs = offset_scan(vf, coords[:, inside], v_conv[:, inside], offsets)
    k = int(np.argmax(np.abs(cs).sum(1)))
    if int(offsets[k]) != offset or np.any(np.sign(cs[k]) != csign_lps):
        log(f"WARNING: with the refined translation the offset/signs change to {int(offsets[k])}, {np.sign(cs[k])}")
    fine = np.round(np.arange(offset - 1.0, offset + 1.0001, 0.1), 2)
    cf = offset_scan(vf, coords[:, inside], v_conv[:, inside], fine) * csign_lps
    offset_frac = float(fine[int(np.argmax(cf.sum(1)))])
    log("   fractional offset scan (linear in time), sum of corr: " + " ".join(f"{o:+.1f}:{s:.3f}" for o, s in zip(fine, cf.sum(1))))
    log(f"   best fractional offset {offset_frac:+.2f} frames ({1e3 * offset_frac * float(np.median(np.diff(prep['t_s']))):.0f} ms)")
    M = np.array([[np.corrcoef(sample_frames(vf, coords[:, inside], np.arange(T) + offset)[..., i].ravel(),
                               v_conv[:, inside, j].ravel())[0, 1] for j in range(3)] for i in range(3)])
    log("   component correlation matrix (all frames), rows 4D-FlowP (Vy, Vz, Vx) unsigned, cols conventional LPS (x, y, z):")
    for i in range(3):
        log("     " + "  ".join(f"{M[i, j]:+.3f}" for j in range(3)))
    log(f"   prepared voxels in the 4D-FlowP FOV: {int(fov.sum())} of {N}; inside (FOV and dilated vessel mask): {int(inside.sum())}")

    # ---- 3. sample velocity and acceleration at all prepared voxels; LPS -> local frame
    def to_local(lps):
        return np.einsum("tnc,ac->tna", lps * csign_lps, E).astype(np.float32)

    vel_flowp = to_local(sample_frames(vf, coords, np.arange(T) + offset))
    vel_flowp_frac = to_local(sample_frames(vf, coords, np.arange(T) + offset_frac))
    del vf
    af = zf["acceleration"].astype(np.float32)  # (3, X, Y, Z, T) m/s^2
    del zf
    acc = to_local(sample_frames(af, coords, np.arange(T) + offset))
    acc_frac = to_local(sample_frames(af, coords, np.arange(T) + offset_frac))
    del af

    log(f"\n   velocity agreement in systole (frames {sysf.tolist()}), {int(inside.sum())} inside voxels, integer offset {offset:+d}"
        f" [fractional {offset_frac:+.2f} in brackets]:")
    vcorr, vslope = np.zeros(3), np.zeros(3)
    for a in range(3):
        c_ = vel_conv[sysf][:, inside, a].ravel()
        f_ = vel_flowp[sysf][:, inside, a].ravel()
        vcorr[a], vslope[a] = corr_slope(f_, c_)
        r2, s2 = corr_slope(vel_flowp_frac[sysf][:, inside, a].ravel(), c_)
        log(f"   local {LOCAL[a]:12s}: corr {vcorr[a]:+.3f} [{r2:+.3f}], slope FlowP/conv {vslope[a]:+.3f} [{s2:+.3f}],"
            f" RMS FlowP {np.sqrt(np.mean(f_**2)):.3f}, conv {np.sqrt(np.mean(c_**2)):.3f} m/s")
    r_all, s_all = corr_slope(vel_flowp[:, inside].ravel(), vel_conv[:, inside].ravel())
    log(f"   all frames and components: corr {r_all:+.3f}, slope {s_all:+.3f}")

    # ---- 5. checks
    vox_m3 = float(np.prod(prep["voxel_mm"])) * 1e-9
    h_m = float(np.mean(prep["voxel_mm"])) * 1e-3
    lut = {tuple(k): n for n, k in enumerate(np.round(prep["pos_mm"] / prep["voxel_mm"]).astype(int))}
    cells = [lut.get(tuple(k)) for k in np.round(prep["aao_mm"] / prep["voxel_mm"]).astype(int)]
    if any(c is None for c in cells):
        log("WARNING: some upstream-plane cells are not prepared voxels")
    cells = np.array([c for c in cells if c is not None])
    nrm = prep["aao_normal"]

    def flux(v):
        return (v[:, cells] @ nrm).sum(-1) * vox_m3 / h_m * 1e6  # mL/s, as flux_cells in prepare_case

    q_ref = prep["aao_flux_frames_mLs"]
    q_fp, q_fp_frac = flux(vel_flowp), flux(vel_flowp_frac)
    log(f"\n5. flux through the upstream plane ({len(cells)} cells, {int(inside[cells].sum())} inside), mL/s per frame:")
    log("   prepared file      : " + " ".join(f"{x:5.0f}" for x in q_ref))
    log("   recomputed (conv)  : " + " ".join(f"{x:5.0f}" for x in flux(vel_conv)))
    log("   4D-FlowP           : " + " ".join(f"{x:5.0f}" for x in q_fp))
    log("   4D-FlowP fractional: " + " ".join(f"{x:5.0f}" for x in q_fp_frac))
    log(f"   mean: prepared {q_ref.mean():.1f}, 4D-FlowP {q_fp.mean():.1f}, meter {float(prep['meter_flow_mLs']):.1f} mL/s;"
        f" corr over frames {np.corrcoef(q_fp, q_ref)[0, 1]:+.3f} [fractional {np.corrcoef(q_fp_frac, q_ref)[0, 1]:+.3f}]")
    amag = np.linalg.norm(acc[:, inside], axis=-1)
    log("   p99 |a| inside, m/s^2, frames 0-8: " + " ".join(f"{np.percentile(amag[t], 99):5.1f}" for t in range(min(9, T))))
    calm = np.sort(np.argsort([np.linalg.norm(vel_conv[t], axis=-1).mean() for t in range(T)])[:5])
    noise_c = np.sqrt(np.mean(acc[calm][:, inside] ** 2, axis=(0, 1)))
    noise = float(np.sqrt(np.mean(acc[calm][:, inside] ** 2)))
    log(f"   noise: calmest frames {calm.tolist()}, RMS a per local component {np.round(noise_c, 2)}, overall {noise:.2f} m/s^2")

    # the measured acceleration against the conventional scan's own material acceleration dv/dt + (v . grad) v
    # (local frame; grid axes (k, i, j) carry the local components (c, b, a); periodic central differences in time)
    vol = np.zeros((T,) + mask.shape + (3,), np.float32)
    vol[:, K, I, J] = vel_conv
    dt = float(np.median(np.diff(prep["t_s"])))
    h = (sp_j * 1e-3, sp_i * 1e-3, dk * 1e-3)
    gax = (2, 1, 0)
    a_mat = np.zeros((T, N, 3), np.float32)
    for t in range(T):
        dvdt = (vol[(t + 1) % T] - vol[(t - 1) % T]) / (2 * dt)
        conv = sum(vol[t][..., b : b + 1] * np.gradient(vol[t], h[b], axis=gax[b]) for b in range(3))
        a_mat[t] = (dvdt + conv)[K, I, J]
    del vol
    s2 = inside & ndimage.binary_erosion(mask, iterations=1)[K, I, J]
    log(f"   measured a vs the conventional scan's dv/dt + (v.grad)v (unsmoothed), systole, {int(s2.sum())} interior voxels,"
        " corr / slope [fractional offset]:")
    for a in range(3):
        r, s = corr_slope(acc[sysf][:, s2, a].ravel(), a_mat[sysf][:, s2, a].ravel())
        rf, sf = corr_slope(acc_frac[sysf][:, s2, a].ravel(), a_mat[sysf][:, s2, a].ravel())
        log(f"     local {LOCAL[a]:12s}: {r:+.3f} / {s:+.3f}   [{rf:+.3f} / {sf:+.3f}]")

    # ---- 4. save
    np.savez_compressed(out_path, acc=acc, vel_flowp=vel_flowp, acc_frac=acc_frac, vel_flowp_frac=vel_flowp_frac, inside=inside,
                        frame_offset=np.int64(offset), frame_offset_frac=np.float64(offset_frac),
                        axis_signs=signs.astype(np.int64), component_signs=comp_signs.astype(np.int64),
                        translation_mm=trans, translation_mask_mm=trans_mask, affine_lps=affine(signs, trans),
                        dice=np.float64(dice), speed_corr=np.float64(scorr), runner_up_dice=np.float64(runner["dice"]),
                        runner_up_speed_corr=np.float64(runner["speed_corr"]), vel_corr=vcorr, vel_slope=vslope,
                        acc_noise_ms2=np.float64(noise), acc_noise_comp_ms2=noise_c, aao_flux_flowp_mLs=q_fp, case=np.array(case))
    log(f"\nsaved {out_path} ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
