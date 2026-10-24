"""Register the 4D-FlowP scan of a coarctation phantom case onto the conventional 4D flow scan (CPU, read-only inputs).

The 4D-FlowP export (Imaging/<case>/4DFlowP/data.npz) has velocity AND acceleration but no position or orientation:
its array axes are (FH, RL, AP) with unknown direction and origin, and its vector components (Vx, Vy, Vz) along those
axes have unknown signs. The conventional scan (the DICOM the network trains on, pinnflow.coa_phantom) has full
patient (LPS) geometry. Both image the same phantom in the same session. This script finds the rigid mapping between
the two from the data and samples the 4D-FlowP acceleration and velocity at the voxels and frames of the prepared
training file, in its local frame.

    1. Spatial mapping: axis permutation fixed (FH -> LPS z, RL -> LPS x, AP -> LPS y); for each of the 8 axis sign
       choices, the translation that maximises the overlap of the two lumen masks (FFT cross-correlation on a
       patient-aligned grid with the 4D-FlowP spacing), refined to sub-voxel by a parabolic fit of the overlap peak.
       Scored by Dice of the masks and the correlation of the time-mean speed; all 8 are printed.
    2. Timing and vector component signs. Both scans go on one clock over the beat: period = the prepared period_s,
       conventional frames at the prepared t_s, 4D-FlowP frame k at tau + k * dt (dt from its readme, 60 ms), periodic,
       linear interpolation in time between 4D-FlowP frames (the last one interpolates to frame 0 one period later).
       tau is scanned over the whole beat for the best agreement of the 4D-FlowP velocity with the conventional velocity
       at the lumen voxels (all frames, per LPS component); the signs of those correlations are the component signs.
       With equal frame counts the integer frame offset (-3..+3) is also scanned, as before.
       The translation is then refined (sub-voxel, Nelder-Mead) to maximise the systolic velocity correlation. Guard:
       a move over one voxel on any axis is redone with the move bounded to one voxel, and the refinement is kept only
       if the Dice stays within DICE_TOL of the mask-only Dice and the held-out correlation improves.
    3. Trilinear sampling of acceleration and velocity at every prepared voxel and every 4D-FlowP frame, mapping to
       the conventional frames, conversion to the prepared local frame (row_dir, col_dir, normal), and an `inside` flag
       (in the 4D-FlowP field of view and in its vessel mask dilated by one voxel, 6-connected).
    4. Output: data/stanford_coa/prepared/<case>_flowp.npz (or --out)
    5. Checks: flux of the sampled 4D-FlowP velocity through the prepared upstream plane against the prepared file's,
       per-frame p99 |a|, noise estimates, and the sampled acceleration against the conventional scan's own
       dv/dt + (v . grad) v.

    python "claude helper/diagnostics/flowp_register.py"                        # P2_baseline
    python "claude helper/diagnostics/flowp_register.py" --case P1_severe
    python "claude helper/diagnostics/flowp_register.py" --case P2_severe_lowflow --out data/stanford_coa/prepared/P2_severe_lowflow_flowp_new.npz

Output keys (N prepared voxels in the prepared file's order; F = the prepared file's frames, in its order):
    acc (F, N, 3) m/s^2 and vel_flowp (F, N, 3) m/s in the local frame (row_dir, col_dir, normal):
        equal frame counts: 4D-FlowP frame (t + frame_offset) mod F at prepared frame t, no time interpolation
        (acc_mode "integer"); unequal counts: interpolated in time at the best clock offset (acc_mode "time");
    acc_frac, vel_flowp_frac: always the best-clock-offset version (identical to acc when acc_mode is "time");
    acc_nearest, vel_flowp_nearest: whole 4D-FlowP frames (no blending in time), at each conventional frame the frame
        nearest on the beat clock at the best tau; nearest_frame (F,) its index, nearest_err_ms (F,) its time minus the
        conventional frame time (one timing rule for all cases; blending flattens systolic peaks);
    acc_floor, vel_flowp_floor, floor_frame, floor_err_ms: the same with the latest 4D-FlowP frame at or before each
        conventional frame time (always early; equals the integer-offset `acc` when the frame counts are equal);
    time_offset_ms (tau: conventional trigger time of 4D-FlowP frame 0), frame_offset (-1 if not used),
    frame_offset_frac (tau as the equivalent 4D-FlowP frame shift: mean over frames of 4D-FlowP index - prepared index);
    inside (N,) bool; axis_signs (s_FH, s_RL, s_AP); component_signs (c_FH, c_RL, c_AP);
    translation_mm (LPS x, y, z of 4D-FlowP voxel (0, 0, 0)); translation_mask_mm (mask-only); translation_refined;
    affine_lps (4 x 4, 4D-FlowP index (ix, iy, iz, 1) -> LPS mm); dice, speed_corr (final), dice_mask, speed_corr_mask,
    runner_up_dice, runner_up_speed_corr; vel_corr, vel_slope (3,) per local component in systole for `acc`'s timing;
    acc_noise_ms2, acc_noise_comp_ms2 (of `acc`), acc_noise_raw_ms2 (raw 4D-FlowP frames); aao_flux_flowp_mLs.
Mapping: LPS (mm) of 4D-FlowP voxel (ix, iy, iz) = translation_mm + (s_RL * dy * iy, s_AP * dz * iz, s_FH * dx * ix),
(dx, dy, dz) = (1.7, 1.8, 1.8) mm; the 4D-FlowP vector (Vx, Vy, Vz) in LPS (x, y, z) = (c_RL * Vy, c_AP * Vz, c_FH * Vx).
"""
from __future__ import annotations

import argparse
import os
import re
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

LPS_OF_AXIS = np.array([2, 0, 1])  # 4D-FlowP axis / component (FH, RL, AP) -> LPS coordinate index (z, x, y)
AXIS_OF_LPS = np.argsort(LPS_OF_AXIS)  # LPS coordinate (x, y, z) -> 4D-FlowP axis (RL, AP, FH)
MAX_OFFSET = 3
DICE_TOL = 0.02  # the velocity refinement may cost at most this much Dice
LOCAL = ("a (row_dir)", "b (col_dir)", "c (normal)")


def log(*a):
    print(*a, flush=True)


def read_readme(path: Path):
    """Voxel size (mm, along FH, RL, AP) and frame interval (s) from the 4D-FlowP readme; defaults if absent."""
    sp, dt = np.array([1.7, 1.8, 1.8]), 0.060
    if path.exists():
        txt = path.read_text(errors="ignore")
        m = re.search(r"Resolution \(mm\).*?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)", txt)
        if m:
            sp = np.array([float(g) for g in m.groups()])
        m = re.search(r"Time resolution \(ms\)\s*:\s*([\d.]+)", txt)
        if m:
            dt = float(m.group(1)) / 1000.0
    return sp, dt


# ------------------------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------------------------
def lps_to_flowp_index(p_lps: np.ndarray, signs, trans, sp) -> np.ndarray:
    """(..., 3) LPS mm -> (3, ...) fractional 4D-FlowP indices (ix, iy, iz) under the mapping (signs, translation)."""
    q = p_lps - trans
    return np.stack([q[..., LPS_OF_AXIS[a]] / (signs[a] * sp[a]) for a in range(3)])


def affine(signs, trans, sp) -> np.ndarray:
    A = np.eye(4)
    A[:3, :3] = 0.0
    for a in range(3):
        A[LPS_OF_AXIS[a], a] = signs[a] * sp[a]
    A[:3, 3] = trans
    return A


def sample(vol: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Trilinear sample of a (X, Y, Z) volume at fractional indices (3, ...); zero outside the field of view."""
    return ndimage.map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)


def reference_grid(d, mask, smean, sp):
    """The conventional mask and time-mean speed resampled onto a grid aligned with the patient axes, ordered like the
    4D-FlowP array (axis 0 = LPS z, 1 = LPS x, 2 = LPS y; all increasing) with the 4D-FlowP spacing.
    Returns the grid origin in (z, x, y), the LPS points (nz, nx, ny, 3), the mask and the speed."""
    xyz = d["xyz_mm"]
    lo, hi = xyz.reshape(-1, 3).min(0), xyz.reshape(-1, 3).max(0)
    g0 = lo[LPS_OF_AXIS]
    n = (np.floor((hi - lo)[LPS_OF_AXIS] / sp) + 1).astype(int)
    Gz, Gx, Gy = np.meshgrid(*[g0[a] + sp[a] * np.arange(n[a]) for a in range(3)], indexing="ij")
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


def mask_scores(gp, gm, gs, fmask, fsmean, signs, trans, sp):
    """Dice of the two lumen masks and Pearson correlation of the time-mean speed over their union, on the reference
    grid within the 4D-FlowP field of view, with the 4D-FlowP fields sampled through the (sub-voxel) mapping."""
    c = lps_to_flowp_index(gp, signs, trans, sp)
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
    """4D-FlowP vector field (3, X, Y, Z, T') at fractional indices (3, n) and the given integer frames, components
    rearranged to LPS (x, y, z) order (signs not applied): (len(frames), n, 3)."""
    frames = range(vf.shape[-1]) if frames is None else frames
    out = np.zeros((len(frames), coords.shape[1], 3), np.float32)
    for n, t in enumerate(frames):
        for c in range(3):
            out[n, :, LPS_OF_AXIS[c]] = sample(vf[c, ..., t], coords)
    return out


# ------------------------------------------------------------------------------------------------------------
# timing: one clock over the beat
# ------------------------------------------------------------------------------------------------------------
def time_weights(t_conv, tau, dt_f, n_f, period):
    """For each conventional frame time, the two 4D-FlowP frames around it and the weight of the second one.
    4D-FlowP frame k sits at tau + k * dt_f on the conventional clock, periodic with `period`; between the last frame
    and frame 0 of the next beat the interval is period - (n_f - 1) * dt_f."""
    if period <= (n_f - 1) * dt_f:
        raise ValueError(f"period {period:.3f} s is shorter than the 4D-FlowP frames span {(n_f - 1) * dt_f:.3f} s")
    s = np.mod(np.asarray(t_conv) - tau, period)
    k0 = np.minimum(np.floor(s / dt_f).astype(int), n_f - 1)
    last = k0 == n_f - 1
    k1 = np.where(last, 0, k0 + 1)
    w = np.where(last, (s - (n_f - 1) * dt_f) / (period - (n_f - 1) * dt_f), s / dt_f - k0)
    return k0, k1, w.astype(np.float32)


def interp_time(F, k0, k1, w):
    """(n_f, n, 3) 4D-FlowP samples -> (len(k0), n, 3) at the conventional frames."""
    return (1 - w)[:, None, None] * F[k0] + w[:, None, None] * F[k1]


def nearest_frames(t_conv, tau, dt_f, n_f, period):
    """For each conventional frame time, the 4D-FlowP frame whose time tau + k * dt_f is nearest on the periodic beat
    clock, and the signed timing error (4D-FlowP frame time - conventional frame time, s, wrapped to half a period)."""
    tk = tau + dt_f * np.arange(n_f)
    err = np.mod(tk[None, :] - np.asarray(t_conv)[:, None] + period / 2, period) - period / 2  # (frames, n_f)
    k = np.argmin(np.abs(err), axis=1)
    return k, err[np.arange(len(k)), k]


def comp_corr(f, c):
    """Pearson correlation per component of two (frames, n, 3) arrays, over frames and voxels."""
    return np.array([np.corrcoef(f[..., j].ravel(), c[..., j].ravel())[0, 1] for j in range(3)])


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
    fdir = COA_DIR / "Imaging" / case / "4DFlowP"
    sp, dt_f = read_readme(fdir / "readme.txt")
    h_lps = sp[AXIS_OF_LPS]  # 4D-FlowP voxel size along LPS (x, y, z)

    # ---- the conventional scan and the prepared training file
    d = load_4dflow(case)
    mask = segment_lumen(d)
    prep = dict(np.load(COA_DIR / "prepared" / f"{case}.npz"))
    sp_i, sp_j, dk = d["voxel_mm"]
    K, I, J = np.nonzero(mask)
    pos_mm = np.stack([J * sp_j, I * sp_i, K * dk], -1)
    E = np.stack([d["row_dir"], d["col_dir"], d["normal"]])  # rows: local axes (a, b, c) in LPS
    if len(K) == len(prep["pos_mm"]) and np.abs(pos_mm - prep["pos_mm"]).max() < 1e-3:
        log(f"{case}: conventional mask reproduces the prepared voxels: N = {len(K)}")
    else:
        log("WARNING: segment_lumen does not reproduce the prepared voxels; using the prepared pos_mm with the scan axes")
        K, I, J = (np.round(prep["pos_mm"][:, 2] / dk).astype(int), np.round(prep["pos_mm"][:, 1] / sp_i).astype(int),
                   np.round(prep["pos_mm"][:, 0] / sp_j).astype(int))
    p_lps = d["xyz_mm"][0, 0, 0] + prep["pos_mm"].astype(np.float64) @ E  # (N, 3) LPS of the prepared voxels
    N = len(p_lps)
    vel_conv = prep["vel"].astype(np.float32)  # (F, N, 3) local frame
    v_conv = np.einsum("tna,ac->tnc", vel_conv, E).astype(np.float32)  # LPS
    T = v_conv.shape[0]
    t_conv, period = np.asarray(prep["t_s"], float), float(prep["period_s"])
    p99 = np.array([np.percentile(np.linalg.norm(vel_conv[t], axis=-1), 99) for t in range(T)])
    sysf = np.where(p99 > 0.5 * p99.max())[0]
    g0, gp, gm, gs = reference_grid(d, mask, np.linalg.norm(d["velocity"], axis=-1).mean(0), sp)
    del d
    log(f"reference grid (z, x, y) {gm.shape}, {int(gm.sum())} lumen voxels, origin LPS (z, x, y) {np.round(g0, 1)} mm;"
        f" conventional {T} frames at {np.round(1e3 * t_conv[[0, -1]]).astype(int).tolist()} ms, period {1e3 * period:.0f} ms")

    # ---- the 4D-FlowP mask and velocity
    zf = np.load(fdir / "data.npz", allow_pickle=False)
    fmask = zf["approximate_vessel_mask"].astype(bool)
    vf = (zf["velocity"] / 100.0).astype(np.float32)  # (3, X, Y, Z, T') cm/s -> m/s
    Tf = vf.shape[-1]
    fsmean = np.sqrt((vf**2).sum(0)).mean(-1)
    shape = np.array(fmask.shape)
    log(f"4D-FlowP grid {tuple(int(s) for s in shape)}, voxel {sp.tolist()} mm, {Tf} frames every {1e3 * dt_f:.0f} ms"
        f" ({1e3 * Tf * dt_f:.0f} ms of the {1e3 * period:.0f} ms beat), {int(fmask.sum())} vessel voxels")

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
        t_zxy = g0 + sp * (dtot + np.where(signs < 0, shape - 1, 0))
        trans = np.empty(3)
        trans[LPS_OF_AXIS] = t_zxy  # LPS (x, y, z) of 4D-FlowP voxel (0, 0, 0)
        dice, scorr = mask_scores(gp, gm, gs, fmask, fsmean, signs, trans, sp)
        cands.append({"signs": signs, "trans": trans, "dice": dice, "speed_corr": scorr, "overlap": float(c[idx])})
    cands.sort(key=lambda r: -r["dice"])
    log("\n1. axis signs (FH, RL, AP)   Dice   speed corr   overlap   translation LPS (x, y, z) mm")
    for r in cands:
        log(f"   {str(tuple(int(s) for s in r['signs'])):24s} {r['dice']:.3f}   {r['speed_corr']:+.3f}     {r['overlap']:8.0f}   {np.round(r['trans'], 2)}")
    best, runner = cands[0], cands[1]
    signs, trans_mask = best["signs"], best["trans"]
    log(f"   best {tuple(int(s) for s in signs)}: Dice {best['dice']:.3f}, speed corr {best['speed_corr']:+.3f};"
        f" runner-up {tuple(int(s) for s in runner['signs'])}: Dice {runner['dice']:.3f}, speed corr {runner['speed_corr']:+.3f}")

    dil = ndimage.binary_dilation(fmask)

    def inside_of(trans):
        coords = lps_to_flowp_index(p_lps, signs, trans, sp)
        fov = np.all((coords >= 0) & (coords <= (shape - 1)[:, None]), axis=0)
        near = np.clip(np.round(coords).astype(int), 0, (shape - 1)[:, None])
        return coords, fov, fov & dil[near[0], near[1], near[2]]

    # ---- 2. timing and component signs at the mask translation
    coords, fov, inside = inside_of(trans_mask)
    Fv = sample_frames(vf, coords[:, inside])  # (Tf, n_inside, 3) LPS order, unsigned
    vin = v_conv[:, inside]

    def tau_scan(Fv, vin, taus):
        return np.array([comp_corr(interp_time(Fv, *time_weights(t_conv, tau, dt_f, Tf, period)), vin) for tau in taus])

    coarse = np.arange(0.0, period, 0.005)
    cc = tau_scan(Fv, vin, coarse)
    tau0 = coarse[int(np.argmax(np.abs(cc).sum(1)))]
    fine = tau0 + np.arange(-0.006, 0.0061, 0.001)
    cf = tau_scan(Fv, vin, fine)
    kf = int(np.argmax(np.abs(cf).sum(1)))
    tau, csign_lps = float(np.mod(fine[kf], period)), np.sign(cf[kf])
    comp_signs = csign_lps[LPS_OF_AXIS]  # per 4D-FlowP component (FH, RL, AP)
    log("\n2. clock offset tau (conventional trigger time of 4D-FlowP frame 0), sum |corr| over the beat (every 40 ms):")
    log("   " + " ".join(f"{1e3 * t:.0f}:{s:.2f}" for t, s in zip(coarse[::8], np.abs(cc).sum(1)[::8])))
    log(f"   best tau {1e3 * tau:.0f} ms: corr (x <- Vy, y <- Vz, z <- Vx) {np.round(cf[kf], 3)};"
        f" component signs (FH, RL, AP) {tuple(int(s) for s in comp_signs)}")
    equal = Tf == T
    offset = -1
    if equal:
        offsets = np.arange(-MAX_OFFSET, MAX_OFFSET + 1)
        ci = np.array([comp_corr(Fv[(np.arange(T) + o) % T], vin) for o in offsets])
        log("   equal frame counts: integer offset scan, corr per LPS component, sum |corr|:")
        for o, r in zip(offsets, ci):
            log(f"   {o:+d}   {np.round(r, 3)}   {np.abs(r).sum():.3f}")
        k = int(np.argmax(np.abs(ci).sum(1)))
        offset = int(offsets[k])
        if np.any(np.sign(ci[k]) != csign_lps):
            log(f"WARNING: the integer-offset signs {np.sign(ci[k])} differ from the clock-offset signs {csign_lps}")
        log(f"   integer offset {offset:+d}")
    del Fv

    # ---- the translation refined on the systolic velocity, with the guard
    fit = rng.choice(np.where(inside)[0], min(args.fit_voxels, int(inside.sum())), replace=False)
    held = np.setdiff1d(np.where(inside)[0], fit)
    if equal:  # as before: frames (t + offset) mod T
        k0s, k1s, ws = (sysf + offset) % T, (sysf + offset) % T, np.zeros(len(sysf), np.float32)
    else:
        k0s, k1s, ws = time_weights(t_conv[sysf], tau, dt_f, Tf, period)
    need = np.unique(np.concatenate([k0s, k1s]))
    pos_of = {k: n for n, k in enumerate(need)}

    def sys_corr(trans, idx):
        F = sample_frames(vf, lps_to_flowp_index(p_lps[idx], signs, trans, sp), need) * csign_lps
        f = interp_time(F, np.array([pos_of[k] for k in k0s]), np.array([pos_of[k] for k in k1s]), ws)
        return comp_corr(f, v_conv[sysf][:, idx])

    nm = {"xatol": 0.05, "fatol": 1e-4}
    res = optimize.minimize(lambda t: -sys_corr(t, fit).sum(), trans_mask, method="Nelder-Mead",
                            options={**nm, "initial_simplex": trans_mask + np.vstack([np.zeros(3), np.eye(3)])})
    trans_ref = res.x
    bounded = False
    if np.any(np.abs(trans_ref - trans_mask) > h_lps):  # over a voxel: redo with the move bounded to 0.95 voxel
        bounded = True
        lim = 0.95 * h_lps
        r2 = optimize.minimize(lambda u: -sys_corr(trans_mask + lim * np.tanh(u), fit).sum(), np.zeros(3), method="Nelder-Mead",
                               options={**nm, "initial_simplex": np.vstack([np.zeros(3), 0.5 * np.eye(3)])})
        log(f"   unbounded refinement moved {np.round(trans_ref - trans_mask, 2)} mm (over a voxel): redone bounded")
        trans_ref = trans_mask + lim * np.tanh(r2.x)
    c_mask, c_ref = sys_corr(trans_mask, held), sys_corr(trans_ref, held)
    dice_ref, scorr_ref = mask_scores(gp, gm, gs, fmask, fsmean, signs, trans_ref, sp)
    accept = dice_ref >= best["dice"] - DICE_TOL and c_ref.sum() > c_mask.sum() + 0.01
    log(f"   refinement{' (bounded)' if bounded else ''}: move {np.round(trans_ref - trans_mask, 2)} mm; held-out systolic corr (x, y, z)"
        f" {np.round(c_mask, 3)} -> {np.round(c_ref, 3)}; Dice {best['dice']:.3f} -> {dice_ref:.3f}, speed corr"
        f" {best['speed_corr']:+.3f} -> {scorr_ref:+.3f}: {'ACCEPTED' if accept else 'REJECTED, mask-only translation kept'}")
    trans, dice, scorr = (trans_ref, dice_ref, scorr_ref) if accept else (trans_mask, best["dice"], best["speed_corr"])
    coords, fov, inside = inside_of(trans)
    log(f"   prepared voxels in the 4D-FlowP FOV: {int(fov.sum())} of {N}; inside (FOV and dilated vessel mask): {int(inside.sum())}")

    # ---- 3. sample velocity and acceleration at all prepared voxels and 4D-FlowP frames; then the conventional frames
    Fv = sample_frames(vf, coords) * csign_lps  # (Tf, N, 3) LPS, signed
    del vf
    af = zf["acceleration"].astype(np.float32)  # (3, X, Y, Z, T') m/s^2
    del zf
    Fa = sample_frames(af, coords) * csign_lps
    del af
    if accept:  # re-check the clock at the final translation
        cf2 = tau_scan(Fv[:, inside], v_conv[:, inside], fine)
        tau2 = float(np.mod(fine[int(np.argmax(np.abs(cf2).sum(1)))], period))
        if abs(tau2 - tau) > 0.0015:
            log(f"   clock re-check at the final translation: best tau {1e3 * tau2:.0f} ms (was {1e3 * tau:.0f}); using it")
            tau = tau2
    kw = time_weights(t_conv, tau, dt_f, Tf, period)
    # the clock offset as a 4D-FlowP frame shift (equal frame counts only): mean of (4D-FlowP index - prepared index)
    frac_equiv = float(np.mean(np.mod(kw[0] + kw[2] - np.arange(T) + Tf / 2, Tf) - Tf / 2)) if equal else float("nan")

    def to_local(lps):
        return np.einsum("tnc,ac->tna", lps, E).astype(np.float32)

    vel_flowp_frac, acc_frac = to_local(interp_time(Fv, *kw)), to_local(interp_time(Fa, *kw))
    if equal:
        fr = (np.arange(T) + offset) % T
        vel_flowp, acc, mode = to_local(Fv[fr]), to_local(Fa[fr]), "integer"
    else:
        vel_flowp, acc, mode = vel_flowp_frac, acc_frac, "time"
    # whole 4D-FlowP frames, no blending in time: the nearest frame on the beat clock (keeps systolic peaks unflattened)
    k_near, err_near = nearest_frames(t_conv, tau, dt_f, Tf, period)
    vel_flowp_nearest, acc_nearest = to_local(Fv[k_near]), to_local(Fa[k_near])
    used = np.bincount(k_near, minlength=Tf)
    log(f"\n   nearest-frame rule: 4D-FlowP frames {k_near.tolist()}; timing error ms {np.round(1e3 * err_near).astype(int).tolist()}"
        f" (max |err| {1e3 * np.abs(err_near).max():.0f}, RMS {1e3 * np.sqrt(np.mean(err_near**2)):.0f});"
        f" used twice {np.where(used > 1)[0].tolist()}, skipped {np.where(used == 0)[0].tolist()}")
    if equal:
        diff = np.where(k_near != fr)[0]
        same = len(diff) == 0 and np.array_equal(acc_nearest, acc)
        log(f"   acc_nearest identical to the integer-offset acc: {same}"
            + ("" if same else "; frames that differ " + ", ".join(
                f"{t}: nearest {k_near[t]} (err {1e3 * err_near[t]:+.0f} ms) vs integer {fr[t]} (err "
                f"{1e3 * (np.mod(tau + dt_f * fr[t] - t_conv[t] + period / 2, period) - period / 2):+.0f} ms)" for t in diff)))
    # floor rule: the latest 4D-FlowP frame at or before each conventional frame time (whole frames, always early; with
    # equal frame counts this is the integer-offset `acc` the development case was trained on)
    k_floor = kw[0]
    err_floor = np.mod(tau + dt_f * k_floor - t_conv + period / 2, period) - period / 2
    vel_flowp_floor, acc_floor = to_local(Fv[k_floor]), to_local(Fa[k_floor])
    used_f = np.bincount(k_floor, minlength=Tf)
    log(f"   floor rule: 4D-FlowP frames {k_floor.tolist()}; timing error ms {np.round(1e3 * err_floor).astype(int).tolist()};"
        f" used twice {np.where(used_f > 1)[0].tolist()}, skipped {np.where(used_f == 0)[0].tolist()}"
        + (f"; identical to the integer-offset acc: {np.array_equal(acc_floor, acc)}" if equal else ""))
    log(f"\n3. acc_mode {mode}: tau {1e3 * tau:.1f} ms (equivalent frame shift {frac_equiv:+.2f}); 4D-FlowP fractional frame"
        f" per conventional frame (past {Tf - 1} interpolates to frame 0): " + " ".join(f"{k + w:.2f}" for k, w in zip(kw[0], kw[2])))

    log(f"   velocity agreement in systole (frames {sysf.tolist()}), {int(inside.sum())} inside voxels, stored `acc` timing"
        f" [best clock offset in brackets]:")
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
    log("   4D-FlowP (acc)     : " + " ".join(f"{x:5.0f}" for x in q_fp))
    if equal:
        log("   4D-FlowP best clock: " + " ".join(f"{x:5.0f}" for x in q_fp_frac))
    log(f"   mean: prepared {q_ref.mean():.1f}, 4D-FlowP {q_fp.mean():.1f}, meter {float(prep['meter_flow_mLs']):.1f} mL/s;"
        f" corr over frames {np.corrcoef(q_fp, q_ref)[0, 1]:+.3f} [best clock {np.corrcoef(q_fp_frac, q_ref)[0, 1]:+.3f}]")
    amag = np.linalg.norm(acc[:, inside], axis=-1)
    log("   p99 |a| inside, m/s^2, frames 0-8: " + " ".join(f"{np.percentile(amag[t], 99):5.1f}" for t in range(min(9, T))))
    pk = int(np.argmax(p99))  # the conventional peak-flow frame

    def p99a(x, t):
        return float(np.percentile(np.linalg.norm(x[t][inside], axis=-1), 99))

    peaks = {k: (p99a(x, pk), max(p99a(x, t) for t in range(T))) for k, x in (("acc", acc), ("acc_nearest", acc_nearest), ("acc_floor", acc_floor), ("acc_frac", acc_frac))}
    log(f"   p99 |a| inside at the peak-flow frame {pk} [max over frames], m/s^2: "
        + ", ".join(f"{k} {a:.1f} [{b:.1f}]" for k, (a, b) in peaks.items()))
    calm = np.sort(np.argsort([np.linalg.norm(vel_conv[t], axis=-1).mean() for t in range(T)])[:5])
    noise_c = np.sqrt(np.mean(acc[calm][:, inside] ** 2, axis=(0, 1)))
    noise = float(np.sqrt(np.mean(acc[calm][:, inside] ** 2)))
    calm_f = np.sort(np.argsort([np.linalg.norm(Fv[k, inside], axis=-1).mean() for k in range(Tf)])[:5])
    noise_raw = float(np.sqrt(np.mean(Fa[calm_f][:, inside] ** 2)))
    log(f"   noise: calmest frames {calm.tolist()}, RMS a per local component {np.round(noise_c, 2)}, overall {noise:.2f} m/s^2;"
        f" raw 4D-FlowP frames {calm_f.tolist()}: {noise_raw:.2f} m/s^2")

    # the measured acceleration against the conventional scan's own material acceleration dv/dt + (v . grad) v
    # (local frame; grid axes (k, i, j) carry the local components (c, b, a); periodic central differences in time)
    vol = np.zeros((T,) + mask.shape + (3,), np.float32)
    vol[:, K, I, J] = vel_conv
    h = (sp_j * 1e-3, sp_i * 1e-3, dk * 1e-3)
    gax = (2, 1, 0)
    a_mat = np.zeros((T, N, 3), np.float32)
    for t in range(T):
        span = np.mod(t_conv[(t + 1) % T] - t_conv[(t - 1) % T], period)
        dvdt = (vol[(t + 1) % T] - vol[(t - 1) % T]) / span
        conv = sum(vol[t][..., b : b + 1] * np.gradient(vol[t], h[b], axis=gax[b]) for b in range(3))
        a_mat[t] = (dvdt + conv)[K, I, J]
    del vol
    s2 = inside & ndimage.binary_erosion(mask, iterations=1)[K, I, J]
    log(f"   measured a vs the conventional scan's dv/dt + (v.grad)v (unsmoothed), systole, {int(s2.sum())} interior voxels,"
        " corr / slope [best clock]:")
    for a in range(3):
        r, s = corr_slope(acc[sysf][:, s2, a].ravel(), a_mat[sysf][:, s2, a].ravel())
        rf, sf = corr_slope(acc_frac[sysf][:, s2, a].ravel(), a_mat[sysf][:, s2, a].ravel())
        log(f"     local {LOCAL[a]:12s}: {r:+.3f} / {s:+.3f}   [{rf:+.3f} / {sf:+.3f}]")

    # ---- 4. save
    np.savez_compressed(out_path, acc=acc, vel_flowp=vel_flowp, acc_frac=acc_frac, vel_flowp_frac=vel_flowp_frac, inside=inside,
                        acc_nearest=acc_nearest, vel_flowp_nearest=vel_flowp_nearest, nearest_frame=k_near.astype(np.int64),
                        nearest_err_ms=1e3 * err_near, acc_floor=acc_floor, vel_flowp_floor=vel_flowp_floor,
                        floor_frame=k_floor.astype(np.int64), floor_err_ms=1e3 * err_floor,
                        acc_mode=np.array(mode), time_offset_ms=np.float64(1e3 * tau), frame_offset=np.int64(offset),
                        frame_offset_frac=np.float64(frac_equiv), flowp_frames=np.int64(Tf), flowp_dt_s=np.float64(dt_f),
                        period_s=np.float64(period), axis_signs=signs.astype(np.int64), component_signs=comp_signs.astype(np.int64),
                        translation_mm=trans, translation_mask_mm=trans_mask, translation_refined=np.bool_(accept),
                        affine_lps=affine(signs, trans, sp), dice=np.float64(dice), speed_corr=np.float64(scorr),
                        dice_mask=np.float64(best["dice"]), speed_corr_mask=np.float64(best["speed_corr"]),
                        runner_up_dice=np.float64(runner["dice"]), runner_up_speed_corr=np.float64(runner["speed_corr"]),
                        vel_corr=vcorr, vel_slope=vslope, acc_noise_ms2=np.float64(noise), acc_noise_comp_ms2=noise_c,
                        acc_noise_raw_ms2=np.float64(noise_raw), aao_flux_flowp_mLs=q_fp,
                        aao_flux_corr=np.float64(np.corrcoef(q_fp, q_ref)[0, 1]), case=np.array(case))
    log(f"\nsaved {out_path} ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
