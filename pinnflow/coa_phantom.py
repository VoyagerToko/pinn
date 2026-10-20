"""The public 4D-FlowP aortic coarctation phantom (Stanford Digital Repository, doi:10.25740/ms528wr9373, CC BY-NC 4.0):
real MRI of 3D-printed compliant aortas with a coarctation, in a pulsatile flow loop, with catheter pressure.

    data/stanford_coa/CathPressure/<case>/<case>_cath.mat   LabChart export, 1 kHz: upstream (AAo) and downstream (DAo)
                                                            pressure, their difference, flows; 5 downstream catheter
                                                            positions, one per recording block (comments "DAo cath
                                                            location k"); location 3 is the one the publication uses
    data/stanford_coa/Imaging/<case>/4DFlow_highRes/        Siemens product 4D flow DICOM, 1.8 mm isotropic

Cases: P1_baseline, P1_severe, P2_baseline, P2_severe, each also at a lower flow rate (*_lowflow).
Fetch: python scripts/remote_zip.py https://stacks.stanford.edu/file/druid:ms528wr9373/Imaging.zip --get ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .data import DATA_DIR

COA_DIR = DATA_DIR / "stanford_coa"
PREP_VERSION = 3  # 3: background phase correction + speed-based segmentation; older prepared files are redone
CASES = ["P1_baseline", "P1_severe", "P2_baseline", "P2_severe",
         "P1_baseline_lowflow", "P1_severe_lowflow", "P2_baseline_lowflow", "P2_severe_lowflow"]


def load_catheter(case: str, root: Path = COA_DIR) -> Dict:
    """Catheter recordings of one case: per downstream location k (1..5), the channels of the block whose comment
    is "DAo cath location k" (time in s from the block start, pressures in mmHg, flows in L/min)."""
    import scipy.io

    m = scipy.io.loadmat(root / "CathPressure" / case / f"{case}_cath.mat", squeeze_me=True)
    titles = [str(s).strip() for s in np.atleast_1d(m["titles"])]
    data = np.asarray(m["data"], np.float64)
    ds, de = np.atleast_2d(m["datastart"]).astype(int), np.atleast_2d(m["dataend"]).astype(int)
    fs = np.atleast_2d(m["samplerate"])
    comtext = [str(s).strip() for s in np.atleast_1d(m["comtext"])]
    com = np.atleast_2d(m["com"])
    out = {"titles": titles, "locations": {}}
    for row in com:
        block = int(row[1]) - 1                      # 1-based block number
        text = comtext[int(row[4]) - 1]               # 1-based comment text index
        if not text.lower().startswith("dao cath location"):
            continue
        k = int(text.split()[-1])
        ch = {}
        for c, name in enumerate(titles):
            a, b = ds[c, block], de[c, block]
            if a <= 0 or b < a:
                continue
            y = data[a - 1 : b]                       # 1-based inclusive indices
            ch[name] = (np.arange(len(y)) / fs[c, block], y)
        out["locations"][k] = ch
    return out


def beat_average(t: np.ndarray, y: np.ndarray, ref: np.ndarray, n: int = 200):
    """Ensemble-average a periodic signal over its beats. Beats start at the upward crossings of the reference signal's
    mean (the inflow); returns (phase in s from the beat start (n,), mean (n,), std (n,), period in s, beats)."""
    r = ref - ref.mean()
    up = np.where((r[:-1] < 0) & (r[1:] >= 0))[0]
    if len(up) < 3:
        raise ValueError("fewer than two whole beats in the recording")
    per = np.median(np.diff(t[up]))
    keep = [i for i, j in zip(up[:-1], up[1:]) if 0.8 * per < t[j] - t[i] < 1.2 * per]
    ph = np.linspace(0.0, per, n, endpoint=False)
    beats = np.stack([np.interp(t[i] + ph, t, y) for i in keep])
    return ph, beats.mean(0), beats.std(0), float(per), len(keep)


def catheter_pressure_drop(case: str, location: int = 3, root: Path = COA_DIR) -> Dict:
    """Beat-averaged trans-coarctation pressure drop (upstream minus downstream) at one downstream location."""
    rec = load_catheter(case, root)["locations"][location]
    t, up = rec["Pressure Upstream"]
    _, dn = rec["Pressure Downstream"]
    _, q = rec["Total flow"]
    n = min(len(up), len(dn), len(q))
    t, up, dn, q = t[:n], up[:n], dn[:n], q[:n]
    ph, dp, dp_sd, per, nb = beat_average(t, up - dn, q)
    _, qm, _, _, _ = beat_average(t, q, q)
    _, upm, _, _, _ = beat_average(t, up, q)
    return {"phase_s": ph, "dp_mmHg": dp, "dp_sd": dp_sd, "flow_Lmin": qm, "p_up_mmHg": upm, "period_s": per, "beats": nb,
            "dp_peak_mmHg": float(dp.max()), "dp_mean_mmHg": float(dp.mean())}


# ------------------------------------------------------------------------------------------------------------
# 4D flow MRI (Siemens product sequence, enhanced MR multi-frame DICOM: one file = one slice x all cardiac frames)
# ------------------------------------------------------------------------------------------------------------
FLUID = {"rho": 1100.0, "mu": 0.0042}  # glycerol-water 40/60 (Nair et al., medRxiv 2026); nu = 3.8e-6 m^2/s


def _series_dir(case: str, series: str, suffix: str, root: Path) -> Path:
    base = root / "Imaging" / case / series
    hits = sorted(p for p in base.iterdir() if p.is_dir() and p.name.endswith(suffix) and (suffix or "_" not in p.name.split("venc")[-1]))
    if not hits:
        raise FileNotFoundError(f"no {series}*{suffix} series under {base}")
    return hits[0]


def _frames_of(path: Path):
    """(frame trigger times ms, slice position (3,), row dir, col dir, pixel spacing, rescale, venc dir, venc, pixels)."""
    import pydicom

    ds = pydicom.dcmread(path)
    sh = ds.SharedFunctionalGroupsSequence[0]
    pf = ds.PerFrameFunctionalGroupsSequence

    def g(name, i=0):
        return pf[i][name][0] if name in pf[i] else sh[name][0]

    trig = np.array([float(pf[i].CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime) for i in range(len(pf))])
    pos = np.array(g("PlanePositionSequence").ImagePositionPatient, float)
    ori = np.array(g("PlaneOrientationSequence").ImageOrientationPatient, float)
    sp = np.array(g("PixelMeasuresSequence").PixelSpacing, float)
    pvt = g("PixelValueTransformationSequence")
    slope, icpt = float(pvt.RescaleSlope), float(pvt.RescaleIntercept)
    venc_dir, venc = None, None
    if "MRVelocityEncodingSequence" in pf[0] or "MRVelocityEncodingSequence" in sh:
        mv = g("MRVelocityEncodingSequence")
        venc_dir = np.array(mv.VelocityEncodingDirection, float)
        venc = float(mv.VelocityEncodingMaximumValue)  # cm/s
    px = ds.pixel_array.astype(np.float64) * slope + icpt  # (frames, rows, cols)
    return trig, pos, ori[:3], ori[3:], sp, venc_dir, venc, px


def load_4dflow(case: str, series: str = "4DFlow_highRes", root: Path = COA_DIR) -> Dict:
    """Velocity (frames, nk, ni, nj, 3) in m/s along patient axes (LPS), magnitude (frames, nk, ni, nj), the voxel
    centres' patient coordinates in mm (nk, ni, nj, 3), trigger times in s, voxel size in mm and VENC in m/s.

    Index order: k = slice, i = image row, j = image column. The phase images map linearly to [-VENC, +VENC] over the
    signed 12-bit range [-4096, 4096) after rescaling (Siemens convention).
    """
    pdir = _series_dir(case, series, "_P", root)
    mdir = _series_dir(case, series, "_MAG", root)
    files = sorted(pdir.glob("*.dcm"))
    by_dir: Dict[tuple, list] = {}
    meta = None
    for f in files:
        trig, pos, rdir, cdir, sp, vdir, venc, px = _frames_of(f)
        key = tuple(np.round(vdir, 3))
        by_dir.setdefault(key, []).append((pos, px))
        meta = (trig, rdir, cdir, sp, venc)
    trig, rdir, cdir, sp, venc = meta
    normal = np.cross(rdir, cdir)
    vel = np.zeros(None or (len(trig),) + (len(next(iter(by_dir.values()))),) + next(iter(by_dir.values()))[0][1].shape[1:] + (3,))
    positions = None
    for key, items in by_dir.items():
        items.sort(key=lambda it: float(np.dot(it[0], normal)))
        stack = np.stack([px for _, px in items], axis=1)  # (frames, nk, ni, nj)
        v = stack / 4096.0 * venc / 100.0  # m/s along the encoding direction
        vel += v[..., None] * np.asarray(key)[None, None, None, None, :]
        positions = np.stack([p for p, _ in items])  # (nk, 3)
    mag_items = []
    for f in sorted(mdir.glob("*.dcm")):
        trig_m, pos, _, _, _, vdir, _, px = _frames_of(f)
        mag_items.append((pos, px))
    mag_items.sort(key=lambda it: float(np.dot(it[0], normal)))
    # three magnitude images per slice (one per encoding): average them
    nk = positions.shape[0]
    mag = np.stack([px for _, px in mag_items], axis=1)
    mag = mag.reshape(mag.shape[0], nk, -1, *mag.shape[2:]).mean(axis=2) if mag.shape[1] == 3 * nk else mag
    ni, nj = vel.shape[2], vel.shape[3]
    I, J = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    xyz = positions[:, None, None, :] + (J[None, ..., None] * sp[1]) * rdir + (I[None, ..., None] * sp[0]) * cdir
    dk = float(np.dot(positions[1] - positions[0], normal))
    vel, bg = correct_background(vel, mag)
    return {"velocity": vel, "magnitude": mag, "background": bg, "xyz_mm": xyz, "t_s": trig / 1000.0, "voxel_mm": (float(sp[0]), float(sp[1]), abs(dk)),
            "venc_ms": venc / 100.0, "row_dir": rdir, "col_dir": cdir, "normal": normal, "case": case, "series": series}


# ------------------------------------------------------------------------------------------------------------
# segmentation and the training set for one case
# ------------------------------------------------------------------------------------------------------------
def _shading_corrected(mag: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mmean = mag.mean(0)
    return mmean / (ndimage.gaussian_filter(mmean, 12) + 1e-6)


def correct_background(vel: np.ndarray, mag: np.ndarray, order: int = 2):
    """Eddy-current background phase correction, the standard first step of 4D-flow processing.

    Static tissue (here the gel around the phantom) should read zero velocity, but eddy currents add a smooth,
    time-constant offset to the phase. Static voxels: enough signal, not vessel-bright, and among the steadiest over
    the beat. A polynomial of the given order in the three voxel indices is fitted to their time-mean velocity, per
    component, and subtracted from every frame. Returns the corrected velocity and a summary of the fit."""
    norm = _shading_corrected(mag)
    mmean = mag.mean(0)
    sd = vel.std(0).max(-1)  # (k, i, j) largest temporal std over components
    signal = (mmean > np.percentile(mmean, 40)) & (norm < 1.25)
    steady = sd < np.percentile(sd[signal], 30) if signal.any() else signal
    static = signal & steady
    K, I, J = np.nonzero(static)
    if len(K) < 200:
        return vel, {"order": order, "static_voxels": int(len(K)), "applied": False}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(K), size=min(len(K), 60000), replace=False)
    shape = np.array(vel.shape[1:4], float)

    def basis(k, i, j):
        x, y, z = (k / shape[0] - 0.5) * 2, (i / shape[1] - 0.5) * 2, (j / shape[2] - 0.5) * 2
        cols = [np.ones_like(x)]
        for p in range(1, order + 1):
            for a_ in range(p + 1):
                for b_ in range(p + 1 - a_):
                    c_ = p - a_ - b_
                    cols.append(x**a_ * y**b_ * z**c_)
        return np.stack(cols, -1)

    A = basis(K[pick].astype(float), I[pick].astype(float), J[pick].astype(float))
    target = vel.mean(0)[K[pick], I[pick], J[pick]]  # (n, 3)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    kk, ii, jj = np.meshgrid(*[np.arange(n, dtype=float) for n in vel.shape[1:4]], indexing="ij")
    field = basis(kk, ii, jj) @ coef  # (k, i, j, 3)
    before = float(np.abs(vel.mean(0)[static]).mean())
    out = vel - field[None]
    after = float(np.abs(out.mean(0)[static]).mean())
    return out, {"order": order, "static_voxels": int(len(K)), "applied": True, "static_mean_abs_before_ms": before, "static_mean_abs_after_ms": after}


def segment_lumen(d: Dict, bright: float = 1.5, grow: float = 1.3, speed_seed: float = 0.12, speed_grow: float = 0.06) -> np.ndarray:
    """Lumen mask from the background-corrected scan.

    Seed: contrast-bright fluid (shading-corrected magnitude > ``bright``) that moves (time-mean speed > ``speed_seed``
    m/s); the largest connected piece is the aorta. It then grows into connected voxels that are moderately bright and
    still moving (> ``grow``, > ``speed_grow``), which adds the head-and-neck branches but not the static gel or the
    coil-shaded band at the edge of the volume. Holes are filled."""
    from scipy import ndimage

    norm = _shading_corrected(d["magnitude"])
    msp = np.linalg.norm(d["velocity"], axis=-1).mean(0)
    seed = ndimage.binary_opening((norm > bright) & (msp > speed_seed), iterations=1)
    lab, n = ndimage.label(seed)
    if n == 0:
        raise ValueError(f"{d.get('case')}: no moving, contrast-bright fluid found")
    seed = lab == (int(np.argmax(ndimage.sum(seed, lab, range(1, n + 1)))) + 1)
    grow_ok = ndimage.binary_opening((norm > grow) & (msp > speed_grow), iterations=1)
    lum = ndimage.binary_propagation(seed, mask=grow_ok | seed)
    return ndimage.binary_fill_holes(lum)


def wall_faces(mask: np.ndarray):
    """Centres (in voxel index units, k i j) and outward normals of the faces between lumen and non-lumen voxels.
    Faces on the volume border are open vessel ends (flow leaves the field of view there) and are not walls."""
    pts, nrm = [], []
    for ax in range(3):
        for s in (1, -1):
            nb = np.roll(mask, -s, axis=ax)
            border = np.zeros_like(mask)
            idx = [slice(None)] * 3
            idx[ax] = -1 if s == 1 else 0
            border[tuple(idx)] = True
            face = mask & ~nb & ~border
            k = np.argwhere(face).astype(np.float64)
            k[:, ax] += 0.5 * s
            n = np.zeros((len(k), 3))
            n[:, ax] = s
            pts.append(k)
            nrm.append(n)
    return np.concatenate(pts), np.concatenate(nrm)


def prepare_case(case: str, out: Path, root: Path = COA_DIR, z_aao: float = None, z_dao: float = None) -> Dict:
    """One case as a training set (npz): lumen voxels' velocities per frame in a local orthonormal frame aligned with
    the scan (a = image column, b = image row, c = slice; metres), wall face points, the two catheter planes, and the
    beat-averaged catheter pressure drop. Planes default to the 2D phase-contrast inlet (AAo) and outlet (DAo) planes."""
    import pydicom

    d = load_4dflow(case, root=root)
    mask = segment_lumen(d)
    rdir, cdir, nrm = d["row_dir"], d["col_dir"], d["normal"]
    sp_i, sp_j, dk = d["voxel_mm"]
    v = d["velocity"]
    loc = np.stack([np.einsum("tkijc,c->tkij", v, e) for e in (rdir, cdir, nrm)], -1)  # (T, nk, ni, nj, 3) local axes
    K, I, J = np.nonzero(mask)
    pos_mm = np.stack([J * sp_j, I * sp_i, K * dk], -1)  # local (a, b, c) in mm, origin at voxel (0, 0, 0)
    origin = d["xyz_mm"][0, 0, 0]
    # the catheter planes: the 2D phase-contrast inlet (AAo) and outlet (DAo) planes, which are tilted. The plane's cells
    # are the lumen voxels within half a voxel of it; the plane cuts both the ascending and the descending aorta, so the
    # connected piece is chosen by its flux along the plane normal (patient +z is head-wards): ascending flows head-wards.
    def plane_of(series):
        fs = sorted((root / "Imaging" / case / series).rglob("*.dcm"))
        ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
        pf = ds.PerFrameFunctionalGroupsSequence[0]
        sh = ds.SharedFunctionalGroupsSequence[0]
        pos = np.array(pf.PlanePositionSequence[0].ImagePositionPatient, float)
        ori = np.array((pf.get("PlaneOrientationSequence") or sh.PlaneOrientationSequence)[0].ImageOrientationPatient, float)
        n = np.cross(ori[:3], ori[3:])
        return pos, n / np.linalg.norm(n) * np.sign(n[2] if abs(n[2]) > 1e-6 else 1.0)

    def plane_cells(series, ascending):
        from scipy import ndimage

        pos, n = plane_of(series)
        dist = (d["xyz_mm"] - pos) @ n  # (nk, ni, nj) signed distance in mm
        slab = mask & (np.abs(dist) <= 0.5 * float(np.mean(d["voxel_mm"])))
        lab, nc = ndimage.label(slab)
        vn = np.einsum("tkijc,c->tkij", v, n)  # velocity along the plane normal
        best, best_q = None, None
        for c in range(1, nc + 1):
            cm = lab == c
            if cm.sum() < 10:
                continue
            q = float(vn[:, cm].sum(-1).mean())
            if best is None or (q > best_q if ascending else q < best_q):
                best, best_q = cm, q
        if best is None:
            raise ValueError(f"{case}: the {series} plane does not cut the lumen")
        kk, ii, jj = np.nonzero(best)
        return np.stack([jj * sp_j, ii * sp_i, kk * dk], -1), n

    aao, n_aao = plane_cells("2DPC_Inlet", True)
    dao, n_dao = plane_cells("2DPC_Outlet", False)
    # The scan's flow through the upstream plane is checked against the phantom's total-flow meter. Close to the arch,
    # complex flow loses 4D-flow signal and the measured flux falls short of the meter; if it is off by more than 15%,
    # the upstream plane moves down the ascending aorta (a horizontal cut, wide and slow there: the pressure changes by
    # well under 1 mmHg along it) to the nearest height where the flux agrees within 10%. The shift is recorded.
    cath0 = catheter_pressure_drop(case, 3, root)
    q_meter = float(cath0["flow_Lmin"].mean()) * 1000 / 60  # mL/s
    vox_m3 = sp_i * sp_j * dk * 1e-9

    def flux_cells(cells_mm, n_world):
        kk = np.round(cells_mm[:, 2] / dk).astype(int)
        ii = np.round(cells_mm[:, 1] / sp_i).astype(int)
        jj = np.round(cells_mm[:, 0] / sp_j).astype(int)
        vn = np.einsum("tnc,c->tn", v[:, kk, ii, jj, :], n_world)
        return float(vn.sum(-1).mean()) * vox_m3 / (np.mean([sp_i, sp_j, dk]) * 1e-3) * 1e6

    q_plane = flux_cells(aao, n_aao)
    aao_shift_mm = 0.0
    if not 0.85 < q_plane / q_meter < 1.15:
        from scipy import ndimage

        i0 = int(round(aao[:, 1].mean() / sp_i))
        head = -cdir  # head-wards
        for di in list(range(1, 40)):
            i = i0 + di  # down the ascending aorta, away from the arch
            if i >= mask.shape[1]:
                break
            lab, nc = ndimage.label(mask[:, i, :])
            comps = [(float(np.einsum("tnc,c->tn", v[:, lab == c][:, :, :] if False else v[:, :, i, :][:, lab == c], head).sum(-1).mean()), c)
                     for c in range(1, nc + 1) if (lab == c).sum() >= 10]
            if not comps:
                continue
            q_row, c = max(comps)
            q_row *= sp_j * dk * 1e-6 * 1e6
            if 0.9 < q_row / q_meter < 1.1:
                kk, jj = np.nonzero(lab == c)
                aao = np.stack([jj * sp_j, np.full(len(kk), i * sp_i), kk * dk], -1)
                n_aao = head
                aao_shift_mm = di * sp_i
                q_plane = q_row
                break
    b_aao, b_dao = float(aao[:, 1].mean()), float(dao[:, 1].mean())
    # the normals in the local (a, b, c) frame, for flux checks
    n_aao_loc = np.array([n_aao @ rdir, n_aao @ cdir, n_aao @ nrm])
    n_dao_loc = np.array([n_dao @ rdir, n_dao @ cdir, n_dao @ nrm])
    wf, wn = wall_faces(mask)
    wall_mm = np.stack([wf[:, 2] * sp_j, wf[:, 1] * sp_i, wf[:, 0] * dk], -1)
    cath = catheter_pressure_drop(case, 3, root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t_s=d["t_s"], vel=loc[:, K, I, J, :].astype(np.float32), pos_mm=pos_mm.astype(np.float32),
                        wall_mm=wall_mm.astype(np.float32), aao_mm=aao.astype(np.float32), dao_mm=dao.astype(np.float32),
                        prep_version=np.int64(PREP_VERSION),
                        aao_normal=n_aao_loc, dao_normal=n_dao_loc, aao_shift_mm=np.float64(aao_shift_mm),
                        aao_flux_mLs=np.float64(q_plane), meter_flow_mLs=np.float64(q_meter),
                        voxel_mm=np.array([sp_j, sp_i, dk]), frame_window_s=np.float64(np.median(np.diff(d["t_s"]))),
                        period_s=np.float64(cath["period_s"]), venc_ms=np.float64(d["venc_ms"]),
                        cath_phase_s=cath["phase_s"], cath_dp_mmHg=cath["dp_mmHg"], cath_dp_peak_mmHg=np.float64(cath["dp_peak_mmHg"]),
                        rho=np.float64(FLUID["rho"]), mu=np.float64(FLUID["mu"]), case=np.array(case))
    return {"voxels": int(len(K)), "frames": int(v.shape[0]), "wall_faces": int(len(wf)), "aao_cells": int(len(aao)),
            "dao_cells": int(len(dao)), "b_aao_mm": b_aao, "b_dao_mm": b_dao, "aao_shift_mm": aao_shift_mm,
            "aao_flux_mLs": q_plane, "meter_flow_mLs": q_meter, "period_s": cath["period_s"], "cath_dp_peak": cath["dp_peak_mmHg"]}


# ------------------------------------------------------------------------------------------------------------
# baselines from the same scan: the conventional pressure Poisson estimate, and simplified Bernoulli
# ------------------------------------------------------------------------------------------------------------
def ppe_pressure_drop(case: str, root: Path = COA_DIR, mask: np.ndarray = None) -> Dict:
    """Conventional 4D-flow pressure mapping on the voxel grid (pressure Poisson estimator).

    b = -rho (dv/dt + (v . grad) v) + mu lap v from the measured velocity (central differences; periodic in time), then
    a finite-volume Poisson solve on the lumen voxels, sum over interior faces of (p_n - p) / h = average(b) . n, which
    carries the Neumann condition dp/dn = b . n on the wall exactly. One voxel of the downstream plane is the gauge.
    Returns the section-averaged pressure difference between the two catheter planes per frame, in mmHg.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    d = load_4dflow(case, root=root)
    mask = segment_lumen(d) if mask is None else mask
    prep = dict(np.load(root / "prepared" / f"{case}.npz"))
    rho, mu = FLUID["rho"], FLUID["mu"]
    rdir, cdir, nrm = d["row_dir"], d["col_dir"], d["normal"]
    v = np.stack([np.einsum("tkijc,c->tkij", d["velocity"], e) for e in (rdir, cdir, nrm)], -1)  # (T, k, i, j, comp a b c)
    v = v * mask[None, ..., None]
    sp_i, sp_j, dk = (s / 1000.0 for s in d["voxel_mm"])
    h = {"a": sp_j, "b": sp_i, "c": dk}
    axis = {"a": 3, "b": 2, "c": 1}  # array axes of (T, k, i, j)
    t = d["t_s"]
    T = len(t)
    per = float(prep["period_s"])
    tt = np.concatenate([t[-1:] - per, t, t[:1] + per])
    vv = np.concatenate([v[-1:], v, v[:1]], 0)
    dvdt = (vv[2:] - vv[:-2]) / (tt[2:] - tt[:-2])[:, None, None, None, None]

    def grad(f, ax_name):
        return np.gradient(f, h[ax_name], axis=axis[ax_name])

    b = np.zeros_like(v)
    for comp in range(3):
        f = v[..., comp]
        conv = sum(v[..., j] * grad(f, n) for j, n in enumerate("abc"))
        lap = sum(np.gradient(grad(f, n), h[n], axis=axis[n]) for n in "abc")
        b[..., comp] = -rho * (dvdt[..., comp] + conv) + mu * lap
    b *= mask[None, ..., None]
    aao_vox = _plane_voxels(prep["aao_mm"], sp_i, sp_j, dk)
    dao_vox = _plane_voxels(prep["dao_mm"], sp_i, sp_j, dk)
    dp = poisson_plane_drop(mask, b, (dk, sp_i, sp_j), aao_vox, dao_vox) / 133.322
    dp = np.asarray(dp)
    return {"t_s": t, "dp_mmHg": dp, "dp_peak_mmHg": float(dp.max()), "dp_mean_mmHg": float(dp.mean())}


def _plane_voxels(plane_mm, sp_i, sp_j, dk):
    return np.round(np.stack([plane_mm[:, 2] / (dk * 1000), plane_mm[:, 1] / (sp_i * 1000), plane_mm[:, 0] / (sp_j * 1000)], -1)).astype(int)


def poisson_plane_drop(mask: np.ndarray, b: np.ndarray, h_kij, aao_vox: np.ndarray, dao_vox: np.ndarray) -> np.ndarray:
    """Solve grad p = b (least squares, as a finite-volume Poisson problem on the lumen voxels with dp/dn = b . n on the
    wall) for each frame of b (T, k, i, j, 3; components along a = j, b = i, c = k), and return the section-mean
    pressure difference between the two planes' voxels (Pa). The gauge is one downstream-plane voxel."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    idx = -np.ones(mask.shape, int)
    idx[mask] = np.arange(mask.sum())
    n_cells = int(mask.sum())
    rows, cols, vals = [], [], []
    diag = np.zeros(n_cells)
    faces = []
    for comp, ax in ((0, 2), (1, 1), (2, 0)):  # component a lives on array axis j (2), b on i (1), c on k (0)
        hh = h_kij[ax]
        src = np.argwhere(mask)
        dst = src.copy()
        dst[:, ax] += 1
        ok = dst[:, ax] < mask.shape[ax]
        ok[ok] = mask[tuple(dst[ok].T)]
        s_, d_ = src[ok], dst[ok]
        a_, n_ = idx[tuple(s_.T)], idx[tuple(d_.T)]
        rows += [a_, n_]
        cols += [n_, a_]
        vals += [np.full(len(a_), 1.0 / hh**2)] * 2
        np.add.at(diag, a_, -1.0 / hh**2)
        np.add.at(diag, n_, -1.0 / hh**2)
        faces.append((s_, d_, a_, n_, comp, hh))
    rows.append(np.arange(n_cells))
    cols.append(np.arange(n_cells))
    vals.append(diag)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_cells, n_cells)).tolil()
    g = idx[tuple(dao_vox[len(dao_vox) // 2])]
    A[g, :] = 0
    A[g, g] = 1.0
    lu = splu(A.tocsc())
    ia, idd = idx[tuple(aao_vox.T)], idx[tuple(dao_vox.T)]
    out = []
    for f in range(b.shape[0]):
        rhs = np.zeros(n_cells)
        for s_, d_, a_, n_, comp, hh in faces:
            bf = 0.5 * (b[f][tuple(s_.T)][:, comp] + b[f][tuple(d_.T)][:, comp])
            np.add.at(rhs, a_, bf / hh)
            np.add.at(rhs, n_, -bf / hh)
        rhs[g] = 0.0
        p = lu.solve(rhs)
        out.append(p[ia].mean() - p[idd].mean())
    return np.asarray(out)


BERNOULLI_VERSION = 2  # 2: 3x3x3 median filter, 99.5th percentile (isolated noisy or aliased voxels cannot set the peak)


def bernoulli_peak(case: str, root: Path = COA_DIR, mask: np.ndarray = None, q: float = 99.5) -> Dict:
    """Simplified Bernoulli from the scan: 4 v_peak^2 (mmHg, m/s). v_peak is the q-th percentile speed in the lumen at
    the fastest frame, after a 3x3x3 median filter, the way a Doppler reading reports a jet's core rather than one
    outlier: single voxels with phase noise or aliasing would otherwise reach sqrt(3) x VENC."""
    from scipy import ndimage

    d = load_4dflow(case, root=root)
    mask = segment_lumen(d) if mask is None else mask
    sp = np.linalg.norm(d["velocity"], axis=-1)
    vpk = max(float(np.percentile(ndimage.median_filter(sp[f], size=3)[mask], q)) for f in range(sp.shape[0]))
    return {"v_peak_ms": vpk, "dp_peak_mmHg": 4.0 * vpk**2}
