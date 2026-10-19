"""Real-world use case: the pressure drop across an arterial stenosis, inferred from MRI-like velocity data.

A synthetic, controlled study (not clinical). Three pieces:

* ``StenosisVessel``: a 2D vessel of diameter D = 1 with a smooth cosine narrowing and a pulsatile, heartbeat-shaped
  inflow. Everything is non-dimensional (lengths in D, velocities in the mean inflow speed U, time in D/U, pressure
  in rho U^2); ``physical()`` gives the factors to m/s, s and mmHg for a small artery (D = 6 mm, blood
  nu = 3.5e-6 m^2/s, rho = 1060 kg/m^3).
* ``solve_cfd``: the ground truth. Incompressible Navier-Stokes on a staggered (MAC) grid with the vessel wall as a
  staircase of solid cells, third-order upwind convection, Adams-Bashforth 2 in time and a projection step whose
  Poisson matrix is factorised once (scipy splu). Parabolic pulsatile inflow, p = 0 at the outlet.
* ``mri_from_cfd``: what a 4D-flow MRI scan of that flow would deliver: velocity averaged over voxels (0.1 D =
  0.6 mm) and over cardiac phases (24 per beat), only in voxels segmented as lumen, with Gaussian phase noise of
  sigma = sqrt(2) VENC / (pi SNR) per component.

The network (``pinnflow.problems.stenosis_inverse``) sees only the MRI data and the wall geometry, and must infer the
pressure; the scoring compares its trans-stenotic pressure drop with the CFD truth and with the simplified Bernoulli
rule used clinically with Doppler ultrasound, dp [mmHg] = 4 v_peak [m/s]^2.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

MMHG = 133.322  # Pa


@dataclass(frozen=True)
class StenosisVessel:
    length: float = 14.0          # domain length, in diameters (the outlet is far downstream of the field of view)
    xc: float = 3.0               # stenosis centre
    half_len: float = 1.0         # half-length of the narrowing
    severity: float = 0.6         # diameter reduction at the throat (0.6: the throat is 40% of D)
    Re: float = 150.0             # U D / nu with U the cycle-mean inflow speed
    period: float = 12.0          # cardiac period in D / U (12 -> 0.82 s, 73 beats per minute, at the scale below)
    harmonics: Tuple[Tuple[float, float], ...] = ((0.6, 0.0), (0.2, 0.5))  # (amplitude, phase) of U(t)/U - 1
    fov: Tuple[float, float] = (0.5, 9.5)  # the scanned field of view along the vessel
    proximal: float = 1.0         # pressure station upstream of the throat (x = xc - 2)
    D_m: float = 6e-3
    nu_m2s: float = 3.5e-6
    rho: float = 1060.0

    # ---- geometry --------------------------------------------------------------------------------------
    def half_width(self, x, xp=np):
        s = xp.clip((x - self.xc) / self.half_len, -1.0, 1.0)
        bump = 0.5 * (1.0 + xp.cos(np.pi * s))
        return 0.5 * (1.0 - self.severity * bump)

    def inside(self, x, y, xp=np):
        return xp.abs(y) < self.half_width(x, xp)

    @property
    def throat(self) -> float:
        return self.xc

    @property
    def station_up(self) -> float:
        return self.proximal

    # ---- inflow ----------------------------------------------------------------------------------------
    def flow(self, t, xp=np):
        """U(t) / U_mean: the cycle-mean is exactly 1."""
        w = 2 * np.pi / self.period
        out = 1.0
        for k, (a, ph) in enumerate(self.harmonics, start=1):
            out = out + a * xp.sin(k * w * t + ph)
        return out

    def inflow_profile(self, t, y, xp=np):
        return 1.5 * self.flow(t, xp) * (1.0 - (y / 0.5) ** 2)

    # ---- physical scale --------------------------------------------------------------------------------
    def physical(self) -> Dict[str, float]:
        U = self.Re * self.nu_m2s / self.D_m
        return {
            "U_ms": U,
            "D_mm": self.D_m * 1e3,
            "period_s": self.period * self.D_m / U,
            "heart_rate_bpm": 60.0 / (self.period * self.D_m / U),
            "mmHg_per_unit": self.rho * U**2 / MMHG,
        }

    def to_json(self) -> Dict:
        d = asdict(self)
        d["harmonics"] = [list(h) for h in self.harmonics]
        d["fov"] = list(self.fov)
        d["physical"] = self.physical()
        return d

    @staticmethod
    def from_json(d: Dict) -> "StenosisVessel":
        d = {k: v for k, v in d.items() if k != "physical"}
        d["harmonics"] = tuple(tuple(h) for h in d.get("harmonics", ()))
        d["fov"] = tuple(d.get("fov", (0.5, 9.5)))
        return StenosisVessel(**d)


# ------------------------------------------------------------------------------------------------------------
# ground truth: staggered-grid projection solver
# ------------------------------------------------------------------------------------------------------------
def _pad_odd(f, axis, n=2):
    """Pad with antisymmetric ghosts (a no-slip wall half a cell beyond the last value)."""
    pad = [(0, 0)] * f.ndim
    pad[axis] = (n, n)
    g = np.pad(f, pad, mode="symmetric")
    sl_lo = [slice(None)] * f.ndim
    sl_hi = [slice(None)] * f.ndim
    sl_lo[axis] = slice(0, n)
    sl_hi[axis] = slice(-n, None)
    g[tuple(sl_lo)] *= -1
    g[tuple(sl_hi)] *= -1
    return g


def _upwind3(f_pad, a, axis, h):
    """a * df/dx, third-order upwind-biased, from f padded by 2 on ``axis``."""
    n = f_pad.shape[axis] - 4

    def s(k):
        sl = [slice(None)] * f_pad.ndim
        sl[axis] = slice(2 + k, 2 + k + n)
        return f_pad[tuple(sl)]

    c4 = (-s(2) + 8 * s(1) - 8 * s(-1) + s(-2)) / (12 * h)
    d4 = (s(2) - 4 * s(1) + 6 * s(0) - 4 * s(-1) + s(-2)) / (12 * h)
    return a * c4 + np.abs(a) * d4


def solve_cfd(v: StenosisVessel, N: int = 50, cfl: float = 0.3, n_periods: int = 2, frames: int = 96,
              log: Optional[Callable[[str], None]] = print, t_end: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Integrate ``n_periods`` cardiac cycles from rest and return the last one as ``frames`` snapshots.

    Returns cell-centred u, v, p of shape (frames, nx, ny) (NaN in solid cells), the cell centres, the fluid mask,
    and the snapshot times shifted to [0, period). With ``t_end`` the run stops there and returns one snapshot
    (used by the tests).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    h = 1.0 / N
    nx, ny = int(round(v.length * N)), N
    xc = (np.arange(nx) + 0.5) * h
    yc = -0.5 + (np.arange(ny) + 0.5) * h
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    F = np.abs(YC) < v.half_width(XC)
    Fu = np.zeros((nx + 1, ny), bool)
    Fu[1:nx] = F[:-1] & F[1:]
    Fv = np.zeros((nx, ny + 1), bool)
    Fv[:, 1:ny] = F[:, :-1] & F[:, 1:]
    rows_in, rows_out = F[0].copy(), F[-1].copy()

    # Poisson matrix on fluid cells: Neumann at walls and inlet, Dirichlet (phi = 0) at the outlet face
    idx = -np.ones((nx, ny), int)
    idx[F] = np.arange(F.sum())
    I, J, V = [], [], []
    diag = np.zeros(F.sum())
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        src = np.argwhere(F)
        ni, nj = src[:, 0] + di, src[:, 1] + dj
        inb = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        nb_fluid = np.zeros(len(src), bool)
        nb_fluid[inb] = F[ni[inb], nj[inb]]
        a = idx[src[:, 0], src[:, 1]]
        I.append(a[nb_fluid])
        J.append(idx[ni[nb_fluid], nj[nb_fluid]])
        V.append(np.ones(nb_fluid.sum()))
        diag[a[nb_fluid]] -= 1.0
        if di == 1:  # the outlet face
            out = ni == nx
            diag[a[out]] -= 2.0
    I.append(np.arange(F.sum()))
    J.append(np.arange(F.sum()))
    V.append(diag)
    A = coo_matrix((np.concatenate(V) / h**2, (np.concatenate(I), np.concatenate(J))), shape=(F.sum(), F.sum())).tocsc()
    lu = splu(A)

    u = np.zeros((nx + 1, ny))
    w = np.zeros((nx, ny + 1))
    p = np.zeros((nx, ny))
    nu = 1.0 / v.Re

    umax = 1.5 * (1.0 + sum(a for a, _ in v.harmonics)) * 0.5 / v.half_width(v.xc)
    # convective CFL, and explicit diffusion: AB2 is stable on the negative real axis only down to -1, and the
    # 5-point Laplacian (with no-slip ghosts) reaches about -10 nu/h^2, so nu dt / h^2 must stay below ~0.1
    dt_cfl = min(cfl * h / umax, 0.08 * h**2 * v.Re)
    T = v.period
    frame_dt = T / frames
    n_sub = int(np.ceil(frame_dt / dt_cfl))
    dt = frame_dt / n_sub
    total_frames = n_periods * frames if t_end is None else int(np.ceil(t_end / frame_dt))
    if log:
        log(f"[cfd] grid {nx}x{ny} (h = {h:.4f}), fluid cells {F.sum()}, dt = {dt:.2e}, steps = {total_frames * n_sub}")

    def explicit(u, w):
        # u-momentum at u faces
        w_at_u = np.zeros_like(u)
        wc = 0.5 * (w[:, :-1] + w[:, 1:])                     # v at cell centres (nx, ny)
        w_at_u[1:nx] = 0.5 * (wc[:-1] + wc[1:])
        u_px = np.pad(u, ((2, 2), (0, 0)), mode="edge")
        u_py = _pad_odd(u, 1)
        Hu = -(_upwind3(u_px, u, 0, h) + _upwind3(u_py, w_at_u, 1, h))
        uN = np.where(np.pad(Fu, ((0, 0), (0, 1)))[:, 1:], np.pad(u, ((0, 0), (0, 1)))[:, 1:], -u)
        uS = np.where(np.pad(Fu, ((0, 0), (1, 0)))[:, :-1], np.pad(u, ((0, 0), (1, 0)))[:, :-1], -u)
        uE = np.pad(u, ((0, 1), (0, 0)), mode="edge")[1:]
        uW = np.pad(u, ((1, 0), (0, 0)), mode="edge")[:-1]
        Hu += nu * (uE + uW + uN + uS - 4 * u) / h**2
        Hu[~Fu] = 0.0
        # v-momentum at v faces
        u_at_w = np.zeros_like(w)
        ucell = 0.5 * (u[:-1] + u[1:])                         # u at cell centres (nx, ny)
        u_at_w[:, 1:ny] = 0.5 * (ucell[:, :-1] + ucell[:, 1:])
        w_px = np.pad(w, ((2, 2), (0, 0)), mode="edge")
        w_py = np.pad(w, ((0, 0), (2, 2)), mode="edge")
        Hw = -(_upwind3(w_px, u_at_w, 0, h) + _upwind3(w_py, w, 1, h))
        Fw_cells_E = np.pad(F, ((0, 1), (0, 0)))[1:]  # is the cell to the east fluid
        sideE = np.zeros_like(Fv)
        sideE[:, 1:ny] = Fw_cells_E[:, :-1] & Fw_cells_E[:, 1:]
        Fw_cells_W = np.pad(F, ((1, 0), (0, 0)))[:-1]
        sideW = np.zeros_like(Fv)
        sideW[:, 1:ny] = Fw_cells_W[:, :-1] & Fw_cells_W[:, 1:]
        wE_raw = np.pad(w, ((0, 1), (0, 0)), mode="edge")[1:]
        wW_raw = np.pad(w, ((1, 0), (0, 0)))[:-1]
        wE = np.where(sideE | (np.arange(nx)[:, None] == nx - 1), wE_raw, -w)  # outlet: zero gradient
        wW = np.where(sideW, wW_raw, -w)                                       # inlet and side walls: no slip
        wN = np.pad(w, ((0, 0), (0, 1)))[:, 1:]
        wS = np.pad(w, ((0, 0), (1, 0)))[:, :-1]
        Hw += nu * (wE + wW + wN + wS - 4 * w) / h**2
        Hw[~Fv] = 0.0
        return Hu, Hw

    def project(us, ws, t_new):
        us[0, rows_in] = v.inflow_profile(t_new, yc[rows_in])
        us[0, ~rows_in] = 0.0
        us[nx, rows_out] = us[nx - 1, rows_out]
        us[nx, ~rows_out] = 0.0
        us[~Fu & ~np.isin(np.arange(nx + 1), [0, nx])[:, None]] = 0.0
        ws[~Fv] = 0.0
        div = (us[1:] - us[:-1]) / h + (ws[:, 1:] - ws[:, :-1]) / h
        phi = np.zeros((nx, ny))
        phi[F] = lu.solve(div[F] / dt)
        us[1:nx] -= dt * (phi[1:] - phi[:-1]) / h * Fu[1:nx]
        us[nx, rows_out] -= dt * (-2.0 * phi[-1, rows_out]) / h
        ws[:, 1:ny] -= dt * (phi[:, 1:] - phi[:, :-1]) / h * Fv[:, 1:ny]
        return us, ws, phi

    t = 0.0
    Hu_old = Hw_old = None
    keep_from = (n_periods - 1) * frames if t_end is None else total_frames - 1
    out_u, out_v, out_p, out_t = [], [], [], []
    i_up, i_th = int(np.argmin(np.abs(xc - v.station_up))), int(np.argmin(np.abs(xc - v.throat)))
    dp_all = []  # the trans-stenotic pressure drop at every frame of every cycle: the periodicity check
    t0 = time.time()
    for fr in range(total_frames):
        for _ in range(n_sub):
            Hu, Hw = explicit(u, w)
            if Hu_old is None:
                us, ws = u + dt * Hu, w + dt * Hw
            else:
                us, ws = u + dt * (1.5 * Hu - 0.5 * Hu_old), w + dt * (1.5 * Hw - 0.5 * Hw_old)
            Hu_old, Hw_old = Hu, Hw
            t += dt
            u, w, p = project(us, ws, t)
        if not np.isfinite(u).all():
            raise FloatingPointError(f"[cfd] the solution blew up at t = {t:.3f}; lower cfl or raise N")
        dp_all.append(p[i_up, F[i_up]].mean() - p[i_th, F[i_th]].mean())
        if fr >= keep_from:
            uc = 0.5 * (u[:-1] + u[1:])
            vc = 0.5 * (w[:, :-1] + w[:, 1:])
            pc = p.copy()
            for a in (uc, vc, pc):
                a[~F] = np.nan
            out_u.append(uc.astype(np.float32))
            out_v.append(vc.astype(np.float32))
            out_p.append(pc.astype(np.float32))
            out_t.append(t)
        if log and (fr + 1) % max(1, total_frames // 8) == 0:
            cfl_now = max(np.abs(u).max(), np.abs(w).max()) * dt / h
            div = (u[1:] - u[:-1]) / h + (w[:, 1:] - w[:, :-1]) / h
            log(f"[cfd] t = {t:7.3f} / {total_frames * frame_dt:.1f}   CFL {cfl_now:.2f}   max|div| {np.abs(div[F]).max():.1e}   {time.time() - t0:.0f} s")
    times = np.asarray(out_t)
    U, W, P = np.stack(out_u), np.stack(out_v), np.stack(out_p)
    if t_end is None:  # the true phase of each kept frame in [0, T), in order
        times = np.mod(np.round(times / frame_dt) * frame_dt, T)
        order = np.argsort(times)
        times, U, W, P = times[order], U[order], W[order], P[order]
    return {"x": xc, "y": yc, "fluid": F, "t": times.astype(np.float32), "u": U, "v": W, "p": P, "h": h,
            "dp_all": np.asarray(dp_all, np.float32), "frames_per_period": frames}


def periodicity_error(truth: Dict[str, np.ndarray]) -> float:
    """max |dp(last cycle) - dp(previous cycle)| / max |dp|: how far the run is from a periodic state."""
    f = int(truth["frames_per_period"])
    d = np.asarray(truth["dp_all"])
    if len(d) < 2 * f:
        return float("nan")
    a, b = d[-f:], d[-2 * f : -f]
    return float(np.abs(a - b).max() / np.abs(a).max())


# ------------------------------------------------------------------------------------------------------------
# what an MRI scan would deliver
# ------------------------------------------------------------------------------------------------------------
def mri_from_cfd(truth: Dict[str, np.ndarray], v: StenosisVessel, voxel: float = 0.1, phases: int = 24,
                 snr: float = 20.0, seed: int = 0, min_fluid_fraction: float = 0.5) -> Dict[str, np.ndarray]:
    """Voxel- and phase-averaged velocities with phase noise, in the field of view, lumen voxels only."""
    h = float(truth["h"])
    k = int(round(voxel / h))
    if abs(k * h - voxel) > 1e-6:
        raise ValueError(f"voxel {voxel} must be a multiple of the CFD cell {h}")
    nfr = truth["u"].shape[0]
    if nfr % phases:
        raise ValueError(f"{nfr} CFD frames do not split into {phases} phases")
    r = nfr // phases
    i0 = int(round(v.fov[0] / h))
    i1 = i0 + ((int(round(v.fov[1] / h)) - i0) // k) * k
    ny = truth["u"].shape[2]
    nvy = ny // k
    nvx = (i1 - i0) // k
    F = truth["fluid"][i0:i1, : nvy * k].astype(np.float64)

    def block(a):  # (frames, nx, ny) -> (phases, nvx, nvy) mean over fluid cells
        a = np.nan_to_num(a[:, i0:i1, : nvy * k].astype(np.float64)) * F
        s = a.reshape(phases, r, nvx, k, nvy, k).sum(axis=(1, 3, 5))
        return s / np.maximum(F.reshape(nvx, k, nvy, k).sum(axis=(1, 3)) * r, 1e-12)

    frac = F.reshape(nvx, k, nvy, k).mean(axis=(1, 3))
    lumen = frac >= min_fluid_fraction
    u_c, v_c = block(truth["u"]), block(truth["v"])
    venc = 1.2 * float(np.sqrt(u_c**2 + v_c**2)[:, lumen].max())
    sigma = np.sqrt(2.0) * venc / (np.pi * snr)
    rng = np.random.default_rng(seed)
    u_n = u_c + sigma * rng.standard_normal(u_c.shape)
    v_n = v_c + sigma * rng.standard_normal(v_c.shape)
    for a in (u_c, v_c, u_n, v_n):
        a[:, ~lumen] = np.nan
    xv = truth["x"][i0] - h / 2 + (np.arange(nvx) + 0.5) * voxel
    yv = -0.5 + (np.arange(nvy) + 0.5) * voxel
    frame_dt = v.period / nfr
    t_phase = (np.arange(phases) * r + (r - 1) / 2.0) * frame_dt
    return {
        "x": xv.astype(np.float32), "y": yv.astype(np.float32), "t": t_phase.astype(np.float32),
        "u": u_n.astype(np.float32), "v": v_n.astype(np.float32), "u_clean": u_c.astype(np.float32), "v_clean": v_c.astype(np.float32),
        "lumen": lumen, "fluid_fraction": frac.astype(np.float32),
        "voxel": np.float32(voxel), "phase_window": np.float32(v.period / phases), "venc": np.float32(venc), "sigma": np.float32(sigma), "snr": np.float32(snr),
    }


def mri_samples(mri: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Flatten the lumen voxels of every phase into training samples (t, x, y, u, v)."""
    P, nvx, nvy = mri["u"].shape
    T, X, Y = np.meshgrid(mri["t"], mri["x"], mri["y"], indexing="ij")
    m = np.broadcast_to(mri["lumen"][None], (P, nvx, nvy))
    return {k: a[m].astype(np.float32) for k, a in (("t", T), ("x", X), ("y", Y), ("u", mri["u"]), ("v", mri["v"]))}


# ------------------------------------------------------------------------------------------------------------
# pressure drop: truth, network, simplified Bernoulli
# ------------------------------------------------------------------------------------------------------------
def section_mean(p_cells: np.ndarray, x: np.ndarray, xs: float) -> np.ndarray:
    """Mean of (frames, nx, ny) cell values over the lumen of the column nearest to xs -> (frames,)."""
    i = int(np.argmin(np.abs(x - xs)))
    return np.nanmean(p_cells[:, i, :], axis=1)


def truth_pressure_drop(truth: Dict[str, np.ndarray], v: StenosisVessel) -> np.ndarray:
    """dp(t) = section-mean pressure upstream minus at the throat, non-dimensional, per CFD frame."""
    return section_mean(truth["p"], truth["x"], v.station_up) - section_mean(truth["p"], truth["x"], v.throat)


def bernoulli_pressure_drop(mri: Dict[str, np.ndarray], v: StenosisVessel) -> np.ndarray:
    """Simplified Bernoulli from the scan: 4 v_peak^2 [mmHg, m/s], v_peak the fastest voxel near the throat."""
    sel = (mri["x"] > v.xc - 1.0) & (mri["x"] < v.xc + 2.0)
    speed = np.sqrt(mri["u"] ** 2 + mri["v"] ** 2)[:, sel, :]
    vpk = np.nanmax(speed.reshape(speed.shape[0], -1), axis=1) * v.physical()["U_ms"]
    return 4.0 * vpk**2


# ------------------------------------------------------------------------------------------------------------
# files
# ------------------------------------------------------------------------------------------------------------
def save_dataset(out_dir: Path, v: StenosisVessel, truth: Dict, mri: Dict, meta: Optional[Dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "truth.npz", **{k: np.asarray(a) for k, a in truth.items()})
    np.savez_compressed(out_dir / "mri.npz", **{k: np.asarray(a) for k, a in mri.items()})
    info = {"vessel": v.to_json(), **(meta or {})}
    (out_dir / "vessel.json").write_text(json.dumps(info, indent=2))


def load_dataset(data_dir: Path) -> Tuple[StenosisVessel, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data_dir = Path(data_dir)
    if not (data_dir / "mri.npz").exists():
        raise FileNotFoundError(f"no stenosis dataset in {data_dir}; create it with: python scripts/make_stenosis_data.py")
    info = json.loads((data_dir / "vessel.json").read_text())
    v = StenosisVessel.from_json(info["vessel"])
    truth = dict(np.load(data_dir / "truth.npz"))
    mri = dict(np.load(data_dir / "mri.npz"))
    return v, truth, mri
