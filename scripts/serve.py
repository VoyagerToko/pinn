"""Live inference server for the visual application (handbook STEP 9.5, server-side variant).

Loads the trained checkpoints once onto the GPU and answers field / curve requests from the browser.
Every value returned under "pred" is a network evaluation; "ref" values are the independent references
(exact solution, finite-difference field, Nektar data, FEATFLOW and DNS curves).

    python scripts/serve.py --port 8000                      # all models found under runs/
    python scripts/serve.py --models tgv2d,cavity_ok         # a subset
    python scripts/serve.py --runs /path/to/runs             # another checkpoint root

Endpoints (interactive docs at /docs):
    GET /health
    GET /models                                   registry: id, title, status, gate, metrics, domain, controls
    GET /field/{model}?t=&Re=&n=&ny=&nz=          binary: 4-byte little-endian header length, JSON header, float32 arrays
    GET /curve/{model}?kind=                      JSON curves (energy/dissipation, flow rate, drag, centrelines)
    GET /points/{model}                           JSON scattered points (sparse sensors of the inverse problem)
    POST /velocity/{model}?t=&Re=&dim=            raw float32 points (N, dim) in, predicted velocities (N, dim) out
    GET /bundle                                   tar.gz of every scene's checkpoints + references (extract at repo root)

The binary field layout is documented in :func:`pack_fields`.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from fastapi import Request  # module level: endpoint annotations are resolved from module globals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow import metrics as M  # noqa: E402
from pinnflow.benchmarks import ghia_tables  # noqa: E402
from pinnflow.data import DATA_DIR, load_cylinder_wake, load_featflow_series  # noqa: E402
from pinnflow.physics import velocity_gradient, vorticity_from_grad  # noqa: E402
from pinnflow.problems import PROBLEMS, DFGCylinderPINN  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402

# --------------------------------------------------------------------------------------
# registry: which run backs which scene, and the verdict from STATUS.md
# --------------------------------------------------------------------------------------
REGISTRY: Dict[str, Dict] = {
    "tgv2d": {
        "run": "tgv2d_G",
        "benchmark": "tgv2d",
        "title": "Taylor-Green vortex (2D)",
        "status": "passed",
        "gate": "relative L2 of velocity < 1e-4",
        "measured": "4.0e-5 (pressure 2.6e-4, divergence exactly 0)",
        "summary": "Four counter-rotating vortices decaying under viscosity. The exact solution is known, so this is the correctness gate of the whole method.",
        "reference": "analytic solution",
        "controls": {"t": [0.0, 1.0]},
    },
    "cavity_ok": {
        "run": "cavity_C_r2",
        "benchmark": "cavity",
        "title": "Lid-driven cavity (soft boundary, row C)",
        "status": "passed_field",
        "gate": "Ghia centreline error < 2% at Re = 1000",
        "measured": "u 0.63%, v 2.05%; 0.27% from a converged finite-difference solution of the same problem",
        "summary": "A box of fluid stirred by its moving lid. Misses the Ghia v gate by 0.05 points, but a converged reference solution misses it too (2.3%): the gate is tighter than Ghia's table.",
        "reference": "finite-difference solution (N = 512) and Ghia et al. (1982)",
        "controls": {"Re": [100, 400, 1000]},
    },
    "cavity_hard": {
        "run": "cavity_F_r2",
        "benchmark": "cavity",
        "title": "Lid-driven cavity (hard lid constraint, row F)",
        "status": "failed",
        "gate": "Ghia centreline error < 2% at Re = 1000",
        "measured": "u 19.8%, v 20.2%",
        "summary": "The same lid, with its velocity imposed exactly. The exact lid and side walls cannot satisfy continuity in the two top corners, so the continuity residual keeps a floor (mean square about 0.04) and the vortex comes out too weak.",
        "reference": "finite-difference solution (N = 512)",
        "controls": {"Re": [100, 400, 1000]},
    },
    "inverse": {
        "run": "cylinder_inverse",
        "benchmark": "cylinder_inverse",
        "title": "Cylinder wake reconstructed from sparse data (inverse problem)",
        "status": "passed",
        "gate": "recover lambda_1, lambda_2 and the hidden pressure",
        "measured": "lambda_1 error 0.004%, lambda_2 (viscosity) error 0.065%, hidden pressure 2.4%",
        "summary": "5000 scattered velocity measurements of a vortex street at Re = 100. The network recovers the full velocity field, the pressure that was never measured, and the two unknown coefficients of the equations.",
        "reference": "Nektar spectral-element data (Raissi et al. 2019)",
        "controls": {"t": [0.0, 19.9]},
    },
    "cylinder_fwd": {
        "run": "cylinder_sf_2x6k",
        "benchmark": "cylinder",
        "title": "Flow past a cylinder, forward solve (failed)",
        "status": "failed",
        "gate": "Strouhal within 3% of 0.30, Cd_max within 5% of 3.23",
        "measured": "no vortex shedding; Cd_max 1.54; 25% of the inflow mass lost along the channel",
        "summary": "The DFG benchmark from rest with a ramped inflow. At the reduced budget the network never develops shedding, and the flow rate decays downstream instead of staying constant.",
        "reference": "FEATFLOW drag and lift series",
        "controls": {"t": [0.0, 1.0]},
    },
    "tgv3d": {
        "run": "tgv3d_H",
        "benchmark": "tgv3d",
        "title": "Taylor-Green vortex (3D, SPINN)",
        "status": "failed",
        "gate": "dissipation peak within 5% of the DNS reference (0.0128 at t = 9)",
        "measured": "no transition: dissipation largest at t = 0, enstrophy never grows",
        "summary": "The 3D vortex should break into turbulence around t = 9. The separable network keeps the flow two-dimensional and smooth; energy leaks through the residual instead of through viscosity.",
        "reference": "DNS dissipation curve (DeBonis 2013, digitised)",
        "controls": {"t": [0.0, 20.0]},
    },
    "deeponet": {
        "run": "deeponet_cavity_cont4",
        "benchmark": "deeponet_cavity",
        "title": "Cavity for any lid speed (PI-DeepONet)",
        "status": "passed",
        "gate": "zero-shot error < 10% at Re = 400 (never seen in training)",
        "measured": "9.65% at Re = 400; 2.1% at Re = 100; 30% at Re = 1000",
        "summary": "One network for every Reynolds number between 100 and 1000. Re in [300, 530] was excluded from training, so Re = 400 is a true zero-shot query.",
        "reference": "finite-difference solutions at Re = 100, 400, 1000",
        "controls": {"Re": [100.0, 1000.0], "holdout": [300.0, 530.0]},
    },
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def load_cfg(workdir: Path) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(json.loads((workdir / "config.json").read_text()))


def read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def pack_fields(fields: Dict[str, np.ndarray], header: Dict, dtype: str = "float32") -> bytes:
    """Binary payload: uint32 LE header length, UTF-8 JSON header, then the arrays in header['fields'] order.

    header gets 'fields' (names), 'shape' (grid shape, 'ij' order: first axis x) and 'dtype'
    ('float32', or 'float16' to halve the payload for display; NaN marks points outside the fluid).
    """
    names = list(fields)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    header = dict(header, fields=names, dtype=np.dtype(np_dtype).name)
    arrays = [np.ascontiguousarray(np.asarray(fields[k], dtype=np_dtype)).tobytes() for k in names]
    hb = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(hb)) + hb + b"".join(arrays)


def grid2d(extent, nx: int, ny: int):
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


# --------------------------------------------------------------------------------------
# model wrappers
# --------------------------------------------------------------------------------------
class Scene:
    """One served model: loads the checkpoint(s), builds jitted evaluators, knows its references."""

    def __init__(self, sid: str, spec: Dict, runs: Path):
        self.id = sid
        self.spec = spec
        self.workdir = runs / spec["run"]
        self.cfg = load_cfg(self.workdir)
        self.bench_name = spec["benchmark"]
        self.metrics = {
            "final_eval": read_json(self.workdir / "final_eval.json"),
            "eval": read_json(self.workdir / "eval.json"),
            "eval_fd": read_json(self.workdir / "eval_fd.json"),
            "cost": read_json(self.workdir / "cost.json"),
        }
        self._jit: Dict = {}
        self._cache: Dict = {}
        self.load()

    # ---- loading ----------------------------------------------------------------
    def _restore(self, problem, path: Path):
        params, _ = load_params(path, template=problem.init_params(jax.random.PRNGKey(0)))
        return params

    def load(self):
        b = self.bench_name
        if b == "cavity":
            self.problem = PROBLEMS["cavity"](self.cfg)
            self.params = {}
            for ck in sorted(self.workdir.glob("Re*.msgpack")):
                self.params[int(ck.stem[2:])] = self._restore(self.problem, ck)
            if not self.params:
                self.params[int(round(float(self.cfg.problem.Re)))] = self._restore(self.problem, self.workdir / "latest.msgpack")
            self.domain = [0.0, 1.0, 0.0, 1.0]
        elif b == "cylinder":
            dt = float(self.cfg.problem.window_dt)
            self.windows = []
            for w in sorted(self.workdir.glob("window_*")):
                idx = int(w.name.split("_")[1])
                p = DFGCylinderPINN(self.cfg, t0=idx * dt, t1=(idx + 1) * dt)
                self.windows.append((p, self._restore(p, w / "latest.msgpack")))
            self.problem = self.windows[0][0]
            self.dt = dt
            self.t_end = len(self.windows) * dt
            self.spec["controls"]["t"] = [0.0, self.t_end]
            bch = self.problem.bench
            self.domain = [0.0, bch.length, 0.0, bch.height]
        else:
            self.problem = PROBLEMS[b](self.cfg)
            self.params = self._restore(self.problem, self.workdir / "latest.msgpack")
            if b == "tgv2d":
                self.domain = [0.0, 2 * np.pi, 0.0, 2 * np.pi]
            elif b == "cylinder_inverse":
                self.domain = [1.0, 8.0, -2.0, 2.0]
                self.wake = load_cylinder_wake()
            elif b == "deeponet_cavity":
                self.domain = [0.0, 1.0, 0.0, 1.0]
            elif b == "tgv3d":
                self.domain = [0.0, 2 * np.pi] * 3

    # ---- 2D point-wise evaluation -----------------------------------------------
    def _field_fn(self, key, vel_fn: Callable, unsteady: bool, extra: Optional[Callable] = None):
        """jitted pts (N, d) -> (u, v, p, vorticity[, extra])."""
        if key not in self._jit:
            g = velocity_gradient(vel_fn, dim=2, unsteady=unsteady)

            def one(z):
                uvp = vel_fn(z)
                _, grad_u = g(z)
                out = [uvp[0], uvp[1], uvp[2], vorticity_from_grad(grad_u)]
                if extra is not None:
                    out.append(extra(z))
                return jnp.stack(out)

            self._jit[key] = jax.jit(jax.vmap(one))
        return self._jit[key]

    def _eval_pts(self, fn, pts: np.ndarray, chunk: int = 16384) -> np.ndarray:
        outs = [np.asarray(fn(jnp.asarray(pts[s : s + chunk], jnp.float32))) for s in range(0, pts.shape[0], chunk)]
        return np.concatenate(outs, axis=0)

    # ---- field requests -----------------------------------------------------------
    def field(self, t: float, Re: float, n: int, ny: Optional[int], nz: Optional[int], dtype: str = "float32") -> bytes:
        b = self.bench_name
        if b == "tgv3d":
            return self._field_tgv3d(t, n, dtype)
        ext = self.domain
        aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
        nx = int(n)
        ny = int(ny) if ny else max(8, int(round(nx * aspect)))
        x, y, X, Y = grid2d(ext, nx, ny)
        header = {"model": self.id, "shape": [nx, ny], "extent": ext, "x": x.tolist(), "y": y.tolist()}
        if "t" in self.spec["controls"]:
            header["t"] = t
        pts2 = np.stack([X.ravel(), Y.ravel()], -1)
        fields: Dict[str, np.ndarray] = {}

        if b == "tgv2d":
            pts = np.concatenate([np.full((pts2.shape[0], 1), t), pts2], 1)
            out = self._eval_pts(self._field_fn("f", self.problem.velocity_fn(self.params), True), pts)
            bch = self.problem.bench
            ue, ve, pe = (np.asarray(a) for a in bch.exact(t, X.ravel(), Y.ravel()))
            we = 2 * np.cos(X.ravel()) * np.cos(Y.ravel()) * np.exp(-2 * t / bch.Re)
            p = out[:, 2] - out[:, 2].mean() + pe.mean()
            fields.update(u=out[:, 0], v=out[:, 1], p=p, vort=out[:, 3], u_ref=ue, v_ref=ve, p_ref=pe, vort_ref=we)

        elif b == "cavity":
            Re_i = min(self.params, key=lambda r: abs(r - Re))
            header["Re"] = Re_i
            self.problem.set_Re(Re_i)
            params = self.params[Re_i]
            div_fn = lambda z: self.problem.residual_fn(params, float(Re_i))(z)[1]
            out = self._eval_pts(self._field_fn(("f", Re_i), self.problem.velocity_fn(params), False, extra=div_fn), pts2)
            fields.update(u=out[:, 0], v=out[:, 1], p=out[:, 2] - out[:, 2].mean(), vort=out[:, 3], div=out[:, 4])
            ref = self._cavity_fd(Re_i, x, y)
            if ref is not None:
                fields.update(u_ref=ref[0], v_ref=ref[1])

        elif b == "deeponet_cavity":
            Re_f = float(np.clip(Re, *self.spec["controls"]["Re"]))
            header["Re"] = Re_f
            vel = self.problem.net(self.params, Re_f)
            if "f" not in self._jit:
                g = velocity_gradient(lambda z: self.problem.net(self.params, z[2])(z[:2]), dim=2, unsteady=False)

                def one(z3):
                    uvp = self.problem.net(self.params, z3[2])(z3[:2])
                    _, grad_u = g(z3)
                    return jnp.stack([uvp[0], uvp[1], uvp[2], vorticity_from_grad(grad_u)])

                self._jit["f"] = jax.jit(jax.vmap(one))
            pts3 = np.concatenate([pts2, np.full((pts2.shape[0], 1), Re_f)], 1)
            out = self._eval_pts(self._jit["f"], pts3)
            fields.update(u=out[:, 0], v=out[:, 1], p=out[:, 2] - out[:, 2].mean(), vort=out[:, 3])
            nearest = min((100, 400, 1000), key=lambda r: abs(r - Re_f))
            if abs(nearest - Re_f) < 1.0:
                ref = self._cavity_fd(nearest, x, y)
                if ref is not None:
                    fields.update(u_ref=ref[0], v_ref=ref[1])
            header["holdout"] = self.spec["controls"].get("holdout")

        elif b == "cylinder_inverse":
            pts = np.concatenate([np.full((pts2.shape[0], 1), t), pts2], 1)
            out = self._eval_pts(self._field_fn("f", self.problem.velocity_fn(self.params), True), pts)
            ref = self._wake_ref(t, x, y)
            p = out[:, 2] - out[:, 2].mean() + ref[2].mean()
            fields.update(u=out[:, 0], v=out[:, 1], p=p, vort=out[:, 3], u_ref=ref[0], v_ref=ref[1], p_ref=ref[2])
            header["t_ref"] = float(self.wake["t"][self._nearest_snapshot(t)])

        elif b == "cylinder":
            idx = min(int(t // self.dt), len(self.windows) - 1)
            problem, params = self.windows[idx]
            pts = np.concatenate([np.full((pts2.shape[0], 1), t), pts2], 1)
            out = self._eval_pts(self._field_fn(("f", idx), problem.velocity_dim_fn(params), True), pts)
            bch = problem.bench
            mask = ~np.asarray(bch.in_fluid(X.ravel(), Y.ravel()))
            for k, col in zip(("u", "v", "p", "vort"), range(4)):
                a = out[:, col].copy()
                a[mask] = np.nan
                fields[k] = a
            header["cylinder"] = {"center": list(bch.center), "radius": bch.radius}
            header["window"] = idx
        return pack_fields(fields, header, dtype)

    def _field_tgv3d(self, t: float, n: int, dtype: str = "float32") -> bytes:
        n = int(n)
        f = self.problem.grid_fields(self.params, float(t), n)
        g = np.linspace(0, 2 * np.pi, n, endpoint=False)
        header = {"model": self.id, "shape": [n, n, n], "extent": self.domain, "t": t, "axis": g.tolist()}
        fields = {"u": f["velocity"][:, 0], "v": f["velocity"][:, 1], "w": f["velocity"][:, 2], "speed": f["speed"], "q": f["qcriterion"], "vort_mag": np.linalg.norm(f["vorticity"], axis=1)}
        return pack_fields(fields, header, dtype)

    # ---- references -------------------------------------------------------------------
    def _cavity_fd(self, Re: int, x: np.ndarray, y: np.ndarray):
        key = ("fd", Re, len(x), len(y))
        if key in self._cache:
            return self._cache[key]
        files = sorted((DATA_DIR / "cavity_fd").glob(f"cavity_fd_Re{Re}_regularised_N*.npz"), key=lambda p: int(p.stem.split("_N")[-1]))
        if not files:
            self._cache[key] = None
            return None
        from scipy.interpolate import RegularGridInterpolator

        d = np.load(files[-1])
        X, Y = np.meshgrid(x, y, indexing="ij")
        q = np.stack([X.ravel(), Y.ravel()], -1)
        fu = RegularGridInterpolator((d["x"], d["y"]), d["u"], bounds_error=False, fill_value=None)
        fv = RegularGridInterpolator((d["x"], d["y"]), d["v"], bounds_error=False, fill_value=None)
        self._cache[key] = (fu(q), fv(q))
        return self._cache[key]

    def _nearest_snapshot(self, t: float) -> int:
        return int(np.argmin(np.abs(self.wake["t"] - t)))

    def _wake_ref(self, t: float, x: np.ndarray, y: np.ndarray):
        k = self._nearest_snapshot(t)
        key = ("wake", k, len(x), len(y))
        if key in self._cache:
            return self._cache[key]
        from scipy.interpolate import griddata

        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = self.wake["X_star"]
        vals = [self.wake["U_star"][:, 0, k], self.wake["U_star"][:, 1, k], self.wake["p_star"][:, k]]
        ref = tuple(griddata(pts, v, (X.ravel(), Y.ravel()), method="linear", fill_value=0.0) for v in vals)
        self._cache[key] = ref
        return ref

    # ---- curves ---------------------------------------------------------------------
    def curve(self, kind: str) -> Dict:
        b = self.bench_name
        if b == "tgv3d":
            return self._curve_tgv3d()
        if b == "cylinder":
            return self._curve_cylinder(kind)
        if b == "cavity":
            return self._curve_cavity()
        if b == "deeponet_cavity":
            return self._curve_deeponet()
        if b == "tgv2d":
            return self._curve_tgv2d()
        if b == "cylinder_inverse":
            return self._curve_inverse()
        return {}

    def _curve_tgv3d(self) -> Dict:
        if "curve" in self._cache:
            return self._cache["curve"]
        hist_file = self.workdir / "energy_history.csv"
        if hist_file.exists():
            h = np.loadtxt(hist_file, delimiter=",", skiprows=1)
            t, Ek, zeta = h[:, 0], h[:, 1], h[:, 2]
        else:
            times = np.linspace(0, self.problem.bench.T, 81)
            hist = self.problem.energy_history(self.params, times, n=24)
            t, Ek, zeta = hist["t"], hist["Ek"], hist["enstrophy"]
        eps_E = M.dissipation_from_energy(t, Ek)
        eps_Z = M.dissipation_from_enstrophy(zeta, self.problem.bench.nu)
        ref = np.loadtxt(DATA_DIR / "tgv3d_re1600" / "debonis2013_fig4a_ref_dissipation.csv", delimiter=",", skiprows=1)
        out = {
            "t": t.tolist(),
            "series": {"E_k": Ek.tolist(), "dissipation_from_energy": eps_E.tolist(), "dissipation_from_enstrophy": np.asarray(eps_Z).tolist()},
            "ref": {"t": ref[:, 0].tolist(), "dissipation_dns": ref[:, 1].tolist()},
            "gate": {"peak_value": 0.01282, "peak_time": 9.0, "tolerance": 0.05},
        }
        self._cache["curve"] = out
        return out

    def _curve_cylinder(self, kind: str) -> Dict:
        key = ("curve", kind)
        if key in self._cache:
            return self._cache[key]
        bch = self.problem.bench
        times = np.linspace(0.0, self.t_end - 1e-6, 41)
        stations = [0.0, 0.3, 0.6, 1.2, 1.8, 2.2]
        ys = np.linspace(0, bch.height, 201)
        Q = np.zeros((len(times), len(stations)))
        Cd, Cl = np.zeros(len(times)), np.zeros(len(times))
        for i, t in enumerate(times):
            idx = min(int(t // self.dt), len(self.windows) - 1)
            problem, params = self.windows[idx]
            vel = problem.velocity_dim_fn(params)
            fn = self._field_fn(("f", idx), vel, True)
            for j, xs in enumerate(stations):
                pts = np.stack([np.full_like(ys, t), np.full_like(ys, xs), ys], -1)
                u = self._eval_pts(fn, pts)[:, 0]
                fluid = np.asarray(bch.in_fluid(np.full_like(ys, xs), ys))
                Q[i, j] = np.trapezoid(np.where(fluid, u, 0.0), ys)
            cd, cl, _ = M.drag_lift(vel, float(t), bch, n_theta=128)
            Cd[i], Cl[i] = float(cd), float(cl)
        Q_in = Q[:, 0:1]
        ratio = np.where(np.abs(Q_in) > 1e-6, Q / np.maximum(Q_in, 1e-9), np.nan)
        out = {"t": times.tolist(), "stations": stations, "flow_rate": Q.tolist(), "flow_rate_ratio": ratio.tolist(), "Cd": Cd.tolist(), "Cl": Cl.tolist(), "ref": {}}
        try:
            ff = load_featflow_series("draglift", "2D-2")
            n = len(ff["t"])
            out["ref"] = {"Cd_max": float(ff["Cd"][n // 2 :].max()), "Cd_mean": float(ff["Cd"][n // 2 :].mean()), "Cl_max": float(ff["Cl"][n // 2 :].max()), "St": 0.30}
        except FileNotFoundError:
            pass
        self._cache[key] = out
        return out

    def _curve_cavity(self) -> Dict:
        if "curve" in self._cache:
            return self._cache["curve"]
        out = {"Re": {}}
        yq = np.linspace(0, 1, 201)
        for Re, params in self.params.items():
            self.problem.set_Re(Re)
            vel = jax.jit(jax.vmap(self.problem.velocity_fn(params)))
            u_c = np.asarray(vel(jnp.stack([jnp.full(201, 0.5), jnp.asarray(yq)], -1)))[:, 0]
            v_c = np.asarray(vel(jnp.stack([jnp.asarray(yq), jnp.full(201, 0.5)], -1)))[:, 1]
            entry = {"s": yq.tolist(), "u_centre": u_c.tolist(), "v_centre": v_c.tolist()}
            try:
                tab = ghia_tables(int(Re))
                entry["ghia"] = {"y": tab["y"].tolist(), "u": tab["u"].tolist(), "x": tab["x"].tolist(), "v": tab["v"].tolist()}
            except KeyError:
                pass
            out["Re"][str(Re)] = entry
        self._cache["curve"] = out
        return out

    def _curve_deeponet(self) -> Dict:
        if "curve" in self._cache:
            return self._cache["curve"]
        yq = np.linspace(0, 1, 201)
        Res = [100, 200, 400, 700, 1000]
        out = {"Re": {}}
        for Re in Res:
            vel = jax.jit(jax.vmap(self.problem.net(self.params, float(Re))))
            u_c = np.asarray(vel(jnp.stack([jnp.full(201, 0.5), jnp.asarray(yq)], -1)))[:, 0]
            v_c = np.asarray(vel(jnp.stack([jnp.asarray(yq), jnp.full(201, 0.5)], -1)))[:, 1]
            entry = {"s": yq.tolist(), "u_centre": u_c.tolist(), "v_centre": v_c.tolist()}
            try:
                tab = ghia_tables(Re)
                entry["ghia"] = {"y": tab["y"].tolist(), "u": tab["u"].tolist(), "x": tab["x"].tolist(), "v": tab["v"].tolist()}
            except KeyError:
                pass
            out["Re"][str(Re)] = entry
        self._cache["curve"] = out
        return out

    def _curve_tgv2d(self) -> Dict:
        if "curve" in self._cache:
            return self._cache["curve"]
        bch = self.problem.bench
        times = np.linspace(0, bch.T, 41)
        g = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        X, Y = np.meshgrid(g, g, indexing="ij")
        fn = self._field_fn("f", self.problem.velocity_fn(self.params), True)
        Ek, Ek_ref, err = [], [], []
        for t in times:
            pts = np.stack([np.full(X.size, t), X.ravel(), Y.ravel()], -1)
            out = self._eval_pts(fn, pts)
            ue, ve, _ = (np.asarray(a) for a in bch.exact(t, X.ravel(), Y.ravel()))
            Ek.append(float(np.mean(0.5 * (out[:, 0] ** 2 + out[:, 1] ** 2))))
            Ek_ref.append(float(np.mean(0.5 * (ue**2 + ve**2))))
            err.append(float(np.linalg.norm(out[:, :2] - np.stack([ue, ve], -1)) / np.linalg.norm(np.stack([ue, ve], -1))))
        self._cache["curve"] = {"t": times.tolist(), "series": {"E_k": Ek, "rel_l2_velocity": err}, "ref": {"E_k": Ek_ref}}
        return self._cache["curve"]

    def _curve_inverse(self) -> Dict:
        lam1, lam2 = (float(v) for v in self.problem.lambdas(self.params))
        return {"lambda_1": lam1, "lambda_2": lam2, "lambda_1_true": 1.0, "lambda_2_true": 0.01}

    # ---- points -------------------------------------------------------------------------
    def points(self) -> Dict:
        if self.bench_name != "cylinder_inverse":
            return {}
        pts = np.asarray(self.problem.train_pts)
        return {"t": pts[:, 0].tolist(), "x": pts[:, 1].tolist(), "y": pts[:, 2].tolist(), "n": int(pts.shape[0])}

    # ---- velocity at arbitrary points (real-time particle advection) ------------------------
    def velocity(self, pts: np.ndarray, t: float, Re: float) -> np.ndarray:
        """Predicted velocity at scattered points: (N, 2) -> (N, 2) for 2D scenes, (N, 3) -> (N, 3) for 3D.

        Coordinates are the scene's own (dimensional for the forward cylinder, as in /field). The batch is
        padded to a power of two so particle counts that change from frame to frame do not recompile; t and Re
        are traced arguments, so moving the time or Reynolds number never recompiles either.
        """
        b = self.bench_name
        n, d = pts.shape
        cap = max(256, 1 << int(np.ceil(np.log2(max(n, 1)))))
        padded = np.zeros((cap, d), np.float32)
        padded[:n] = pts
        if b == "cavity":
            Re_i = min(self.params, key=lambda r: abs(r - Re))
            key = ("vel", Re_i)
            if key not in self._jit:
                self.problem.set_Re(Re_i)
                vf = self.problem.velocity_fn(self.params[Re_i])
                self._jit[key] = jax.jit(jax.vmap(lambda p, t, Re: vf(p)[:2], in_axes=(0, None, None)))
        elif b == "deeponet_cavity":
            key = ("vel",)
            if key not in self._jit:
                net = self.problem.net
                params = self.params
                self._jit[key] = jax.jit(jax.vmap(lambda p, t, Re: net(params, Re)(p)[:2], in_axes=(0, None, None)))
            Re = float(np.clip(Re, *self.spec["controls"]["Re"]))
        elif b == "cylinder":
            idx = min(int(max(t, 0.0) // self.dt), len(self.windows) - 1)
            key = ("vel", idx)
            if key not in self._jit:
                problem, params = self.windows[idx]
                vf = problem.velocity_dim_fn(params)
                self._jit[key] = jax.jit(jax.vmap(lambda p, t, Re: vf(jnp.concatenate([jnp.reshape(t, (1,)), p]))[:2], in_axes=(0, None, None)))
        else:  # tgv2d, cylinder_inverse (t, x, y) and tgv3d (t, x, y, z)
            key = ("vel",)
            if key not in self._jit:
                vf = self.problem.velocity_fn(self.params)
                k = 3 if b == "tgv3d" else 2
                self._jit[key] = jax.jit(jax.vmap(lambda p, t, Re: vf(jnp.concatenate([jnp.reshape(t, (1,)), p]))[:k], in_axes=(0, None, None)))
        out = self._jit[key](jnp.asarray(padded), jnp.asarray(t, jnp.float32), jnp.asarray(Re, jnp.float32))
        return np.asarray(out, dtype=np.float32)[:n]

    # ---- metadata -----------------------------------------------------------------------
    def describe(self) -> Dict:
        d = {k: v for k, v in self.spec.items()}
        d.update(id=self.id, domain=self.domain, metrics=self.metrics, dims=3 if self.bench_name == "tgv3d" else 2)
        if self.bench_name == "cavity":
            d["controls"] = {"Re": sorted(self.params)}
        return d


# --------------------------------------------------------------------------------------
# bundle: everything another machine needs to serve the same scenes locally
# --------------------------------------------------------------------------------------
BUNDLE_PATTERNS = (
    "config.json", "final_eval.json", "eval.json", "eval_fd.json", "cost.json", "energy_history.csv",
    "latest.msgpack", "Re*.msgpack", "window_*/latest.msgpack",
)


def build_bundle(runs: Path, out: Path) -> Path:
    """tar.gz with the registry runs (configs, final checkpoints, metric files) and the finest
    finite-difference cavity reference per Re. Extract it at the repository root of the receiving machine:
    files land in runs/<run>/ and data/cavity_fd/, where serve.py looks for them."""
    import tarfile

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        for run in sorted({spec["run"] for spec in REGISTRY.values()}):
            wd = runs / run
            if not wd.exists():
                continue
            for pat in BUNDLE_PATTERNS:
                for f in sorted(wd.glob(pat)):
                    tar.add(f, arcname=f"runs/{run}/{f.relative_to(wd).as_posix()}")
        best: Dict[str, tuple] = {}
        for f in (DATA_DIR / "cavity_fd").glob("cavity_fd_Re*_regularised_N*.npz"):
            Re, N = f.stem.split("_")[2], int(f.stem.split("_N")[-1])
            if Re not in best or N > best[Re][0]:
                best[Re] = (N, f)
        for _, f in sorted(best.values()):
            tar.add(f, arcname=f"data/cavity_fd/{f.name}")
    tmp.replace(out)
    return out


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------
def build_app(runs: Path, only: Optional[List[str]] = None, warm: bool = True):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response

    app = FastAPI(title="pinnflow live server", version="0.1")
    from fastapi.middleware.gzip import GZipMiddleware

    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])
    scenes: Dict[str, Scene] = {}
    for sid, spec in REGISTRY.items():
        if only and sid not in only:
            continue
        wd = runs / spec["run"]
        if not (wd / "config.json").exists():
            print(f"[serve] skip {sid}: {wd} not found")
            continue
        t0 = time.time()
        try:
            scenes[sid] = Scene(sid, dict(spec), runs)
            print(f"[serve] loaded {sid} <- {wd.name} ({time.time() - t0:.1f}s)")
        except Exception as e:  # keep serving the others
            print(f"[serve] FAILED {sid}: {type(e).__name__}: {e}")

    def get(sid: str) -> Scene:
        if sid not in scenes:
            raise HTTPException(404, f"unknown model {sid!r}; available: {sorted(scenes)}")
        return scenes[sid]

    @app.get("/health")
    def health():
        return {"ok": True, "device": str(jax.devices()[0]), "models": sorted(scenes)}

    @app.get("/models")
    def models():
        return [s.describe() for s in scenes.values()]

    @app.get("/field/{sid}")
    def field(sid: str, t: float = 0.0, Re: float = 1000.0, n: int = Query(128, ge=8, le=512), ny: Optional[int] = None, nz: Optional[int] = None, dtype: str = Query("float32", pattern="^(float32|float16)$")):
        s = get(sid)
        t0 = time.time()
        payload = s.field(t, Re, n, ny, nz, dtype)
        return Response(content=payload, media_type="application/octet-stream", headers={"X-Compute-Seconds": f"{time.time() - t0:.3f}"})

    @app.get("/curve/{sid}")
    def curve(sid: str, kind: str = "default"):
        return get(sid).curve(kind)

    @app.get("/points/{sid}")
    def points(sid: str):
        return get(sid).points()

    @app.post("/velocity/{sid}")
    async def velocity(sid: str, request: Request, t: float = 0.0, Re: float = 1000.0, dim: int = Query(2, ge=2, le=3)):
        """Body: raw little-endian float32 points, shape (N, dim). Response: float32 velocities, same shape."""
        s = get(sid)
        body = await request.body()
        if len(body) % (4 * dim):
            raise HTTPException(400, f"body length {len(body)} is not a multiple of {4 * dim} bytes")
        pts = np.frombuffer(body, dtype="<f4").reshape(-1, dim)
        t0 = time.time()
        from starlette.concurrency import run_in_threadpool

        vel = await run_in_threadpool(s.velocity, pts, t, Re)  # GPU work off the event loop: one scene compiling never blocks another
        return Response(content=vel.astype("<f4").tobytes(), media_type="application/octet-stream", headers={"X-Compute-Seconds": f"{time.time() - t0:.4f}"})

    @app.get("/bundle")
    def bundle(rebuild: bool = False):
        """Download the checkpoints and references of every scene as one tar.gz (see build_bundle)."""
        out = runs / "_bundle" / "pinn_bundle.tar.gz"
        if rebuild or not out.exists():
            build_bundle(runs, out)
        return FileResponse(out, media_type="application/gzip", filename="pinn_bundle.tar.gz")

    if warm:  # trigger the XLA compiles now rather than on the first browser request
        for sid, s in scenes.items():
            t0 = time.time()
            try:
                n = 32 if s.bench_name == "tgv3d" else 48
                s.field(s.spec["controls"].get("t", [0.0])[0], s.spec["controls"].get("Re", [1000])[-1], n, None, None)
                print(f"[serve] warmed {sid} ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"[serve] warm-up failed for {sid}: {type(e).__name__}: {e}")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--models", default=None, help="comma-separated subset of model ids")
    ap.add_argument("--no-warm", action="store_true", help="skip the start-up compile of every model")
    args = ap.parse_args()
    import uvicorn

    only = args.models.split(",") if args.models else None
    app = build_app(Path(args.runs), only, warm=not args.no_warm)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
