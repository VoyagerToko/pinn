"""Real data: the pressure drop across an aortic coarctation from 4D flow MRI of a physical phantom, scored against
catheter pressure (Stanford 4D-FlowP phantom, see :mod:`pinnflow.coa_phantom`).

The network maps (t, a, b, c) to (u, v, w, p) in a scan-aligned orthonormal frame. Scales: length 10 mm, velocity
1 m/s, time 10 ms, pressure rho (1 m/s)^2 = 1100 Pa = 8.25 mmHg, so Re = 0.01 / nu = 2619 with the measured fluid.
It is exactly periodic over the pump beat. Loss terms:

    u_data, v_data, w_data   each lumen voxel and frame against the network averaged over that voxel cube and the frame's
                             acquisition window (``problem.sub_samples`` points)
    r_u, r_v, r_w, r_c       3D Navier-Stokes momentum and continuity at collocation points in the lumen
    wall                     no slip on the faces between lumen and outside voxels (the segmented wall)
    p_anchor                 p = 0 at the downstream plane's centre (the gauge; excluded from adaptive weighting)
    nu_t_reg                 (with problem.eddy_viscosity) the mean learned eddy viscosity, a small fixed-weight penalty

With ``problem.eddy_viscosity`` the network has a fifth output, an eddy-viscosity field nu_t(t, x) >= 0, and the momentum
equation carries the full stress divergence, (nu + nu_t) lap u + grad(nu_t) . (grad u + grad u^T). Turbulent mixing in a
coarctation jet happens far below the voxel size; its momentum exchange (how fast the jet spreads and decays in the
data) is what nu_t is inferred from, as the wake benchmark infers a viscosity. Without it, a laminar momentum balance at
the molecular viscosity recovers almost all the jet's pressure downstream and the trans-coarctation drop vanishes.

The pressure is never measured by the scan. ``evaluate`` returns the section-averaged pressure difference between the
upstream (AAo) and downstream (DAo) catheter planes over the beat, in mmHg, against the catheter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from .. import physics
from ..data import DATA_DIR
from ..losses import mse
from .base import Problem

L_REF, U_REF = 0.01, 1.0  # m, m/s
T_REF = L_REF / U_REF  # s


class CoarctationMRIInversePINN(Problem):
    input_dim = 4  # (t, a, b, c)

    def __init__(self, config):
        p = config.problem
        f = Path(p.get("data_file", None) or DATA_DIR / "stanford_coa" / "prepared" / f"{p.get('case', 'P2_baseline')}.npz")
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; prepare it with: python scripts/prepare_coa.py --case {p.get('case', 'P2_baseline')}")
        d = dict(np.load(f))
        self.d = d
        self.case = str(d["case"])
        self.T = float(d["period_s"]) / T_REF
        self.rho, self.mu = float(d["rho"]), float(d["mu"])
        self.Re = self.rho * U_REF * L_REF / self.mu
        self.mmHg_per_unit = self.rho * U_REF**2 / 133.322
        with config.unlocked():
            config.arch.periodicity = ml_collections.ConfigDict({"period": (2 * np.pi / self.T,), "axis": (0,), "trainable": (False,)})
            config.arch.out_dim = 5 if bool(p.get("eddy_viscosity", False)) else 4
        super().__init__(config)
        if bool(p.get("eddy_viscosity", False)):
            self.init_weights["nu_t_reg"] = float(p.get("nu_t_reg_weight", 0.01))
        mm = 1e-3 / L_REF  # mm -> length units
        self.voxel = np.asarray(d["voxel_mm"], np.float32) * mm
        self.window = float(d["frame_window_s"]) / T_REF
        pos = np.asarray(d["pos_mm"], np.float32) * mm  # (N, 3)
        t = np.asarray(d["t_s"], np.float32) / T_REF  # (F,)
        vel = np.asarray(d["vel"], np.float32) / U_REF  # (F, N, 3)
        F, N = vel.shape[:2]
        self.train_c = jnp.asarray(np.concatenate([np.repeat(t, N)[:, None], np.tile(pos, (F, 1))], 1))
        self.train_uvw = jnp.asarray(vel.reshape(F * N, 3))
        self.lumen = jnp.asarray(pos)
        self.wall = jnp.asarray(np.asarray(d["wall_mm"], np.float32) * mm)
        self.aao = np.asarray(d["aao_mm"], np.float32) * mm
        self.dao = np.asarray(d["dao_mm"], np.float32) * mm
        lo, hi = pos.min(0) - self.voxel, pos.max(0) + self.voxel
        self.in_shift = jnp.asarray(np.concatenate([[0.0], 0.5 * (lo + hi)]).astype(np.float32))
        self.in_scale = jnp.asarray(np.concatenate([[1.0], 0.5 * (hi - lo)]).astype(np.float32))
        self.p_scale = float(p.get("p_scale", 3.0))
        self.eddy = bool(p.get("eddy_viscosity", False))
        self.nu_t_scale = float(p.get("nu_t_scale", 0.01))  # ~30 x the molecular 1/Re at O(1) network output
        self.n_sub = int(p.get("sub_samples", 4))
        self.cath_phase = np.asarray(d["cath_phase_s"])
        self.cath_dp = np.asarray(d["cath_dp_mmHg"])
        self.cath_peak = float(d["cath_dp_peak_mmHg"])

    # ------------------------------------------------------------------ network
    def net(self, params):
        if self.eddy:  # (u, v, w, p, nu_t): softplus keeps the eddy viscosity non-negative
            def f(z):
                raw = self.arch.apply(params, (z - self.in_shift) / self.in_scale)
                return jnp.concatenate([raw[:3], self.p_scale * raw[3:4], self.nu_t_scale * jax.nn.softplus(raw[4:5])])
            return f
        scale = jnp.array([1.0, 1.0, 1.0, self.p_scale])
        return lambda z: self.arch.apply(params, (z - self.in_shift) / self.in_scale) * scale

    def velocity_fn(self, params):
        """(t, a, b, c) -> (u, v, w, p), non-dimensional."""
        return self.net(params)

    def residual_fn(self, params):
        if not self.eddy:
            return physics.ns_vp_residual(self.net(params), Re=self.Re, dim=3, unsteady=True)
        f = self.net(params)
        nu = 1.0 / self.Re

        def r(z):
            out = f(z)
            u, nut = out[:3], out[4]
            u_t = physics.directional(f, z, 0)[:3]
            conv, lap, gp, div, cols = jnp.zeros(3), jnp.zeros(3), [], 0.0, []
            for j, s in enumerate((1, 2, 3)):
                f_s, f_ss = physics.directional(f, z, s, second=True)
                conv = conv + u[j] * f_s[:3]
                lap = lap + f_ss[:3]
                gp.append(f_s[3])
                div = div + f_s[j]
                cols.append(f_s)
            G = jnp.stack([c[:3] for c in cols], axis=1)   # G[i, j] = d u_i / d x_j
            gnu = jnp.stack([c[4] for c in cols])          # d nu_t / d x_j
            stress = (G + G.T) @ gnu                        # sum_j (du_i/dx_j + du_j/dx_i) dnu_t/dx_j
            r_mom = u_t + conv + jnp.stack(gp) - (nu + nut) * lap - stress
            return r_mom, div

        return r

    # ------------------------------------------------------------------ sampling
    def _jitter_voxels(self, key, centres):
        k1, k2 = jax.random.split(key)
        return centres + (jax.random.uniform(k1, centres.shape) - 0.5) * jnp.asarray(self.voxel), k2

    def uniform_collocation(self, key, n):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (n,), 0, self.lumen.shape[0])
        x, _ = self._jitter_voxels(k2, self.lumen[idx])
        t = jax.random.uniform(k3, (n, 1), minval=0.0, maxval=self.T)
        return jnp.concatenate([t, x], 1)

    def sample_batch(self, key):
        cfg = self.config.training
        ks = jax.random.split(key, 8)
        B, S = int(cfg.data_batch_size), self.n_sub
        idx = jax.random.randint(ks[0], (B,), 0, self.train_c.shape[0])
        c = self.train_c[idx]
        sub = jnp.repeat(c, S, axis=0)
        jit_x = (jax.random.uniform(ks[1], (B * S, 3)) - 0.5) * jnp.asarray(self.voxel)
        jit_t = (jax.random.uniform(ks[2], (B * S, 1)) - 0.5) * self.window
        data = sub + jnp.concatenate([jit_t, jit_x], 1)
        nw = int(cfg.bc_batch_size)
        wi = jax.random.randint(ks[3], (nw,), 0, self.wall.shape[0])
        tw = jax.random.uniform(ks[4], (nw, 1), minval=0.0, maxval=self.T)
        da = jnp.asarray(self.dao.mean(0))
        ta = jax.random.uniform(ks[5], (nw, 1), minval=0.0, maxval=self.T)
        return {
            "data": data, "data_uvw": self.train_uvw[idx],
            "res": self.draw_collocation(ks[6], int(cfg.res_batch_size)),
            "wall": jnp.concatenate([tw, self.wall[wi]], 1),
            "anchor": jnp.concatenate([ta, jnp.broadcast_to(da, (nw, 3))], 1),
            **self._common_batch_fields(),
        }

    # ------------------------------------------------------------------ losses
    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        f = jax.vmap(self.net(params))
        B = batch["data_uvw"].shape[0]
        pred = f(batch["data"])[:, :3].reshape(B, self.n_sub, 3).mean(1)
        out = {k: mse(pred[:, i], batch["data_uvw"][:, i]) for i, k in enumerate(("u_data", "v_data", "w_data"))}
        r_mom, r_c = self.vmap_pointwise(self.residual_fn(params))(batch["res"])
        out.update(r_u=mse(r_mom[:, 0]), r_v=mse(r_mom[:, 1]), r_w=mse(r_mom[:, 2]), r_c=mse(r_c))
        out["wall"] = mse(f(batch["wall"])[:, :3])
        out["p_anchor"] = mse(f(batch["anchor"])[:, 3])
        if self.eddy:
            out["nu_t_reg"] = jnp.mean(f(batch["res"])[:, 4]) / self.nu_t_scale
        return out

    # ------------------------------------------------------------------ evaluation
    def predict(self, params, pts: np.ndarray, chunk: int = 8192) -> np.ndarray:
        if not hasattr(self, "_predict_jit"):
            self._predict_jit = jax.jit(lambda prm, z: jax.vmap(self.net(prm))(z))
        pts = np.asarray(pts, np.float32)
        out = []
        for s in range(0, len(pts), chunk):
            c = pts[s : s + chunk]
            pad = chunk - len(c)
            if pad:
                c = np.concatenate([c, np.repeat(c[-1:], pad, 0)], 0)
            out.append(np.asarray(self._predict_jit(params, jnp.asarray(c)))[: chunk - pad])
        return np.concatenate(out, 0)

    def pressure_drop(self, params, times: np.ndarray) -> np.ndarray:
        """Section-averaged p(AAo plane) - p(DAo plane), in mmHg, at the given non-dimensional times."""
        res = []
        for plane in (self.aao, self.dao):
            T, P = np.meshgrid(np.asarray(times, np.float32), np.arange(len(plane)), indexing="ij")
            pts = np.concatenate([T.reshape(-1, 1), plane[P.ravel()]], 1)
            res.append(self.predict(params, pts)[:, 3].reshape(len(times), len(plane)).mean(1))
        return (res[0] - res[1]) * self.mmHg_per_unit

    def evaluate(self, params) -> Dict[str, float]:
        times = np.linspace(0.0, self.T, 64, endpoint=False)
        dp = self.pressure_drop(params, times)
        # velocity fit: relative L2 against the (noisy) scan on a fixed subset of voxels and frames
        n = min(20000, self.train_c.shape[0])
        sel = np.linspace(0, self.train_c.shape[0] - 1, n).astype(int)
        pred = self.predict(params, np.asarray(self.train_c)[sel])[:, :3]
        ref = np.asarray(self.train_uvw)[sel]
        fit = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
        peak = float(dp.max())
        extra = {}
        if self.eddy:
            nut = self.predict(params, np.asarray(self.uniform_collocation(jax.random.PRNGKey(0), 8192)))[:, 4]
            extra = {"nu_t_mean_over_nu": float(nut.mean() * self.Re), "nu_t_p99_over_nu": float(np.percentile(nut, 99) * self.Re)}
        extra["data_prep_version"] = int(self.d.get("prep_version", 0))
        return {**extra,
            "dp_peak_pinn_mmHg": peak, "dp_peak_cath_mmHg": self.cath_peak,
            "dp_peak_err_pct": 100 * abs(peak - self.cath_peak) / self.cath_peak,
            "dp_mean_pinn_mmHg": float(dp.mean()), "dp_mean_cath_mmHg": float(self.cath_dp.mean()),
            "velocity_fit_rel": fit,
        }
