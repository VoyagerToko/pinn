"""Real-world use case: infer the blood pressure field, and the pressure drop across a stenosis, from 4D-flow-MRI-like
velocity data (see :mod:`pinnflow.vessel` for the vessel, the CFD truth and the synthetic scan).

The network maps (t, x, y) to (psi, p) with u = psi_y, v = -psi_x, so the flow is exactly incompressible, and it is
exactly periodic in time with the cardiac period (period embedding on the t axis). With ``problem.hard_walls`` (the
default) the segmented wall is imposed exactly:

    psi = Q(t) (3 eta - eta^3) / 4 + (1 - eta^2)^2 N(t, x, y),     eta = y / h(x)

so psi is constant along each wall and its gradient vanishes there (u = v = 0 on the wall), and the flow rate Q(t)
is a learned Fourier series over the heartbeat, initialised from the scan's flux. The pressure output is scaled by
``problem.p_scale`` (the expected pressure-drop scale, 1/2 of the throat speed squared) so the network works at O(1).
Loss terms:

    u_data, v_data   each MRI voxel against the network's velocity *averaged over that voxel and its phase window*
                     (``problem.sub_samples`` random points per voxel inside the lumen), the way a scan averages spins
    r_u, r_v         Navier-Stokes momentum at collocation points in the lumen, at the known Reynolds number
    u_wall, v_wall   no slip on the segmented wall y = +-h(x) (only with hard_walls = False)
    p_anchor         p = 0 at one point of the distal end of the field of view (pressure is only defined up to a
                     function of time; this pins the gauge and is excluded from adaptive weighting)

The pressure is never measured. ``evaluate`` scores the velocity and pressure against the CFD truth and the
trans-stenotic pressure drop dp(t) = p(proximal) - p(throat) (section means) over the cardiac cycle, in mmHg at the
physical scale of the vessel, next to the simplified Bernoulli estimate from the same scan.
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
from ..losses import align_pressure_gauge, mse
from ..metrics import relative_l2
from ..vessel import bernoulli_pressure_drop, load_dataset, mri_samples, truth_pressure_drop
from .base import Problem


class StenosisMRIInversePINN(Problem):
    input_dim = 3  # (t, x, y)

    def __init__(self, config):
        p = config.problem
        data_dir = Path(p.get("data_dir", None) or DATA_DIR / "stenosis")
        self.vessel, self.truth, self.mri = load_dataset(data_dir)
        v = self.vessel
        self.T = float(v.period)
        self.Re = float(v.Re)
        with config.unlocked():  # exactly periodic in time with the cardiac period
            config.arch.periodicity = ml_collections.ConfigDict({"period": (2 * np.pi / self.T,), "axis": (0,), "trainable": (False,)})
        super().__init__(config)
        self.phys = v.physical()
        mri = self.mri
        self.voxel = float(mri["voxel"])
        self.window = float(mri["phase_window"])
        smp = mri_samples(mri)
        self.train_c = jnp.asarray(np.stack([smp["t"], smp["x"], smp["y"]], -1))
        self.train_uv = jnp.asarray(np.stack([smp["u"], smp["v"]], -1))
        self.n_sub = int(p.get("sub_samples", 4))
        self.x_lo = float(mri["x"][0] - self.voxel / 2)
        self.x_hi = float(mri["x"][-1] + self.voxel / 2)
        mid, half = 0.5 * (self.x_lo + self.x_hi), 0.5 * (self.x_hi - self.x_lo)
        self.in_shift = jnp.array([0.0, mid, 0.0])
        self.in_scale = jnp.array([1.0, half, 0.5])  # t stays raw: the period embedding handles it
        self.hard_walls = bool(p.get("hard_walls", True))
        self.p_scale = float(p.get("p_scale", 10.0))
        self.n_harm = int(p.get("flow_harmonics", 6))
        if self.hard_walls:  # exactly zero terms must not enter the adaptive weighting
            self.init_weights = {k: w for k, w in self.init_weights.items() if k not in ("u_wall", "v_wall")}
        self.q_init = self._flux_fourier_from_scan()

        # validation: every 4th CFD frame (24 per beat), every 2nd cell inside the field of view
        tr = self.truth
        xs, ys = tr["x"], tr["y"]
        self.val_frames = np.arange(0, tr["t"].shape[0], max(1, tr["t"].shape[0] // 24))
        ix = np.where((xs > self.x_lo) & (xs < self.x_hi))[0][::2]
        iy = np.arange(len(ys))[::2]
        F = tr["fluid"][np.ix_(ix, iy)]
        XX, YY = np.meshgrid(xs[ix], ys[iy], indexing="ij")
        self._val_xy = np.stack([XX[F], YY[F]], -1).astype(np.float32)
        self._val_idx = (ix, iy, F)
        self.dp_true_frames = truth_pressure_drop(tr, v)  # (frames,) non-dimensional
        self.dp_bernoulli = bernoulli_pressure_drop(mri, v)  # (phases,) mmHg

    # ------------------------------------------------------------------ network
    def _flux_fourier_from_scan(self) -> np.ndarray:
        """Least-squares Fourier coefficients (a0, a_1..K, b_1..K) of the flow rate the scan measures upstream."""
        mri, v = self.mri, self.vessel
        col = int(np.argmin(np.abs(mri["x"] - v.station_up)))
        q = np.nansum(np.nan_to_num(mri["u"][:, col, :]), axis=1) * self.voxel  # (phases,)
        w = 2 * np.pi / self.T
        t = mri["t"].astype(np.float64)
        A = np.concatenate([np.ones((len(t), 1))] + [np.stack([np.cos(k * w * t), np.sin(k * w * t)], -1) for k in range(1, self.n_harm + 1)], 1)
        coef, *_ = np.linalg.lstsq(A, q, rcond=None)
        a0, rest = coef[0], coef[1:].reshape(self.n_harm, 2)
        return np.concatenate([[a0], rest[:, 0], rest[:, 1]]).astype(np.float32)

    def init_params(self, key):
        net = self.arch.init(key, jnp.zeros(self.input_dim))
        return {"net": net, "Q": jnp.asarray(self.q_init)} if self.hard_walls else net

    def flow_rate(self, params, t):
        c = params["Q"]
        K = self.n_harm
        k = jnp.arange(1, K + 1)
        w = 2 * np.pi / self.T
        return c[0] + jnp.sum(c[1 : K + 1] * jnp.cos(k * w * t)) + jnp.sum(c[K + 1 :] * jnp.sin(k * w * t))

    def net(self, params):
        if not self.hard_walls:
            return lambda z: self.arch.apply(params, (z - self.in_shift) / self.in_scale) * jnp.array([1.0, self.p_scale])

        def f(z):
            raw = self.arch.apply(params["net"], (z - self.in_shift) / self.in_scale)
            eta = z[2] / self._h(z[1])
            psi = self.flow_rate(params, z[0]) * (3 * eta - eta**3) / 4 + (1 - eta**2) ** 2 * raw[0]
            return jnp.stack([psi, self.p_scale * raw[1]])

        return f

    def velocity_fn(self, params):
        """(t, x, y) -> (u, v, p), non-dimensional."""
        return physics.streamfunction_velocity(self.net(params), unsteady=True)

    def residual_fn(self, params):
        return physics.ns_vp_residual(self.velocity_fn(params), Re=self.Re, dim=2, unsteady=True)

    def predict(self, params, pts: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """(N, 3) -> (N, 3) u, v, p. One compiled function, params traced, chunks padded to one shape."""
        if not hasattr(self, "_predict_jit"):
            self._predict_jit = jax.jit(lambda prm, z: jax.vmap(self.velocity_fn(prm))(z))
        pts = np.asarray(pts, np.float32)
        n = pts.shape[0]
        out = []
        for s in range(0, n, chunk):
            c = pts[s : s + chunk]
            pad = chunk - c.shape[0]
            if pad:
                c = np.concatenate([c, np.repeat(c[-1:], pad, 0)], 0)
            out.append(np.asarray(self._predict_jit(params, jnp.asarray(c)))[: chunk - pad])
        return np.concatenate(out, 0)

    # ------------------------------------------------------------------ sampling
    def _h(self, x):
        return self.vessel.half_width(x, xp=jnp)

    def uniform_collocation(self, key, n):
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.uniform(k1, (n,), minval=self.x_lo, maxval=self.x_hi)
        eta = jax.random.uniform(k2, (n,), minval=-1.0, maxval=1.0)
        t = jax.random.uniform(k3, (n,), minval=0.0, maxval=self.T)
        return jnp.stack([t, x, eta * self._h(x)], -1)

    def _wall_points(self, key, n):
        k1, k2, k3 = jax.random.split(key, 3)
        x = jax.random.uniform(k1, (n,), minval=self.x_lo, maxval=self.x_hi)
        side = jnp.where(jax.random.bernoulli(k2, 0.5, (n,)), 1.0, -1.0)
        t = jax.random.uniform(k3, (n,), minval=0.0, maxval=self.T)
        return jnp.stack([t, x, side * self._h(x)], -1)

    def _voxel_subpoints(self, key, centres):
        """(B, 3) voxel centres -> (B * S, 3) points spread over each voxel's lumen and phase window."""
        B, S, d = centres.shape[0], self.n_sub, self.voxel
        k1, k2, k3 = jax.random.split(key, 3)
        t = centres[:, None, 0] + self.window * (jax.random.uniform(k1, (B, S)) - 0.5)
        x = centres[:, None, 1] + d * (jax.random.uniform(k2, (B, S)) - 0.5)
        hx = self._h(x)
        y_lo = jnp.maximum(centres[:, None, 2] - d / 2, -hx)
        y_hi = jnp.minimum(centres[:, None, 2] + d / 2, hx)
        y_hi = jnp.maximum(y_hi, y_lo)
        y = y_lo + (y_hi - y_lo) * jax.random.uniform(k3, (B, S))
        return jnp.stack([t, x, y], -1).reshape(B * S, 3)

    def sample_batch(self, key):
        cfg = self.config.training
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        B = int(cfg.data_batch_size)
        idx = jax.random.choice(k1, self.train_c.shape[0], (B,), replace=False)
        nb = int(cfg.bc_batch_size)
        t_anchor = jax.random.uniform(k5, (nb,), minval=0.0, maxval=self.T)
        anchor = jnp.stack([t_anchor, jnp.full((nb,), self.x_hi), jnp.zeros(nb)], -1)
        return {
            "data": self._voxel_subpoints(k2, self.train_c[idx]),
            "data_uv": self.train_uv[idx],
            "res": self.draw_collocation(k3, int(cfg.res_batch_size)),
            "wall": self._wall_points(k4, nb),
            "anchor": anchor,
            **self._common_batch_fields(),
        }

    # ------------------------------------------------------------------ losses
    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        vel = jax.vmap(self.velocity_fn(params))
        B = batch["data_uv"].shape[0]
        pred = vel(batch["data"])[:, :2].reshape(B, self.n_sub, 2).mean(axis=1)
        out = {"u_data": mse(pred[:, 0], batch["data_uv"][:, 0]), "v_data": mse(pred[:, 1], batch["data_uv"][:, 1])}
        r_mom, _ = self.vmap_pointwise(self.residual_fn(params))(batch["res"])
        out["r_u"] = mse(r_mom[:, 0])
        out["r_v"] = mse(r_mom[:, 1])
        if not self.hard_walls:
            w = vel(batch["wall"])
            out["u_wall"] = mse(w[:, 0])
            out["v_wall"] = mse(w[:, 1])
        out["p_anchor"] = mse(vel(batch["anchor"])[:, 2])
        return out

    # ------------------------------------------------------------------ evaluation
    def pressure_drop(self, params, times: np.ndarray, ny: int = 64) -> np.ndarray:
        """dp(t) = section-mean pressure at the proximal station minus at the throat, non-dimensional."""
        v, xs = self.vessel, self.truth["x"]
        out = []
        for xs_i in (v.station_up, v.throat):
            xcol = float(xs[np.argmin(np.abs(xs - xs_i))])  # the same column the CFD section mean uses
            hcol = float(v.half_width(xcol))
            yq = np.linspace(-hcol, hcol, ny + 2)[1:-1]
            T, Y = np.meshgrid(np.asarray(times, np.float32), yq, indexing="ij")
            pts = np.stack([T.ravel(), np.full(T.size, xcol), Y.ravel()], -1).astype(np.float32)
            p = self.predict(params, pts)[:, 2]
            out.append(p.reshape(len(times), ny).mean(axis=1))
        return out[0] - out[1]

    def evaluate(self, params) -> Dict[str, float]:
        tr, k_mmHg = self.truth, self.phys["mmHg_per_unit"]
        ix, iy, F = self._val_idx
        pu, pv, pp, ru, rv, rp = [], [], [], [], [], []
        for f in self.val_frames:
            t = float(tr["t"][f])
            pts = np.concatenate([np.full((len(self._val_xy), 1), t, np.float32), self._val_xy], 1)
            pred = self.predict(params, pts)
            ref_u, ref_v, ref_p = (tr[k][f][np.ix_(ix, iy)][F] for k in ("u", "v", "p"))
            pu.append(pred[:, 0]); pv.append(pred[:, 1]); ru.append(ref_u); rv.append(ref_v); rp.append(ref_p)
            pp.append(np.asarray(align_pressure_gauge(jnp.asarray(pred[:, 2]), jnp.asarray(ref_p))))  # gauge per time slice
        cat = np.concatenate
        times = tr["t"][self.val_frames]
        dp_net = self.pressure_drop(params, times) * k_mmHg
        dp_true = self.dp_true_frames[self.val_frames] * k_mmHg
        peak_true = float(np.max(dp_true))
        return {
            "rel_l2_u": float(relative_l2(cat(pu), cat(ru))),
            "rel_l2_v": float(relative_l2(cat(pv), cat(rv))),
            "rel_l2_p": float(relative_l2(cat(pp), cat(rp))),
            "dp_peak_true_mmHg": peak_true,
            "dp_peak_pinn_mmHg": float(np.max(dp_net)),
            "dp_peak_err_pct": float(100 * abs(np.max(dp_net) - peak_true) / peak_true),
            "dp_cycle_rms_err_pct": float(100 * np.sqrt(np.mean((dp_net - dp_true) ** 2)) / peak_true),
            "dp_mean_true_mmHg": float(np.mean(dp_true)),
            "dp_mean_pinn_mmHg": float(np.mean(dp_net)),
            "dp_peak_bernoulli_mmHg": float(np.max(self.dp_bernoulli)),
            "dp_peak_bernoulli_err_pct": float(100 * abs(np.max(self.dp_bernoulli) - peak_true) / peak_true),
        }
