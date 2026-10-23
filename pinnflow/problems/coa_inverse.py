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

With ``problem.smagorinsky_cs`` > 0 the eddy viscosity is not learned but computed from the resolved flow,
nu_t = (C_s Delta)^2 |S| with |S| = sqrt(2 S_ij S_ij) and Delta the voxel size, in the same stress divergence. It cannot be
switched off where the flow shears (the learned field collapsed to ~0 in every run) and needs no penalty.

With ``problem.accel_data`` the loss also matches the network's material acceleration Du/Dt = u_t + (u . grad) u,
averaged over each voxel and acquisition window like the velocity, to the acceleration measured by 4D-FlowP (terms
acc_u, acc_v, acc_w). The pressure gradient balances rho Du/Dt, so this constrains the pressure where the blurred
velocity cannot: a voxel-averaged acceleration keeps the jet's sub-voxel convective and turbulent part, which the
acceleration of the voxel-averaged velocity loses.

With ``problem.pressure_net`` the pressure comes from a second, separate network instead of the velocity network's fourth
output: exactly periodic over the beat through explicit time harmonics, with isotropic multi-scale Fourier features in
space (``PressureMLP``, ``_p_features``). Trained jointly with the velocity, it is what the momentum terms act on. The
shared output did not learn the pressure: the first phantom networks left the flow's systolic acceleration in the
momentum residual (0.3 mmHg), while a pressure fitted alone to the same frozen velocity gives 7.3 mmHg and a third less
residual (claude helper/diagnostics/coa_pressure_fit.py, docs/STATUS.md 2026-10-23). Run configs without the key keep the shared output, so
older checkpoints load as they were trained.

The pressure is never measured by the scan. ``evaluate`` returns the section-averaged pressure difference between the
upstream (AAo) and downstream (DAo) catheter planes over the beat, in mmHg, against the catheter. The catheter's beat
starts at the flow meter's upstroke, the scan's at the MRI trigger: ``cath_t`` is the catheter waveform on the scan's
clock (prep v5 measures the offset; peak and mean, the scores, do not depend on it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import flax.linen as nn
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


class PressureMLP(nn.Module):
    """The separate pressure network's trunk: tanh layers on the features of ``_p_features``, one output."""

    width: int = 128
    depth: int = 4

    @nn.compact
    def __call__(self, h):
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)[0]


class CoarctationMRIInversePINN(Problem):
    input_dim = 4  # (t, a, b, c)

    def __init__(self, config):
        p = config.problem
        f = Path(p.get("data_file", None) or DATA_DIR / "stanford_coa" / "prepared" / f"{p.get('case', 'P2_baseline')}.npz")
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; prepare it with: python scripts/data/prepare_coa.py --case {p.get('case', 'P2_baseline')}")
        d = dict(np.load(f))
        self.d = d
        self.case = str(d["case"])
        self.T = float(d["period_s"]) / T_REF
        self.rho, self.mu = float(d["rho"]), float(d["mu"])
        self.Re = self.rho * U_REF * L_REF / self.mu
        self.mmHg_per_unit = self.rho * U_REF**2 / 133.322
        self.pressure_net = bool(p.get("pressure_net", False))  # absent in older run configs: the shared fourth output
        with config.unlocked():
            config.arch.periodicity = ml_collections.ConfigDict({"period": (2 * np.pi / self.T,), "axis": (0,), "trainable": (False,)})
            # the velocity network's outputs: (u, v, w[, p][, raw nu_t])
            config.arch.out_dim = 3 + (0 if self.pressure_net else 1) + (1 if bool(p.get("eddy_viscosity", False)) else 0)
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
        # Smagorinsky eddy viscosity (problem.smagorinsky_cs > 0): computed from the resolved flow, not learned; Delta is
        # the scan's resolution (geometric mean of the voxel edges) unless problem.smagorinsky_delta_mm is given
        self.smag_cs = float(p.get("smagorinsky_cs", 0.0))
        if self.smag_cs > 0 and self.eddy:
            raise ValueError("problem.smagorinsky_cs and problem.eddy_viscosity are alternatives; set one")
        dmm = p.get("smagorinsky_delta_mm", None)
        self.smag_delta = (float(dmm) * mm) if dmm else float(np.prod(self.voxel) ** (1.0 / 3.0))
        # measured material acceleration (problem.accel_data): the 4D-FlowP acquisition encodes Du/Dt, registered onto
        # this scan's voxels and frames by claude helper/diagnostics/flowp_register.py (<case>_flowp.npz: acc (F, N, 3) m/s^2 in the local
        # frame, inside (N,) where 4D-FlowP covers the lumen)
        self.accel = bool(p.get("accel_data", False))
        # problem.accel_pressure: the pressure gradient balances the measured acceleration directly, grad p = -Du/Dt
        # (the 4D-FlowP pressure estimate's principle; viscous terms are a few per cent of it and left out)
        self.accel_p = bool(p.get("accel_pressure", False))
        if self.accel or self.accel_p:
            fa = Path(p.get("accel_file", None) or f.with_name(f"{self.case}_flowp.npz"))
            if not fa.exists():
                raise FileNotFoundError(f"{fa} missing; register 4D-FlowP with: python \"claude helper/diagnostics/flowp_register.py\" --case {self.case}")
            da = np.load(fa)
            acc = np.asarray(da[str(p.get("accel_key", "acc"))], np.float32) * (L_REF / U_REF**2)  # m/s^2 -> length / time^2
            if acc.shape[:2] != (F, N):
                raise ValueError(f"{fa}: acc {acc.shape} does not match the scan's (frames, voxels) = {(F, N)}")
            inside = np.asarray(da["inside"], bool)
            fi, ni = np.nonzero(np.broadcast_to(inside[None], (F, N)))
            self.acc_idx = jnp.asarray((fi * N + ni).astype(np.int32))  # voxel-frames of train_c with a measurement
            self.train_acc = jnp.asarray(acc.reshape(F * N, 3))
            for k in (("acc_u", "acc_v", "acc_w") if self.accel else ()) + (("pacc_u", "pacc_v", "pacc_w") if self.accel_p else ()):
                self.init_weights.setdefault(k, 1.0)
        self.cath_phase = np.asarray(d["cath_phase_s"])
        self.cath_dp = np.asarray(d["cath_dp_mmHg"])
        self.cath_peak = float(d["cath_dp_peak_mmHg"])
        # the catheter waveform on the scan's clock: catheter phase = (scan time + shift) mod period (0 before prep v5)
        self.cath_shift = float(d.get("cath_shift_s", 0.0))
        period = float(d["period_s"])
        order = np.argsort((self.cath_phase - self.cath_shift) % period)
        self.cath_t = ((self.cath_phase - self.cath_shift) % period)[order]
        self.cath_dp_scan = self.cath_dp[order]
        self.data_pool = None  # voxel-frame indices drawn by data misfit (update_collocation); None: uniform
        if self.pressure_net:
            self.p_harmonics = int(p.get("pressure_net_harmonics", 12))
            self.p_mlp = PressureMLP(width=int(p.get("pressure_net_width", 128)), depth=int(p.get("pressure_net_depth", 4)))
            # isotropic space features: one length for all axes, so a feature's wavelength is the same in every direction
            self.p_centre = jnp.asarray(0.5 * (pos.min(0) + pos.max(0)))
            self.p_len = float(0.5 * (pos.max(0) - pos.min(0)).max())
            sig = tuple(float(s) for s in p.get("pressure_net_sigmas", (1.0, 4.0, 12.0)))
            self.p_B = jnp.concatenate([jax.random.normal(jax.random.PRNGKey(i), (3, 32)) * s for i, s in enumerate(sig)], 1)

    # ------------------------------------------------------------------ the separate pressure network
    def _p_features(self, z):
        """(t, a, b, c) -> time harmonics of the beat (exactly periodic), the coordinates, and their Fourier features."""
        k = jnp.arange(1, self.p_harmonics + 1)
        wt = k * (2 * jnp.pi / self.T) * z[0]
        x = (z[1:] - self.p_centre) / self.p_len
        xb = x @ self.p_B
        return jnp.concatenate([jnp.cos(wt), jnp.sin(wt), x, jnp.cos(xb), jnp.sin(xb)])

    def init_params(self, key):
        if not self.pressure_net:
            return super().init_params(key)
        k1, k2 = jax.random.split(key)
        z0 = jnp.zeros(self.input_dim)
        return {"vel": self.arch.init(k1, z0), "p": self.p_mlp.init(k2, self._p_features(z0))}

    # ------------------------------------------------------------------ network
    def net(self, params):
        """(t, a, b, c) -> (u, v, w, p[, nu_t]) whichever networks carry them."""
        if self.pressure_net:
            def g(z):
                raw = self.arch.apply(params["vel"], (z - self.in_shift) / self.in_scale)
                p = self.p_mlp.apply(params["p"], self._p_features(z))[None]
                if self.eddy:  # softplus keeps the eddy viscosity non-negative
                    return jnp.concatenate([raw[:3], p, self.nu_t_scale * jax.nn.softplus(raw[3:4])])
                return jnp.concatenate([raw[:3], p])
            return g
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

    def smagorinsky_fn(self, params):
        """z -> nu_t = (C_s Delta)^2 |S|, |S| = sqrt(2 S_ij S_ij), S the resolved strain rate (non-dimensional)."""
        f = self.net(params)
        coef = (self.smag_cs * self.smag_delta) ** 2

        def nut(z):
            G = jnp.stack([physics.directional(f, z, s)[:3] for s in (1, 2, 3)], axis=1)  # G[i, j] = d u_i / d x_j
            S = 0.5 * (G + G.T)
            return coef * jnp.sqrt(2.0 * jnp.sum(S * S) + 1e-10)

        return nut

    def residual_fn(self, params):
        if not (self.eddy or self.smag_cs > 0):
            return physics.ns_vp_residual(self.net(params), Re=self.Re, dim=3, unsteady=True)
        f = self.net(params)
        nu = 1.0 / self.Re
        smag = self.smagorinsky_fn(params) if self.smag_cs > 0 else None

        def r(z):
            out = f(z)
            u = out[:3]
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
            if smag is not None:  # nu_t from the resolved strain rate; its gradient by forward mode through |S|
                nut = smag(z)
                gnu = jnp.stack([physics.directional(smag, z, s) for s in (1, 2, 3)])
            else:  # the learned field, the velocity network's last output
                nut = out[4]
                gnu = jnp.stack([c[4] for c in cols])      # d nu_t / d x_j
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

    # ------------------------------------------------------------------ adaptive sampling (RAD, handbook 5.5)
    def residual_magnitude_fn(self, params):
        """|momentum residual| + |continuity residual| at one point: RAD of the collocation points."""
        r = self.residual_fn(params)

        def mag(z):
            r_mom, r_c = r(z)
            return jnp.sqrt(jnp.sum(r_mom**2) + r_c**2)

        return mag

    def update_collocation(self, key, params):
        """RAD of the collocation points, and the same rule for the data (``training.rad_data``): the voxel-frames the
        network fits worst are drawn more often.

        The physics residual alone cannot find an under-fitted jet: a smooth flow that leaves the jet out satisfies the
        equations everywhere (the first phantom runs: residuals ~1e-4, velocity off by 34 %, no pressure drop). The data
        misfit does find it, so the data pool is resampled by it, p ~ err^k / mean(err^k) + c as for the residual."""
        k1, k2, k3, k4 = jax.random.split(key, 4)
        super().update_collocation(k1, params)
        cfg = self.config.training
        if not bool(cfg.get("rad_data", True)):
            return
        n_all = self.train_c.shape[0]
        nc = min(int(cfg.get("rad_candidates", 100_000)), n_all)
        cand = jax.random.choice(k2, n_all, (nc,), replace=False)
        pred = self.predict(params, np.asarray(self.train_c[cand]))[:, :3]
        err = jnp.linalg.norm(jnp.asarray(pred) - self.train_uvw[cand], axis=1)
        k, c = float(cfg.get("rad_k", 1.0)), float(cfg.get("rad_c", 1.0))
        p = err**k / (jnp.mean(err**k) + 1e-12) + c
        n_pool = min(int(cfg.get("rad_pool_size", 65536)), nc)
        n_adapt = int(round(n_pool * (1.0 - float(cfg.get("rad_uniform_frac", 0.2)))))
        pick = cand[jax.random.choice(k3, nc, (n_adapt,), replace=False, p=p / p.sum())]
        self.data_pool = jnp.concatenate([pick, jax.random.randint(k4, (n_pool - n_adapt,), 0, n_all)])

    def sample_batch(self, key):
        cfg = self.config.training
        ks = jax.random.split(key, 8)
        B, S = int(cfg.data_batch_size), self.n_sub
        if self.data_pool is None:
            idx = jax.random.randint(ks[0], (B,), 0, self.train_c.shape[0])
        else:  # the RAD data pool (update_collocation)
            idx = self.data_pool[jax.random.randint(ks[0], (B,), 0, self.data_pool.shape[0])]
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
        batch = {
            "data": data, "data_uvw": self.train_uvw[idx],
            "res": self.draw_collocation(ks[6], int(cfg.res_batch_size)),
            "wall": jnp.concatenate([tw, self.wall[wi]], 1),
            "anchor": jnp.concatenate([ta, jnp.broadcast_to(da, (nw, 3))], 1),
            **self._common_batch_fields(),
        }
        if self.accel or self.accel_p:  # measured accelerations, averaged over the voxel cube and window like the velocity
            ka = jax.random.split(ks[7], 3)
            Ba = int(cfg.get("acc_batch_size", B))
            ia = self.acc_idx[jax.random.randint(ka[0], (Ba,), 0, self.acc_idx.shape[0])]
            ax = (jax.random.uniform(ka[1], (Ba * S, 3)) - 0.5) * jnp.asarray(self.voxel)
            at = (jax.random.uniform(ka[2], (Ba * S, 1)) - 0.5) * self.window
            batch["acc"] = jnp.repeat(self.train_c[ia], S, axis=0) + jnp.concatenate([at, ax], 1)
            batch["acc_target"] = self.train_acc[ia]
        return batch

    def material_acceleration_fn(self, params):
        """z -> Du/Dt = u_t + (u . grad) u of the network's velocity (non-dimensional), by forward-mode directions."""
        g = self.net(params)

        def acc(z):
            u = g(z)[:3]
            a = physics.directional(g, z, 0)[:3]
            for j, s in enumerate((1, 2, 3)):
                a = a + u[j] * physics.directional(g, z, s)[:3]
            return a

        return acc

    def pressure_gradient_fn(self, params):
        """z -> grad p (non-dimensional), through the pressure network alone when there is one."""
        if self.pressure_net:
            def p(z):
                return self.p_mlp.apply(params["p"], self._p_features(z))
        else:
            g = self.net(params)

            def p(z):
                return g(z)[3]

        return lambda z: jnp.stack([physics.directional(p, z, s) for s in (1, 2, 3)])

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
        if self.accel:
            Ba = batch["acc_target"].shape[0]
            pa = self.vmap_pointwise(self.material_acceleration_fn(params))(batch["acc"]).reshape(Ba, self.n_sub, 3).mean(1)
            out.update({k: mse(pa[:, i], batch["acc_target"][:, i]) for i, k in enumerate(("acc_u", "acc_v", "acc_w"))})
        if self.accel_p:  # the voxel- and window-averaged pressure gradient against minus the measured acceleration
            Ba = batch["acc_target"].shape[0]
            gp = jax.vmap(self.pressure_gradient_fn(params))(batch["acc"]).reshape(Ba, self.n_sub, 3).mean(1)
            out.update({k: mse(gp[:, i], -batch["acc_target"][:, i]) for i, k in enumerate(("pacc_u", "pacc_v", "pacc_w"))})
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
        if self.eddy or self.smag_cs > 0:
            pts = self.uniform_collocation(jax.random.PRNGKey(0), 8192)
            if self.eddy:
                nut = self.predict(params, np.asarray(pts))[:, 4]
            else:
                nut = np.asarray(jax.jit(jax.vmap(self.smagorinsky_fn(params)))(pts))
            extra = {"nu_t_mean_over_nu": float(nut.mean() * self.Re), "nu_t_p99_over_nu": float(np.percentile(nut, 99) * self.Re)}
        if self.accel:  # material acceleration at the voxel centres against the measured one (noisy: ~3 m/s^2 per component)
            ia = np.asarray(self.acc_idx)[np.linspace(0, len(self.acc_idx) - 1, min(8192, len(self.acc_idx))).astype(int)]
            if not hasattr(self, "_acc_jit"):
                self._acc_jit = jax.jit(lambda prm, z: jax.vmap(self.material_acceleration_fn(prm))(z))
            pa = np.asarray(self._acc_jit(params, self.train_c[ia]))
            ra = np.asarray(self.train_acc)[ia]
            extra["accel_fit_rel"] = float(np.linalg.norm(pa - ra) / np.linalg.norm(ra))
        if self.accel_p:  # grad p against minus the measured acceleration at the voxel centres
            ia = np.asarray(self.acc_idx)[np.linspace(0, len(self.acc_idx) - 1, min(8192, len(self.acc_idx))).astype(int)]
            if not hasattr(self, "_gp_jit"):
                self._gp_jit = jax.jit(lambda prm, z: jax.vmap(self.pressure_gradient_fn(prm))(z))
            gp = np.asarray(self._gp_jit(params, self.train_c[ia]))
            ra = np.asarray(self.train_acc)[ia]
            extra["pacc_fit_rel"] = float(np.linalg.norm(gp + ra) / np.linalg.norm(ra))
        extra["data_prep_version"] = int(self.d.get("prep_version", 0))
        extra["pressure_net"] = int(self.pressure_net)
        extra["cath_shift_ms"] = 1e3 * self.cath_shift
        return {**extra,
            "dp_peak_pinn_mmHg": peak, "dp_peak_cath_mmHg": self.cath_peak,
            "dp_peak_err_pct": 100 * abs(peak - self.cath_peak) / self.cath_peak,
            "dp_mean_pinn_mmHg": float(dp.mean()), "dp_mean_cath_mmHg": float(self.cath_dp.mean()),
            "velocity_fit_rel": fit,
        }
