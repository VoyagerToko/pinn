"""Benchmark A - 2D Taylor-Green vortex. Validates the residual implementation to near machine precision.

Formulations: ``vp`` (u, v, p) or ``streamfunction`` (psi, p; exactly divergence-free).
Periodicity is exact through the PeriodEmbs embedding on (x, y). Loss terms:
    u_ic, v_ic           initial condition against the analytic solution
    r_u, r_v[, r_c]      momentum (and continuity for VP) residuals, optionally causal
    p_anchor             zero-mean pressure (closed periodic domain, handbook 5.6)
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .. import physics
from ..benchmarks import TaylorGreen2D
from ..losses import causal_residual_losses, mse, pressure_anchor_closed, align_pressure_gauge
from ..metrics import max_divergence, relative_l2
from ..sampling import uniform_box
from ..utils import chunked_vmap, meshgrid_points
from .base import Problem


class TaylorGreen2DPINN(Problem):
    input_dim = 3  # (t, x, y)

    def __init__(self, config):
        super().__init__(config)
        self.bench = TaylorGreen2D(Re=float(config.problem.Re), T=float(config.problem.T))
        self.formulation = config.problem.get("formulation", "streamfunction")
        self.dom = jnp.asarray(self.bench.domain)
        # held-out validation grid (never trained on): 5 times x 64 x 64
        g = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ts = np.linspace(0, self.bench.T, 5)
        self.eval_pts = meshgrid_points(jnp.asarray(ts), jnp.asarray(g), jnp.asarray(g))

    # ------------------------------------------------------------------
    def velocity_fn(self, params):
        """Point-wise (u, v, p) regardless of formulation."""
        f = self.net(params)
        if self.formulation == "streamfunction":
            return physics.streamfunction_velocity(f, unsteady=True)
        return f

    def residual_fn(self, params, Re):
        f = self.net(params)
        if self.formulation == "streamfunction":
            return physics.ns_streamfunction_residual(f, Re, unsteady=True)
        return physics.ns_vp_residual(f, Re, dim=2, unsteady=True)

    # ------------------------------------------------------------------
    def uniform_collocation(self, key, n):
        return uniform_box(key, self.dom, n)

    def residual_magnitude_fn(self, params):
        r = self.residual_fn(params, self.bench.Re)
        return lambda z: jnp.linalg.norm(r(z)[0])

    def sample_batch(self, key):
        cfg = self.config.training
        k1, k2 = jax.random.split(key)
        res = self.draw_collocation(k1, int(cfg.res_batch_size))
        ic = uniform_box(k2, self.dom[1:], int(cfg.ic_batch_size))
        return {"res": res, "ic": ic, **self._common_batch_fields()}

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        Re = self.bench.Re
        vel = jax.vmap(self.velocity_fn(params))
        z_ic = jnp.concatenate([jnp.zeros((batch["ic"].shape[0], 1)), batch["ic"]], axis=1)
        pred_ic = vel(z_ic)
        u0, v0, _ = self.bench.exact(z_ic[:, 0], z_ic[:, 1], z_ic[:, 2])
        out = {"u_ic": mse(pred_ic[:, 0], u0), "v_ic": mse(pred_ic[:, 1], v0)}

        r_mom, r_c = self.vmap_pointwise(self.residual_fn(params, Re))(batch["res"])
        terms = [r_mom[:, 0] ** 2, r_mom[:, 1] ** 2] + ([r_c**2] if self.formulation == "vp" else [])
        names = ["r_u", "r_v"] + (["r_c"] if self.formulation == "vp" else [])
        if self.use_causal:
            vals, _ = causal_residual_losses(batch["res"][:, 0], terms, self.num_chunks, batch["causal_eps"])
        else:
            vals = [jnp.mean(t) for t in terms]
        out.update(dict(zip(names, vals)))
        p = vel(batch["res"])[:, 2]
        out["p_anchor"] = pressure_anchor_closed(p)
        return out

    def causal_min_weight(self, params, batch):
        r_mom, r_c = jax.vmap(self.residual_fn(params, self.bench.Re))(batch["res"])
        _, gamma = causal_residual_losses(batch["res"][:, 0], [r_mom[:, 0] ** 2, r_mom[:, 1] ** 2], self.num_chunks, batch["causal_eps"])
        return gamma.min()

    def ntk_diags(self, params, batch):
        f = self.net(params)
        r = self.residual_fn(params, self.bench.Re)
        vel = self.velocity_fn(params)
        z_ic = jnp.concatenate([jnp.zeros((batch["ic"].shape[0], 1)), batch["ic"]], axis=1)
        from ..losses import ntk_diag

        apply_vel = lambda p, z, i: self.velocity_fn(p)(z)[i]
        apply_res = lambda p, z, i: self.residual_fn(p, self.bench.Re)(z)[0][i]
        d = {
            "u_ic": ntk_diag(lambda p, z: apply_vel(p, z, 0), params, z_ic),
            "v_ic": ntk_diag(lambda p, z: apply_vel(p, z, 1), params, z_ic),
            "r_u": ntk_diag(lambda p, z: apply_res(p, z, 0), params, batch["res"]),
            "r_v": ntk_diag(lambda p, z: apply_res(p, z, 1), params, batch["res"]),
        }
        return d

    # ------------------------------------------------------------------
    def evaluate(self, params) -> Dict[str, float]:
        vel = self.velocity_fn(params)
        pred = chunked_vmap(vel, self.eval_pts)
        z = self.eval_pts
        u, v, p = self.bench.exact(z[:, 0], z[:, 1], z[:, 2])
        if self.formulation == "vp":
            div = chunked_vmap(lambda zz: self.residual_fn(params, self.bench.Re)(zz)[1], z)
        else:
            div = jnp.zeros(1)
        return {
            "rel_l2_u": float(relative_l2(pred[:, 0], u)),
            "rel_l2_v": float(relative_l2(pred[:, 1], v)),
            "rel_l2_p": float(relative_l2(align_pressure_gauge(pred[:, 2], p), p)),
            "rel_l2_vel": float(relative_l2(pred[:, :2], jnp.stack([u, v], -1))),
            "max_div": float(max_divergence(div)),
        }
