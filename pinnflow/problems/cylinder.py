"""Benchmark C - DFG 2D-2 / 2D-3 flow past a cylinder (unsteady), the real test.

Non-dimensionalisation: U = U_mean = 1, L = D = 0.1  =>  x* in [0, 22], y* in [0, 4.1], Re = 100.
The network sees inputs rescaled to O(1) (t*/T*, x*/22, y*/4.1); residuals are taken with respect
to the non-dimensional coordinates (t*, x*, y*).

Boundary treatment (handbook 4.6b):
    hard_bc=True   u_hat = g + phi N on (u, v) with
                   phi = x y (0.41 - y) ((x-0.2)^2 + (y-0.2)^2 - 0.05^2)   (all Dirichlet boundaries),
                   g   = u_in(t, y) d_cyl / (d_cyl + x)                      (inflow, zero on walls/cylinder)
                   -> only the outflow condition remains a soft loss.
    hard_bc=False  soft losses on inflow, walls, cylinder.
Outflow (soft): do-nothing  (1/Re) u_x - p = 0,  v_x = 0   (DFG definition; ``outflow="neumann_p0"``
gives u_x = 0, p = 0 instead, the handbook's wording).

Time is trained in windows [t0, t1] (handbook 6.4); the first window starts from rest (exact for 2D-3,
an impulsive start for 2D-2, which is also how FEATFLOW produces its reference). Later windows take the
previous window's terminal state through ``ic_fn(x, y) -> (u, v, p)`` in *dimensional* units.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .. import physics
from ..benchmarks import DFGCylinder
from ..constraints import cylinder_inflow_extension, hard_dirichlet, phi_channel_cylinder, phi_channel_cylinder_bounded
from ..losses import causal_residual_losses, mse
from ..metrics import drag_lift
from ..sampling import channel_boundaries, channel_initial, channel_interior
from .base import Problem


class DFGCylinderPINN(Problem):
    input_dim = 3  # (t*, x*, y*)

    def __init__(self, config, t0: float = 0.0, t1: Optional[float] = None, ic_fn: Optional[Callable] = None):
        super().__init__(config)
        self.bench = DFGCylinder(variant=config.problem.get("variant", "2D-2"))
        self.t0 = float(t0)
        self.t1 = float(t1 if t1 is not None else config.problem.get("window_dt", 0.5))
        self.ic_fn = ic_fn  # dimensional (x, y) -> (u, v, p)
        self.hard_bc = bool(config.problem.get("hard_bc", True))
        self.outflow = config.problem.get("outflow", "do_nothing")
        self.Re = self.bench.Re
        L, U = self.bench.L_ref, self.bench.U_ref
        self.scale = jnp.array([U / L, 1.0 / L, 1.0 / L])  # dimensional -> nondimensional
        self.T_star = (self.t1 - self.t0) * U / L
        # network input normalisation (nondim -> O(1))
        self.in_shift = jnp.array([self.t0 * U / L, 0.0, 0.0])
        self.in_scale = jnp.array([max(self.T_star, 1e-6), self.bench.length / L, self.bench.height / L])

    # ------------------------------------------------------------------
    # coordinates
    def to_nondim(self, z_dim):
        return z_dim * self.scale

    def to_dim(self, z_nd):
        return z_nd / self.scale

    def inflow_nd(self, t_nd, y_nd):
        """Non-dimensional inflow profile evaluated from nondimensional (t*, y*)."""
        t_dim = t_nd / self.scale[0]
        y_dim = y_nd / self.scale[2]
        return self.bench.inflow(t_dim, y_dim) / self.bench.U_ref

    # ------------------------------------------------------------------
    def net(self, params):
        def raw(z):
            return self.arch.apply(params, (z - self.in_shift) / self.in_scale)

        if not self.hard_bc:
            return raw
        b = self.bench
        g_dim = cylinder_inflow_extension(lambda t, y: b.inflow(t, y) / b.U_ref, b.center, b.radius)

        def g(z):
            zd = self.to_dim(z)
            return g_dim(zd[0], zd[1], zd[2])

        literal = self.config.problem.get("phi", "bounded") == "handbook"

        def phi(z):
            zd = self.to_dim(z)
            if literal:  # handbook 4.6b polynomial, rescaled by L^4 to be dimensionless
                return phi_channel_cylinder(zd[1], zd[2], b.center, b.radius, b.height, include_inlet=True) / (b.L_ref**4)
            return phi_channel_cylinder_bounded(zd[1], zd[2], b.center, b.radius, b.height, include_inlet=True)

        return hard_dirichlet(raw, g, phi, constrained=(0, 1))

    def velocity_fn(self, params):
        return self.net(params)

    def velocity_dim_fn(self, params):
        """Dimensional (t, x, y) -> dimensional (u, v, p); what :func:`pinnflow.metrics.drag_lift` needs."""
        f = self.net(params)
        b = self.bench

        def vel(z_dim):
            out = f(self.to_nondim(z_dim))
            return jnp.concatenate([out[:2] * b.U_ref, out[2:3] * b.rho * b.U_ref**2])

        return vel

    def residual_fn(self, params):
        return physics.ns_vp_residual(self.net(params), self.Re, dim=2, unsteady=True)

    def outflow_residual_fn(self, params):
        f = self.net(params)

        def r(z):
            out, J = physics.derivatives(f, z, 1)
            u_x, v_x, p = J[0, 1], J[1, 1], out[2]
            if self.outflow == "neumann_p0":
                return jnp.stack([u_x, p, v_x])
            return jnp.stack([u_x / self.Re - p, v_x, jnp.zeros(())])

        return r

    def terminal_state_fn(self, params) -> Callable:
        """Dimensional (x, y) -> (u, v, p) at t = t1, for the next time window (handbook 6.4)."""
        vel = self.velocity_dim_fn(params)
        return lambda x, y: vel(jnp.array([self.t1, x, y]))

    # ------------------------------------------------------------------
    def uniform_collocation(self, key, n):
        return self.to_nondim(channel_interior(key, self.bench, n, self.t0, self.t1))

    def residual_magnitude_fn(self, params):
        r = self.residual_fn(params)
        return lambda z: jnp.linalg.norm(r(z)[0])

    def sample_batch(self, key):
        cfg = self.config.training
        k = jax.random.split(key, 3)
        res = self.draw_collocation(k[0], int(cfg.res_batch_size))
        bnd = channel_boundaries(k[1], self.bench, int(cfg.bc_batch_size), self.t0, self.t1)
        ic = channel_initial(k[2], self.bench, int(cfg.ic_batch_size), self.t0)
        batch = {"res": res, "ic": self.to_nondim(ic), **{f"bc_{k_}": self.to_nondim(v) for k_, v in bnd.items()}}
        if self.ic_fn is not None:
            ic_vals = jax.vmap(self.ic_fn)(ic[:, 1], ic[:, 2])  # dimensional (u, v, p)
            batch["ic_vals"] = jnp.stack([ic_vals[:, 0] / self.bench.U_ref, ic_vals[:, 1] / self.bench.U_ref, ic_vals[:, 2] / (self.bench.rho * self.bench.U_ref**2)], -1)
        else:
            batch["ic_vals"] = jnp.zeros((ic.shape[0], 3))
        batch.update(self._common_batch_fields())
        return batch

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        f = jax.vmap(self.net(params))
        out = {}
        # initial condition
        pred_ic = f(batch["ic"])
        out["u_ic"] = mse(pred_ic[:, 0], batch["ic_vals"][:, 0])
        out["v_ic"] = mse(pred_ic[:, 1], batch["ic_vals"][:, 1])
        if self.ic_fn is not None:
            out["p_ic"] = mse(pred_ic[:, 2], batch["ic_vals"][:, 2])
        # Dirichlet boundaries (soft only)
        if not self.hard_bc:
            pin = f(batch["bc_inlet"])
            u_in = self.inflow_nd(batch["bc_inlet"][:, 0], batch["bc_inlet"][:, 2])
            out["u_in"] = mse(pin[:, 0], u_in)
            out["v_in"] = mse(pin[:, 1])
            for name in ("walls", "cylinder"):
                p = f(batch[f"bc_{name}"])
                out[f"u_{name}"] = mse(p[:, 0])
                out[f"v_{name}"] = mse(p[:, 1])
        # outflow (soft)
        r_out = jax.vmap(self.outflow_residual_fn(params))(batch["bc_outlet"])
        out["u_out"] = mse(r_out[:, 0])
        out["v_out"] = mse(r_out[:, 1])
        if self.outflow == "neumann_p0":
            out["p_out"] = mse(r_out[:, 2])
        # residuals
        r_mom, r_c = jax.vmap(self.residual_fn(params))(batch["res"])
        terms = [r_mom[:, 0] ** 2, r_mom[:, 1] ** 2, r_c**2]
        if self.use_causal:
            vals, _ = causal_residual_losses(batch["res"][:, 0], terms, self.num_chunks, batch["causal_eps"])
        else:
            vals = [jnp.mean(t) for t in terms]
        out.update(dict(zip(["r_u", "r_v", "r_c"], vals)))
        return out

    def causal_min_weight(self, params, batch):
        r_mom, r_c = jax.vmap(self.residual_fn(params))(batch["res"])
        _, gamma = causal_residual_losses(batch["res"][:, 0], [r_mom[:, 0] ** 2, r_mom[:, 1] ** 2, r_c**2], self.num_chunks, batch["causal_eps"])
        return gamma.min()

    def ntk_diags(self, params, batch):
        from ..losses import ntk_diag

        net_i = lambda i: (lambda p, z: self.net(p)(z)[i])
        res_i = lambda i: (lambda p, z: self.residual_fn(p)(z)[0][i])
        d = {
            "u_ic": ntk_diag(net_i(0), params, batch["ic"]),
            "v_ic": ntk_diag(net_i(1), params, batch["ic"]),
            "u_out": ntk_diag(lambda p, z: self.outflow_residual_fn(p)(z)[0], params, batch["bc_outlet"]),
            "v_out": ntk_diag(lambda p, z: self.outflow_residual_fn(p)(z)[1], params, batch["bc_outlet"]),
            "r_u": ntk_diag(res_i(0), params, batch["res"]),
            "r_v": ntk_diag(res_i(1), params, batch["res"]),
            "r_c": ntk_diag(lambda p, z: self.residual_fn(p)(z)[1], params, batch["res"]),
        }
        if not self.hard_bc:
            for name in ("inlet", "walls", "cylinder"):
                tag = {"inlet": "in"}.get(name, name)
                d[f"u_{tag}"] = ntk_diag(net_i(0), params, batch[f"bc_{name}"])
                d[f"v_{tag}"] = ntk_diag(net_i(1), params, batch[f"bc_{name}"])
        return d

    # ------------------------------------------------------------------
    def evaluate(self, params) -> Dict[str, float]:
        vel = self.velocity_dim_fn(params)
        out = {}
        for frac in (0.5, 1.0):
            t = self.t0 + frac * (self.t1 - self.t0)
            cd, cl, dp = drag_lift(vel, t, self.bench, n_theta=128)
            out[f"Cd@{t:.2f}"] = float(cd)
            out[f"Cl@{t:.2f}"] = float(cl)
            out[f"dP@{t:.2f}"] = float(dp)
        return out
