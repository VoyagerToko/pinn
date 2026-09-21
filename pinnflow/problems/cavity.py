"""Benchmark B - steady lid-driven cavity with the regularised lid, Re curriculum 100 -> 400 -> 1000.

Formulations:
    vp                 (u, v, p); boundary either soft (u_bc, v_bc losses) or hard via
                       u_hat = g + phi N with g = (u_lid(x) y, 0), phi = x(1-x)y(1-y)  [handbook 4.6b]
    streamfunction     (psi, p) exactly divergence-free; soft BCs on (psi_y, -psi_x)
Pressure is anchored to zero mean (closed domain). ``set_Re`` switches the Reynolds number for the
curriculum without re-initialising the network (warm start).
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .. import physics
from ..benchmarks import LidDrivenCavity, ghia_tables
from ..constraints import hard_dirichlet, lid_extension, phi_unit_square
from ..data import cavity_centerlines, load_jaxpi_cavity
from ..losses import mse, pressure_anchor_closed
from ..metrics import relative_l2
from ..sampling import square_boundary, uniform_box
from ..utils import chunked_vmap, meshgrid_points
from .base import Problem


class CavityPINN(Problem):
    input_dim = 2  # (x, y)

    def __init__(self, config):
        super().__init__(config)
        self.bench = LidDrivenCavity(Re=float(config.problem.Re))
        self.Re = float(config.problem.Re)
        self.formulation = config.problem.get("formulation", "vp")
        self.hard_bc = bool(config.problem.get("hard_bc", False)) and self.formulation == "vp"
        self.dom = jnp.asarray(self.bench.domain)
        self._ref_cache: Dict[int, Dict] = {}

    def set_Re(self, Re: float):
        self.Re = float(Re)
        self.bench = LidDrivenCavity(Re=float(Re))

    # ------------------------------------------------------------------
    def net(self, params):
        raw = lambda z: self.arch.apply(params, z)
        if self.hard_bc:
            g = lid_extension(self.bench.lid_profile)
            return hard_dirichlet(raw, lambda z: g(z[0], z[1]), lambda z: phi_unit_square(z[0], z[1]), constrained=(0, 1))
        return raw

    def velocity_fn(self, params):
        f = self.net(params)
        if self.formulation == "streamfunction":
            return physics.streamfunction_velocity(f, unsteady=False)
        return f

    def residual_fn(self, params, Re):
        f = self.net(params)
        if self.formulation == "streamfunction":
            return physics.ns_streamfunction_residual(f, Re, unsteady=False)
        return physics.ns_vp_residual(f, Re, dim=2, unsteady=False)

    # ------------------------------------------------------------------
    def uniform_collocation(self, key, n):
        return uniform_box(key, self.dom, n)

    def residual_magnitude_fn(self, params):
        r = self.residual_fn(params, self.Re)
        return lambda z: jnp.linalg.norm(r(z)[0])

    def sample_batch(self, key):
        cfg = self.config.training
        k1, k2 = jax.random.split(key)
        res = self.draw_collocation(k1, int(cfg.res_batch_size))
        b = square_boundary(k2, int(cfg.bc_batch_size) // 4)
        bc = jnp.concatenate([b["bottom"], b["top"], b["left"], b["right"]], axis=0)
        return {"res": res, "bc": bc, "Re": jnp.asarray(self.Re, jnp.float32), **self._common_batch_fields()}

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        Re = batch["Re"]
        out = {}
        vel = jax.vmap(self.velocity_fn(params))
        if not self.hard_bc:
            pred = vel(batch["bc"])
            u_bc, v_bc = self.bench.boundary_velocity(batch["bc"][:, 0], batch["bc"][:, 1])
            out["u_bc"] = mse(pred[:, 0], u_bc)
            out["v_bc"] = mse(pred[:, 1], v_bc)
        r_mom, r_c = jax.vmap(self.residual_fn(params, Re))(batch["res"])
        out["r_u"] = mse(r_mom[:, 0])
        out["r_v"] = mse(r_mom[:, 1])
        if self.formulation == "vp":
            out["r_c"] = mse(r_c)
        out["p_anchor"] = pressure_anchor_closed(vel(batch["res"])[:, 2])
        return out

    def ntk_diags(self, params, batch):
        from ..losses import ntk_diag

        vel_i = lambda i: (lambda p, z: self.velocity_fn(p)(z)[i])
        res_i = lambda i: (lambda p, z: self.residual_fn(p, batch["Re"])(z)[0][i])
        d = {"r_u": ntk_diag(res_i(0), params, batch["res"]), "r_v": ntk_diag(res_i(1), params, batch["res"])}
        if not self.hard_bc:
            d["u_bc"] = ntk_diag(vel_i(0), params, batch["bc"])
            d["v_bc"] = ntk_diag(vel_i(1), params, batch["bc"])
        if self.formulation == "vp":
            d["r_c"] = ntk_diag(lambda p, z: self.residual_fn(p, batch["Re"])(z)[1], params, batch["res"])
        return d

    # ------------------------------------------------------------------
    def _reference(self, Re: int):
        if Re not in self._ref_cache:
            try:
                self._ref_cache[Re] = load_jaxpi_cavity(Re)
            except Exception:
                self._ref_cache[Re] = None
        return self._ref_cache[Re]

    def evaluate(self, params) -> Dict[str, float]:
        Re = int(round(self.Re))
        vel = self.velocity_fn(params)
        out = {}
        ref = self._reference(Re)
        if ref is not None:
            pts = meshgrid_points(jnp.asarray(ref["x"]), jnp.asarray(ref["y"]))
            pred = chunked_vmap(vel, pts)
            U_pred = jnp.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2).reshape(ref["u"].shape)
            U_ref = np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)
            out["rel_l2_speed_vs_jaxpi"] = float(relative_l2(U_pred, jnp.asarray(U_ref)))
        try:
            tab = ghia_tables(Re)
            yq = jnp.asarray(tab["y"][1:-1])  # skip the lid point (regularised vs unit lid) and the wall
            xq = jnp.asarray(tab["x"][1:-1])
            u_c = chunked_vmap(vel, jnp.stack([jnp.full_like(yq, 0.5), yq], -1))[:, 0]
            v_c = chunked_vmap(vel, jnp.stack([xq, jnp.full_like(xq, 0.5)], -1))[:, 1]
            out["ghia_u_rel_err"] = float(relative_l2(u_c, jnp.asarray(tab["u"][1:-1])))
            out["ghia_v_rel_err"] = float(relative_l2(v_c, jnp.asarray(tab["v"][1:-1])))
        except KeyError:
            pass
        return out
