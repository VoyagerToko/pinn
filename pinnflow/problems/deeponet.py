"""Component 7 - PI-DeepONet over the Reynolds number for the lid-driven cavity.

    G_theta(a)(y) = sum_k b_k(a) t_k(y) + b_0,   a = log10(Re) (scaled),  y = (x, y)

The physics loss is applied to ``G(a)(y)`` for Re sampled log-uniformly in ``[Re_min, Re_max]``;
boundary conditions are hard (same g/phi as the cavity). Zero-shot inference at an unseen Re is a
single forward pass. Gate (STEP 10): zero-shot error at unseen Re < 10%.
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .. import physics
from ..benchmarks import LidDrivenCavity, ghia_tables
from ..constraints import hard_dirichlet, lid_extension, phi_unit_square, phi_unit_square_open_top
from ..data import load_jaxpi_cavity
from ..losses import mse, pressure_anchor_closed
from ..metrics import relative_l2
from ..sampling import uniform_box
from ..utils import chunked_vmap, meshgrid_points
from .base import Problem


class ParametricCavityDeepONet(Problem):
    input_dim = 2

    def __init__(self, config):
        super().__init__(config)
        self.bench = LidDrivenCavity()
        self.Re_range = (float(config.problem.Re_min), float(config.problem.Re_max))
        self.n_Re = int(config.training.n_Re_per_batch)
        self.dom = jnp.asarray(self.bench.domain)
        self.eval_Re = tuple(int(r) for r in config.problem.get("eval_Re", (100, 400, 1000)))
        # zero-shot test: Re values inside ``holdout_Re_band`` are never sampled during training
        band = config.problem.get("holdout_Re_band", None)
        self.holdout = None if band is None else (float(band[0]), float(band[1]))
        self.zero_shot_Re = config.problem.get("zero_shot_Re", None)
        # same boundary treatment options as CavityPINN (see problems/cavity.py for the lid_bc="soft" rationale)
        self.soft_lid = config.problem.get("lid_bc", "hard") == "soft"
        if self.soft_lid:
            self.init_weights.setdefault("u_lid", 1.0)

    def init_params(self, key):
        return self.arch.init(key, jnp.zeros(1), jnp.zeros(self.input_dim))

    def encode_Re(self, Re):
        return jnp.atleast_1d((jnp.log10(Re) - 2.0) / 2.0)  # Re=100 -> 0, Re=10^4 -> 1

    def net(self, params, Re):
        raw = lambda z: self.arch.apply(params, self.encode_Re(Re), z)
        g = lid_extension(self.bench.lid_profile, power=float(self.config.problem.get("lid_power", 8.0)))
        if self.soft_lid:
            phi = lambda z: jnp.stack([phi_unit_square_open_top(z[0], z[1]), phi_unit_square(z[0], z[1])])
        else:
            phi = lambda z: phi_unit_square(z[0], z[1])
        return hard_dirichlet(raw, lambda z: g(z[0], z[1]), phi, constrained=(0, 1))

    def residual_fn(self, params, Re):
        return physics.ns_vp_residual(self.net(params, Re), Re, dim=2, unsteady=False)

    def sample_batch(self, key):
        cfg = self.config.training
        k1, k2 = jax.random.split(key)
        lo, hi = np.log10(self.Re_range[0]), np.log10(self.Re_range[1])
        if self.holdout is None:
            Re = 10 ** jax.random.uniform(k1, (self.n_Re,), minval=lo, maxval=hi)
        else:
            # log-uniform on [lo, hi] minus the held-out band: draw on the shortened interval, then
            # shift the part above the band's lower edge past the band
            b0, b1 = np.log10(self.holdout[0]), np.log10(self.holdout[1])
            s = jax.random.uniform(k1, (self.n_Re,), minval=lo, maxval=hi - (b1 - b0))
            Re = 10 ** jnp.where(s < b0, s, s + (b1 - b0))
        k2, k3 = jax.random.split(k2)
        res = uniform_box(k2, self.dom, int(cfg.res_batch_size))
        x_lid = jax.random.uniform(k3, (int(cfg.get("bc_batch_size", 2048)) // 4,))
        return {"Re": Re, "res": res, "lid": jnp.stack([x_lid, jnp.ones_like(x_lid)], -1)}

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        def per_Re(Re):
            r_mom, r_c = self.vmap_pointwise(self.residual_fn(params, Re))(batch["res"])
            p = jax.vmap(self.net(params, Re))(batch["res"])[:, 2]
            u_top = jax.vmap(self.net(params, Re))(batch["lid"])[:, 0]
            return mse(r_mom[:, 0]), mse(r_mom[:, 1]), mse(r_c), pressure_anchor_closed(p), mse(u_top, self.bench.lid_profile(batch["lid"][:, 0]))

        ru, rv, rc, pa, ul = jax.vmap(per_Re)(batch["Re"])
        out = {"r_u": ru.mean(), "r_v": rv.mean(), "r_c": rc.mean(), "p_anchor": pa.mean()}
        if self.soft_lid:
            out["u_lid"] = ul.mean()
        return out

    def evaluate(self, params) -> Dict[str, float]:
        out = {}
        for Re in self.eval_Re:
            try:
                ref = load_jaxpi_cavity(Re)
            except Exception:
                continue
            pts = meshgrid_points(jnp.asarray(ref["x"]), jnp.asarray(ref["y"]))
            pred = chunked_vmap(self.net(params, float(Re)), pts)
            U_pred = jnp.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2).reshape(ref["u"].shape)
            U_ref = np.sqrt(ref["u"] ** 2 + ref["v"] ** 2)
            out[f"rel_l2_speed_Re{Re}"] = float(relative_l2(U_pred, jnp.asarray(U_ref)))
            out[f"rel_l2_uv_Re{Re}"] = float(relative_l2(pred[:, :2], jnp.stack([jnp.asarray(ref["u"]).ravel(), jnp.asarray(ref["v"]).ravel()], -1)))
            try:
                tab = ghia_tables(Re)
                yq, xq = jnp.asarray(tab["y"][1:-1]), jnp.asarray(tab["x"][1:-1])
                f = self.net(params, float(Re))
                u_c = chunked_vmap(f, jnp.stack([jnp.full_like(yq, 0.5), yq], -1))[:, 0]
                v_c = chunked_vmap(f, jnp.stack([xq, jnp.full_like(xq, 0.5)], -1))[:, 1]
                out[f"ghia_u_rel_err_Re{Re}"] = float(relative_l2(u_c, jnp.asarray(tab["u"][1:-1])))
                out[f"ghia_v_rel_err_Re{Re}"] = float(relative_l2(v_c, jnp.asarray(tab["v"][1:-1])))
            except KeyError:
                pass
        return out
