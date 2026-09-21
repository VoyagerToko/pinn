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
from ..benchmarks import LidDrivenCavity
from ..constraints import hard_dirichlet, lid_extension, phi_unit_square
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

    def init_params(self, key):
        return self.arch.init(key, jnp.zeros(1), jnp.zeros(self.input_dim))

    def encode_Re(self, Re):
        return jnp.atleast_1d((jnp.log10(Re) - 2.0) / 2.0)  # Re=100 -> 0, Re=10^4 -> 1

    def net(self, params, Re):
        raw = lambda z: self.arch.apply(params, self.encode_Re(Re), z)
        g = lid_extension(self.bench.lid_profile)
        return hard_dirichlet(raw, lambda z: g(z[0], z[1]), lambda z: phi_unit_square(z[0], z[1]), constrained=(0, 1))

    def residual_fn(self, params, Re):
        return physics.ns_vp_residual(self.net(params, Re), Re, dim=2, unsteady=False)

    def sample_batch(self, key):
        cfg = self.config.training
        k1, k2 = jax.random.split(key)
        lo, hi = np.log10(self.Re_range[0]), np.log10(self.Re_range[1])
        Re = 10 ** jax.random.uniform(k1, (self.n_Re,), minval=lo, maxval=hi)
        res = uniform_box(k2, self.dom, int(cfg.res_batch_size))
        return {"Re": Re, "res": res}

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        def per_Re(Re):
            r_mom, r_c = jax.vmap(self.residual_fn(params, Re))(batch["res"])
            p = jax.vmap(self.net(params, Re))(batch["res"])[:, 2]
            return mse(r_mom[:, 0]), mse(r_mom[:, 1]), mse(r_c), pressure_anchor_closed(p)

        ru, rv, rc, pa = jax.vmap(per_Re)(batch["Re"])
        return {"r_u": ru.mean(), "r_v": rv.mean(), "r_c": rc.mean(), "p_anchor": pa.mean()}

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
        return out
