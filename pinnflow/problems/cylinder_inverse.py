"""Benchmark C (inverse) - recover the Navier-Stokes parameters and the hidden pressure from sparse
velocity data in the Raissi et al. (2019) cylinder wake (Nektar, Re = 100).

    r_u = u_t + lambda_1 (u u_x + v u_y) + p_x - lambda_2 (u_xx + u_yy)
    r_v = v_t + lambda_1 (u v_x + v v_y) + p_y - lambda_2 (v_xx + v_yy)

with trainable ``lambda_1`` (target 1.0) and ``lambda_2`` (target nu = 0.01). Network outputs
(psi, p) so continuity is exact (Raissi's formulation) or (u, v, p) with ``formulation="vp"``.
This is the *data-assimilation mode*: L_data enters the loss on purpose (handbook STEP 3).
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .. import physics
from ..benchmarks import CylinderWakeInverse
from ..data import load_cylinder_wake, sample_sparse
from ..losses import align_pressure_gauge, mse
from ..metrics import relative_l2
from ..sampling import uniform_box
from .base import Problem


class CylinderWakeInversePINN(Problem):
    input_dim = 3  # (t, x, y)

    def __init__(self, config):
        super().__init__(config)
        self.bench = CylinderWakeInverse()
        self.formulation = config.problem.get("formulation", "streamfunction")
        data = load_cylinder_wake()
        self.data = data
        sp = sample_sparse(data["flat"], int(config.problem.get("n_train", 5000)), seed=int(config.seed))
        self.train_pts = jnp.stack([sp["t"], sp["x"], sp["y"]], -1)
        self.train_uv = jnp.stack([sp["u"], sp["v"]], -1)
        self.dom = jnp.asarray(self.bench.domain)
        self.in_scale = jnp.array([self.bench.domain[0, 1], 8.0, 2.0])
        self.noise = float(config.problem.get("noise", 0.0))
        if self.noise > 0:
            rng = np.random.default_rng(int(config.seed))
            self.train_uv = self.train_uv + self.noise * jnp.std(self.train_uv, axis=0) * jnp.asarray(rng.standard_normal(self.train_uv.shape))
        # validation snapshot (t index 100): all 5000 points
        k = 100
        self.val_pts = jnp.concatenate([jnp.full((data["X_star"].shape[0], 1), float(data["t"][k])), jnp.asarray(data["X_star"])], axis=1)
        self.val_uvp = jnp.stack([data["U_star"][:, 0, k], data["U_star"][:, 1, k], data["p_star"][:, k]], -1)

    # ------------------------------------------------------------------
    def init_params(self, key):
        # both unknown parameters start at 0 (Raissi et al. 2019) and are learned jointly with the network
        return {"net": self.arch.init(key, jnp.zeros(self.input_dim)), "lambda": jnp.array([0.0, 0.0])}

    def lambdas(self, params):
        lam = params["lambda"]
        return lam[0], lam[1]

    def net(self, params):
        return lambda z: self.arch.apply(params["net"], z / self.in_scale)

    def velocity_fn(self, params):
        f = self.net(params)
        if self.formulation == "streamfunction":
            return physics.streamfunction_velocity(f, unsteady=True)
        return f

    def residual_fn(self, params):
        vel = self.velocity_fn(params)
        lam1, lam2 = self.lambdas(params)
        # r = u_t + lambda_1 (u . grad) u + grad p - lambda_2 lap u  (Raissi et al. 2019)
        return physics.ns_vp_residual(vel, Re=None, dim=2, unsteady=True, conv_coeff=lam1, visc=lam2)

    # ------------------------------------------------------------------
    def uniform_collocation(self, key, n):
        return uniform_box(key, self.dom, n)

    def sample_batch(self, key):
        cfg = self.config.training
        k1, k2 = jax.random.split(key)
        idx = jax.random.choice(k1, self.train_pts.shape[0], (int(cfg.data_batch_size),), replace=False)
        return {"data": self.train_pts[idx], "data_uv": self.train_uv[idx], "res": self.draw_collocation(k2, int(cfg.res_batch_size)), **self._common_batch_fields()}

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        vel = jax.vmap(self.velocity_fn(params))
        pred = vel(batch["data"])
        out = {"u_data": mse(pred[:, 0], batch["data_uv"][:, 0]), "v_data": mse(pred[:, 1], batch["data_uv"][:, 1])}
        pts = jnp.concatenate([batch["data"], batch["res"]], axis=0)
        r_mom, r_c = self.vmap_pointwise(self.residual_fn(params))(pts)
        out["r_u"] = mse(r_mom[:, 0])
        out["r_v"] = mse(r_mom[:, 1])
        if self.formulation == "vp":
            out["r_c"] = mse(r_c)
        return out

    def evaluate(self, params) -> Dict[str, float]:
        lam1, lam2 = self.lambdas(params)
        pred = jax.vmap(self.velocity_fn(params))(self.val_pts)
        return {
            "lambda1": float(lam1),
            "lambda2": float(lam2),
            "lambda1_err_pct": float(100 * abs(lam1 - self.bench.lambda1_true) / self.bench.lambda1_true),
            "lambda2_err_pct": float(100 * abs(lam2 - self.bench.nu_true) / self.bench.nu_true),
            "rel_l2_u": float(relative_l2(pred[:, 0], self.val_uvp[:, 0])),
            "rel_l2_v": float(relative_l2(pred[:, 1], self.val_uvp[:, 1])),
            "rel_l2_p": float(relative_l2(align_pressure_gauge(pred[:, 2], self.val_uvp[:, 2]), self.val_uvp[:, 2])),
        }
