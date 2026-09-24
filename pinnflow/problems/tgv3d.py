"""Benchmark D - 3D Taylor-Green vortex at Re = 1600 with a separable PINN (SPINN).

Collocation is a separable grid ``t x X x Y x Z`` (handbook 5.2: 64^4 effective through
``sum n_i`` network evaluations). Residuals are computed with forward-mode ``jvp`` along each
1-D axis (:func:`pinnflow.physics.ns_vp_residual_grid`).

Formulations: ``vp`` (u, v, w, p) or ``vector_potential`` (A1, A2, A3, p; u = curl A exactly
divergence-free, plus the weak gauge term ``lambda_g ||div A||^2``, handbook 4.6a).
Periodicity in x, y, z is exact through the SPINN periodic axis embedding.

Loss terms: ic_u, ic_v, ic_w (and ic_A for the vector potential), r_u, r_v, r_w[, r_c], p_anchor[, gauge].
Causal weighting acts along the t axis of the grid (M chunks of n_t / M time samples).
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .. import physics
from ..benchmarks import TaylorGreen3D
from ..losses import causal_weights, mse, pressure_anchor_closed
from ..metrics import enstrophy, kinetic_energy, relative_l2
from ..sampling import linspace_axes, spinn_axes
from .base import Problem


class TaylorGreen3DSPINN(Problem):
    input_dim = 4  # (t, x, y, z)

    def __init__(self, config):
        super().__init__(config)
        self.bench = TaylorGreen3D(Re=float(config.problem.Re), T=float(config.problem.T))
        self.formulation = config.problem.get("formulation", "vp")
        self.gauge_weight = float(config.problem.get("gauge_weight", 1e-3))
        self.dom = jnp.asarray(self.bench.domain)
        self.n_axes = tuple(int(n) for n in config.training.n_per_axis)  # (n_t, n_x, n_y, n_z)
        self.n_ic = int(config.training.ic_grid)
        if self.use_causal:
            assert self.n_axes[0] % self.num_chunks == 0, "n_t must be divisible by num_chunks for causal weighting"
        self.eval_axes = linspace_axes(self.dom[1:], (24, 24, 24))

    # ------------------------------------------------------------------
    def init_params(self, key):
        return self.arch.init(key, [jnp.zeros(2) for _ in range(self.input_dim)], mode="grid")

    def grid_fn(self, params):
        """``f(t, x, y, z) -> (4, n_t, n_x, n_y, n_z)`` raw network outputs."""
        return lambda *coords: self.arch.apply(params, list(coords), mode="grid")

    def velocity_grid(self, params, coords):
        """(u, v, w, p) on the grid, regardless of formulation."""
        f = self.grid_fn(params)
        if self.formulation == "vector_potential":
            u = physics.curl_grid(f, coords, unsteady=True)
            return jnp.concatenate([u, f(*coords)[3:4]], axis=0)
        return f(*coords)

    def velocity_fn(self, params):
        """Point-wise (u, v, w, p) for z = (t, x, y, z): used by metrics and STEP 9 visualisation."""
        raw = lambda z: self.arch.apply(params, z, mode="points")
        if self.formulation == "vector_potential":
            return physics.vector_potential_velocity(raw, unsteady=True)
        return raw

    # ------------------------------------------------------------------
    def sample_batch(self, key):
        k1, k2 = jax.random.split(key)
        axes = spinn_axes(k1, self.dom, self.n_axes)
        ic_axes = spinn_axes(k2, self.dom[1:], (self.n_ic,) * 3)
        return {"axes": axes, "ic_axes": ic_axes, **self._common_batch_fields()}

    def _causal_mean(self, r_sq: jnp.ndarray, eps) -> jnp.ndarray:
        """r_sq (n_t, ...) -> causal-weighted mean over M chunks along t (t is sorted)."""
        per_t = r_sq.reshape(r_sq.shape[0], -1).mean(axis=1)
        chunks = per_t.reshape(self.num_chunks, -1).mean(axis=1)
        w = causal_weights(chunks, eps)
        return jnp.mean(w * chunks), w

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        out = {}
        coords = batch["axes"]
        f = self.grid_fn(params)
        if self.formulation == "vector_potential":
            r_mom, r_c, gauge = physics.ns_vector_potential_residual_grid(f, coords, self.bench.Re, unsteady=True)
            out["gauge"] = self.gauge_weight * mse(gauge)
        else:
            r_mom, r_c = physics.ns_vp_residual_grid(f, coords, self.bench.Re, unsteady=True)
        names = ["r_u", "r_v", "r_w"]
        terms = [r_mom[i] ** 2 for i in range(3)]
        if self.formulation == "vp":
            names.append("r_c")
            terms.append(r_c**2)
        if self.use_causal:
            for n, t in zip(names, terms):
                out[n], _ = self._causal_mean(t, batch["causal_eps"])
        else:
            out.update({n: jnp.mean(t) for n, t in zip(names, terms)})
        # pressure gauge on the grid
        out["p_anchor"] = pressure_anchor_closed(f(*coords)[3])
        # initial condition on a (t=0) x n_ic^3 grid
        ic_coords = [jnp.zeros(1), *batch["ic_axes"]]
        pred0 = self.velocity_grid(params, ic_coords)[:, 0]  # (4, nx, ny, nz)
        X, Y, Z = jnp.meshgrid(*batch["ic_axes"], indexing="ij")
        u0, v0, w0, _ = self.bench.initial_condition(X, Y, Z)
        out["ic_u"] = mse(pred0[0], u0)
        out["ic_v"] = mse(pred0[1], v0)
        out["ic_w"] = mse(pred0[2], w0)
        return out

    def causal_min_weight(self, params, batch):
        r_mom, _ = physics.ns_vp_residual_grid(self.velocity_grid_fn(params), batch["axes"], self.bench.Re, unsteady=True)
        _, w = self._causal_mean(jnp.sum(r_mom**2, axis=0), batch["causal_eps"])
        return w.min()

    def velocity_grid_fn(self, params):
        return lambda *c: self.velocity_grid(params, list(c))

    # ------------------------------------------------------------------
    def energy_history(self, params, times, n: int = 32) -> Dict[str, np.ndarray]:
        """E_k(t), enstrophy(t) on an n^3 grid (STEP 7.4) - used by scripts/evaluate.py."""
        axes = linspace_axes(self.dom[1:], (n, n, n))
        vf = self.velocity_grid_fn(params)
        Ek, Z = [], []
        for t in times:
            coords = [jnp.asarray([float(t)]), *axes]
            vel = vf(*coords)[:3, 0]
            _, first, _ = physics.grid_derivatives(lambda *c: vf(*c)[:3], coords)
            dU = jnp.stack([first[s][:, 0] for s in (1, 2, 3)], axis=1)  # dU[i, j] = d u_i / d x_j
            omega = jnp.einsum("ijk,kj...->i...", physics._EPS, dU)
            Ek.append(float(kinetic_energy(vel)))
            Z.append(float(enstrophy(omega)))
        return {"t": np.asarray(times), "Ek": np.asarray(Ek), "enstrophy": np.asarray(Z)}

    def grid_fields(self, params, t: float, n: int) -> Dict[str, np.ndarray]:
        """Velocity, speed, vorticity and Q on an n^3 grid at time t in separable (grid) mode - one network
        evaluation per axis instead of n^3 point-wise Jacobians. Arrays are flattened in 'ij' order
        (x slowest), the layout of :func:`pinnflow.viz.sample_grid`."""
        if not hasattr(self, "_grid_fields_jit"):
            self._grid_fields_jit = {}
        if n not in self._grid_fields_jit:
            axes = linspace_axes(self.dom[1:], (n, n, n))

            def core(params, t):
                coords = [jnp.reshape(t, (1,)), *axes]
                vf = self.velocity_grid_fn(params)
                vel = vf(*coords)[:3, 0]
                _, first, _ = physics.grid_derivatives(lambda *c: vf(*c)[:3], coords)
                dU = jnp.stack([first[s][:, 0] for s in (1, 2, 3)], axis=1)  # dU[i, j] = d u_i / d x_j, (3, 3, n, n, n)
                omega = jnp.einsum("ijk,kj...->i...", physics._EPS, dU)
                S = 0.5 * (dU + jnp.swapaxes(dU, 0, 1))
                O = 0.5 * (dU - jnp.swapaxes(dU, 0, 1))
                q = 0.5 * (jnp.sum(O * O, axis=(0, 1)) - jnp.sum(S * S, axis=(0, 1)))
                return vel, omega, q

            self._grid_fields_jit[n] = jax.jit(core)
        vel, omega, q = self._grid_fields_jit[n](params, jnp.asarray(float(t)))
        v = np.asarray(vel).reshape(3, -1).T
        return {"velocity": v, "speed": np.linalg.norm(v, axis=1), "vorticity": np.asarray(omega).reshape(3, -1).T, "qcriterion": np.asarray(q).ravel()}

    def evaluate(self, params) -> Dict[str, float]:
        axes = self.eval_axes
        vel0 = self.velocity_grid(params, [jnp.zeros(1), *axes])[:3, 0]
        X, Y, Z = jnp.meshgrid(*axes, indexing="ij")
        u0, v0, w0, _ = self.bench.initial_condition(X, Y, Z)
        out = {"rel_l2_ic": float(relative_l2(vel0, jnp.stack([u0, v0, w0])))}
        hist = self.energy_history(params, np.linspace(0, self.bench.T, 5), n=24)
        for t, e in zip(hist["t"], hist["Ek"]):
            out[f"Ek@{t:.1f}"] = float(e)
        return out
