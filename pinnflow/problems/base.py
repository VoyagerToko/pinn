"""Problem interface shared by all benchmarks (see :mod:`pinnflow.training` for the contract)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

from ..archs import create_arch
from ..losses import rad_resample
from ..utils import chunked_vmap


class Problem:
    input_dim: int = 3

    def __init__(self, config):
        self.config = config
        self.arch = create_arch(config.arch)
        self.init_weights: Dict[str, float] = dict(config.weighting.init_weights)
        self.causal_eps = float(config.weighting.get("causal_tol", 1.0))
        self.use_causal = bool(config.weighting.get("use_causal", False))
        self.num_chunks = int(config.weighting.get("num_chunks", 32))
        self.rad_pool: Optional[jnp.ndarray] = None

    # --- parameters -------------------------------------------------------------------
    def init_params(self, key):
        return self.arch.init(key, jnp.zeros(self.input_dim))

    def net(self, params) -> Callable:
        """Point-wise network ``z -> outputs`` (override to add input scaling / hard constraints)."""
        return lambda z: self.arch.apply(params, z)

    # --- knobs the Trainer can turn ----------------------------------------------------
    def set_causal_eps(self, eps: float):
        self.causal_eps = float(eps)

    def _common_batch_fields(self) -> Dict[str, jnp.ndarray]:
        """Scalars that must reach the jitted loss as traced values (not baked-in constants)."""
        return {"causal_eps": jnp.asarray(self.causal_eps, dtype=jnp.float32)}

    # --- RAD hook (handbook 5.5) -----------------------------------------------------
    def residual_magnitude_fn(self, params) -> Optional[Callable]:
        """Return ``pts (N, d) -> |r|`` for RAD, or None if the problem does not support it."""
        return None

    def uniform_collocation(self, key, n: int) -> jnp.ndarray:
        raise NotImplementedError

    def update_collocation(self, key, params):
        fn = self.residual_magnitude_fn(params)
        if fn is None:
            return
        cfg = self.config.training
        self.rad_pool = rad_resample(
            key,
            lambda pts: chunked_vmap(lambda p: fn(p), pts, chunk=int(cfg.get("rad_chunk", 8192))),
            self.uniform_collocation,
            n_points=int(cfg.get("rad_pool_size", 8 * cfg.get("res_batch_size", 8192))),
            n_candidates=int(cfg.get("rad_candidates", 100_000)),
            k=float(cfg.get("rad_k", 1.0)),
            c=float(cfg.get("rad_c", 1.0)),
            uniform_frac=float(cfg.get("rad_uniform_frac", 0.2)),
        )

    def draw_collocation(self, key, n: int) -> jnp.ndarray:
        """Uniform points, or points from the RAD pool once it exists."""
        if self.rad_pool is None:
            return self.uniform_collocation(key, n)
        idx = jax.random.choice(key, self.rad_pool.shape[0], (n,), replace=False)
        return self.rad_pool[idx]

    # --- memory --------------------------------------------------------------------------
    def vmap_pointwise(self, fn):
        """``jax.vmap`` of a per-point function, evaluated in rematerialised chunks.

        Residuals with second/third-order derivatives create ~50 network-forward-equivalents of
        intermediates per point (about 0.6 MB per point for a 6 x 256 network). Reverse mode needs
        them all live at once, so a plain ``vmap`` over 8192 points allocates ~5 GB in a single
        buffer. Here the batch is split into chunks of ``training.res_chunk`` points, each chunk is
        ``jax.checkpoint``-ed and the chunks are processed by ``lax.map`` (a scan). The backward pass
        then recomputes and differentiates one chunk at a time, so peak memory scales with the
        chunk size instead of the batch size (about +30% compute).
        Set ``training.remat=False`` for a plain ``vmap`` when memory is plentiful.
        """
        vf = jax.vmap(fn)
        if not bool(self.config.training.get("remat", True)):
            return vf
        cf = jax.checkpoint(vf)
        chunk = int(self.config.training.get("res_chunk", 2048))

        def run(pts):
            n = pts.shape[0]
            if n <= chunk or n % chunk != 0:
                return cf(pts)
            out = jax.lax.map(cf, pts.reshape(n // chunk, chunk, *pts.shape[1:]))
            return jax.tree_util.tree_map(lambda x: x.reshape(n, *x.shape[2:]), out)

        return run

    # --- interface ---------------------------------------------------------------------
    def sample_batch(self, key):
        raise NotImplementedError

    def losses(self, params, batch) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    def evaluate(self, params) -> Dict[str, float]:
        return {}
