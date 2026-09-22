"""STEP 5 - the loss function, term by term.

    L(theta) = lambda_r L_res + lambda_c L_div + lambda_b L_bc + lambda_i L_ic + lambda_d L_data

5.3  adaptive loss weighting: gradient-norm annealing, NTK weighting, self-adaptive (SA-PINN) masks
5.4  causal training weights over M temporal windows
5.5  residual-based adaptive collocation resampling (RAD)
5.6  pressure anchoring
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.flatten_util import ravel_pytree

Array = jnp.ndarray


def mse(x: Array, y: Optional[Array] = None) -> Array:
    return jnp.mean(x**2) if y is None else jnp.mean((x - y) ** 2)


def weighted_total(losses: Dict[str, Array], weights: Dict[str, float]) -> Array:
    """sum_i lambda_i L_i ; terms missing from ``weights`` get weight 1."""
    return sum(losses[k] * weights.get(k, 1.0) for k in losses)


# --------------------------------------------------------------------------------------
# 5.4 causal weights
# --------------------------------------------------------------------------------------
def causal_chunk_losses(t: Array, r_sq: Array, num_chunks: int) -> Array:
    """Sort squared residuals by time and return the mean loss in each of ``M`` equal windows."""
    order = jnp.argsort(t)
    r_sq = r_sq[order]
    n = (r_sq.shape[0] // num_chunks) * num_chunks
    return r_sq[:n].reshape(num_chunks, -1).mean(axis=1)


def causal_weights(chunk_losses: Array, eps: float) -> Array:
    """w_i = exp(-eps * sum_{k<i} L_k)  (stop-gradient; window i waits for earlier windows)."""
    M = chunk_losses.shape[0]
    strictly_lower = jnp.tril(jnp.ones((M, M)), k=-1)  # [i, k] = 1 if k < i
    return lax.stop_gradient(jnp.exp(-eps * (strictly_lower @ chunk_losses)))


def causal_residual_losses(t: Array, r_sq_terms: Sequence[Array], num_chunks: int, eps: float) -> Tuple[List[Array], Array]:
    """Causal loss for several residual equations sharing the same collocation times.

    Returns ``([mean_i w_i L_i^(eq) for each eq], gamma)`` where ``gamma`` is the element-wise
    minimum of the per-equation weights (JAX-PI convention) and is what the annealer monitors.
    """
    chunks = [causal_chunk_losses(t, r, num_chunks) for r in r_sq_terms]
    gamma = jnp.stack([causal_weights(c, eps) for c in chunks]).min(axis=0)
    return [jnp.mean(gamma * c) for c in chunks], gamma


@dataclass
class CausalAnnealer:
    """Anneal eps through ``schedule`` (handbook 5.4), advancing when ``min_i w_i > threshold``."""

    schedule: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    threshold: float = 0.99
    idx: int = 0

    @property
    def eps(self) -> float:
        return float(self.schedule[self.idx])

    def update(self, min_weight: float) -> bool:
        if min_weight > self.threshold and self.idx < len(self.schedule) - 1:
            self.idx += 1
            return True
        return False


# --------------------------------------------------------------------------------------
# 5.3 adaptive weighting
# --------------------------------------------------------------------------------------
def grad_norm_weights(losses_fn: Callable, params, *args, reference: Optional[str] = None, eps: float = 1e-12) -> Dict[str, Array]:
    """5.3(a)  lambda_i = ||grad L_ref|| / ||grad L_i||.

    ``reference=None`` uses the mean gradient norm over all terms as the numerator (JAX-PI);
    ``reference="r_u"`` reproduces the handbook formula literally with L_res as the reference.
    """
    # One backward pass per term (instead of jacrev over the dict, which batches all cotangents and
    # multiplies the peak memory of the residual graph by the number of terms). Runs every ~1000 steps.
    keys = list(jax.eval_shape(lambda p: losses_fn(p, *args), params).keys())
    norms = {}
    for k in keys:
        g = jax.grad(lambda p, k=k: losses_fn(p, *args)[k])(params)
        norms[k] = jnp.linalg.norm(ravel_pytree(g)[0])
    num = norms[reference] if reference is not None else jnp.mean(jnp.stack(list(norms.values())))
    return {k: num / (n + eps) for k, n in norms.items()}


def ntk_diag(scalar_fn: Callable, params, *batched_args, chunk: int = 256) -> Array:
    """Diagonal of the NTK of a scalar-output function over a batch: K_ii = ||d f(x_i)/d theta||^2.

    Evaluated in chunks of ``chunk`` points (a full-batch ``vmap`` would hold batch x n_params floats).
    """

    def one(*a):
        g = jax.grad(scalar_fn)(params, *a)
        g, _ = ravel_pytree(g)
        return jnp.dot(g, g)

    n = batched_args[0].shape[0]
    if n <= chunk or n % chunk != 0:
        return jax.vmap(one)(*batched_args)
    chunks = tuple(a.reshape(n // chunk, chunk, *a.shape[1:]) for a in batched_args)
    return jax.lax.map(lambda c: jax.vmap(one)(*c), chunks).reshape(n)


def ntk_weights(ntk_diags: Dict[str, Array], eps: float = 1e-12) -> Dict[str, Array]:
    """5.3(b)  lambda_i = tr(K) / tr(K_i), with traces replaced by per-point means so batch sizes cancel."""
    means = {k: jnp.mean(v) for k, v in ntk_diags.items()}
    total = jnp.mean(jnp.stack(list(means.values())))
    return {k: total / (m + eps) for k, m in means.items()}


def ema_update(old: Dict[str, Array], new: Dict[str, Array], momentum: float) -> Dict[str, Array]:
    """lambda <- (1 - alpha) lambda + alpha lambda_hat, written as an EMA with momentum = 1 - alpha... in JAX-PI's
    convention ``momentum`` is the weight on the *old* value (0.9 keeps 90% of the old weight)."""
    return {k: lax.stop_gradient(momentum * old[k] + (1 - momentum) * new[k]) for k in old}


# 5.3(c) self-adaptive point-wise weights (SA-PINN)
def sa_mask(w: Array, kind: str = "softplus") -> Array:
    return jax.nn.softplus(w) if kind == "softplus" else jax.nn.sigmoid(w)


def sa_weighted_loss(w: Array, r_sq: Array, kind: str = "softplus") -> Array:
    """sum_i m(w_i) r_i^2 / N ; minimise over theta, *maximise* over w (use a negative-lr optimiser for w)."""
    return jnp.mean(sa_mask(w, kind) * r_sq)


# --------------------------------------------------------------------------------------
# 5.5 residual-based adaptive distribution (RAD)
# --------------------------------------------------------------------------------------
def rad_resample(
    key,
    residual_magnitude_fn: Callable[[Array], Array],
    uniform_sampler: Callable[[jax.Array, int], Array],
    n_points: int,
    n_candidates: int = 100_000,
    k: float = 1.0,
    c: float = 1.0,
    uniform_frac: float = 0.2,
) -> Array:
    """Draw ``n_points`` collocation points with p(x) ∝ eps(x)^k / E[eps^k] + c (Wu et al. 2023).

    ``residual_magnitude_fn`` maps candidates (Nc, d) -> |residual| (Nc,); ``uniform_sampler(key, n)``
    draws uniform points in the domain. ``uniform_frac`` of the batch stays uniform to avoid
    collapsing onto a single hot spot.
    """
    k1, k2, k3 = random.split(key, 3)
    cands = uniform_sampler(k1, n_candidates)
    eps = jnp.abs(residual_magnitude_fn(cands))
    p = eps**k / (jnp.mean(eps**k) + 1e-12) + c
    p = p / p.sum()
    n_adapt = int(round(n_points * (1.0 - uniform_frac)))
    idx = random.choice(k2, n_candidates, shape=(n_adapt,), replace=False, p=p)
    uni = uniform_sampler(k3, n_points - n_adapt)
    return jnp.concatenate([cands[idx], uni], axis=0)


# --------------------------------------------------------------------------------------
# 5.6 pressure anchoring
# --------------------------------------------------------------------------------------
def pressure_anchor_closed(p_values: Array) -> Array:
    """Closed domain: L_p = ( (1/|Omega|) sum_i p(x_i) )^2  (Monte-Carlo zero mean)."""
    return jnp.mean(p_values) ** 2


def align_pressure_gauge(p_pred: Array, p_ref: Array) -> Array:
    """Shift the predicted pressure so its mean matches the reference before computing errors."""
    return p_pred - jnp.mean(p_pred) + jnp.mean(p_ref)
