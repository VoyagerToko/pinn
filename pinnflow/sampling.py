"""Collocation, boundary and initial-condition samplers (STEP 5.2 budgets live in configs)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .benchmarks import DFGCylinder

Array = jnp.ndarray


def uniform_box(key, dom: Array, n: int) -> Array:
    """Uniform points in the box ``dom`` (d, 2) -> (n, d)."""
    dom = jnp.asarray(dom)
    return random.uniform(key, (n, dom.shape[0]), minval=dom[:, 0], maxval=dom[:, 1])


def uniform_time(key, n: int, t0: float, t1: float) -> Array:
    return random.uniform(key, (n, 1), minval=t0, maxval=t1)


def square_boundary(key, n_per_side: int, lo: float = 0.0, hi: float = 1.0) -> Dict[str, Array]:
    """Random points on the four sides of a square; keys: bottom (y=lo), top (y=hi), left, right."""
    k = random.split(key, 4)
    s = [random.uniform(kk, (n_per_side,), minval=lo, maxval=hi) for kk in k]
    return {
        "bottom": jnp.stack([s[0], jnp.full_like(s[0], lo)], -1),
        "top": jnp.stack([s[1], jnp.full_like(s[1], hi)], -1),
        "left": jnp.stack([jnp.full_like(s[2], lo), s[2]], -1),
        "right": jnp.stack([jnp.full_like(s[3], hi), s[3]], -1),
    }


def channel_boundaries(key, bench: DFGCylinder, n: int, t0: float, t1: float) -> Dict[str, Array]:
    """Boundary samples (t, x, y) in *dimensional* units for Benchmark C.

    keys: inlet (x=0), outlet (x=L), walls (y=0 and y=H), cylinder (surface).
    """
    k = random.split(key, 8)
    H, L = bench.height, bench.length
    t = lambda kk: random.uniform(kk, (n,), minval=t0, maxval=t1)
    y_in = random.uniform(k[0], (n,), minval=0.0, maxval=H)
    y_out = random.uniform(k[1], (n,), minval=0.0, maxval=H)
    x_w = random.uniform(k[2], (n,), minval=0.0, maxval=L)
    wall_y = jnp.where(random.bernoulli(k[3], 0.5, (n,)), 0.0, H)
    theta = random.uniform(k[4], (n,), minval=0.0, maxval=2 * jnp.pi)
    cx, cy, r = bench.center[0], bench.center[1], bench.radius
    return {
        "inlet": jnp.stack([t(k[5]), jnp.zeros(n), y_in], -1),
        "outlet": jnp.stack([t(k[6]), jnp.full(n, L), y_out], -1),
        "walls": jnp.stack([t(k[7]), x_w, wall_y], -1),
        "cylinder": jnp.stack([t(k[0]), cx + r * jnp.cos(theta), cy + r * jnp.sin(theta)], -1),
    }


def channel_interior(key, bench: DFGCylinder, n: int, t0: float, t1: float, near_frac: float = 0.25, near_radius: float = 4.0) -> Array:
    """Interior (t, x, y) points outside the cylinder. A fraction is concentrated in an annulus of
    ``near_radius * r`` around the cylinder where the boundary layer and near wake live."""
    k1, k2, k3, k4 = random.split(key, 4)
    n_near = int(n * near_frac)
    n_far = n - n_near
    # far: rejection sampling with a static shape (oversample 2x, sort valid first)
    pts = uniform_box(k1, jnp.array([[t0, t1], [0.0, bench.length], [0.0, bench.height]]), 2 * n_far)
    valid = bench.in_fluid(pts[:, 1], pts[:, 2])
    order = jnp.argsort(~valid)  # valid points first
    far = pts[order[:n_far]]
    # near: annulus, clipped to the channel
    rho = bench.radius * jnp.sqrt(random.uniform(k2, (n_near,), minval=1.0, maxval=near_radius**2))
    th = random.uniform(k3, (n_near,), minval=0.0, maxval=2 * jnp.pi)
    x = jnp.clip(bench.center[0] + rho * jnp.cos(th), 0.0, bench.length)
    y = jnp.clip(bench.center[1] + rho * jnp.sin(th), 0.0, bench.height)
    near = jnp.stack([random.uniform(k4, (n_near,), minval=t0, maxval=t1), x, y], -1)
    return jnp.concatenate([far, near], axis=0)


def channel_initial(key, bench: DFGCylinder, n: int, t0: float = 0.0) -> Array:
    """(t0, x, y) points in the fluid for the initial-condition loss."""
    pts = channel_interior(key, bench, n, t0, t0, near_frac=0.2)
    return pts.at[:, 0].set(t0)


def spinn_axes(key, dom: Array, n_per_axis: Sequence[int], sort: bool = True) -> List[Array]:
    """Independent uniform 1-D samples per axis for separable (SPINN) collocation.

    ``dom`` (d, 2); returns ``[x_1 (n_1,), ..., x_d (n_d,)]``. Effective points = prod n_i.
    """
    dom = jnp.asarray(dom)
    keys = random.split(key, dom.shape[0])
    axes = [random.uniform(k, (int(n),), minval=dom[i, 0], maxval=dom[i, 1]) for i, (k, n) in enumerate(zip(keys, n_per_axis))]
    return [jnp.sort(a) for a in axes] if sort else axes


def linspace_axes(dom: Array, n_per_axis: Sequence[int], endpoint: bool = False) -> List[Array]:
    dom = np.asarray(dom)
    return [jnp.linspace(dom[i, 0], dom[i, 1], int(n), endpoint=endpoint) for i, n in enumerate(n_per_axis)]
