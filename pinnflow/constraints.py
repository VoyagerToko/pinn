"""STEP 4.6 - hard constraints.

(a) exactly divergence-free velocity: handled by the stream-function / vector-potential
    residuals in :mod:`pinnflow.physics`.
(b) exactly satisfied Dirichlet boundary conditions:

        u_hat(x) = g(x) + phi(x) * N_theta(x)

    ``g`` is any smooth extension of the boundary data, ``phi`` a smooth approximate distance
    function (zero on the Dirichlet boundary, positive inside). R-functions (Rvachev) combine
    distance functions of primitives: ``r_and(a, b)`` is zero where either is zero.

Also here: :func:`fold_rwf_params` (fold W = g*V back into a plain kernel for inference).
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp

Array = jnp.ndarray


# --------------------------------------------------------------------------------------
# R-functions
# --------------------------------------------------------------------------------------
def r_and(a: Array, b: Array) -> Array:
    """R-conjunction: >0 iff a>0 and b>0; 0 where either vanishes."""
    return a + b - jnp.sqrt(a * a + b * b)


def r_or(a: Array, b: Array) -> Array:
    """R-disjunction: >0 iff a>0 or b>0."""
    return a + b + jnp.sqrt(a * a + b * b)


def r_and_smooth(a: Array, b: Array, m: int = 2) -> Array:
    """Normalised R-conjunction of order m (Rvachev), smoother near corners than r_and."""
    return a + b - (a**m + b**m) ** (1.0 / m)


# --------------------------------------------------------------------------------------
# distance functions for the benchmark geometries
# --------------------------------------------------------------------------------------
def phi_unit_square(x: Array, y: Array) -> Array:
    """Benchmark B: zero on all four walls of [0,1]^2."""
    return x * (1 - x) * y * (1 - y)


def cylinder_signed_distance_sq(x: Array, y: Array, center=(0.2, 0.2), radius=0.05) -> Array:
    """(x-cx)^2 + (y-cy)^2 - r^2 : zero on the cylinder, positive outside."""
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 - radius**2


def phi_channel_cylinder(
    x: Array, y: Array, center=(0.2, 0.2), radius=0.05, height=0.41, include_inlet: bool = True
) -> Array:
    """Benchmark C distance function (handbook 4.6b):

        phi = y (0.41 - y) ((x-0.2)^2 + (y-0.2)^2 - 0.05^2)          [handbook form]

    With ``include_inlet=True`` an extra factor ``x`` makes phi vanish on the inlet too, so
    *all* Dirichlet boundaries (walls, cylinder, inflow) are enforced exactly and only the
    outflow (Neumann / p = 0) remains a soft loss.
    """
    phi = y * (height - y) * cylinder_signed_distance_sq(x, y, center, radius)
    if include_inlet:
        phi = phi * x
    return phi


def phi_channel_cylinder_bounded(x: Array, y: Array, center=(0.2, 0.2), radius=0.05, height=0.41, include_inlet: bool = True) -> Array:
    """Same zero set as :func:`phi_channel_cylinder` but every factor is saturated into [0, 1]:

        phi = tanh(x / 0.1) * 4 y (H - y) / H^2 * tanh(d_cyl / r^2)

    so the multiplier on the network output stays O(1) throughout the channel (the literal
    polynomial form grows to ~10^4 x r^4 near the outlet, which conditions the network badly).
    """
    d = cylinder_signed_distance_sq(x, y, center, radius)
    phi = 4.0 * y * (height - y) / height**2 * jnp.tanh(d / radius**2)
    if include_inlet:
        phi = phi * jnp.tanh(x / (2 * radius))
    return phi


def cylinder_inflow_extension(u_in_fn: Callable, center=(0.2, 0.2), radius=0.05) -> Callable:
    """Smooth ``g(t, x, y)`` for Benchmark C that equals the inflow profile at x=0, is zero on the
    walls (because ``u_in(y)`` is) and zero on the cylinder:

        g_u = u_in(t, y) * d_cyl / (d_cyl + x),   g_v = 0

    ``d_cyl`` is the squared signed distance to the cylinder, positive in the fluid.
    """

    def g(t, x, y):
        d = cylinder_signed_distance_sq(x, y, center, radius)
        gu = u_in_fn(t, y) * d / (d + x)
        return jnp.stack([gu, jnp.zeros_like(gu)])

    return g


def lid_extension(u_lid_fn: Callable) -> Callable:
    """Smooth ``g(x, y) = (u_lid(x) * y, 0)`` for the regularised cavity lid (zero at the corners)."""

    def g(x, y):
        return jnp.stack([u_lid_fn(x) * y, jnp.zeros_like(x)])

    return g


# --------------------------------------------------------------------------------------
# the constraint wrapper
# --------------------------------------------------------------------------------------
def hard_dirichlet(net_fn: Callable, g_fn: Callable, phi_fn: Callable, constrained: Tuple[int, ...] = (0, 1)) -> Callable:
    """Return ``u_hat(z) = g(z) + phi(z) * N(z)`` on the ``constrained`` output components.

    All callables take the same point vector ``z``. ``g_fn`` returns a vector with one entry per
    constrained component; other outputs (e.g. pressure) pass through unchanged so the
    boundary loss for velocity disappears from the objective entirely.
    """
    constrained = tuple(constrained)

    def u_hat(z):
        n = net_fn(z)
        phi = phi_fn(z)
        g = g_fn(z)
        out = n
        for k, c in enumerate(constrained):
            out = out.at[c].set(g[k] + phi * n[c])
        return out

    return u_hat


# --------------------------------------------------------------------------------------
# RWF fold-back (handbook 4.4: "costs nothing at inference")
# --------------------------------------------------------------------------------------
def fold_rwf_params(params):
    """Replace every factorised kernel ``(g, V)`` by the dense kernel ``g * V``.

    The result is loadable by the same architecture built with ``reparam=None``.
    """

    def _fold(tree):
        if isinstance(tree, dict) or hasattr(tree, "items"):
            out = {}
            for k, v in tree.items():
                if k == "kernel" and isinstance(v, (tuple, list)) and len(v) == 2:
                    g, V = v
                    out[k] = g * V
                else:
                    out[k] = _fold(v)
            return out
        return tree

    return _fold(params)
