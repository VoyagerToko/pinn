"""STEP 1 - the physics, written down exactly.

Non-dimensional incompressible Navier-Stokes (Re = U L / nu):

    momentum    u_t + (u . grad) u = -grad p + (1/Re) lap u
    continuity  div u = 0

Every derivative below is an automatic-differentiation call on the network. Nothing is
discretised.

Two families of helpers:

1. **Point-wise** residuals for networks ``fn(z) -> outputs`` with ``z`` a single point
   ``(t, x, y[, z])`` (or ``(x, y[, z])`` for steady problems). Derivatives use nested
   forward-mode ``jax.jacfwd`` (cheap for a handful of inputs) and are meant to be ``vmap``-ed.
   Formulations (handbook 1.2): VP (velocity-pressure), stream function (2D, exactly
   divergence-free), vector potential (3D, exactly divergence-free).

2. **Grid** residuals for separable networks (SPINN) that return whole tensors on a
   Cartesian grid. Derivatives use ``jax.jvp`` along each 1-D coordinate axis with an
   all-ones tangent - the forward-mode trick that makes 3D+time feasible.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import jvp

Array = jnp.ndarray

# Levi-Civita symbol for curls in 3D
_EPS = jnp.zeros((3, 3, 3))
_EPS = _EPS.at[0, 1, 2].set(1).at[1, 2, 0].set(1).at[2, 0, 1].set(1)
_EPS = _EPS.at[0, 2, 1].set(-1).at[2, 1, 0].set(-1).at[1, 0, 2].set(-1)


def _indices(unsteady: bool, dim: int):
    """Return (time index or None, tuple of spatial indices) into the input vector."""
    if unsteady:
        return 0, tuple(range(1, dim + 1))
    return None, tuple(range(dim))


def derivatives(fn: Callable, z: Array, order: int):
    """Return ``(fn(z), J, H, T)[:order+1]`` - value, Jacobian, Hessian, third derivative tensor.

    Shapes: J[o, i], H[o, i, j], T[o, i, j, k] for output ``o`` and input coordinates ``i, j, k``.
    Forward-over-forward mode throughout.
    """
    outs = [fn(z)]
    if order >= 1:
        outs.append(jax.jacfwd(fn)(z))
    if order >= 2:
        outs.append(jax.jacfwd(jax.jacfwd(fn))(z))
    if order >= 3:
        outs.append(jax.jacfwd(jax.jacfwd(jax.jacfwd(fn)))(z))
    return tuple(outs)


# --------------------------------------------------------------------------------------
# 1.2 VP formulation
# --------------------------------------------------------------------------------------
def ns_vp_residual(fn: Callable, Re: float, dim: int = 2, unsteady: bool = True) -> Callable:
    """Velocity-pressure residual for ``fn(z) -> (u_1..u_dim, p)``.

    Returns ``r(z) -> (r_mom (dim,), r_c ())``. With ``dim=2``: r_u, r_v, r_c (handbook 1.1).
    """
    it, sp = _indices(unsteady, dim)
    spi = jnp.array(sp)

    def r(z):
        out, J, H = derivatives(fn, z, 2)
        u = out[:dim]
        grad_u = J[:dim][:, spi]  # (dim, dim)   grad_u[i, j] = d u_i / d x_j
        grad_p = J[dim][spi]  # (dim,)
        lap_u = jnp.stack([sum(H[i, s, s] for s in sp) for i in range(dim)])
        u_t = J[:dim, it] if unsteady else jnp.zeros(dim)
        conv = grad_u @ u  # (u . grad) u_i = sum_j u_j d u_i / d x_j
        r_mom = u_t + conv + grad_p - lap_u / Re
        r_c = jnp.trace(grad_u)
        return r_mom, r_c

    return r


# --------------------------------------------------------------------------------------
# 1.2 stream function formulation (2D): psi, p with u = psi_y, v = -psi_x
# --------------------------------------------------------------------------------------
def ns_streamfunction_residual(fn: Callable, Re: float, unsteady: bool = True) -> Callable:
    """Residual for ``fn(z) -> (psi, p)``; divergence is identically zero (returned as a check).

    Third-order derivatives of psi are needed (handbook 1.2, 4.6a).
    Returns ``r(z) -> (r_mom (2,), r_c ())`` where ``r_c`` is the (machine-zero) divergence.
    """
    it, (ix, iy) = _indices(unsteady, 2)

    def r(z):
        out, J, H, T = derivatives(fn, z, 3)
        psi_J, psi_H, psi_T = J[0], H[0], T[0]
        u, v = psi_J[iy], -psi_J[ix]
        u_x, u_y = psi_H[iy, ix], psi_H[iy, iy]
        v_x, v_y = -psi_H[ix, ix], -psi_H[ix, iy]
        u_xx, u_yy = psi_T[iy, ix, ix], psi_T[iy, iy, iy]
        v_xx, v_yy = -psi_T[ix, ix, ix], -psi_T[ix, iy, iy]
        u_t = psi_H[iy, it] if unsteady else 0.0
        v_t = -psi_H[ix, it] if unsteady else 0.0
        p_x, p_y = J[1, ix], J[1, iy]
        r_u = u_t + u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
        r_v = v_t + u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
        r_c = u_x + v_y
        return jnp.stack([r_u, r_v]), r_c

    return r


def streamfunction_velocity(fn: Callable, unsteady: bool = True) -> Callable:
    """``fn(z) -> (psi, p)``  =>  ``vel(z) -> (u, v, p)``."""
    it, (ix, iy) = _indices(unsteady, 2)

    def vel(z):
        out, J = derivatives(fn, z, 1)
        return jnp.stack([J[0, iy], -J[0, ix], out[1]])

    return vel


# --------------------------------------------------------------------------------------
# 1.2 vector potential formulation (3D): A, p with u = curl A
# --------------------------------------------------------------------------------------
def vector_potential_velocity(fn: Callable, unsteady: bool = True) -> Callable:
    """``fn(z) -> (A1, A2, A3, p)``  =>  ``vel(z) -> (u, v, w, p)`` with ``u = curl A``."""
    it, sp = _indices(unsteady, 3)
    sp = jnp.array(sp)

    def vel(z):
        out, J = derivatives(fn, z, 1)
        JA = J[:3][:, sp]  # JA[k, j] = d A_k / d x_j
        u = jnp.einsum("ijk,kj->i", _EPS, JA)
        return jnp.concatenate([u, out[3:4]])

    return vel


def ns_vector_potential_residual(fn: Callable, Re: float, unsteady: bool = True) -> Callable:
    """Residual for ``fn(z) -> (A1, A2, A3, p)`` with ``u = curl A`` (handbook 4.6a, 3D).

    Returns ``r(z) -> (r_mom (3,), r_c (), gauge ())`` where ``r_c = div u`` (identically 0)
    and ``gauge = div A`` for the weak gauge penalty ``lambda_g ||div A||^2``.
    """
    it, sp = _indices(unsteady, 3)
    spi = jnp.array(sp)

    def r(z):
        out, J, H, T = derivatives(fn, z, 3)
        JA = J[:3][:, spi]  # (3, 3)      d A_k / d x_j
        HA = H[:3][:, spi][:, :, spi]  # (3, 3, 3)   d^2 A_k / d x_j d x_l
        TA = T[:3][:, spi][:, :, spi][:, :, :, spi]  # (3,3,3,3) d^3 A_k / dx_j dx_l dx_m
        u = jnp.einsum("ijk,kj->i", _EPS, JA)
        grad_u = jnp.einsum("ijk,kjl->il", _EPS, HA)  # d u_i / d x_l
        lap_u = jnp.einsum("ijk,kjll->i", _EPS, TA)
        if unsteady:
            HA_t = H[:3][:, spi, it]  # d^2 A_k / d x_j d t
            u_t = jnp.einsum("ijk,kj->i", _EPS, HA_t)
        else:
            u_t = jnp.zeros(3)
        grad_p = J[3][spi]
        r_mom = u_t + grad_u @ u + grad_p - lap_u / Re
        r_c = jnp.trace(grad_u)
        gauge = jnp.trace(JA)
        return r_mom, r_c, gauge

    return r


# --------------------------------------------------------------------------------------
# derived point-wise quantities (STEP 7 / STEP 9)
# --------------------------------------------------------------------------------------
def velocity_gradient(fn: Callable, dim: int, unsteady: bool = True) -> Callable:
    """``fn(z) -> (u..., p)``  =>  ``g(z) -> (u (dim,), grad_u (dim, dim))`` by AD."""
    it, sp = _indices(unsteady, dim)
    spi = jnp.array(sp)

    def g(z):
        out, J = derivatives(fn, z, 1)
        return out[:dim], J[:dim][:, spi]

    return g


def vorticity_from_grad(grad_u: Array) -> Array:
    """2D: scalar ``v_x - u_y``. 3D: vector ``curl u``."""
    if grad_u.shape[0] == 2:
        return grad_u[1, 0] - grad_u[0, 1]
    return jnp.einsum("ijk,kj->i", _EPS, grad_u)


def q_criterion(grad_u: Array) -> Array:
    """9.2  Q = 1/2 (||Omega||_F^2 - ||S||_F^2), S/Omega the symmetric/antisymmetric parts of grad u."""
    S = 0.5 * (grad_u + grad_u.T)
    O = 0.5 * (grad_u - grad_u.T)
    return 0.5 * (jnp.sum(O * O) - jnp.sum(S * S))


def lambda2_criterion(grad_u: Array) -> Array:
    """9.2  second-largest eigenvalue of S^2 + Omega^2 (vortex core where lambda_2 < 0)."""
    S = 0.5 * (grad_u + grad_u.T)
    O = 0.5 * (grad_u - grad_u.T)
    eig = jnp.linalg.eigvalsh(S @ S + O @ O)  # ascending
    return eig[1]


# --------------------------------------------------------------------------------------
# forward-mode helpers on separable grids (SPINN)
# --------------------------------------------------------------------------------------
def hvp_fwdfwd(f: Callable, primals: Tuple[Array], tangents: Tuple[Array], return_primals: bool = False):
    """Forward-over-forward: returns ``(f'(x) . v, f''(x)[v, v])`` (SPINN repo convention)."""
    g = lambda p: jvp(f, (p,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


def grid_derivatives(f: Callable, coords: Sequence[Array], second: Sequence[int] = ()):
    """First (and optionally second) partial derivatives of ``f(*coords)`` along every axis.

    ``f`` maps ``d`` 1-D coordinate arrays to a tensor whose grid axes follow the coordinate
    order (leading axes may be output components). Because each grid value depends on exactly
    one entry of each 1-D array, a ``jvp`` with an all-ones tangent along ``coords[i]`` is the
    exact partial derivative with respect to ``x_i`` at every grid point.

    Returns ``(value, first, second_dict)`` with ``first[i]`` = d f / d x_i and
    ``second_dict[i]`` = d^2 f / d x_i^2 for ``i in second``.
    """
    coords = list(coords)
    value = f(*coords)
    first, sec = [], {}
    for i, c in enumerate(coords):
        ones = jnp.ones_like(c)

        def fi(ci, i=i):
            cc = list(coords)
            cc[i] = ci
            return f(*cc)

        if i in second:
            d1, d2 = hvp_fwdfwd(fi, (c,), (ones,), return_primals=True)
            sec[i] = d2
        else:
            d1 = jvp(fi, (c,), (ones,))[1]
        first.append(d1)
    return value, first, sec


def ns_vp_residual_grid(f: Callable, coords: Sequence[Array], Re: float, unsteady: bool = True):
    """VP residual on a separable grid. ``f(*coords) -> (dim+1, n_1, ..., n_d)`` = (u..., p).

    ``coords`` = ``(t, x, y[, z])`` or ``(x, y[, z])``. Returns ``(r_mom (dim, grid...), r_c (grid...))``.
    """
    d_in = len(coords)
    dim = d_in - 1 if unsteady else d_in
    it, sp = _indices(unsteady, dim)
    value, first, sec = grid_derivatives(f, coords, second=sp)
    u = value[:dim]
    u_t = first[it][:dim] if unsteady else 0.0
    conv = sum(u[j] * first[s][:dim] for j, s in enumerate(sp))
    grad_p = jnp.stack([first[s][dim] for s in sp])
    lap = sum(sec[s][:dim] for s in sp)
    r_mom = u_t + conv + grad_p - lap / Re
    r_c = sum(first[s][j] for j, s in enumerate(sp))
    return r_mom, r_c


def curl_grid(fA: Callable, coords: Sequence[Array], unsteady: bool = True) -> Array:
    """``u = curl A`` on a separable grid; ``fA(*coords) -> (>=3, grid...)`` (first 3 = A)."""
    dim = 3
    it, sp = _indices(unsteady, dim)
    _, first, _ = grid_derivatives(lambda *c: fA(*c)[:3], coords)
    dA = jnp.stack([first[s] for s in sp], axis=1)  # dA[k, j] = d A_k / d x_j
    return jnp.einsum("ijk,kj...->i...", _EPS, dA)


def ns_vector_potential_residual_grid(f: Callable, coords: Sequence[Array], Re: float, unsteady: bool = True):
    """Vector-potential VP residual on a separable grid. ``f(*coords) -> (4, grid...)`` = (A1, A2, A3, p).

    Returns ``(r_mom (3, grid...), r_c (grid...), gauge (grid...))``.
    """

    def vel_p(*c):
        u = curl_grid(f, c, unsteady)
        return jnp.concatenate([u, f(*c)[3:4]], axis=0)

    r_mom, r_c = ns_vp_residual_grid(vel_p, coords, Re, unsteady)
    it, sp = _indices(unsteady, 3)
    _, first, _ = grid_derivatives(lambda *c: f(*c)[:3], coords)
    gauge = sum(first[s][k] for k, s in enumerate(sp))
    return r_mom, r_c, gauge
