"""STEP 1 - the physics, written down exactly.

Non-dimensional incompressible Navier-Stokes (Re = U L / nu):

    momentum    u_t + (u . grad) u = -grad p + (1/Re) lap u
    continuity  div u = 0

Every derivative below is an automatic-differentiation call on the network. Nothing is
discretised.

Two families of helpers:

1. **Point-wise** residuals for networks ``fn(z) -> outputs`` with ``z`` a single point
   ``(t, x, y[, z])`` (or ``(x, y[, z])`` for steady problems). Derivatives are *directional*
   forward-mode ``jvp`` calls along unit coordinate vectors (forward-over-forward for second
   order), so only the derivatives the residual needs are ever formed; meant to be ``vmap``-ed.
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


def _unit(d: int, i: int) -> Array:
    return jnp.zeros(d).at[i].set(1.0)


def directional(fn: Callable, z: Array, i: int, second: bool = False):
    """Directional derivative(s) of ``fn`` along coordinate ``i`` by forward mode.

    ``second=False`` -> ``d fn / d z_i``;  ``second=True`` -> ``(d fn / d z_i, d^2 fn / d z_i^2)``.
    Forward-over-forward with a unit tangent: two nested ``jvp`` calls, no Jacobian tensors. This
    is what keeps the memory of a 6 x 256 network with third-order derivatives in the low GB range
    (the full ``jacfwd`` tensors need > 60 GB for the same batch).
    """
    e = _unit(z.shape[0], i)
    if second:
        return hvp_fwdfwd(fn, (z,), (e,), return_primals=True)
    return jvp(fn, (z,), (e,))[1]


# --------------------------------------------------------------------------------------
# 1.2 VP formulation
# --------------------------------------------------------------------------------------
def ns_vp_residual(fn: Callable, Re, dim: int = 2, unsteady: bool = True, conv_coeff=1.0, visc=None) -> Callable:
    """Velocity-pressure residual for ``fn(z) -> (u_1..u_dim, p)``.

        r_mom = u_t + conv_coeff (u . grad) u + grad p - visc lap u,      visc = 1/Re by default
        r_c   = div u

    Returns ``r(z) -> (r_mom (dim,), r_c ())``. ``conv_coeff``/``visc`` may be traced scalars
    (trainable lambda_1, lambda_2 in the inverse problem). Cost: one first-order and ``dim``
    second-order directional derivatives, all forward mode.
    """
    it, sp = _indices(unsteady, dim)
    nu = (1.0 / Re) if visc is None else visc

    def r(z):
        out = fn(z)
        u = out[:dim]
        u_t = directional(fn, z, it)[:dim] if unsteady else jnp.zeros(dim)
        conv = jnp.zeros(dim)
        lap = jnp.zeros(dim)
        grad_p = []
        div = 0.0
        for j, s in enumerate(sp):
            f_s, f_ss = directional(fn, z, s, second=True)
            conv = conv + u[j] * f_s[:dim]
            lap = lap + f_ss[:dim]
            grad_p.append(f_s[dim])
            div = div + f_s[j]
        r_mom = u_t + conv_coeff * conv + jnp.stack(grad_p) - nu * lap
        return r_mom, div

    return r


# --------------------------------------------------------------------------------------
# 1.2 stream function formulation (2D): psi, p with u = psi_y, v = -psi_x
# --------------------------------------------------------------------------------------
def streamfunction_velocity(fn: Callable, unsteady: bool = True) -> Callable:
    """``fn(z) -> (psi, p)``  =>  ``vel(z) -> (u, v, p)`` with ``u = psi_y, v = -psi_x``."""
    it, (ix, iy) = _indices(unsteady, 2)

    def vel(z):
        out = fn(z)
        psi_x = directional(fn, z, ix)[0]
        psi_y = directional(fn, z, iy)[0]
        return jnp.stack([psi_y, -psi_x, out[1]])

    return vel


def ns_streamfunction_residual(fn: Callable, Re: float, unsteady: bool = True) -> Callable:
    """Residual for ``fn(z) -> (psi, p)``; the divergence is identically zero (returned as a check).

    Third-order derivatives of psi enter through second derivatives of the derived velocity
    (handbook 1.2, 4.6a). Returns ``r(z) -> (r_mom (2,), r_c ())``.
    """
    return ns_vp_residual(streamfunction_velocity(fn, unsteady), Re, dim=2, unsteady=unsteady)


# --------------------------------------------------------------------------------------
# 1.2 vector potential formulation (3D): A, p with u = curl A
# --------------------------------------------------------------------------------------
def vector_potential_velocity(fn: Callable, unsteady: bool = True) -> Callable:
    """``fn(z) -> (A1, A2, A3, p)``  =>  ``vel(z) -> (u, v, w, p)`` with ``u = curl A``."""
    it, sp = _indices(unsteady, 3)

    def vel(z):
        out = fn(z)
        dA = jnp.stack([directional(fn, z, s)[:3] for s in sp], axis=1)  # dA[k, j] = d A_k / d x_j
        u = jnp.einsum("ijk,kj->i", _EPS, dA)
        return jnp.concatenate([u, out[3:4]])

    return vel


def ns_vector_potential_residual(fn: Callable, Re: float, unsteady: bool = True) -> Callable:
    """Residual for ``fn(z) -> (A1, A2, A3, p)`` with ``u = curl A`` (handbook 4.6a, 3D).

    Returns ``r(z) -> (r_mom (3,), r_c (), gauge ())`` where ``r_c = div u`` (identically 0)
    and ``gauge = div A`` for the weak gauge penalty ``lambda_g ||div A||^2``.
    """
    it, sp = _indices(unsteady, 3)
    r_vp = ns_vp_residual(vector_potential_velocity(fn, unsteady), Re, dim=3, unsteady=unsteady)

    def r(z):
        r_mom, r_c = r_vp(z)
        gauge = sum(directional(fn, z, s)[k] for k, s in enumerate(sp))
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
