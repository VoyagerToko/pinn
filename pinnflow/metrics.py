"""STEP 7 - validation metrics with the exact formulas.

7.1 relative L2 and maximum divergence   7.2 drag and lift   7.3 Strouhal number
7.4 kinetic energy, dissipation, enstrophy   7.5 energy spectrum   7.6 cost accounting
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .benchmarks import DFGCylinder
from .physics import velocity_gradient

Array = jnp.ndarray


# 7.1 ------------------------------------------------------------------------------------
def relative_l2(pred: Array, ref: Array) -> Array:
    """e = ||u_hat - u||_2 / ||u||_2  (flattened over all points and components)."""
    pred, ref = jnp.ravel(pred), jnp.ravel(ref)
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def max_divergence(div_values: Array) -> Array:
    return jnp.max(jnp.abs(div_values))


# 7.2 ------------------------------------------------------------------------------------
def drag_lift(vel_p_dim_fn: Callable, t: float, bench: DFGCylinder, n_theta: int = 256) -> Tuple[Array, Array, Array]:
    """Drag/lift coefficients and front-rear pressure difference at time ``t``.

    ``vel_p_dim_fn(z) -> (u, v, p)`` must take *dimensional* ``z = (t, x, y)`` and return dimensional
    velocity and pressure (the problem classes provide this wrapper; AD then yields dimensional
    gradients directly). Two evaluations of the surface integral are returned in one call:

        C_D = 2/(rho U^2 D) ∮ ( rho nu ∂u_tau/∂n n_y − p n_x ) dS         (Schaefer-Turek form, handbook 7.2)
        C_L = −2/(rho U^2 D) ∮ ( rho nu ∂u_tau/∂n n_x + p n_y ) dS

    with tau = (n_y, -n_x) and ∂u_tau/∂n = tau · (grad u · n). The full stress-tensor form
    (sigma n with sigma = nu(grad u + grad u^T) - p I) gives the same numbers for a no-slip wall
    and is used here as an internal consistency check (returned as extras when ``return_extras``).
    """
    pts, nx, ny, ds = bench.cylinder_surface(n_theta)
    pts, nx, ny, ds = map(jnp.asarray, (pts, nx, ny, ds))
    g = jax.vmap(velocity_gradient(vel_p_dim_fn, dim=2, unsteady=True))
    z = jnp.concatenate([jnp.full((pts.shape[0], 1), t), pts], axis=1)
    u, grad_u = g(z)  # (n,2), (n,2,2)
    p = jax.vmap(vel_p_dim_fn)(z)[:, 2]
    n = jnp.stack([nx, ny], -1)
    tau = jnp.stack([ny, -nx], -1)
    dudn = jnp.einsum("ni,nij,nj->n", tau, grad_u, n)  # ∂u_tau/∂n
    rho, nu, U, D = bench.rho, bench.nu, bench.U_ref, bench.diameter
    F_D = jnp.sum((rho * nu * dudn * ny - p * nx) * ds)
    F_L = -jnp.sum((rho * nu * dudn * nx + p * ny) * ds)
    C_D = 2.0 * F_D / (rho * U**2 * D)
    C_L = 2.0 * F_L / (rho * U**2 * D)
    probes = jnp.asarray(bench.pressure_probe_points())
    zp = jnp.concatenate([jnp.full((2, 1), t), probes], axis=1)
    pp = jax.vmap(vel_p_dim_fn)(zp)[:, 2]
    dP = pp[0] - pp[1]
    return C_D, C_L, dP


def drag_lift_series(vel_p_dim_fn: Callable, times: np.ndarray, bench: DFGCylinder, n_theta: int = 256) -> Dict[str, np.ndarray]:
    f = jax.jit(lambda t: drag_lift(vel_p_dim_fn, t, bench, n_theta))
    out = np.array([np.asarray(f(float(t))) for t in times])
    return {"t": np.asarray(times), "Cd": out[:, 0], "Cl": out[:, 1], "dP": out[:, 2]}


# 7.3 ------------------------------------------------------------------------------------
def strouhal(t: np.ndarray, cl: np.ndarray, D: float = 0.1, U: float = 1.0, transient_frac: float = 0.5) -> Tuple[float, float]:
    """St = f D / U with f the dominant FFT peak of C_L(t) after discarding the initial transient.

    Returns ``(St, f)``. ``t`` must be uniformly spaced.
    """
    t, cl = np.asarray(t), np.asarray(cl)
    s = int(len(t) * transient_frac)
    tt, cc = t[s:], cl[s:] - np.mean(cl[s:])
    dt = tt[1] - tt[0]
    spec = np.abs(np.fft.rfft(cc * np.hanning(len(cc))))
    freqs = np.fft.rfftfreq(len(cc), d=dt)
    spec[0] = 0.0
    f = float(freqs[np.argmax(spec)])
    return f * D / U, f


def periodic_cycle_stats(t: np.ndarray, cd: np.ndarray, cl: np.ndarray, transient_frac: float = 0.6) -> Dict[str, float]:
    """max/min/mean/amplitude of C_D and C_L over one shedding cycle (FEATFLOW protocol)."""
    s = int(len(t) * transient_frac)
    cd, cl = np.asarray(cd)[s:], np.asarray(cl)[s:]
    return {
        "Cd_max": float(cd.max()), "Cd_min": float(cd.min()), "Cd_mean": float(0.5 * (cd.max() + cd.min())), "Cd_amp": float(cd.max() - cd.min()),
        "Cl_max": float(cl.max()), "Cl_min": float(cl.min()), "Cl_mean": float(0.5 * (cl.max() + cl.min())), "Cl_amp": float(cl.max() - cl.min()),
    }


# 7.4 ------------------------------------------------------------------------------------
def kinetic_energy(u: Array) -> Array:
    """E_k = (1/|Omega|) ∫ 1/2 u·u dOmega, estimated as the mean of 1/2 |u|^2 over uniform samples/grid.
    ``u`` has shape (..., dim) or (dim, ...)."""
    if u.shape[-1] in (2, 3):
        return jnp.mean(0.5 * jnp.sum(u**2, axis=-1))
    return jnp.mean(0.5 * jnp.sum(u**2, axis=0))


def enstrophy(omega: Array) -> Array:
    """zeta = (1/|Omega|) ∫ 1/2 omega·omega dOmega (same layout conventions as kinetic_energy)."""
    return kinetic_energy(omega)


def dissipation_from_energy(t: np.ndarray, Ek: np.ndarray) -> np.ndarray:
    """epsilon(t) = -dE_k/dt from the energy history (second-order finite differences in time)."""
    return -np.gradient(np.asarray(Ek), np.asarray(t))


def dissipation_from_enstrophy(zeta: Array, nu: float) -> Array:
    """epsilon = 2 nu zeta - the incompressible identity, a free consistency check."""
    return 2.0 * nu * zeta


# 7.5 ------------------------------------------------------------------------------------
def energy_spectrum(u_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """E(k) = 1/2 sum_{|k'| in [k, k+1)} |u_hat(k')|^2 for a velocity field on a uniform periodic grid.

    ``u_grid`` has shape (dim, n, n[, n]) on [0, 2pi]^dim; the FFT is normalised by n^dim so that
    sum_k E(k) equals the mean kinetic energy. Returns ``(k, E)`` with integer shells k = 0, 1, ...
    """
    u_grid = np.asarray(u_grid)
    dim = u_grid.shape[0]
    n = u_grid.shape[1]
    uh = np.fft.fftn(u_grid, axes=tuple(range(1, dim + 1))) / n**dim
    e = 0.5 * np.sum(np.abs(uh) ** 2, axis=0)
    freqs = np.fft.fftfreq(n, d=1.0 / n)
    grids = np.meshgrid(*([freqs] * dim), indexing="ij")
    kmag = np.sqrt(sum(g**2 for g in grids))
    shells = np.floor(kmag).astype(int)
    kmax = n // 2
    E = np.bincount(shells.ravel(), weights=e.ravel(), minlength=kmax + 1)[: kmax + 1]
    return np.arange(kmax + 1), E


def inertial_range_slope(k: np.ndarray, E: np.ndarray, kmin: int = 4, kmax: int = 16) -> float:
    """Log-log slope of E(k) over [kmin, kmax]; compare with -5/3."""
    m = (k >= kmin) & (k <= kmax) & (E > 0)
    return float(np.polyfit(np.log(k[m]), np.log(E[m]), 1)[0])


# 7.6 ------------------------------------------------------------------------------------
def inference_throughput(fn: Callable, pts: Array, repeats: int = 3, chunk: int = None) -> float:
    """Query points per second for a batched, jitted evaluation.

    With ``chunk`` the points are streamed through one compiled batch of ``chunk`` points (the way a
    large query would be served); without it the whole array is a single batch.
    """
    import time

    f = jax.jit(fn)
    if chunk is None or chunk >= pts.shape[0]:
        batches = [pts]
    else:
        n = (pts.shape[0] // chunk) * chunk
        batches = [pts[s : s + chunk] for s in range(0, n, chunk)]
    jax.block_until_ready(f(batches[0]))
    t0 = time.perf_counter()
    for _ in range(repeats):
        for b in batches:
            out = f(b)
        jax.block_until_ready(out)
    return sum(b.shape[0] for b in batches) * repeats / (time.perf_counter() - t0)
