import jax.numpy as jnp
import numpy as np

from pinnflow import metrics as M
from pinnflow.benchmarks import DFGCylinder


def test_relative_l2_and_max_div():
    a = jnp.array([1.0, 2.0, 2.0])
    assert float(M.relative_l2(a, a)) == 0.0
    np.testing.assert_allclose(float(M.relative_l2(jnp.zeros(3), a)), 1.0)
    assert float(M.max_divergence(jnp.array([-3.0, 2.0]))) == 3.0


def test_drag_from_linear_pressure_field():
    """u = 0, p = x  =>  F_D = -∮ x n_x ds = -pi r^2 (divergence theorem); F_L = 0."""
    b = DFGCylinder()
    vel = lambda z: jnp.stack([0.0 * z[1], 0.0 * z[1], z[1]])
    Cd, Cl, dP = M.drag_lift(vel, 0.0, b, n_theta=512)
    expected = -2.0 * np.pi * b.radius**2 / (b.rho * b.U_ref**2 * b.diameter)
    np.testing.assert_allclose(float(Cd), expected, rtol=1e-4)
    np.testing.assert_allclose(float(Cl), 0.0, atol=1e-6)  # float32 quadrature round-off
    np.testing.assert_allclose(float(dP), 0.15 - 0.25, atol=1e-6)


def test_drag_from_quadratic_shear_flow():
    """u = (y - 0.2)^2, v = 0, p = 0: only the viscous term contributes; F_L = 0 by symmetry.

    grad u = [[0, 2(y-0.2)], [0, 0]]; on the circle y - 0.2 = r sin(th):
    d u_tau/dn = tau . (grad u . n) = n_y * 2 r sin^2(th) = 2 r sin^3(th)
    F_D = rho nu ∮ (2 r sin^3 th) n_y ds = 2 rho nu r^2 ∫ sin^4 th dth = (3 pi / 2) rho nu r^2
    """
    b = DFGCylinder()
    vel = lambda z: jnp.stack([(z[2] - 0.2) ** 2, 0.0 * z[2], 0.0 * z[2]])
    Cd, Cl, _ = M.drag_lift(vel, 0.0, b, n_theta=1024)
    expected = 2.0 * (1.5 * np.pi * b.rho * b.nu * b.radius**2) / (b.rho * b.U_ref**2 * b.diameter)
    np.testing.assert_allclose(float(Cd), expected, rtol=1e-4)
    np.testing.assert_allclose(float(Cl), 0.0, atol=1e-8)


def test_strouhal_recovers_frequency():
    t = np.linspace(0, 20, 4001)
    f0 = 3.0
    cl = 0.3 + 1.0 * np.sin(2 * np.pi * f0 * t) * (1 - np.exp(-t))
    St, f = M.strouhal(t, cl, D=0.1, U=1.0, transient_frac=0.3)
    np.testing.assert_allclose(f, f0, atol=0.1)
    np.testing.assert_allclose(St, f0 * 0.1, atol=0.01)


def test_energy_spectrum_single_mode():
    n = 32
    g = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    u = np.stack([np.sin(3 * X), np.zeros_like(X), np.zeros_like(X)])  # |k| = 3
    k, E = M.energy_spectrum(u)
    assert np.argmax(E) == 3
    np.testing.assert_allclose(E.sum(), np.mean(0.5 * np.sum(u**2, axis=0)), rtol=1e-10)
    np.testing.assert_allclose(float(M.kinetic_energy(jnp.asarray(u))), 0.25, rtol=1e-10)


def test_dissipation_identities():
    t = np.linspace(0, 1, 11)
    Ek = np.exp(-2 * t)
    eps = M.dissipation_from_energy(t, Ek)
    np.testing.assert_allclose(eps[5], 2 * np.exp(-2 * t[5]), rtol=0.02)
    np.testing.assert_allclose(float(M.dissipation_from_enstrophy(jnp.array(3.0), 0.5)), 3.0)
