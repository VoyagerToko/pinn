"""Residual implementation checks - the handbook's STEP 2.1 gate: exact solutions must give
residuals at machine precision *before* any network is trained."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinnflow import physics
from pinnflow.benchmarks import TaylorGreen2D, TaylorGreen3D


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _pts(key, n, dom):
    dom = jnp.asarray(dom)
    return jax.random.uniform(key, (n, dom.shape[0]), minval=dom[:, 0], maxval=dom[:, 1], dtype=jnp.float64)


def test_tgv2d_exact_vp_residual_is_zero(key):
    b = TaylorGreen2D(Re=100.0, T=2.0)
    r = jax.vmap(physics.ns_vp_residual(b.exact_vector, b.Re, dim=2, unsteady=True))
    r_mom, r_c = r(_pts(key, 512, b.domain))
    assert float(jnp.abs(r_mom).max()) < 1e-10
    assert float(jnp.abs(r_c).max()) < 1e-10


def test_tgv2d_exact_streamfunction_residual_is_zero(key):
    b = TaylorGreen2D(Re=50.0, T=1.0)
    r = jax.vmap(physics.ns_streamfunction_residual(b.exact_psi_p, b.Re, unsteady=True))
    r_mom, r_c = r(_pts(key, 512, b.domain))
    assert float(jnp.abs(r_mom).max()) < 1e-10
    assert float(jnp.abs(r_c).max()) < 1e-12
    # velocity recovered from psi equals the exact velocity
    vel = jax.vmap(physics.streamfunction_velocity(b.exact_psi_p))
    z = _pts(key, 64, b.domain)
    np.testing.assert_allclose(np.asarray(vel(z)), np.asarray(jax.vmap(b.exact_vector)(z)), atol=1e-12)


def test_tgv3d_initial_condition_is_divergence_free(key):
    b = TaylorGreen3D()

    def fn(z):  # steady field (x, y, z) -> (u, v, w, p)
        u, v, w, p = b.initial_condition(z[0], z[1], z[2])
        return jnp.stack([u, v, w, p])

    r = jax.vmap(physics.ns_vp_residual(fn, b.Re, dim=3, unsteady=False))
    _, r_c = r(_pts(key, 256, b.domain[1:]))
    assert float(jnp.abs(r_c).max()) < 1e-12


def test_tgv3d_vector_potential_reproduces_initial_velocity(key):
    b = TaylorGreen3D()

    def A_p(z):
        A1, A2, A3 = b.initial_vector_potential(z[0], z[1], z[2])
        p = b.initial_condition(z[0], z[1], z[2])[3]
        return jnp.stack([A1, A2, A3, p])

    z = _pts(key, 128, b.domain[1:])
    vel = jax.vmap(physics.vector_potential_velocity(A_p, unsteady=False))(z)
    u, v, w, _ = b.initial_condition(z[:, 0], z[:, 1], z[:, 2])
    np.testing.assert_allclose(np.asarray(vel[:, :3]), np.stack([u, v, w], -1), atol=1e-12)
    r_mom, r_c, gauge = jax.vmap(physics.ns_vector_potential_residual(A_p, b.Re, unsteady=False))(z)
    assert float(jnp.abs(r_c).max()) < 1e-12  # curl is exactly divergence-free
    assert r_mom.shape == (128, 3) and gauge.shape == (128,)


def test_grid_residual_matches_pointwise_for_separable_field(key):
    """Forward-mode grid derivatives (SPINN path) must agree with point-wise AD."""
    b = TaylorGreen2D(Re=100.0, T=1.0)
    t = jnp.linspace(0.05, 0.9, 4, dtype=jnp.float64)
    x = jnp.linspace(0.1, 6.0, 5, dtype=jnp.float64)
    y = jnp.linspace(0.2, 5.5, 6, dtype=jnp.float64)

    def f_grid(t, x, y):  # exact solution evaluated on the tensor grid -> (3, nt, nx, ny)
        T, X, Y = jnp.meshgrid(t, x, y, indexing="ij")
        u, v, p = b.exact(T, X, Y)
        return jnp.stack([u, v, p])

    r_mom, r_c = physics.ns_vp_residual_grid(f_grid, [t, x, y], b.Re, unsteady=True)
    assert float(jnp.abs(r_mom).max()) < 1e-10 and float(jnp.abs(r_c).max()) < 1e-10

    # a non-solution: compare against point-wise residuals of the same field
    def g_point(z):
        return jnp.stack([jnp.sin(z[1]) * jnp.cos(z[2]) * z[0], jnp.cos(z[1] * z[2]), jnp.exp(-z[0]) * z[1] ** 2])

    def g_grid(t, x, y):
        T, X, Y = jnp.meshgrid(t, x, y, indexing="ij")
        return jnp.stack([jnp.sin(X) * jnp.cos(Y) * T, jnp.cos(X * Y), jnp.exp(-T) * X**2])

    rg_mom, rg_c = physics.ns_vp_residual_grid(g_grid, [t, x, y], 10.0, unsteady=True)
    T, X, Y = jnp.meshgrid(t, x, y, indexing="ij")
    pts = jnp.stack([T.ravel(), X.ravel(), Y.ravel()], -1)
    rp_mom, rp_c = jax.vmap(physics.ns_vp_residual(g_point, 10.0, dim=2, unsteady=True))(pts)
    np.testing.assert_allclose(np.asarray(rg_mom).reshape(2, -1).T, np.asarray(rp_mom), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(rg_c).ravel(), np.asarray(rp_c), rtol=1e-9, atol=1e-9)


def test_q_criterion_signs():
    rotation = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    strain = jnp.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    assert float(physics.q_criterion(rotation)) > 0
    assert float(physics.q_criterion(strain)) < 0
    assert float(physics.lambda2_criterion(rotation)) < 0  # vortex core
    omega = physics.vorticity_from_grad(rotation)
    np.testing.assert_allclose(np.asarray(omega), [0.0, 0.0, 2.0])
