import jax.numpy as jnp
import numpy as np
import pytest

from pinnflow import benchmarks as B
from pinnflow import constraints as C
from pinnflow.data import EXTERNAL_DIR, cavity_centerlines, load_jaxpi_cavity


def test_dfg_inflow_profile():
    b = B.DFGCylinder("2D-2")
    y = jnp.linspace(0, b.height, 20001)
    u = b.inflow(0.0, y)
    assert abs(float(u.max()) - 1.5) < 1e-6
    mean = float(jnp.trapezoid(u, y) / b.height)
    assert abs(mean - 1.0) < 1e-4  # U_mean = 2/3 U_max = 1  -> Re = 100
    assert abs(b.Re - 100.0) < 1e-12
    b3 = B.DFGCylinder("2D-3")
    assert float(jnp.abs(b3.inflow(0.0, y)).max()) == 0.0 and float(b3.inflow(4.0, y).max()) > 1.49


def test_dfg_distance_functions_vanish_on_boundaries():
    b = B.DFGCylinder()
    th = np.linspace(0, 2 * np.pi, 50)
    xc, yc = b.center[0] + b.radius * np.cos(th), b.center[1] + b.radius * np.sin(th)
    for phi in (C.phi_channel_cylinder, C.phi_channel_cylinder_bounded):
        np.testing.assert_allclose(np.asarray(phi(jnp.asarray(xc), jnp.asarray(yc))), 0.0, atol=1e-6)  # float32 round-off: d = r^2 cos^2 + r^2 sin^2 - r^2 ~ 1e-9, tanh(d / r^2) ~ 4e-7
        x = jnp.linspace(0, b.length, 30)
        np.testing.assert_allclose(np.asarray(phi(x, jnp.zeros_like(x))), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(phi(x, jnp.full_like(x, b.height))), 0.0, atol=1e-12)
        y = jnp.linspace(0, b.height, 30)
        np.testing.assert_allclose(np.asarray(phi(jnp.zeros_like(y), y)), 0.0, atol=1e-12)
        assert float(phi(jnp.array(1.0), jnp.array(0.2))) > 0  # positive in the fluid
    assert float(C.phi_channel_cylinder_bounded(jnp.array(2.0), jnp.array(0.2))) <= 1.0


def test_inflow_extension_matches_bcs():
    b = B.DFGCylinder()
    g = C.cylinder_inflow_extension(lambda t, y: b.inflow(t, y), b.center, b.radius)
    y = jnp.linspace(0, b.height, 11)
    gin = jnp.stack([g(0.0, 0.0, yy) for yy in y])
    np.testing.assert_allclose(np.asarray(gin[:, 0]), np.asarray(b.inflow(0.0, y)), rtol=1e-6, atol=1e-9)
    th = np.linspace(0, 2 * np.pi, 9)
    for t_ in th:
        gc = g(0.0, b.center[0] + b.radius * np.cos(t_), b.center[1] + b.radius * np.sin(t_))
        np.testing.assert_allclose(np.asarray(gc), 0.0, atol=1e-9)


def test_cavity_lid_profile_is_zero_at_corners():
    b = B.LidDrivenCavity()
    assert abs(float(b.lid_profile(jnp.array(0.5))) - (1 - 1 / np.cosh(25.0))) < 1e-9
    assert abs(float(b.lid_profile(jnp.array(0.0)))) < 1e-9 and abs(float(b.lid_profile(jnp.array(1.0)))) < 1e-9
    assert float(C.phi_unit_square(jnp.array(0.5), jnp.array(0.5))) == 0.0625


def test_tgv3d_initial_energy():
    b = B.TaylorGreen3D()
    g = jnp.linspace(0, 2 * np.pi, 48, endpoint=False)
    X, Y, Z = jnp.meshgrid(g, g, g, indexing="ij")
    u, v, w, p = b.initial_condition(X, Y, Z)
    Ek = float(jnp.mean(0.5 * (u**2 + v**2 + w**2)))
    assert abs(Ek - b.initial_energy()) < 1e-6
    assert abs(float(jnp.mean(p))) < 1e-6  # zero-mean pressure -> anchor is consistent


def test_cylinder_surface_normals_and_weights():
    b = B.DFGCylinder()
    pts, nx, ny, ds = b.cylinder_surface(64)
    np.testing.assert_allclose(nx**2 + ny**2, 1.0)
    np.testing.assert_allclose(ds.sum(), 2 * np.pi * b.radius)
    np.testing.assert_allclose(np.hypot(pts[:, 0] - b.center[0], pts[:, 1] - b.center[1]), b.radius)


@pytest.mark.parametrize("Re", [100, 400, 1000])
def test_ghia_tables_agree_with_jaxpi_reference_fields(Re):
    """Cross-check the transcribed Ghia et al. tables against an independent reference solution."""
    if not (EXTERNAL_DIR / "jaxpi" / "examples" / "ldc" / "data" / f"ldc_Re{Re}.mat").exists():
        pytest.skip("JAX-PI reference data not present")
    ref = load_jaxpi_cavity(Re)
    tab = B.ghia_tables(Re)
    # skip the lid point (y=1) and both walls, where the reference grid is coarse relative to the gradient
    yq, xq = tab["y"][1:-1], tab["x"][1:-1]
    c = cavity_centerlines(ref["u"], ref["v"], ref["x"], ref["y"], yq, xq)
    err_u = np.linalg.norm(c["u_centerline"] - tab["u"][1:-1]) / np.linalg.norm(tab["u"][1:-1])
    err_v = np.linalg.norm(c["v_centerline"] - tab["v"][1:-1]) / np.linalg.norm(tab["v"][1:-1])
    assert err_u < 0.05 and err_v < 0.05, (err_u, err_v)
