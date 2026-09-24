"""STEP 9 sampling: the separable grid path must reproduce the point-wise AD path, and .vti files must
keep x as x (VTK image data store x fastest)."""
import jax
import numpy as np
import pytest

from pinnflow import viz
from pinnflow.configs import get_config
from pinnflow.problems import PROBLEMS

TINY = {"arch.num_layers": 2, "arch.hidden_dim": 8, "arch.rank": 6, "training.n_per_axis": (16, 8, 8, 8), "training.ic_grid": 8}


@pytest.mark.parametrize("formulation", ["vp", "vector_potential"])
def test_grid_fields_match_pointwise(formulation):
    cfg = get_config("tgv3d", "H", {**TINY, "problem.formulation": formulation})
    problem = PROBLEMS["tgv3d"](cfg)
    params = problem.init_params(jax.random.PRNGKey(3))
    n, t = 6, 2.5
    grid = problem.grid_fields(params, t, n)
    _, pts = viz.sample_grid(n)
    point = viz.sample_field(problem.velocity_fn(params), t, pts)
    for k in ("velocity", "speed", "vorticity", "qcriterion"):
        np.testing.assert_allclose(grid[k], point[k], rtol=2e-3, atol=2e-5 * (1 + np.abs(point[k]).max()))


def test_vti_keeps_axes(tmp_path):
    pv = pytest.importorskip("pyvista")
    n = 5
    g, pts = viz.sample_grid(n)
    spacing = g[1] - g[0]
    grid = viz.fields_to_vti({"x": pts[:, 0], "vec": pts}, n, spacing, tmp_path / "f.vti")
    back = pv.read(str(tmp_path / "f.vti"))
    np.testing.assert_allclose(np.asarray(back["x"]), np.asarray(back.points[:, 0]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(back["vec"]), np.asarray(back.points), atol=1e-5)
