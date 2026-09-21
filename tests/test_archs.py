import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinnflow import archs
from pinnflow.constraints import fold_rwf_params

RWF = {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}


@pytest.mark.parametrize("name", ["Mlp", "ModifiedMlp", "PirateNet"])
def test_pointwise_arch_shapes_and_vmap(name, key):
    arch = archs.create_arch({"arch_name": name, "num_layers": 2, "hidden_dim": 32, "out_dim": 3, "fourier_emb": {"embed_scale": 1.0, "embed_dim": 32}, "reparam": RWF})
    params = arch.init(key, jnp.zeros(3))
    y = arch.apply(params, jnp.ones(3))
    assert y.shape == (3,)
    yb = jax.vmap(lambda z: arch.apply(params, z))(jnp.ones((5, 3)))
    assert yb.shape == (5, 3)
    # gradients w.r.t. inputs are finite
    g = jax.jacfwd(lambda z: arch.apply(params, z))(jnp.ones(3))
    assert np.all(np.isfinite(np.asarray(g)))


def test_multiscale_fourier_and_time_scale(key):
    emb = archs.FourierEmbs(embed_scale=(1.0, 10.0), embed_dim=64, time_axis=0, time_scale=0.5)
    params = emb.init(key, jnp.zeros(3))
    y = emb.apply(params, jnp.ones(3))
    assert y.shape == (64,)
    k0 = params["params"]["kernel_0"]
    assert k0.shape == (3, 16)


def test_period_embedding_is_exactly_periodic(key):
    arch = archs.create_arch({"arch_name": "ModifiedMlp", "num_layers": 2, "hidden_dim": 16, "out_dim": 2, "periodicity": {"period": (1.0, 1.0), "axis": (1, 2), "trainable": (False, False)}})
    params = arch.init(key, jnp.zeros(3))
    z = jnp.array([0.3, 1.1, 2.2])
    shift = jnp.array([0.0, 2 * jnp.pi, -2 * jnp.pi])
    np.testing.assert_allclose(np.asarray(arch.apply(params, z)), np.asarray(arch.apply(params, z + shift)), atol=1e-5)


def test_piratenet_blocks_are_identity_at_init(key):
    arch = archs.create_arch({"arch_name": "PirateNet", "num_layers": 3, "hidden_dim": 32, "out_dim": 2, "nonlinearity": 0.0})
    params = arch.init(key, jnp.zeros(2))
    z = jnp.array([0.4, -0.2])
    y0 = arch.apply(params, z)
    # zero the dense kernels inside the residual blocks: with alpha = 0 nothing changes
    from flax.core import unfreeze

    p = unfreeze(params)
    blocks = [k for k in p["params"] if k.startswith("PIModifiedBottleneck")]
    assert len(blocks) == 3
    for b in blocks:
        for dk in [k for k in p["params"][b] if k.startswith("Dense")]:
            p["params"][b][dk]["kernel"] = jnp.zeros_like(p["params"][b][dk]["kernel"])
    np.testing.assert_allclose(np.asarray(arch.apply(p, z)), np.asarray(y0), atol=1e-6)


def test_rwf_fold_back_matches(key):
    cfg = {"arch_name": "ModifiedMlp", "num_layers": 2, "hidden_dim": 16, "out_dim": 3, "fourier_emb": {"embed_scale": 2.0, "embed_dim": 16}}
    a_rwf = archs.create_arch({**cfg, "reparam": RWF})
    a_plain = archs.create_arch(cfg)
    params = a_rwf.init(key, jnp.zeros(3))
    folded = fold_rwf_params(params)
    z = jnp.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(np.asarray(a_plain.apply(folded, z)), np.asarray(a_rwf.apply(params, z)), rtol=1e-5, atol=1e-6)


def test_spinn_grid_equals_points_and_is_periodic(key):
    arch = archs.create_arch({"arch_name": "SPINN", "in_dim": 4, "num_layers": 2, "hidden_dim": 16, "rank": 8, "out_dim": 4, "periodicity": {"axes": (1, 2, 3), "period": 2 * np.pi, "num_freqs": 2}})
    t, x, y, z = jnp.linspace(0, 1, 3), jnp.linspace(0, 6, 4), jnp.linspace(0, 6, 5), jnp.linspace(0, 6, 2)
    params = arch.init(key, [t, x, y, z], mode="grid")
    grid = arch.apply(params, [t, x, y, z], mode="grid")
    assert grid.shape == (4, 3, 4, 5, 2)
    T, X, Y, Z = jnp.meshgrid(t, x, y, z, indexing="ij")
    pts = jnp.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], -1)
    p = arch.apply(params, pts, mode="points")
    np.testing.assert_allclose(np.asarray(p.T.reshape(grid.shape)), np.asarray(grid), rtol=1e-5, atol=1e-6)
    # single point and periodicity
    one = arch.apply(params, pts[0], mode="points")
    assert one.shape == (4,)
    shifted = arch.apply(params, pts[0] + jnp.array([0.0, 2 * np.pi, 0.0, 0.0]), mode="points")
    np.testing.assert_allclose(np.asarray(one), np.asarray(shifted), atol=1e-5)


def test_deeponet_shapes(key):
    arch = archs.create_arch({"arch_name": "DeepONet", "num_branch_layers": 2, "num_trunk_layers": 2, "hidden_dim": 16, "p": 8, "out_dim": 3})
    params = arch.init(key, jnp.zeros(1), jnp.zeros(2))
    y = arch.apply(params, jnp.array([0.5]), jnp.array([0.1, 0.2]))
    assert y.shape == (3,)
