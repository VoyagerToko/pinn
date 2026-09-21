import jax
import jax.numpy as jnp
import numpy as np

from pinnflow import losses as L


def test_causal_weights_monotone_and_first_is_one():
    chunk = jnp.array([1.0, 0.5, 0.2, 0.1])
    w = L.causal_weights(chunk, eps=1.0)
    assert float(w[0]) == 1.0
    assert np.all(np.diff(np.asarray(w)) <= 0)
    np.testing.assert_allclose(np.asarray(w[1]), np.exp(-1.0), rtol=1e-6)  # float32
    np.testing.assert_allclose(np.asarray(L.causal_weights(chunk, eps=0.0)), 1.0)


def test_causal_chunking_sorts_by_time():
    t = jnp.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
    r = t * 10  # residual grows with time
    chunks = L.causal_chunk_losses(t, r, 4)
    assert np.all(np.diff(np.asarray(chunks)) > 0)
    vals, gamma = L.causal_residual_losses(t, [r, r], 4, eps=1.0)
    assert len(vals) == 2 and gamma.shape == (4,)


def test_causal_annealer():
    a = L.CausalAnnealer(schedule=(0.1, 1.0, 10.0))
    assert a.eps == 0.1
    assert not a.update(0.5) and a.eps == 0.1
    assert a.update(0.995) and a.eps == 1.0
    a.update(1.0)
    assert not a.update(1.0) and a.eps == 10.0  # saturates at the last value


def test_grad_norm_and_ntk_weights(key):
    params = {"w": jnp.array([1.0, 2.0])}

    def losses(p, batch):
        return {"a": jnp.sum(p["w"] ** 2), "b": 100.0 * jnp.sum(p["w"] ** 2)}

    w = L.grad_norm_weights(losses, params, None)
    assert float(w["a"]) > float(w["b"]) > 0
    np.testing.assert_allclose(float(w["a"]) / float(w["b"]), 100.0, rtol=1e-5)
    w2 = L.grad_norm_weights(losses, params, None, reference="a")
    np.testing.assert_allclose(float(w2["a"]), 1.0, rtol=1e-6)
    ntk = L.ntk_weights({"a": jnp.ones(4), "b": 4 * jnp.ones(4)})
    np.testing.assert_allclose(float(ntk["a"]) / float(ntk["b"]), 4.0, rtol=1e-6)
    ema = L.ema_update({"a": jnp.array(1.0)}, {"a": jnp.array(3.0)}, 0.9)
    np.testing.assert_allclose(float(ema["a"]), 1.2, rtol=1e-6)


def test_rad_concentrates_points_where_residual_is_large(key):
    dom = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    sampler = lambda k, n: jax.random.uniform(k, (n, 2), minval=dom[:, 0], maxval=dom[:, 1])
    resid = lambda pts: jnp.where(pts[:, 0] > 0.9, 50.0, 0.01)  # hot strip covering 10% of the domain
    pts = L.rad_resample(key, resid, sampler, n_points=2000, n_candidates=20000, k=1.0, c=1.0, uniform_frac=0.2)
    assert pts.shape == (2000, 2)
    frac = float(jnp.mean(pts[:, 0] > 0.9))
    assert frac > 0.4  # uniform would give 0.1


def test_pressure_anchor_and_gauge():
    p = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(float(L.pressure_anchor_closed(p)), 4.0)
    ref = jnp.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(L.align_pressure_gauge(p, ref)), np.asarray(ref))
