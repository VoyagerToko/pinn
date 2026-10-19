"""Real-world use case: the stenosis CFD truth, the synthetic MRI scan, and the inverse problem end to end."""
import jax
import numpy as np
import pytest

from pinnflow.configs import get_config
from pinnflow.problems import PROBLEMS
from pinnflow.training import Trainer
from pinnflow.vessel import (
    StenosisVessel, bernoulli_pressure_drop, load_dataset, mri_from_cfd, mri_samples, save_dataset, section_mean, solve_cfd,
)


def test_poiseuille():
    """Straight vessel, steady inflow: dp/dx = -12 / Re and a parabolic profile with centreline speed 1.5."""
    v = StenosisVessel(length=4.0, severity=0.0, Re=20.0, harmonics=(), period=4.0)
    r = solve_cfd(v, N=16, t_end=3.0, log=None)
    u, p, x = r["u"][-1], r["p"][-1], r["x"]
    dpdx = (section_mean(p[None], x, 3.0)[0] - section_mean(p[None], x, 1.5)[0]) / 1.5
    assert dpdx == pytest.approx(-12.0 / v.Re, rel=0.02)
    y = r["y"]
    assert np.nanmax(np.abs(u[32] - 1.5 * (1 - (y / 0.5) ** 2))) < 0.02
    assert np.nanmax(np.abs(r["v"][-1])) < 5e-3


def _tiny(tmp_path):
    v = StenosisVessel(length=4.0, xc=1.5, half_len=0.5, Re=50.0, period=2.0, fov=(0.4, 3.6), proximal=0.6)
    truth = solve_cfd(v, N=10, n_periods=1, frames=16, log=None)
    mri = mri_from_cfd(truth, v, voxel=0.2, phases=4, snr=20.0)
    save_dataset(tmp_path, v, truth, mri)
    return v, truth, mri


def test_mass_and_mri(tmp_path):
    v, truth, mri = _tiny(tmp_path)
    # incompressible: the same flux through every section, equal to the inflow
    u = truth["u"][-1]
    flux = np.nansum(u, axis=1) * truth["h"]
    assert np.allclose(flux, flux[0], rtol=1e-6, atol=1e-6)
    assert flux[0] == pytest.approx(v.flow(truth["t"][-1]) * 1.0, rel=0.02)
    # the scan: phases x voxels, lumen only, noise at the requested SNR
    P, nvx, nvy = mri["u"].shape
    assert P == 4 and nvy == 5 and np.isnan(mri["u"][:, ~mri["lumen"]]).all()
    noise = (mri["u"] - mri["u_clean"])[:, mri["lumen"]]
    assert np.std(noise) == pytest.approx(float(mri["sigma"]), rel=0.35)
    s = mri_samples(mri)
    assert len(s["t"]) == P * mri["lumen"].sum()
    assert np.all(bernoulli_pressure_drop(mri, v) > 0)
    v2, truth2, mri2 = load_dataset(tmp_path)
    assert v2 == v and np.array_equal(mri2["lumen"], mri["lumen"])


def test_stenosis_inverse_smoke(tmp_path):
    _tiny(tmp_path)
    cfg = get_config("stenosis_mri", overrides={
        "problem.data_dir": str(tmp_path), "arch.num_layers": 2, "arch.hidden_dim": 16, "arch.fourier_emb.embed_dim": 16,
        "training.res_batch_size": 128, "training.bc_batch_size": 64, "training.data_batch_size": 32,
        "optim.warmup_steps": 2, "weighting.update_every_steps": 2, "logging.log_every_steps": 2,
    })
    problem = PROBLEMS["stenosis_mri"](cfg)
    tr = Trainer(problem, cfg, tmp_path / "run", jax.random.PRNGKey(0))
    tr.train(4, log_every=2, eval_every=4, ckpt_every=None, rad_every=None)
    ev = problem.evaluate(tr.state.params)
    assert {"rel_l2_p", "dp_peak_err_pct", "dp_peak_bernoulli_mmHg"} <= set(ev)
    assert all(np.isfinite(v) for v in ev.values())
    # exactly periodic in time with the cardiac period
    vel = problem.velocity_fn(tr.state.params)
    z = np.array([0.3, 1.2, 0.1], np.float32)
    assert np.allclose(vel(z), vel(z + np.array([problem.T, 0, 0], np.float32)), atol=1e-4)
    # hard walls: zero velocity on the segmented wall, and the learned flow rate through every section
    for xw in (0.6, 1.5, 2.9):
        hw = float(problem.vessel.half_width(xw))
        for sgn in (1.0, -1.0):
            uw = np.asarray(vel(np.array([0.7, xw, sgn * hw], np.float32)))[:2]
            assert np.abs(uw).max() < 1e-4
        ys = np.linspace(-hw, hw, 201)
        q = np.trapezoid([float(vel(np.array([0.7, xw, y], np.float32))[0]) for y in ys], ys)
        assert q == pytest.approx(float(problem.flow_rate(tr.state.params, 0.7)), rel=1e-3)
