"""End-to-end smoke tests: every problem builds, samples, computes finite losses, takes optimiser
steps, evaluates and checkpoints. Networks are tiny so this runs on CPU in well under a minute each."""
import json

import jax
import numpy as np
import pytest

from pinnflow.configs import get_config
from pinnflow.data import DATA_DIR
from pinnflow.problems import PROBLEMS, DFGCylinderPINN
from pinnflow.training import Trainer, train_time_windows

TINY = {
    "arch.num_layers": 2,
    "arch.hidden_dim": 16,
    "training.res_batch_size": 128,
    "training.bc_batch_size": 64,
    "training.ic_batch_size": 64,
    "optim.warmup_steps": 2,
    "weighting.update_every_steps": 2,
    "weighting.num_chunks": 4,
    "logging.log_every_steps": 2,
}


def _run(benchmark, ablation, tmp_path, extra=None, steps=4, **problem_kwargs):
    ov = dict(TINY)
    if benchmark == "deeponet_cavity":
        ov.pop("arch.num_layers")  # DeepONet has branch/trunk depths instead
        ov.update({"arch.num_branch_layers": 2, "arch.num_trunk_layers": 2, "arch.fourier_emb.embed_dim": 16})
    elif benchmark != "tgv3d" and ablation != "A":  # row A has no Fourier features
        ov["arch.fourier_emb.embed_dim"] = 16
    ov.update(extra or {})
    cfg = get_config(benchmark, ablation, ov)
    problem = PROBLEMS[benchmark](cfg, **problem_kwargs)
    tr = Trainer(problem, cfg, tmp_path / f"{benchmark}_{ablation}", jax.random.PRNGKey(0))
    tr.train(steps, log_every=2, eval_every=steps, ckpt_every=None, rad_every=cfg.training.rad_every)
    rows = (tmp_path / f"{benchmark}_{ablation}" / "metrics.csv").read_text().splitlines()
    assert len(rows) >= 2
    assert np.isfinite(float(rows[-1].split(",")[1]))  # loss column
    return tr, problem


@pytest.mark.parametrize("ablation", ["A", "D", "G"])
def test_tgv2d_rows(ablation, tmp_path):
    tr, problem = _run("tgv2d", ablation, tmp_path, extra={"training.rad_every": 2, "training.rad_candidates": 512, "training.rad_pool_size": 256} if ablation == "G" else None)
    ev = problem.evaluate(tr.state.params)
    assert set(ev) >= {"rel_l2_u", "rel_l2_v", "rel_l2_p"}
    # checkpoint round trip
    tr.save("x.msgpack")
    extra = tr.load(tmp_path / f"tgv2d_{ablation}" / "x.msgpack")
    assert "step" in extra


def test_tgv2d_ntk_scheme_and_lbfgs(tmp_path):
    tr, problem = _run("tgv2d", "D", tmp_path, extra={"weighting.scheme": "ntk"})
    tr.lbfgs(max_iters=3, log_every=1)


@pytest.mark.parametrize("ablation", ["C", "D"])
def test_cavity_rows_and_curriculum(ablation, tmp_path):
    tr, problem = _run("cavity", ablation, tmp_path)
    problem.set_Re(400)
    tr.train(2, log_every=1, eval_every=None, ckpt_every=None)
    ev = problem.evaluate(tr.state.params)
    assert "ghia_u_rel_err" in ev


@pytest.mark.parametrize("ablation", ["C", "E"])
def test_cylinder_rows(ablation, tmp_path):
    tr, problem = _run("cylinder", ablation, tmp_path, t1=0.1)
    ev = problem.evaluate(tr.state.params)
    assert any(k.startswith("Cd@") for k in ev)


def test_cylinder_time_marching(tmp_path):
    ov = dict(TINY, **{"arch.fourier_emb.embed_dim": 16, "problem.window_dt": 0.05})
    cfg = get_config("cylinder", "G", ov)

    def make(idx, ic_fn):
        p = DFGCylinderPINN(cfg, t0=idx * 0.05, t1=(idx + 1) * 0.05, ic_fn=ic_fn)
        return Trainer(p, cfg, tmp_path / f"w{idx}", jax.random.PRNGKey(idx))

    trs = train_time_windows(make, 2, 3, log_every=1, eval_every=None, ckpt_every=None)
    assert len(trs) == 2 and trs[1].problem.ic_fn is not None


def test_tgv3d_spinn_vp_and_vector_potential(tmp_path):
    for form in ("vp", "vector_potential"):
        ov = dict(TINY, **{"arch.rank": 4, "training.n_per_axis": (4, 5, 5, 5), "training.ic_grid": 4, "problem.formulation": form, "weighting.use_causal": True})
        if form == "vector_potential":
            ov["weighting.init_weights"] = {"ic_u": 1.0, "ic_v": 1.0, "ic_w": 1.0, "r_u": 1.0, "r_v": 1.0, "r_w": 1.0, "p_anchor": 1.0, "gauge": 1.0}
        cfg = get_config("tgv3d", "H", ov)
        problem = PROBLEMS["tgv3d"](cfg)
        tr = Trainer(problem, cfg, tmp_path / f"tgv3d_{form}", jax.random.PRNGKey(0))
        tr.train(3, log_every=1, eval_every=3, ckpt_every=None)
        vel = problem.velocity_fn(tr.state.params)
        assert vel(jax.numpy.array([0.1, 1.0, 2.0, 3.0])).shape == (4,)
        hist = problem.energy_history(tr.state.params, [0.0, 1.0], n=6)
        assert hist["Ek"].shape == (2,)


def test_cylinder_inverse(tmp_path):
    if not (DATA_DIR / "cylinder_wake" / "cylinder_nektar_wake.mat").exists():
        pytest.skip("cylinder wake data not downloaded")
    tr, problem = _run("cylinder_inverse", "G", tmp_path, extra={"training.data_batch_size": 64, "problem.n_train": 500})
    ev = problem.evaluate(tr.state.params)
    assert "lambda1" in ev and "lambda2" in ev


def test_deeponet_cavity(tmp_path):
    tr, problem = _run("deeponet_cavity", "G", tmp_path, extra={"arch.p": 8, "training.n_Re_per_batch": 2})
    assert isinstance(problem.evaluate(tr.state.params), dict)
