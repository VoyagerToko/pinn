"""STEP 6.6 reference configuration and STEP 8 ablation ladder.

``get_config(benchmark, ablation)`` returns an ``ml_collections.ConfigDict`` for one row of the
ablation table:

    | Config | Fourier | Mod. MLP (+RWF) | Hard BC/div | Causal | Adaptive lambda | RAD | SPINN |
    | A      |   x     |       x         |      x      |   x    |        x        |  x  |   x   |
    | B      |   v     |       x         |      x      |   x    |        x        |  x  |   x   |
    | C      |   v     |       v         |      x      |   x    |        x        |  x  |   x   |
    | D      |   v     |       v         |      v      |   x    |        x        |  x  |   x   |
    | E      |   v     |       v         |      v      |   v    |        x        |  x  |   x   |
    | F      |   v     |       v         |      v      |   v    |        v        |  x  |   x   |
    | G      |   v     |       v         |      v      |   v    |        v        |  v  |   x   |
    | H      |   v     |       v         |      v      |   v    |        v        |  v  |   v   |   (3D only)

Fix the seed, the collocation budget and the wall-clock budget across rows.
"""
from __future__ import annotations

import ml_collections

ABLATION = {
    "A": dict(fourier=False, modified_mlp=False, hard=False, causal=False, adaptive=False, rad=False, spinn=False),
    "B": dict(fourier=True, modified_mlp=False, hard=False, causal=False, adaptive=False, rad=False, spinn=False),
    "C": dict(fourier=True, modified_mlp=True, hard=False, causal=False, adaptive=False, rad=False, spinn=False),
    "D": dict(fourier=True, modified_mlp=True, hard=True, causal=False, adaptive=False, rad=False, spinn=False),
    "E": dict(fourier=True, modified_mlp=True, hard=True, causal=True, adaptive=False, rad=False, spinn=False),
    "F": dict(fourier=True, modified_mlp=True, hard=True, causal=True, adaptive=True, rad=False, spinn=False),
    "G": dict(fourier=True, modified_mlp=True, hard=True, causal=True, adaptive=True, rad=True, spinn=False),
    "H": dict(fourier=True, modified_mlp=True, hard=True, causal=True, adaptive=True, rad=True, spinn=True),
}


def _base() -> ml_collections.ConfigDict:
    c = ml_collections.ConfigDict()
    c.seed = 42
    c.project = "pinnflow"
    c.name = "run"
    c.problem = ml_collections.ConfigDict()

    c.arch = ml_collections.ConfigDict()
    c.arch.arch_name = "ModifiedMlp"
    c.arch.num_layers = 6
    c.arch.hidden_dim = 256
    c.arch.out_dim = 3
    c.arch.activation = "tanh"
    c.arch.periodicity = None
    c.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": 256})
    c.arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})

    c.optim = ml_collections.ConfigDict()
    c.optim.optimizer = "Adam"
    c.optim.learning_rate = 1e-3
    c.optim.warmup_steps = 5000
    c.optim.decay_rate = 0.9
    c.optim.decay_steps = 2000
    c.optim.grad_accum_steps = 0
    c.optim.beta1 = 0.9
    c.optim.beta2 = 0.999
    c.optim.eps = 1e-8
    c.optim.clip_grad_norm = None

    c.training = ml_collections.ConfigDict()
    c.training.max_steps = 200000
    c.training.lbfgs_steps = 20000
    c.training.res_batch_size = 8192
    c.training.bc_batch_size = 2048
    c.training.ic_batch_size = 2048
    c.training.remat = True  # chunked + rematerialised residual evaluation (memory ~ res_chunk, ~30% slower)
    c.training.res_chunk = 2048  # points per rematerialised chunk; lower it on small GPUs
    c.training.rad_every = None
    c.training.rad_candidates = 100000
    c.training.rad_pool_size = 65536
    c.training.rad_chunk = 8192
    c.training.rad_k = 1.0
    c.training.rad_c = 1.0
    c.training.rad_uniform_frac = 0.2

    c.weighting = ml_collections.ConfigDict()
    c.weighting.scheme = "grad_norm"  # grad_norm | ntk | none
    c.weighting.grad_norm_reference = None  # None = JAX-PI mean-norm numerator; "r_u" = handbook 5.3(a) literal
    c.weighting.init_weights = ml_collections.ConfigDict()
    c.weighting.momentum = 0.9
    c.weighting.update_every_steps = 1000
    c.weighting.use_causal = False
    c.weighting.causal_tol = 1.0
    c.weighting.causal_eps_schedule = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    c.weighting.num_chunks = 32

    c.logging = ml_collections.ConfigDict()
    c.logging.log_every_steps = 100
    c.logging.eval_every_steps = 5000
    c.logging.ckpt_every_steps = 10000
    return c


def _benchmark_defaults(c: ml_collections.ConfigDict, benchmark: str) -> None:
    p, t, w = c.problem, c.training, c.weighting
    if benchmark == "tgv2d":
        p.Re = 100.0
        p.T = 1.0
        p.formulation = "streamfunction"
        c.arch.out_dim = 2
        c.arch.periodicity = ml_collections.ConfigDict({"period": (1.0, 1.0), "axis": (1, 2), "trainable": (False, False)})
        c.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
        t.max_steps = 50000
        t.res_batch_size = 8192
        t.ic_batch_size = 2048
        w.init_weights = ml_collections.ConfigDict({"u_ic": 1.0, "v_ic": 1.0, "r_u": 1.0, "r_v": 1.0, "p_anchor": 1.0})
    elif benchmark == "cavity":
        p.Re = 100.0
        p.formulation = "vp"
        p.hard_bc = True
        p.curriculum_Re = (100, 400, 1000)
        p.curriculum_steps = (20000, 40000, 140000)
        c.arch.out_dim = 3
        t.res_batch_size = 8192
        t.bc_batch_size = 2048
        w.init_weights = ml_collections.ConfigDict({"u_bc": 1.0, "v_bc": 1.0, "r_u": 1.0, "r_v": 1.0, "r_c": 1.0, "p_anchor": 1.0})
    elif benchmark == "cylinder":
        p.variant = "2D-2"
        p.hard_bc = True
        p.phi = "bounded"  # or "handbook" for the literal 4.6b polynomial
        p.outflow = "do_nothing"
        p.window_dt = 0.5  # seconds (dimensional); T* = 5 per window
        p.num_time_windows = 16  # 16 x 0.5 s = 8 s
        c.arch.out_dim = 3
        c.arch.activation = "gelu"
        c.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
        t.max_steps = 200000
        t.res_batch_size = 16384
        t.bc_batch_size = 4096
        t.ic_batch_size = 4096
        w.num_chunks = 32
        w.init_weights = ml_collections.ConfigDict({"u_ic": 1.0, "v_ic": 1.0, "u_out": 1.0, "v_out": 1.0, "r_u": 1.0, "r_v": 1.0, "r_c": 1.0})
    elif benchmark == "tgv3d":
        p.Re = 1600.0
        p.T = 20.0
        p.formulation = "vp"
        p.gauge_weight = 1e-3
        c.arch = ml_collections.ConfigDict(
            {
                "arch_name": "SPINN",
                "in_dim": 4,
                "num_layers": 4,
                "hidden_dim": 64,
                "rank": 128,
                "out_dim": 4,
                "activation": "tanh",
                "mlp": "modified_mlp",
                "periodicity": {"axes": (1, 2, 3), "period": 6.283185307179586, "num_freqs": 4},
                "reparam": {"type": "weight_fact", "mean": 1.0, "stddev": 0.1},
            }
        )
        t.max_steps = 300000
        t.n_per_axis = (32, 32, 32, 32)  # 64^4 in the handbook; 32^4 ~ 1M effective points fits a 6 GB GPU
        t.ic_grid = 48
        w.num_chunks = 16
        w.init_weights = ml_collections.ConfigDict({"ic_u": 1.0, "ic_v": 1.0, "ic_w": 1.0, "r_u": 1.0, "r_v": 1.0, "r_w": 1.0, "r_c": 1.0, "p_anchor": 1.0})
    elif benchmark == "cylinder_inverse":
        p.formulation = "streamfunction"
        p.n_train = 5000
        p.noise = 0.0
        c.arch.out_dim = 2
        c.arch.num_layers = 8
        c.arch.hidden_dim = 128
        c.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 128})
        t.max_steps = 100000
        t.data_batch_size = 2048
        t.res_batch_size = 4096
        w.init_weights = ml_collections.ConfigDict({"u_data": 1.0, "v_data": 1.0, "r_u": 1.0, "r_v": 1.0})
    elif benchmark == "deeponet_cavity":
        p.Re_min = 100.0
        p.Re_max = 1000.0
        p.eval_Re = (100, 400, 1000)
        c.arch = ml_collections.ConfigDict(
            {
                "arch_name": "DeepONet",
                "num_branch_layers": 3,
                "num_trunk_layers": 6,
                "hidden_dim": 256,
                "p": 128,
                "out_dim": 3,
                "activation": "tanh",
                "fourier_emb": {"embed_scale": 10.0, "embed_dim": 256},
                "reparam": {"type": "weight_fact", "mean": 1.0, "stddev": 0.1},
            }
        )
        t.max_steps = 200000
        t.res_batch_size = 4096
        t.n_Re_per_batch = 4
        w.init_weights = ml_collections.ConfigDict({"r_u": 1.0, "r_v": 1.0, "r_c": 1.0, "p_anchor": 1.0})
    else:
        raise KeyError(f"unknown benchmark {benchmark!r}")


def _apply_ablation(c: ml_collections.ConfigDict, benchmark: str, row: str) -> None:
    a = ABLATION[row]
    arch_is_pointwise = c.arch.arch_name in ("Mlp", "ModifiedMlp", "PirateNet")
    if arch_is_pointwise:
        c.arch.arch_name = "ModifiedMlp" if a["modified_mlp"] else "Mlp"
        if not a["fourier"]:
            c.arch.fourier_emb = None
        if not a["modified_mlp"]:
            c.arch.reparam = None  # RWF enters together with the Modified MLP (weeks 6-7)
    # hard constraints
    if benchmark == "tgv2d":
        c.problem.formulation = "streamfunction" if a["hard"] else "vp"
        c.arch.out_dim = 2 if a["hard"] else 3
        if a["hard"]:
            c.weighting.init_weights = ml_collections.ConfigDict({"u_ic": 1.0, "v_ic": 1.0, "r_u": 1.0, "r_v": 1.0, "p_anchor": 1.0})
        else:
            c.weighting.init_weights = ml_collections.ConfigDict({"u_ic": 1.0, "v_ic": 1.0, "r_u": 1.0, "r_v": 1.0, "r_c": 1.0, "p_anchor": 1.0})
    elif benchmark == "cavity":
        c.problem.hard_bc = bool(a["hard"])
        keys = {"r_u": 1.0, "r_v": 1.0, "r_c": 1.0, "p_anchor": 1.0}
        if not a["hard"]:
            keys.update({"u_bc": 1.0, "v_bc": 1.0})
        c.weighting.init_weights = ml_collections.ConfigDict(keys)
    elif benchmark == "cylinder":
        c.problem.hard_bc = bool(a["hard"])
        keys = {"u_ic": 1.0, "v_ic": 1.0, "u_out": 1.0, "v_out": 1.0, "r_u": 1.0, "r_v": 1.0, "r_c": 1.0}
        if not a["hard"]:
            keys.update({"u_in": 1.0, "v_in": 1.0, "u_walls": 1.0, "v_walls": 1.0, "u_cylinder": 1.0, "v_cylinder": 1.0})
        c.weighting.init_weights = ml_collections.ConfigDict(keys)
    elif benchmark == "tgv3d":
        c.problem.formulation = "vector_potential" if a["hard"] else "vp"
        keys = {"ic_u": 1.0, "ic_v": 1.0, "ic_w": 1.0, "r_u": 1.0, "r_v": 1.0, "r_w": 1.0, "p_anchor": 1.0}
        if a["hard"]:
            keys["gauge"] = 1.0
        else:
            keys["r_c"] = 1.0
        c.weighting.init_weights = ml_collections.ConfigDict(keys)
        if not a["spinn"]:
            raise ValueError("Benchmark D (tgv3d) is run only as ablation row H (SPINN); handbook STEP 8 runs rows A-G on Benchmarks A-C.")
    # causal / adaptive / RAD
    unsteady = benchmark in ("tgv2d", "cylinder", "tgv3d")
    c.weighting.use_causal = bool(a["causal"] and unsteady)
    c.weighting.scheme = "grad_norm" if a["adaptive"] else "none"
    c.training.rad_every = 1000 if a["rad"] and benchmark != "tgv3d" else None


def get_config(benchmark: str, ablation: str = "G", overrides=None) -> ml_collections.ConfigDict:
    """Full config for ``benchmark`` at ablation row ``ablation`` ('A'..'H'); ``overrides`` is a
    flat dict like ``{"training.max_steps": 1000, "arch.hidden_dim": 128}``."""
    c = _base()
    _benchmark_defaults(c, benchmark)
    row = ablation.upper()
    if benchmark != "tgv3d" and row == "H":
        row = "G"  # the SPINN row only exists for the 3D case
    if benchmark == "tgv3d":
        row = "H"  # Benchmark D is only run with the full 3D configuration
    if benchmark not in ("cylinder_inverse", "deeponet_cavity"):
        _apply_ablation(c, benchmark, row)
    c.name = f"{benchmark}_{row}"
    if overrides:
        with c.unlocked():
            c.update_from_flattened_dict(dict(overrides))
    return c
