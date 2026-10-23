# Project brief for AI assistants and new contributors

This repository implements `docs/PINN_Implementation_Handbook.md` (physics-informed neural networks
for incompressible Navier-Stokes, JAX/Flax). Read the handbook section that a task refers to before
changing code; the code map in `README.md` section 4 says which module implements which step.
`docs/STATUS.md` holds the benchmark results so far and the open issues. Keep both up to date when you
train something or change behaviour.

## Where things go

- Project files only at the top level: `pinnflow/`, `viewer/`, `scripts/` (`train.py`, `evaluate.py`, `serve.py`, then
  `data/`, `results/`, `machines/`), `tests/`, `docs/` (handbook, `STATUS.md`), `paper/`, `data/`. README section 1
  has the map.
- **Everything made for or by the assistants lives in `claude helper/`**: `DESKTOP_TRAINING.md` (the desktop runbook),
  `design/` (the viewer's DESIGN.md, PRODUCT.md and impeccable files), `diagnostics/` (one-off investigation scripts,
  e.g. the phantom's `coa_pressure_fit.py`), `tools/` (machine utilities). New one-off scripts, prompts and notes go
  there, never at the top level or in `scripts/`. The folder name has a space: quote it in shell commands.
- This file stays at the repository root because Claude Code only reads it from there.

## How the code is organised

- `pinnflow/` is the package. One module per handbook step (physics, benchmarks, data, archs,
  constraints, losses, training, metrics, viz, configs). `pinnflow/problems/` has one class per
  benchmark that wires these together; `pinnflow/training.Trainer` is problem-agnostic and consumes
  the interface documented at the top of `pinnflow/training.py`.
- Point-wise networks map one input vector `z = (t, x, y[, z])` to one output vector; batch with
  `jax.vmap`. Derivatives are directional forward-mode `jvp`s (`pinnflow.physics.directional`).
  Never reintroduce full `jacfwd` Jacobian/Hessian tensors in a residual: that needed 63 GB.
- Residuals are evaluated through `Problem.vmap_pointwise`, which chunks the batch with `lax.map` and
  `jax.checkpoint`. Peak GPU memory scales with `training.res_chunk`, not the batch size.
- Configs are `ml_collections.ConfigDict`s built by `pinnflow.configs.get_config(benchmark, row)`;
  `row` is the ablation letter A-H from handbook STEP 8. Override from the CLI with
  `--set key=value` (python literals). Scalars that must reach a jitted loss as traced values
  (causal eps, Re) travel inside the batch dict, not as Python attributes.
- Checkpoints are flax msgpack state dicts (`pinnflow.utils.save_params/load_params`); load with the
  freshly initialised params of the same arch as a template.

## Running things

- Tests: `pytest` (52 tests, ~8 min on GPU, ~10 min on CPU). `tests/test_physics.py` is the
  handbook's STEP 2.1 gate: exact Taylor-Green solutions must give residuals < 1e-10 in float64.
- Train: `python scripts/train.py --benchmark <name> --ablation <A-H>`; evaluate:
  `scripts/evaluate.py`; figures: `scripts/results/plot2d.py` (2D) and `scripts/results/visualize.py` (3D).
- GPU only inside Linux/WSL2 (`scripts/machines/setup_linux_gpu.sh`). There are no CUDA JAX wheels for
  Windows; `pip install "jax[cuda12]"` on Windows silently installs a 2023 CPU-only JAX.
- Always `export XLA_PYTHON_CLIENT_PREALLOCATE=false`. First compile of a run takes minutes
  (XLA autotuning); `XLA_FLAGS=--xla_gpu_autotune_level=2` shortens it.
- Data and reference repos are not committed: `bash scripts/data/download_data.sh` recreates them.
- Real-world use case (README 6c): `python scripts/data/make_stenosis_data.py` makes `data/stenosis/` (CFD truth plus a
  synthetic MRI scan), then `--benchmark stenosis_mri` trains it. On this laptop's WSL GPU,
  `scripts/machines/train_local_wsl.sh <benchmark> [args]` trains detached (setsid) and restarts the live server at the end.

## Conventions and gotchas

- Non-dimensional everywhere. Benchmark C works in nondimensional coordinates internally and
  converts at the boundaries (`DFGCylinderPINN.to_nondim/to_dim`, `velocity_dim_fn`).
- Unsteady pressure is unique only up to a function of time: align the gauge per time slice
  before reporting pressure errors.
- Terms that can reach exactly zero (`p_anchor`, `gauge`) are excluded from adaptive weighting
  (`weighting.fixed_terms`); their gradient norm vanishes and would send the weight to infinity.
- Reynolds curricula restart the optimiser state per stage (weights warm-started).
- Deviations from the handbook are listed in `README.md` section 7 (Strouhal target for the
  confined cylinder, bounded distance function, row H only for 3D, missing DFG paper, etc.).
- Windows line endings: files may show as modified after a checkout on Windows; `git checkout -- .`
  clears it, there is nothing to keep.
- Commit messages end with a `Co-Authored-By` line for the assistant when it wrote the change.

## What to do when a benchmark misses its gate

1. Look at `runs/<name>/metrics.csv`: loss terms (`loss/*`), adaptive weights (`w/*`), causal
   minimum weight, evaluation columns. A weight in the thousands or a loss term stuck at 1e-1 is
   the first clue.
2. Compare ablation rows (C soft BC vs D/F hard BC, E causal vs D not) before changing the method:
   the table exists to attribute failures to components.
3. Check the learning rate actually reaching the stage in question, the collocation budget,
   and whether L-BFGS ran to convergence (it logs `loss/lbfgs`).
