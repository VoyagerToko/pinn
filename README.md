# PINN Fluid Flow Simulation - code base

Implementation of the *PINN Implementation Handbook* (Steps 0-10) for incompressible
Navier-Stokes with physics-informed neural networks in JAX/Flax. Everything up to training is
in place: environments, reference code, datasets, physics, models, losses, training loop,
validation metrics, ablation configs, visualisation pipeline and a test suite that verifies the
residuals against analytic solutions before a single network is trained.

```
pinn/
├── pinnflow/                 the package (one module per handbook step, see below)
│   └── problems/             one PINN problem class per benchmark (A-D, inverse, PI-DeepONet)
├── scripts/                  train.py · evaluate.py · visualize.py · check_env.py · setup_wsl_gpu.sh
├── tests/                    pytest suite (physics gates, archs, losses, metrics, data, smoke training)
├── data/                     downloaded reference data (see "Data inventory")
├── external/                 JAX-PI (main + pirate), SPINN, CausalPINNs  (git clones, --depth 1)
├── environment.yml           native Windows env (CPU JAX) · pyproject.toml (pip install -e .)
└── runs/                     created by scripts/train.py
```

## 1. Environments (STEP 0)

JAX only ships CUDA wheels for Linux, so on this Windows machine there are two environments:

| env | where | JAX backend | use for |
|---|---|---|---|
| `pinn` (conda, Python 3.11) | native Windows | CPU | development, tests, evaluation, PyVista visualisation |
| `pinn` (miniconda, Python 3.11) | WSL2 Ubuntu (`~/miniconda3/envs/pinn`) | **CUDA 12, RTX 3060 6 GB** | all training |

Both contain jax/flax/optax/ml_collections/wandb/hydra, JAX-PI (editable install of `external/jaxpi`),
a CPU-only torch (JAX-PI's samplers import it) and this package (`pip install -e .`).

```powershell
# native (Windows)
conda activate pinn
python scripts/check_env.py                    # report + 20-step smoke run

# GPU (WSL2 Ubuntu) - the code lives on the Windows drive, no copy needed
wsl -d Ubuntu
source ~/miniconda3/bin/activate pinn
cd /mnt/c/Users/mehul/pinn
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # share the 6 GB with the display driver
python scripts/check_env.py                    # must print devices: [CudaDevice(id=0)]
```

Re-create the WSL environment any time with `wsl -d Ubuntu -- bash /mnt/c/Users/mehul/pinn/scripts/setup_wsl_gpu.sh`.

**Any other Linux / WSL2 machine**: `bash scripts/setup_linux_gpu.sh` (installs Miniforge if needed, no hard-coded paths).
Do **not** `pip install "jax[cuda12]"` in a native Windows Python: there are no CUDA wheels for Windows, so pip
silently backtracks to a 2023 CPU-only JAX that this code cannot run on.

Not installed (Linux-only or outside this project's scope, install on demand): PhysicsNeMo, OpenFOAM,
FEniCSx/gmsh (ground-truth generation, STEP 3.1), DeepXDE, ParaView, Blender.
The `pirate` branch of JAX-PI (PirateNet + SOAP) is cloned to `external/jaxpi-pirate`; it uses the same
package name as JAX-PI, so `pip install -e external/jaxpi-pirate` swaps it in when you want SOAP.
This package re-implements PirateNet itself (`pinnflow.archs.PirateNet`), so the swap is only needed for SOAP.

## 2. Week-1 gate: run the JAX-PI cavity example unmodified

```bash
wsl -d Ubuntu -- bash /mnt/c/Users/mehul/pinn/scripts/run_jaxpi_ldc_example.sh
```
(W&B runs offline; logs in `external/jaxpi/examples/ldc/wandb`). Compare the final `l2_error` with the
numbers in `external/jaxpi/examples/ldc/README.md`.

## 3. Data inventory (STEP 3)

| item | path | status |
|---|---|---|
| Cylinder wake, Raissi et al. 2019 (Nektar, Re=100, 5000 pts x 200 steps) | `data/cylinder_wake/cylinder_nektar_wake.mat` | downloaded, structure verified |
| FEATFLOW DFG 2D-2 reference: Cd/Cl series (levels 2-6, dt 1-4) + pressure probes + 2014 official | `data/dfg_benchmark/dfg2d2_*` | downloaded and unzipped |
| FEATFLOW DFG 2D-3 reference: Cd/Cl + pressure, levels 1-6 | `data/dfg_benchmark/dfg2d3_*` | downloaded and unzipped |
| Schaefer & Turek 1996 PDF | - | **not available** at the handbook URL (404 on every mirror tried); the benchmark definition and reference intervals are on the FEATFLOW pages saved next to the data (`dfg2d2_page.html`, `dfg2d3_page.html`) and in `pinnflow.benchmarks.DFGCylinder` |
| HiOCFD4/5 TGV Re=1600 case descriptions | `data/tgv3d_re1600/HiOCFD*_TaylorGreenVortexRe1600.pdf` | downloaded (the numeric DNS curves are not linked from those pages; digitise from the PDFs or the NASA report) |
| DeBonis 2013 NASA 512^3 reference | `data/tgv3d_re1600/DeBonis2013_NASA_TGV.pdf` | downloaded |
| CausalPINNs NS torus data (vorticity form) | `external/CausalPINNs/data/NS.npy` | cloned |
| JAX-PI cavity reference fields Re=100..5000 (128x128) | `external/jaxpi/examples/ldc/data/ldc_Re*.mat` | cloned; used as Benchmark B ground truth |
| Ghia, Ghia & Shin 1982 centreline tables | `pinnflow.benchmarks.ghia_tables` | transcribed; cross-checked against the JAX-PI fields in `tests/test_benchmarks.py` |
| PDEBench, The Well, JHTDB | - | not downloaded (TB-scale / token-gated; only for the optional operator-learning stage) |

Recreate all downloads with `bash scripts/download_data.sh` (WSL/Linux). Loaders: `pinnflow.data` (`load_cylinder_wake`, `load_featflow_series`, `load_jaxpi_cavity`, `load_causal_ns`).

## 4. Code map against the handbook

| step | module | what is implemented |
|---|---|---|
| 1 physics | `pinnflow/physics.py` | non-dimensional NS residuals: VP (2D/3D), stream function (2D, 3rd order), vector potential (3D, u = curl A + weak gauge); forward-mode grid residuals for SPINN via `jax.jvp` along each axis; vorticity, Q and lambda2 criteria by AD |
| 2 benchmarks | `pinnflow/benchmarks.py` | A: TGV-2D exact solution (+ exact stream function) · B: cavity with regularised lid r=50, Ghia tables · C: DFG 2D-2/2D-3 geometry, inflow, surface quadrature, reference intervals · D: TGV-3D Re=1600 IC (+ exact vector potential), E_k(0)=1/8 · inverse wake data set |
| 3 data | `pinnflow/data.py` | loaders above |
| 4 model | `pinnflow/archs.py` | Fourier features (multi-scale, separate time sigma), Modified MLP, RWF, PirateNet (alpha=0 init, PI least-squares init helper), SPINN (grid and point-wise modes, periodic axes), PI-DeepONet; `create_arch(config)` |
| 4.6 hard constraints | `pinnflow/constraints.py` | `u = g + phi N`, R-functions, distance functions for the cavity and the cylinder channel (handbook polynomial and a bounded variant), inflow extension `g`, RWF fold-back |
| 5 loss | `pinnflow/losses.py` | loss terms, grad-norm and NTK weighting with EMA, SA-PINN masks, causal chunk weights + eps annealer (1e-2 ... 100, advance at min w > 0.99), RAD resampling (k=1, c=1, 20% uniform), pressure anchoring |
| 6 training | `pinnflow/training.py` | Adam with 5k warm-up + x0.9/2k decay, `matmul_precision=highest`, grad-norm/NTK updates, causal annealing, RAD hook, CSV/W&B logging, msgpack checkpoints, L-BFGS stage (optax zoom line search = strong Wolfe), Re curriculum, time marching with warm start |
| 7 validation | `pinnflow/metrics.py` | relative L2, max divergence, C_D/C_L/dP (Schaefer-Turek surface form, AD gradients), Strouhal via FFT, E_k, enstrophy, eps = -dE/dt and 2 nu zeta, shell-summed spectrum + inertial slope, throughput |
| 8 ablation | `pinnflow/configs.py` | rows A-H toggling Fourier / Modified MLP+RWF / hard constraints / causal / adaptive lambda / RAD / SPINN |
| 9 visualisation | `pinnflow/viz.py`, `scripts/visualize.py` | grid sampling to `.vti`, Q isosurfaces to `.vtp`, RK45 particle advection on the continuous field, PyVista off-screen render, ffmpeg encode |
| 5-6 problems | `pinnflow/problems/` | `TaylorGreen2DPINN`, `CavityPINN`, `DFGCylinderPINN`, `TaylorGreen3DSPINN`, `CylinderWakeInversePINN`, `ParametricCavityDeepONet` |

## 5. Tests (run before training - STEP 2.1 gate)

```bash
pytest                       # ~2-4 min on CPU
pytest tests/test_physics.py # exact TGV solutions give |residual| < 1e-10 in float64; grid == point-wise AD
```

`tests/test_smoke_train.py` trains every problem for a few steps with tiny networks, exercising grad-norm and
NTK weighting, causal weights, RAD, L-BFGS, curriculum, time marching, checkpoint round trips, both SPINN
formulations and the inverse/DeepONet problems.

## 6. Training (what remains)

```bash
# Benchmark A - gate: rel L2 < 1e-4
python scripts/train.py --benchmark tgv2d --ablation G
# Benchmark B - curriculum Re 100 -> 400 -> 1000 (steps 20k/40k/140k) - gate: within 2% of Ghia at Re=1000
python scripts/train.py --benchmark cavity --ablation F
# Benchmark C - 16 windows x 0.5 s, 200k Adam steps each - gate: St within 3%, Cd within 5%
python scripts/train.py --benchmark cylinder --ablation G
python scripts/train.py --benchmark cylinder --ablation G --set problem.variant='"2D-3"'
# Benchmark D - SPINN (row H), vector potential by default (hard divergence) - gate: dissipation peak within 5%
python scripts/train.py --benchmark tgv3d --ablation H
# ... or the VP formulation first ("get it working", handbook 1.2), then vector potential as the upgrade
python scripts/train.py --benchmark tgv3d --ablation H --set problem.formulation='"vp"'
# inverse problem and parametric surrogate
python scripts/train.py --benchmark cylinder_inverse
python scripts/train.py --benchmark deeponet_cavity
# ablation sweep (rows A-G on A-C)
for r in A B C D E F G; do python scripts/train.py --benchmark tgv2d --ablation $r; done
```

Every run writes `runs/<benchmark>_<row>/{config.json, metrics.csv, latest.msgpack, final_eval.json}`.
`--steps`, `--lbfgs`, `--windows`, `--seed`, `--wandb` and `--set key=value` override the config.

Then: `python scripts/evaluate.py --benchmark cylinder --workdir runs/cylinder_G` (Cd/Cl series, St,
cycle statistics, overlay against FEATFLOW), `--benchmark tgv3d` (E_k, dissipation peak, enstrophy identity,
spectrum slope), and `python scripts/visualize.py --workdir runs/tgv3d_H` for the animation.

### GPU memory notes (6 GB)
* defaults: `res_batch_size` 8192 (2D), 16384 (cylinder), SPINN grid 32^4 (~1M effective points); the
  handbook's 64^4 needs `--set training.n_per_axis="(64,64,64,64)"` and will need gradient accumulation
  (`optim.grad_accum_steps`) or a smaller rank on this card
* set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in WSL

## 7. Deviations from the handbook and things to know

* **Windows**: GPU JAX only inside WSL2. Native env is CPU (fine for tests/eval/viz).
* **Strouhal target**: the handbook quotes St ~ 0.16-0.17 for Re=100; that holds for the *unconfined* cylinder
  (Raissi wake data). For the confined DFG 2D-2 channel the published interval is St in [0.295, 0.305]
  (encoded in `DFGCylinder.reference`, and visible in the FEATFLOW series).
* **Cylinder distance function**: the handbook polynomial `phi = y(0.41-y)((x-0.2)^2+(y-0.2)^2-0.05^2)` is
  available (`problem.phi="handbook"`), but it does not vanish at the inlet and grows to ~10^4 r^4 downstream;
  the default is a bounded variant with the same zero set plus a `tanh(x/0.1)` inlet factor so all Dirichlet
  boundaries are hard and only the outflow stays a soft (do-nothing) loss.
* **SOAP optimiser** is not in optax; it lives in `external/jaxpi-pirate`. Stage 2 here is optax L-BFGS.
* **Grad-norm weighting** follows JAX-PI (mean gradient norm as numerator); set
  `weighting.grad_norm_reference="r_u"` for the handbook's literal `||grad L_res|| / ||grad L_i||`.
* **Ghia tables** were transcribed, not downloaded; the test against the JAX-PI reference fields is the check.
* **Benchmark D** runs as row H only (handbook: "and H on D"); rows A-G raise a clear error for `tgv3d`.
* **Disk**: C: has ~30 GB free; the WSL VHD grows on C:, so keep an eye on it during large runs.
