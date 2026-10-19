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
├── scripts/                  train.py · evaluate.py · plot2d.py · visualize.py · check_env.py · setup_*_gpu.sh
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

Figures and movies for the 2D cases (fields vs exact, error maps, streamlines, Ghia profiles, vortex street):

```bash
python scripts/plot2d.py --benchmark tgv2d    --workdir runs/tgv2d_G --animate 120
python scripts/plot2d.py --benchmark cavity   --workdir runs/cavity_F
python scripts/plot2d.py --benchmark cylinder --workdir runs/cylinder_G --animate 200
```

Then: `python scripts/evaluate.py --benchmark cylinder --workdir runs/cylinder_G` (Cd/Cl series, St,
cycle statistics, overlay against FEATFLOW), `--benchmark tgv3d` (E_k, dissipation peak, enstrophy identity,
spectrum slope), and `python scripts/visualize.py --workdir runs/tgv3d_H` for the animation.

### Memory and speed notes

Point-wise residuals are evaluated in rematerialised chunks (`training.remat=True`, `training.res_chunk=2048`),
so peak memory scales with the chunk size, not the batch size. Measured on an RTX 3060 Laptop (6 GB, WSL2),
Benchmark A row G with the default config (stream function = third-order derivatives, 6 x 256, batch 8192):

| | value |
|---|---|
| peak GPU memory (incl. XLA autotuning) | 5.9 GB |
| step time | ~1.0 s |
| first-step compile + autotune | ~15 min |
| rel. L2 after 400 steps | 6e-2 |

Expect roughly 4x faster steps on a 16 GB desktop card (about 0.25 s/step, 3-4 h for the 50k-step run).
Row A-C configs (VP formulation, second order) are about 3x cheaper per step than rows D-G.

* On 16 GB cards `--set training.remat=False` removes the ~30% recomputation overhead for small 2D runs; for the
  cylinder (16,384 residual points) it runs out of memory in the grad-norm update (measured 2026-09-24).
* Lower `training.res_chunk` (e.g. 1024) if a 6 GB card still runs out of memory; raise it on big cards.
* The grad-norm / NTK weight updates run one backward pass per loss term every `update_every_steps`; NTK
  diagonals are chunked in blocks of 256 points.
* Slow first compile: `export XLA_FLAGS=--xla_gpu_autotune_level=2` cuts autotuning time at a small runtime cost.
* SPINN grid: 32^4 (~1M effective points) is the default (0.044 s/step, 3.1 GB on the RTX 5070 Ti); the
  handbook's 64^4 (`--set training.n_per_axis="(64,64,64,64)"`) does **not** fit in 16 GB (one 11.1 GB buffer).
* Always `export XLA_PYTHON_CLIENT_PREALLOCATE=false` in WSL so the display driver keeps its share.

## 6b. Live inference server (STEP 9.5)

`scripts/serve.py` loads the trained checkpoints onto the GPU once and serves fields, curves and sensor
points over HTTP for the live viewer in `viewer/` (binary float32 fields, JSON curves, CORS enabled):

```bash
pip install fastapi "uvicorn[standard]"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nohup python scripts/serve.py --port 8000 > runs/serve.log 2>&1 &
curl http://localhost:8000/health        # interactive docs at /docs
```

Scenes and the runs that back them are listed in `REGISTRY` at the top of the script; missing runs are skipped.
The first request per model and grid size compiles (seconds on a GPU); results are cached per request key.
To reach the server from another machine on the network, either set `networkingMode=mirrored` under `[wsl2]` in
`%USERPROFILE%\.wslconfig` (then `wsl --shutdown`) or forward the port with `netsh interface portproxy`, and open
TCP 8000 in the Windows firewall.

**Running the scenes on another machine's GPU.** The checkpoints are small (37.7 MB for all seven scenes with
their references). On the training machine the server offers them at `/bundle`; on the other machine
`curl -o pinn_bundle.tar.gz http://<host>:8000/bundle && tar -xzf pinn_bundle.tar.gz` at the repository root,
then `wsl -d Ubuntu -- bash /mnt/c/.../pinn/scripts/serve_local_wsl.sh`, which prints the address to use.
Measured on an RTX 3060 Laptop (`scripts/bench_server.py`): 5,000 particle velocities in 11-14 ms (70-90 updates/s)
via the WSL address, 64-75 ms via `localhost` (Windows' forwarding into WSL); full fields 18-215 ms per 128^2 frame
depending on the scene, 82 ms per 48^3 3D frame; first request per grid size compiles for 20-30 s.

**The live viewer.** `python -m viewer` opens a desktop window (PySide6 and pyvistaqt, native Windows env). It finds
the server itself (the `PINN_SERVER` variable, then the WSL address, then localhost; press S to start it in WSL).
Every frame is a network prediction. Pigment dropped with a click, or combed with a drag, is carried by the
velocity the network predicts at each particle. The reference sits beside it at the same instant and colour
scale, and each failure shows its evidence: the corner divergence, the lost flow rate, the missing dissipation
peak. Options include `--scene`, `--view`, `--Re`, `--play`, `--size WxH` and `--shot file.png`.
The visual system is in `DESIGN.md`: the "marbling bath", with one pigment per scene, procedural paper and ink,
and a 13 px text floor.

| Module | Role |
|---|---|
| `viewer/app.py` | The window. |
| `viewer/panel.py` | The tray and the layout driven by the domain's aspect ratio. |
| `viewer/bath.py` | 2D painting. |
| `viewer/tank.py` | The 3D tank. |
| `viewer/rack.py` | The reference sheet, charts and coefficients. |
| `viewer/controls.py` | The rail, the stamp and the bars. |
| `viewer/workers.py` | Particle advection and dye texture, field and curve threads. |
| `viewer/materials.py` | Procedural paper, shadows and ink. |
| `viewer/theme.py` | Palette, OKLab maps and fonts. |

## 6c. Real-world use case: blood pressure across a stenosis from MRI-like flow data

A synthetic, controlled study, not a clinical tool. It is the inverse problem of the wake benchmark, applied to a
clinical question: can the pressure drop across a narrowed artery be inferred from the velocities a 4D-flow MRI scan
measures? Today that drop is measured with a catheter, or estimated from Doppler ultrasound with the simplified
Bernoulli rule, dp [mmHg] = 4 v_peak^2.

```bash
python scripts/make_stenosis_data.py                 # CFD truth + simulated scan -> data/stenosis/  (~10 min, CPU)
python scripts/train.py --benchmark stenosis_mri     # the inverse network -> runs/stenosis_mri_G/
# or, on this laptop's WSL GPU, in the background, restarting the live server when it is done:
wsl -d Ubuntu -- bash /mnt/c/.../pinn/scripts/train_local_wsl.sh stenosis_mri
pytest tests/test_vessel.py                          # Poiseuille check of the solver, scan statistics, smoke train
```

- **The vessel (`pinnflow/vessel.py`).** A 2D artery, D = 6 mm, with a smooth 60% narrowing. The inflow is
  heartbeat-shaped, 73 beats per minute, at Re = 150 on the mean flow (0.087 m/s for blood).
- **The truth.** A staggered-grid projection solver with 50 cells per diameter. It runs three cycles from rest
  and keeps the last; the last two cycles agree to float32. Its pressure gradient matches Poiseuille flow to 0.6%.
- **The simulated scan.** Velocity averaged over 0.6 mm voxels and 24 cardiac phases, in lumen voxels only, with
  scanner phase noise at SNR 20 (sigma = 1.2 cm/s, VENC 0.51 m/s). That is 20,160 samples.
- **The network (`pinnflow/problems/stenosis_inverse.py`).**
  - It outputs a stream function and pressure, so continuity holds exactly, and it is exactly periodic in time
    with the heartbeat.
  - It is fitted to each voxel through the average of the network over that voxel and phase window.
  - Momentum residuals apply at the known viscosity, the wall is no-slip, and one point pins the pressure gauge.
  - Pressure is never given to it.
- **The score (`final_eval.json`).** The velocity and pressure error against the CFD truth, and the peak and
  cycle RMS error of the pressure drop from 2 diameters upstream to the throat. The gate is the peak within 10%.
  The same file reports the simplified Bernoulli estimate from the same scan, for comparison.
  - The truth peaks at **0.59 mmHg**. The Bernoulli estimate is 0.78 mmHg, 32% too high, because it ignores the
    viscous losses and the inflow kinetic energy.
  - The gradient is mild because Re is kept modest; `--Re` and `--severity` change it.
- **Viewer and server.** The scene lives in the viewer's "Real-world use case" tab.
  - Before training, the server already serves the scan and the CFD truth, labelled as untrained.
  - After training, restart the server: `runs/stenosis_mri_G/latest.msgpack` and `final_eval.json` plug in as they
    are, and the pigment is carried by the network's flow.

## 7. Deviations from the handbook and things to know

See also `CLAUDE.md` (project brief for assistants and contributors), `STATUS.md` (benchmark results and open
issues) and `docs/PINN_Implementation_Handbook.md` (the handbook this code implements).

* **Windows**: GPU JAX only inside WSL2. Native env is CPU (fine for tests/eval/viz).
* **Strouhal target**: the handbook quotes St ~ 0.16-0.17 for Re=100; that holds for the *unconfined* cylinder
  (Raissi wake data). For the confined DFG 2D-2 channel the published interval is St in [0.295, 0.305]
  (encoded in `DFGCylinder.reference`, and visible in the FEATFLOW series).
* **Cylinder distance function**: the handbook polynomial `phi = y(0.41-y)((x-0.2)^2+(y-0.2)^2-0.05^2)` is
  available (`problem.phi="handbook"`), but it does not vanish at the inlet and grows to ~10^4 r^4 downstream;
  the default is a bounded variant with the same zero set plus a `tanh(x/0.1)` inlet factor so all Dirichlet
  boundaries are hard and only the outflow stays a soft (do-nothing) loss.
* **Pressure gauge in unsteady runs**: p is unique only up to a function of time, so pressure errors are
  computed after aligning the mean per time slice (handbook 5.6 says to align the gauge; this is the unsteady form).
* **SOAP optimiser** is not in optax; it lives in `external/jaxpi-pirate`. Stage 2 here is optax L-BFGS.
* **Grad-norm weighting** follows JAX-PI (mean gradient norm as numerator); set
  `weighting.grad_norm_reference="r_u"` for the handbook's literal `||grad L_res|| / ||grad L_i||`.
* **Ghia tables** were transcribed, not downloaded; the test against the JAX-PI reference fields is the check.
* **Benchmark D** runs as row H only (handbook: "and H on D"); rows A-G raise a clear error for `tgv3d`.
* **Benchmark D reference**: the HiOCFD pages publish no numeric curves; the reference dissipation curve is
  digitised from DeBonis (2013) Fig. 4(a) (`scripts/digitize_tgv_reference.py`, peak 0.01282 at t = 9.00).
* **Cavity lid** (STATUS.md D1-D3): imposing the regularised lid exactly (`u = g + phi N` on the lid) is
  incompatible with continuity at the two top corners, and rows D/F fail with it; `problem.lid_bc="soft"` keeps
  no-slip and v = 0 exact and makes only the lid velocity a loss term. Rows D-G of the cavity ablation use it.
  The fully hard runs are kept as evidence.
* **Cavity reference**: besides Ghia et al. and the JAX-PI fields (both unit lid), a finite-difference solution of
  the regularised-lid problem (`scripts/cavity_fd_reference.py`, 512^2 at Re=1000) is the field reference
  (`rel_l2_vel_vs_fd`). A converged solution of this benchmark is 2.29% from Ghia's v table.
* **L-BFGS** runs on one fixed collocation batch; on the cavity it can lower the loss by orders of magnitude
  while the field error grows (STATUS.md D3). Report pre- and post-L-BFGS numbers.
* **Causal eps** starts at the first entry of the annealing schedule (1e-2); before 2026-09-24 it started at
  `causal_tol` = 1.0.
* **PI-DeepONet** never samples Re in [300, 530]; Re = 400 is the zero-shot query.
* **Reduced budgets (2026-09-27)**: to finish within a 12-hour window the cylinder, 3D, inverse, DeepONet and
  ablation runs use the reduced budgets listed in STATUS.md; the cylinder ablation was not run.
* **STEP 9 export**: `.vti` frames written before 2026-09-24 had x and z swapped (VTK stores x fastest).
* **Disk**: C: has ~30 GB free; the WSL VHD grows on C:, so keep an eye on it during large runs.
