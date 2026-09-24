---
title: "PINN Fluid Flow Simulation"
subtitle: "Detailed Implementation Handbook — Formulas, Models, Datasets and Tools"
date: "September 2026"
---

# Contents

| STEP 0 — Environment and Stack
| STEP 1 — Write Down the Physics Exactly
| STEP 2 — Define the Four Benchmarks Exactly
| STEP 3 — Datasets: What to Download, From Where, and For What
| STEP 4 — The Model: Exact Architecture
| STEP 5 — The Loss Function, Term by Term
| STEP 6 — Training Procedure
| STEP 7 — Validation: Exact Metrics and Formulas
| STEP 8 — Ablation Protocol
| STEP 9 — The 3D Visualisation and Animation Pipeline
| STEP 10 — Execution Order and Realistic Expectations
| Resource Index

> This document is the build instruction set for the research methodology. Every step states **what you do**, **the exact mathematics**, **the exact software/dataset with its link**, and **the acceptance check** that lets you move to the next step.

---

# STEP 0 — Environment and Stack

### 0.1 Why JAX and not PyTorch

PINNs are dominated by the cost of taking derivatives of the network with respect to its *inputs*, not its weights. JAX gives you `jax.jvp` (forward-mode AD), `jax.vmap` and XLA fusion, which together make second-order spatial derivatives roughly an order of magnitude cheaper than the equivalent reverse-mode PyTorch graph. Forward-mode AD is what makes the 3D case feasible at all.

Use PyTorch only if you must reuse an existing PyTorch codebase.

### 0.2 Install

```bash
conda create -n pinn python=3.11 -y && conda activate pinn

# JAX with CUDA 12
pip install --upgrade "jax[cuda12]"
pip install flax optax ml_collections wandb hydra-core

# Core PINN framework (clone, do not just pip-install - you will edit it)
git clone https://github.com/PredictiveIntelligenceLab/jaxpi
cd jaxpi && pip install . && cd ..
git clone -b pirate https://github.com/PredictiveIntelligenceLab/jaxpi jaxpi-pirate

# Separable PINN (3D enabler)
git clone https://github.com/stnamjef/SPINN

# Secondary / cross-check frameworks
pip install deepxde                      # https://github.com/lululxvi/deepxde
pip install nvidia-physicsnemo           # https://github.com/NVIDIA/physicsnemo

# CFD ground truth
sudo apt install openfoam                # or https://www.openfoam.com/download
pip install fenics-dolfinx gmsh          # https://fenicsproject.org

# Post-processing, visualisation, export
pip install pyvista vtk meshio h5py scipy matplotlib
pip install onnx onnxruntime tf2onnx
# ParaView:  https://www.paraview.org/download/
# Blender:   https://www.blender.org/download/
```

### 0.3 What each framework is actually for

| Tool | Repository | Use it for |
|---|---|---|
| **JAX-PI** | `github.com/PredictiveIntelligenceLab/jaxpi` | Your **primary** codebase. Already implements Fourier features, modified MLP, causal weighting, NTK weighting, RWF, curriculum, time-marching. Do not re-implement these. |
| **JAX-PI `pirate` branch** | same repo, branch `pirate` | PirateNet architecture + the SOAP second-order optimiser. This branch reports the first PINN results on turbulent flow up to Re ≈ 10,000 — it is the current state of the art and your high-Re path. |
| **SPINN** | `github.com/stnamjef/SPINN` | Separable architecture for 3D+time. Project page: `jwcho5576.github.io/spinn.github.io/` |
| **DeepXDE** | `github.com/lululxvi/deepxde` | Fast prototyping and sanity cross-checks. Has a ready Navier–Stokes inverse example. |
| **PhysicsNeMo** (ex-Modulus) | `github.com/NVIDIA/physicsnemo` + `physicsnemo-sym` | Symbolic PDE definition via SymPy, multi-GPU, industrial geometry. Heavier; use for the final 3D scale-up if you have multiple GPUs. |
| **OpenFOAM / FEniCSx** | `openfoam.com`, `fenicsproject.org` | Ground-truth generation only. |
| **PyVista / ParaView / Blender** | `pyvista.org`, `paraview.org`, `blender.org` | The Step 9 visualisation chain. |

---

# STEP 1 — Write Down the Physics Exactly

### 1.1 Non-dimensional incompressible Navier–Stokes

Always non-dimensionalise. It puts every loss term on a comparable scale and is the cheapest conditioning improvement available.

With reference velocity `U`, reference length `L`, `x → x/L`, `t → tU/L`, `u → u/U`, `p → p/(ρU²)`, `Re = UL/ν`:

**Momentum (3 equations):**
```
∂u/∂t + (u·∇)u = −∇p + (1/Re)∇²u
```
componentwise:
```
r_u = u_t + u·u_x + v·u_y + w·u_z + p_x − (1/Re)(u_xx + u_yy + u_zz)
r_v = v_t + u·v_x + v·v_y + w·v_z + p_y − (1/Re)(v_xx + v_yy + v_zz)
r_w = w_t + u·w_x + v·w_y + w·w_z + p_z − (1/Re)(w_xx + w_yy + w_zz)
```

**Continuity:**
```
r_c = u_x + v_y + w_z
```

Every subscript above is an automatic-differentiation call on the network. Nothing is discretised.

### 1.2 Choose your formulation

| Formulation | Outputs | Derivative order | Notes |
|---|---|---|---|
| **VP** (velocity–pressure) | `u,v,w,p` | 2nd | Simplest. Pressure fixed only up to a constant — must anchor it (Step 6.6). |
| **VV** (velocity–vorticity) | `u,v,w,ω` | 1st–2nd | Better conditioned, no pressure gauge problem, but +3 unknowns in 3D. Used in NSFnets (Jin et al. 2021). |
| **Stream function (2D)** | `ψ, p` | 3rd | `∇·u = 0` exact. Recommended for all 2D work. |
| **Vector potential (3D)** | `A, p` | 3rd | `∇·u = 0` exact. Expensive; use with SPINN. |

**Recommendation:** 2D → stream function. 3D → VP first (get it working), then vector potential as the ablation upgrade.

---

# STEP 2 — Define the Four Benchmarks Exactly

You need problems where the correct answer is independently known. These four are the standard ladder.

### 2.1 Benchmark A — 2D Taylor–Green vortex (analytic)

Domain `[0, 2π]²`, periodic, `t ∈ [0, T]`.

**Exact solution (use this as ground truth — no CFD needed):**
```
u(x,y,t) = −cos(x) sin(y) exp(−2t/Re)
v(x,y,t) =  sin(x) cos(y) exp(−2t/Re)
p(x,y,t) = −(1/4)(cos(2x) + cos(2y)) exp(−4t/Re)
```

Purpose: validates your residual implementation to machine precision. **If you cannot hit relative L2 < 1e-4 here, every later result is meaningless.** Run this first, always.

### 2.2 Benchmark B — Lid-driven cavity (steady)

Domain `[0,1]²`. Walls no-slip. Lid at `y=1` moves in `+x`.

Use the **regularised lid** to remove the corner singularity that plain PINNs cannot represent:
```
u(x, 1) = 1 − cosh(r(x − 0.5)) / cosh(0.5 r),    r = 50
v(x, 1) = 0
```

Reference data: **Ghia, Ghia & Shin (1982)**, *J. Comput. Phys.* 48(3), 387–411 — Tables I and II give centreline `u(y)` along `x=0.5` and `v(x)` along `y=0.5` at Re = 100, 400, 1000, 3200, 5000. These tables are the universally accepted validation target; digitised copies ship with most CFD teaching repos, and JAX-PI includes the cavity example directly.

Run: Re = 100 → 400 → 1000 as a curriculum (Step 6.3).

### 2.3 Benchmark C — Flow past a cylinder (unsteady, the real test)

Use the **DFG 2D-2 / 2D-3 benchmark** of Schäfer & Turek (1996) — this is the reference setup used by FEATFLOW, FEniCSx and deal.II, so your numbers are directly comparable to published values.

**Geometry:** channel `[0, 2.2] × [0, 0.41]`, cylinder centre `(0.2, 0.2)`, radius `0.05`.
**Fluid:** `ν = 0.001`, `ρ = 1`.
**Inflow (2D-2, steady amplitude, Re = 100):**
```
u_in(y) = 4 · U · y(0.41 − y) / 0.41² ,   U = 1.5
```
**Inflow (2D-3, time-modulated):** multiply `U` by `sin(πt/8)`, `t ∈ (0, 8)`.
**Walls and cylinder:** no-slip. **Outflow:** `∂u/∂n = 0`, `p = 0`.

Reference values: Schäfer & Turek 1996 tables, PDF at
`http://www.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf`
FEATFLOW time-series files `bdforces_lv4.txt` / `pointvalues_lv4.txt` are the standard comparison curves.

**Also use the classic PINN wake dataset** (Raissi et al. 2019) for the inverse problem:
`https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat`

Array structure (verified):
```python
from scipy.io import loadmat
d = loadmat("cylinder_nektar_wake.mat")
d["X_star"]   # (N, 2)     spatial points
d["U_star"]   # (N, 2, T)  velocity (u,v)
d["p_star"]   # (N, T)     pressure
d["t"]        # (T, 1)     time
```
This is a spectral-element (Nektar) simulation at Re = 100 in a window of the wake.

### 2.4 Benchmark D — 3D Taylor–Green vortex at Re = 1600

Domain `[0, 2π]³`, triply periodic, `t ∈ [0, 20]`.

**Initial condition:**
```
u(x,y,z,0) =  sin(x) cos(y) cos(z)
v(x,y,z,0) = −cos(x) sin(y) cos(z)
w(x,y,z,0) =  0
p(x,y,z,0) = (1/16)(cos(2x) + cos(2y))(cos(2z) + 2)
```

No analytic solution — it transitions to turbulence around t ≈ 8–9 then decays. It is *the* standard 3D validation case, used in the 1st–5th International Workshops on High-Order CFD Methods.

Reference data (dissipation rate, enstrophy, kinetic-energy spectra):
- HiOCFD4: `https://how4.cenaero.be/content/bs1-dns-taylor-green-vortex-re1600`
- HiOCFD5: `https://how5.cenaero.be/content/ws1-dns-taylor-green-vortex-re1600`
- DeBonis (NASA), 512³ high-order DRP reference: `https://ntrs.nasa.gov/api/citations/20130011044/downloads/20130011044.pdf`

---

# STEP 3 — Datasets: What to Download, From Where, and For What

**Critical distinction.** For a *forward solve* you train on physics only — the dataset is used purely to grade you. For an *inverse / assimilation* problem, sparse data enters the loss on purpose. Never blur the two, and always state which mode a result came from.

| Dataset | Link | Size | What it gives you | Use in this project |
|---|---|---|---|---|
| **Cylinder wake (Nektar)** | `github.com/maziarraissi/PINNs` → `main/Data/cylinder_nektar_wake.mat` | ~40 MB | Re=100 wake, u/v/p on scattered points over time | Inverse problem: recover ν and the pressure field from sparse velocity |
| **CausalPINNs NS data** | `github.com/PredictiveIntelligenceLab/CausalPINNs/tree/main/data` | small | 2D NS on a torus, vorticity form | The reference data SPINN and the causal-training paper both use — direct comparability |
| **PDEBench** | `github.com/pdebench/PDEBench`, DOI `10.18419/darus-2986` | TB-scale, downloadable per-PDE | Compressible & incompressible NS, Darcy, shallow water; HDF5 `[b,t,x1..xd,v]` | Operator-learning stage (Step 5.5): supervised pretraining and baseline comparison against FNO/U-Net |
| **The Well** | `github.com/PolymathicAI/the_well`, HF `polymathic-ai` | 15 TB total, 16 datasets, 6.9 GB–5.1 TB each | Diverse spatiotemporal physics, HDF5, PyTorch loader, streams from HuggingFace without download | Optional: stress-test generalisation of the operator model |
| **JHTDB** | `turbulence.pha.jhu.edu`, `pip install pyJHTDB` | 100+ TB, query by web service | DNS isotropic turbulence 1024³, channel flow, boundary layer. Requires a free access token | Only if you extend to turbulence closure. Use the cutout service, do not try to download |
| **HiOCFD TGV reference** | links in §2.4 | KB | Dissipation-rate and enstrophy curves at Re=1600 | Benchmark D validation |
| **Ghia et al. 1982 tables** | the published paper | KB | Cavity centreline profiles | Benchmark B validation |
| **Your own OpenFOAM/FEniCSx runs** | generated | GB | Everything else, with full provenance | Primary ground truth |

**Streaming The Well without downloading 15 TB:**
```python
from the_well.data import WellDataset
ds = WellDataset(well_base_path="hf://datasets/polymathic-ai/",
                 well_dataset_name="turbulent_radiative_layer_2D",
                 well_split_name="train")
```

### 3.1 Generating your own ground truth

```bash
# OpenFOAM, cylinder DFG 2D-2
cp -r $FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/cylinder2D .
blockMesh && checkMesh
pimpleFoam | tee log.pimpleFoam
foamToVTK            # -> VTK/ folder, load with PyVista
```
Run at three mesh refinements and confirm `Cd` changes by < 0.5% between the two finest. **Mesh independence is not optional** — an unconverged "ground truth" invalidates every error number you report.

---

# STEP 4 — The Model: Exact Architecture

### 4.1 Baseline (build this first so you have something to beat)

```
MLP: 4 hidden layers × 256 units, tanh, Glorot normal init
Input:  (x, y, t)        Output: (u, v, p)
Loss:   L = L_pde + L_bc + L_ic     (all weights = 1)
```
It will fail on Benchmarks C and D. That failure is your baseline row in the ablation table.

### 4.2 Component 1 — Random Fourier feature embedding

**Purpose:** defeat spectral bias (networks fit low frequencies first and may never reach the high ones).

```
γ(x) = [ cos(2π B x) , sin(2π B x) ],     B ∈ R^(m×d),  B_ij ~ N(0, σ²)
```
- `m = 128` (so the embedding is 256-dim)
- **Multi-scale:** use two or three `B` matrices with `σ ∈ {1, 10}` and concatenate the outputs. One scale captures the bulk flow, the other the boundary layer.
- **Separate space and time:** use a different `σ` for the temporal axis (typically smaller, `σ_t = 1`).

Reference: Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions*, NeurIPS 2020. Already implemented in JAX-PI as `arch_name: ModifiedMlp, fourier_emb: {embed_scale: 10.0, embed_dim: 256}`.

### 4.3 Component 2 — Modified MLP (gated architecture)

**Purpose:** fix gradient pathologies; gives markedly better convergence than a plain MLP at the same parameter count.

```
U = tanh(W_U γ(x) + b_U)
V = tanh(W_V γ(x) + b_V)
H¹ = tanh(W¹ γ(x) + b¹)

for k = 1 … L−1:
    Z^k    = tanh(W^k H^k + b^k)
    H^(k+1) = (1 − Z^k) ⊙ U  +  Z^k ⊙ V

output = W_out H^L + b_out
```
`U` and `V` are computed once and injected at every layer — that is the whole trick.

Reference: Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021.

### 4.4 Component 3 — Random Weight Factorisation (RWF)

**Purpose:** accelerate convergence and escape flat regions of the PINN loss landscape.

Replace every weight matrix `W` by
```
W = diag(exp(s)) · V ,      s ~ N(μ, σ²),  μ = 1.0, σ = 0.1
```
and train `s` and `V` jointly. Costs nothing at inference (fold it back). JAX-PI flag: `weight_fact: {type: rwf, mean: 1.0, stddev: 0.1}`.

### 4.5 Component 4 — PirateNet residual blocks (for deep networks)

Plain PINNs get *worse* beyond ~6 layers because of a pathological initialisation. PirateNet fixes this with an adaptive residual:

```
f^l = tanh(W₁ x^l + b₁)
z₁  = f^l ⊙ U + (1 − f^l) ⊙ V
g   = tanh(W₂ z₁ + b₂)
z₂  = g ⊙ U + (1 − g) ⊙ V
h   = tanh(W₃ z₂ + b₃)
x^(l+1) = α^l · h + (1 − α^l) · x^l        with α^l trainable, initialised to 0
```
With `α = 0` the block is the identity at init, so the network starts as a linear model and deepens itself during training. Use the `pirate` branch of JAX-PI.

### 4.6 Component 5 — Hard constraints (this is where you beat standard PINNs)

**(a) Exactly divergence-free velocity**

2D — output the stream function `ψ` instead of `u,v`:
```
u = ∂ψ/∂y ,   v = −∂ψ/∂x       ⟹   ∇·u = ψ_yx − ψ_xy ≡ 0
```
3D — output a vector potential `A = (A₁,A₂,A₃)`:
```
u = ∇ × A                       ⟹   ∇·u = ∇·(∇×A) ≡ 0
```
Add a weak gauge term `λ_g ‖∇·A‖²` with `λ_g ≈ 1e-3` for conditioning (the vector potential is otherwise non-unique).

Cost: derivative order rises by one. Benefit: mass conservation goes from ~1e-3 (penalty) to ~1e-14 (machine precision). This alone is a defensible contribution.

**(b) Exactly satisfied boundary conditions**

```
û(x) = g(x) + φ(x) · N_θ(x)
```
- `g(x)` — any smooth function matching the BC on the boundary
- `φ(x)` — a smooth **approximate distance function**, zero on the boundary, positive inside

For the cylinder channel:
```
φ(x,y) = y (0.41 − y) · ((x−0.2)² + (y−0.2)² − 0.05²)
```
For analytic geometry generally, construct `φ` with **R-functions** (Rvachev). For arbitrary STL geometry, pre-train a small network on a signed distance field from `trimesh` / `gmsh`.

Result: `L_bc` disappears from the loss entirely. One fewer competing objective is one fewer failure mode.

### 4.7 Component 6 — Separable PINN (SPINN) for 3D

**Purpose:** conventional PINNs need `N⁴` collocation points for 3D+time. SPINN needs `4N` network evaluations.

```
u(x,y,z,t) = Σ_{j=1}^{r}  f_j^x(x) · f_j^y(y) · f_j^z(z) · f_j^t(t)
```
- Four independent MLPs, each `R¹ → R^r`
- Rank `r = 64 … 256` (sweep it; `r` is the representational ceiling)
- Combine with **forward-mode AD** (`jax.jvp`) — this is the half of the method that gives the speedup
- Reported: 62× wall-clock and 1394× FLOP reduction at equal collocation count; a (2+1)-D chaotic Navier–Stokes case solved in 9 minutes versus 10 hours for the prior best on the same single GPU; enables >10⁷ collocation points on one commodity GPU

Code: `github.com/stnamjef/SPINN`. **This is the single most important component for your 3D deliverable.**

### 4.8 Component 7 — Operator learning for parametric generalisation

A standard PINN = one network, one Reynolds number. To get a *family* of solutions, use **PI-DeepONet**:

```
Branch:  b(a)  = encoder of the parameter a        (Re, inflow profile, or SDF of the geometry)
Trunk:   t(y)  = encoder of the query point y = (x,y,z,t)

G_θ(a)(y) = Σ_{k=1}^{p} b_k(a) · t_k(y) + b₀ ,      p = 128
```
Then apply the *physics* loss to `G_θ(a)(y)` for sampled `a`. Training is expensive once; inference on a new `Re` is a single forward pass.

Alternative: **Fourier Neural Operator** (`github.com/neuraloperator/neuraloperator`) if you have PDEBench data and want a data-driven baseline to compare against.

**Model selection summary:**

| Situation | Model |
|---|---|
| 2D, single configuration | Modified MLP + Fourier features + stream function |
| 2D, deep network needed | PirateNet |
| 3D + time | SPINN + vector potential |
| High Re (>1000) | PirateNet + SOAP optimiser (`pirate` branch) |
| Many Reynolds numbers | PI-DeepONet |
| You have thousands of CFD solutions | FNO (as the data-driven baseline to beat) |

---

# STEP 5 — The Loss Function, Term by Term

### 5.1 The full objective

```
L(θ) = λ_r L_res + λ_c L_div + λ_b L_bc + λ_i L_ic + λ_d L_data
```

```
L_res  = (1/N_r) Σ_i [ r_u(x_i)² + r_v(x_i)² + r_w(x_i)² ]
L_div  = (1/N_r) Σ_i [ r_c(x_i)² ]
L_bc   = (1/N_b) Σ_j ‖ û(x_j) − u_BC(x_j) ‖²
L_ic   = (1/N_0) Σ_k ‖ û(x_k, 0) − u₀(x_k) ‖²
L_data = (1/N_d) Σ_m ‖ û(x_m) − u_obs(x_m) ‖²      # inverse problems only
```

With hard constraints from §4.6, `L_div` and `L_bc` drop out. That is the goal.

### 5.2 Collocation point budget

| Case | `N_r` per step | `N_b` | `N_0` |
|---|---|---|---|
| 2D steady (cavity) | 8,192 | 2,048 | — |
| 2D unsteady (cylinder) | 16,384 | 4,096 | 4,096 |
| 3D + time (SPINN) | 64⁴ grid via separable eval ≈ 1.6×10⁷ effective | periodic → none | 64³ |

### 5.3 Adaptive loss weighting — pick one, ablate all three

**(a) Gradient-norm annealing** (cheap, robust, start here):
```
λ̂_i = ‖∇_θ L_res‖ / ‖∇_θ L_i‖
λ_i ← (1 − α) λ_i + α λ̂_i ,     α = 0.9,  update every 1000 steps
```

**(b) NTK-based weighting** (principled, equalises convergence rates):
```
λ_i = tr(K) / tr(K_i) ,   K_i = NTK of loss term i,   K = Σ_i K_i
```

**(c) Self-adaptive (SA-PINN)** — a trainable weight per point, updated by gradient *ascent*:
```
θ ← argmin_θ  Σ_i m(w_i) · r(x_i)²
w ← argmax_w  Σ_i m(w_i) · r(x_i)²        m = softplus or sigmoid mask
```
This automatically concentrates effort on the hardest points. Combine with §5.5.

### 5.4 Causal training weights (essential for Benchmarks C and D)

Split `[0,T]` into `M = 32` equal windows with midpoints `t₁ < … < t_M`. Weight each window:
```
w_i = exp( −ε · Σ_{k<i} L_res(t_k) )
L_res = (1/M) Σ_i w_i · L_res(t_i)
```
- `ε` controls strictness. Anneal it: `ε ∈ {1e-2, 1e-1, 1, 10, 100}`, advancing when `min_i w_i > 0.99`.
- Interpretation: window `i` receives essentially zero gradient until every earlier window has converged. This restores the arrow of time, which the naive loss destroys.

Reference: Wang, Sankaran & Perdikaris, *Respecting Causality for Training PINNs*, 2022. Implemented in JAX-PI as `weighting: {use_causal: true, causal_tol: 1.0, num_chunks: 32}`.

### 5.5 Adaptive collocation resampling (RAD)

Uniform sampling wastes capacity on easy regions. Resample with probability proportional to residual:
```
p(x) ∝ ε(x)^k / E[ε(x)^k] + c ,     ε(x) = |PDE residual at x|
k = 1,  c = 1        # standard, well-tested defaults
```
Every 1,000 iterations: draw 100,000 candidates uniformly, evaluate residual, sample `N_r` points from `p(x)`, and **retain 20% uniform** to avoid collapsing onto one hot spot. Points migrate into boundary layers and the vortex street by themselves.

Reference: Wu, Zhu, Tan, Kartha & Lu, CMAME 2023.

### 5.6 Pressure anchoring (VP formulation only)

Incompressible NS determines `p` only up to a constant. Fix it:
```
closed domain:  L_p = ( (1/|Ω|) Σ_i p(x_i) )²         # zero-mean, Monte Carlo
open domain:    p(x_out) = 0                          # reference Dirichlet at outflow
```
When reporting pressure error, align the gauge first and say so explicitly. Papers that skip this report meaningless pressure errors.

---

# STEP 6 — Training Procedure

### 6.1 Optimiser schedule

```
Stage 1 — Adam
  lr           = 1e-3
  warmup       = 5,000 steps (linear 0 → 1e-3)
  decay        = exponential, ×0.9 every 2,000 steps
  iterations   = 200,000 (2D) / 300,000+ (3D)
  precision    = float32 with jax_default_matmul_precision="highest"

Stage 2 — L-BFGS
  line search  = strong Wolfe
  iterations   = 20,000–50,000
  effect       = typically 1–2 further orders of magnitude in residual

Optional Stage 1b — SOAP (second-order, `pirate` branch)
  currently the strongest reported optimiser for PINNs; reaches
  turbulent regimes (Re up to ~10,000) that Adam-trained PINNs do not
```

### 6.2 Precision warning

Set JAX matmul precision to `highest`. Reproducibility of PINN results is genuinely sensitive to this — the JAX-PI authors changed their default for exactly this reason.

### 6.3 Reynolds-number curriculum

```
for Re in [100, 200, 400, 1000]:
    model.load(checkpoint[previous Re])     # warm start
    train(Re, iterations=100_000)
    save(checkpoint[Re])
```
Never train `Re = 1000` from scratch. The loss landscape becomes stiff and convection-dominated, and cold-start training collapses to a trivial or steady solution.

### 6.4 Time marching (sequence-to-sequence)

For long time horizons, do not train `[0, T]` in one shot:
```
ΔT = 0.5
for n in range(int(T/ΔT)):
    IC   = trained_model[n-1](t = n·ΔT)     # terminal state of previous window
    model[n] = train_window([n·ΔT, (n+1)·ΔT], IC)
```
Each window is well-conditioned; error growth is bounded per window rather than compounding freely.

### 6.5 Domain decomposition (XPINN) for large 3D domains

Partition `Ω` into subdomains `Ω_q`, one network each, plus interface losses:
```
L_interface = Σ_q ‖ u_q − u_q̄ ‖²  (continuity)
            + Σ_q ‖ ∂u_q/∂n − ∂u_q̄/∂n ‖²  (flux continuity)
            + Σ_q ‖ r_q − r_q̄ ‖²  (residual continuity)
```
Reference: Jagtap & Karniadakis, *XPINNs*, Commun. Comput. Phys. 2020. Parallelises cleanly across GPUs.

### 6.6 Full reference config (JAX-PI style)

```yaml
arch:
  arch_name: ModifiedMlp          # or PirateNet
  num_layers: 6
  hidden_dim: 256
  out_dim: 3                      # u, v, p  (or psi, p)
  activation: tanh
  fourier_emb: {embed_scale: 10.0, embed_dim: 256}
  reparam: {type: weight_fact, mean: 1.0, stddev: 0.1}

optim:
  optimizer: Adam
  learning_rate: 1.0e-3
  warmup_steps: 5000
  decay_rate: 0.9
  decay_steps: 2000
  grad_accum_steps: 0

training:
  max_steps: 200000
  batch_size_per_device: 8192

weighting:
  scheme: grad_norm               # or ntk
  init_weights: {u_ic: 1.0, v_ic: 1.0, r_u: 1.0, r_v: 1.0, r_c: 1.0}
  momentum: 0.9
  update_every_steps: 1000
  use_causal: true
  causal_tol: 1.0
  num_chunks: 32

logging:
  log_every_steps: 100
  eval_every_steps: 5000
```

---

# STEP 7 — Validation: Exact Metrics and Formulas

### 7.1 Field accuracy
```
Relative L2 error:    e = ‖û − u‖₂ / ‖u‖₂
Max divergence:       D = max_x |∇·û(x)|
```
Evaluate on a dense held-out grid, never on the collocation points you trained on.

### 7.2 Drag and lift (Benchmark C)
```
C_D = (2 / (ρ Ū² D)) ∮_S ( ρν (∂u_t/∂n) n_y − p n_x ) dS
C_L = −(2 / (ρ Ū² D)) ∮_S ( ρν (∂u_t/∂n) n_x + p n_y ) dS
```
with `u_t` the tangential velocity, `D = 0.1`, `Ū = 1.0` (DFG 2D-2 mean inflow). Evaluate the contour integral by quadrature on the cylinder surface — the PINN gives you `∂u_t/∂n` analytically, which is actually cleaner than a finite-volume reconstruction.

### 7.3 Strouhal number
```
St = f · D / Ū
```
`f` = dominant peak of the FFT of the `C_L(t)` time series, after discarding the initial transient. Target for Re = 100: `St ≈ 0.16–0.17`.

### 7.4 Energy and dissipation (Benchmark D)
```
E_k(t) = (1/|Ω|) ∫_Ω  ½ u·u  dΩ
ε(t)   = −dE_k/dt                        # from the energy history
ζ(t)   = (1/|Ω|) ∫_Ω  ½ ω·ω  dΩ          # enstrophy
ε(t)   = 2ν ζ(t)                         # incompressible identity — a free consistency check
```
Plot `ε(t)` against the HiOCFD reference. The dissipation peak near `t ≈ 9` is the acceptance criterion; a PINN that smooths the small scales will under-predict that peak, which is exactly the diagnostic you want.

### 7.5 Energy spectrum
```
E(k) = ½ Σ_{|k'| ∈ [k, k+1)} |û(k')|²
```
Take the FFT of the PINN sampled on a uniform grid. Compare the inertial-range slope against `k^(−5/3)` and against DNS. **This is the honest test of whether small scales are resolved or merely smoothed away** — relative L2 alone will not reveal this.

### 7.6 Cost accounting (report this, do not hide it)

| Quantity | How to measure |
|---|---|
| Training wall-clock | seconds to reach target L2 |
| Inference throughput | query points per second, batched |
| Peak GPU memory | `nvidia-smi` during training |
| OpenFOAM equivalent | wall-clock for the same case at equivalent accuracy |

Expect to find that OpenFOAM wins on the forward solve. Say so. The contribution is in inverse problems, parametric amortisation and continuous querying — not in beating a finite-volume solver at its own game.

---

# STEP 8 — Ablation Protocol

Run each configuration on Benchmarks A–C (and H on D). Report relative L2 and iterations-to-target.

| Config | Fourier | Mod. MLP | Hard BC/div | Causal | Adaptive λ | RAD | SPINN |
|---|---|---|---|---|---|---|---|
| A — baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| B | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| C | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| D | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| E | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| G | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| H — full 3D | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Fix the seed, fix the collocation budget, fix the wall-clock budget. Changing two things at once makes the table worthless.

---

# STEP 9 — The 3D Visualisation and Animation Pipeline

The payoff. The trained PINN is a continuous function of `(x,y,z,t)`, so you can sample at any resolution and any framerate with **no interpolation** — something a CFD run saved every 100 timesteps cannot do.

### 9.1 Sample the field

```python
import numpy as np, jax.numpy as jnp, pyvista as pv

n = 256
g = np.linspace(0, 2*np.pi, n)
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")

for frame, t in enumerate(np.linspace(0, 20, 600)):        # 600 frames @ 60fps = 10 s
    pts = jnp.stack([X.ravel(), Y.ravel(), Z.ravel(),
                     jnp.full(X.size, t)], axis=-1)
    uvw = model.batched_apply(params, pts)                 # (N,3)
    grid = pv.ImageData(dimensions=(n, n, n),
                        spacing=(g[1]-g[0],)*3)
    grid["velocity"] = np.asarray(uvw)
    grid["speed"]    = np.linalg.norm(uvw, axis=1)
    grid.save(f"out/frame_{frame:04d}.vti")
```

### 9.2 Vorticity and vortex identification

Compute `ω = ∇ × u` **by automatic differentiation**, not finite differences on the grid — it is exact and free.

**Q-criterion** (the standard vortex detector):
```
∇u = S + Ω ,   S = ½(∇u + ∇uᵀ) ,   Ω = ½(∇u − ∇uᵀ)
Q  = ½( ‖Ω‖²_F − ‖S‖²_F )
```
Isosurface at `Q > 0` (typically `Q = 0.1 · Q_max`) gives the vortex tubes.

**λ₂-criterion** (alternative, for cross-checking):
```
λ₂ = second-largest eigenvalue of (S² + Ω²);  vortex core where λ₂ < 0
```

```python
grid = grid.compute_derivative(scalars="velocity", qcriterion=True)
tubes = grid.contour(isosurfaces=[0.1*grid["qcriterion"].max()],
                     scalars="qcriterion")
tubes.save(f"out/q_{frame:04d}.vtp")
```

### 9.3 Particle advection

```python
from scipy.integrate import solve_ivp

def velocity(t, y):
    p = jnp.concatenate([y.reshape(-1,3), jnp.full((y.size//3,1), t)], axis=1)
    return np.asarray(model.batched_apply(params, p)).ravel()

seeds = np.random.uniform(0, 2*np.pi, (5000, 3))
sol = solve_ivp(velocity, (0, 20), seeds.ravel(),
                method="RK45", t_eval=np.linspace(0, 20, 600), rtol=1e-6)
```
Because `u_θ` is continuous in space *and* time, the integrator needs no grid interpolation and no timestep snapping. Trajectories are smooth by construction. Colour by speed, local vorticity, or particle age.

### 9.4 Render

**PyVista (scripted, publication-grade):**
```python
pl = pv.Plotter(off_screen=True, window_size=(1920,1080))
pl.add_volume(grid, scalars="speed", cmap="plasma", opacity="sigmoid")
pl.add_mesh(tubes, color="cyan", opacity=0.6)
pl.camera.azimuth = frame * 0.5                 # slow orbit
pl.screenshot(f"render/{frame:04d}.png")
```

**Blender (cinematic):** export per-frame OpenVDB volumes, import as a volume object, use Principled Volume + emission for the speed field, Cycles render with motion blur. This is the version you put in the presentation.

**Encode:**
```bash
ffmpeg -framerate 60 -i render/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4
```

### 9.5 Interactive browser demo

```bash
python -m tf2onnx.convert --saved-model model_tf --output pinn.onnx
# or jax2tf → SavedModel → ONNX
```
Serve with `onnxruntime-web` (WebGPU backend) and render with Three.js. The user drags a time slider or a Reynolds-number slider, and the field is **re-inferred live in the browser**. Only possible because the model *is* the solution. This is the single most persuasive demo you can build from this project.

---

# STEP 10 — Execution Order and Realistic Expectations

| Week | Step | Gate before moving on |
|---|---|---|
| 1 | Env setup, run JAX-PI cavity example unmodified | Example reproduces the repo's reported error |
| 2 | Benchmark A, your own residual code | Relative L2 < 1e-4 |
| 3–4 | OpenFOAM/FEniCSx ground truth for B and C | Mesh independence < 0.5% on Cd |
| 5 | Baseline PINN on B and C | Document the failures — this is data, not defeat |
| 6–7 | Fourier + Modified MLP + RWF | Cavity Re=1000 within 2% of Ghia |
| 8–9 | Hard constraints (stream function, distance BC) | Max divergence < 1e-10 |
| 10–11 | Causal weighting + adaptive λ + RAD | Cylinder Re=100: St within 3%, Cd within 5% |
| 12–15 | SPINN 3D, Benchmark D | Dissipation peak within 5% of HiOCFD |
| 16–18 | PI-DeepONet parametric | Zero-shot error at unseen Re < 10% |
| 19–20 | Ablation sweep | Complete table A–H |
| 21–23 | Visualisation pipeline and animations | All six visual deliverables rendered |
| 24–25 | Report, code release | Reproducible from a fresh clone |

### Honest expectations

- Benchmark A: easy. Benchmark B: manageable. **Benchmark C at Re ≥ 200 is where most projects stall** — causal weighting plus curriculum is what gets you through.
- 3D will consume more time than you plan. Start SPINN early.
- Your PINN will likely be slower than OpenFOAM on the forward solve. Report it, and lead with the inverse problem, the parametric surrogate and the continuous-field animation instead. That framing is both true and stronger.

---

# Resource Index

**Code**
- JAX-PI — `https://github.com/PredictiveIntelligenceLab/jaxpi` (PirateNet + SOAP: branch `pirate`)
- SPINN — `https://github.com/stnamjef/SPINN` · `https://jwcho5576.github.io/spinn.github.io/`
- DeepXDE — `https://github.com/lululxvi/deepxde`
- NVIDIA PhysicsNeMo — `https://github.com/NVIDIA/physicsnemo` · `https://github.com/NVIDIA/physicsnemo-sym`
- Original PINNs — `https://github.com/maziarraissi/PINNs`
- Hidden Fluid Mechanics — `https://maziarraissi.github.io/HFM/`
- CausalPINNs data — `https://github.com/PredictiveIntelligenceLab/CausalPINNs/tree/main/data`

**Data**
- PDEBench — `https://github.com/pdebench/PDEBench` · DOI `10.18419/darus-2986`
- The Well — `https://github.com/PolymathicAI/the_well` · `https://polymathic-ai.org/the_well/`
- JHTDB — `https://turbulence.pha.jhu.edu/` (`pip install pyJHTDB`, token required)
- Cylinder wake — `https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat`
- DFG benchmark — `http://www.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf`
- TGV Re=1600 — `https://how4.cenaero.be/content/bs1-dns-taylor-green-vortex-re1600` · `https://how5.cenaero.be/content/ws1-dns-taylor-green-vortex-re1600` · `https://ntrs.nasa.gov/api/citations/20130011044/downloads/20130011044.pdf`

**Papers**
1. Raissi, Perdikaris & Karniadakis (2019) — PINNs. *J. Comput. Phys.* 378.
2. Raissi, Yazdani & Karniadakis (2020) — Hidden Fluid Mechanics. *Science* 367.
3. Jin et al. (2021) — NSFnets. *J. Comput. Phys.* 426.
4. Wang, Teng & Perdikaris (2021) — Gradient flow pathologies. *SIAM J. Sci. Comput.*
5. Wang, Yu & Perdikaris (2022) — NTK perspective on PINN failure.
6. Wang, Sankaran & Perdikaris (2022) — Respecting causality.
7. Krishnapriyan et al. (2021) — Failure modes in PINNs. *NeurIPS*.
8. Jagtap & Karniadakis (2020) — XPINNs. *Commun. Comput. Phys.*
9. Cho et al. (2023) — Separable PINNs. *NeurIPS* (Spotlight).
10. Wang et al. (2024) — PirateNets. arXiv:2402.00326.
11. Wang, Sankaran, Wang & Perdikaris (2023) — An Expert's Guide to Training PINNs.
12. Wu et al. (2023) — Adaptive sampling for PINNs. *CMAME*.
13. Tancik et al. (2020) — Fourier features. *NeurIPS*.
14. Lu et al. (2021) — DeepONet. *Nature Machine Intelligence*.
15. Li et al. (2021) — Fourier Neural Operator. *ICLR*.
16. Ghia, Ghia & Shin (1982) — Lid-driven cavity benchmark. *J. Comput. Phys.* 48.
17. Schäfer & Turek (1996) — Laminar flow around a cylinder benchmark. DFG.
