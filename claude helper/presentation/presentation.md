# PINNs for incompressible flow: the presentation

Presenters, in order: **Atharva Palande** (Taylor-Green 2D and 3D), **Ishwari** (the cavity and PI-DeepONet),
**Mantavya** (the cylinder, forward and inverse) and **Mehul** (blood pressure from MRI). Everyone reads part 0
first. All numbers are from `docs/STATUS.md`.

---

## Part 0. Presentation: who explains what, and the basics everyone needs

Every presenter reads this part first. It covers what a PINN is, the training toolkit the benchmarks refer
to, and the lessons that come up in more than one benchmark. All numbers are from `docs/STATUS.md`.

### The split and the running order

| order | presenter | part | benchmarks | verdict |
|---|---|---|---|---|
| 1 | Atharva Palande | the exact-solution flows | A: Taylor-Green 2D · D: Taylor-Green 3D | passed · failed |
| 2 | Ishwari | the cavity | B: lid-driven cavity (soft lid, hard lid) · PI-DeepONet (any lid speed) | gate missed, failed · passed |
| 3 | Mantavya | the cylinder | C: DFG cylinder (forward) · the inverse wake | failed · passed |
| 4 | Mehul | the real world | blood pressure from MRI: synthetic stenosis, real phantom | passed · in progress |

Why this order: it climbs the handbook's ladder from the easiest test to the hardest. Part 3 ends with an
*inverse* problem that recovers a hidden pressure from velocity data, which is exactly what the MRI models do, so it
hands straight over to part 4.

Each part has: purpose, setup, how we trained it, results against the gate, why it passed or failed, what to show in
the live viewer, and likely questions with answers.

### What a PINN is (one slide)

A **physics-informed neural network** is a network that takes a point in space and time and returns the flow there:

```
N_theta : (t, x, y[, z])  ->  (u, v[, w], p)        theta = the network's weights
```

The network is a smooth function, so its derivatives (du/dt, grad u, laplacian u, grad p) can be computed exactly by
automatic differentiation. That lets the training loss contain the physics itself: the incompressible
**Navier-Stokes equations**, Newton's law F = m a for a fluid:

```
du/dt + (u . grad) u = -grad p + (1/Re) laplacian u        (momentum)
div u = 0                                                   (mass conservation)
Re = U L / nu                                               (Reynolds number: inertia against friction)
```

Training minimises a weighted sum of mean squared errors:

```
L = lambda_r * mean|momentum residual|^2 + lambda_c * mean|div u|^2
  + lambda_bc * mean|boundary error|^2 + lambda_ic * mean|initial-condition error|^2 [+ lambda_d * mean|data error|^2]
```

- In a **forward** problem (most benchmarks) there is no data: only the equations, the boundary and initial conditions.
  The reference solution is used only to grade the result.
- In an **inverse** problem (the wake, the MRI models) measured data enters the loss, and the network infers what was
  not measured (a coefficient, the pressure).

### The training toolkit (the "ablation rows")

The handbook defines eight configurations, rows A to H. Each adds one technique to the one before:

| row | adds | what the technique does |
|---|---|---|
| A | plain MLP | the baseline network |
| B | **Fourier features** | feeds the inputs through cos(Bx), sin(Bx) with random B ~ N(0, sigma^2). Plain networks learn smooth shapes first and sharp ones very slowly ("spectral bias"); the waves fix that |
| C | **Modified MLP + weight factorisation** | a deeper-gated network architecture (from JAX-PI) that trains more stably |
| D | **hard constraints** | builds boundary conditions or div u = 0 into the network's output exactly, e.g. a stream function psi with u = psi_y, v = -psi_x, which is divergence-free by construction |
| E | **causal weights** | for time-dependent problems: later times are only trained once earlier times are right (weights exp(-eps * accumulated earlier loss), eps annealed 0.01 -> 100) |
| F | **grad-norm adaptive weights** | every 1000 steps, rebalances the lambdas so each loss term pulls on the weights equally hard |
| G | **RAD** | every 1000 steps, redraws the collocation points where the residual is largest |
| H | **SPINN** | a separable network for 3D: one small network per axis, combined, evaluated on a whole grid at once |

Common settings: Adam with 5k warm-up steps then x0.9 every 2k steps, peak learning rate 1e-3, highest matmul
precision, seed 42; an optional second stage of **L-BFGS** (a quasi-Newton optimiser) on one fixed batch.

### Budget and hardware

- Desktop RTX 5070 Ti 16 GB (training), laptop RTX 3060 6 GB (development). JAX on GPU runs inside WSL2 (Linux).
- The handbook's full budgets (cylinder 16 x 200k steps, SPINN 300k, the full ablation) need about 150-160 GPU hours.
  On request everything was finished in a **12-hour window** with reduced budgets, stated per run. The whole study
  used **30.5 GPU hours**.
- A 60-test suite checks the physics before any training: exact Taylor-Green solutions must give residuals below
  1e-10 in double precision (`tests/test_physics.py`, the handbook's STEP 2.1 gate).

### Lessons that recur across benchmarks (good for Q&A)

1. **Incompatible constraints break PINNs.** The exact cavity lid contradicts mass conservation in two corners; the
   impulsive cylinder start contradicts incompressibility in time. The network pays with a residual floor it can
   never remove. Fix the problem setup, not the optimiser.
2. **Adaptive weighting can backfire.** Grad-norm gives terms with tiny gradients enormous weights: the cavity lid
   weight reached ~465, the cylinder outflow ~2.4e5, the 3D initial w-velocity 6e9. Fixed weights often did better.
3. **L-BFGS on a fixed batch overfits.** It can lower the loss by 1000x while the true error triples.
4. **Small errors everywhere can add up to a large global error.** A residual that is tiny at each point but points
   the same way integrates to a big mistake: lost mass along the cylinder channel, missing pressure drop in the MRI
   phantom. A mean-squared loss cannot see it.
5. **Inverse problems are easier than forward ones.** Data pins the flow down; the physics only has to supply the
   few unknowns. That is why the wake (and the MRI work) succeed where forward solves of similar flows fail.

### The live viewer

`python -m viewer`: three tabs (benchmarks, the synthetic MRI study, the real MRI phantom). Keys 1-8 pick a scene,
V cycles the views (pigment, speed, vorticity, pressure, divergence, error), Space plays time, click drops pigment.
Every frame is computed by the trained network on the GPU.

---

## Part 1. Atharva Palande: the Taylor-Green vortex, in 2D (passed) and 3D (failed)

Read part 0 (shared basics) first. Numbers are from `docs/STATUS.md`.

**One-line story:** the same family of flows is the easiest test in the project and the hardest. In 2D, where an exact
solution exists, the network matches it to 4 parts in 100,000. In 3D, where the flow turns turbulent, it never does:
it keeps the flow smooth and two-dimensional.

---

### 1a. Benchmark A, Taylor-Green 2D (passed)

#### Purpose

The handbook's first gate: *"If you cannot hit relative L2 < 1e-4 here, every later result is meaningless."* It has
an exact solution, so it checks that our physics code (the residuals) and our training loop are correct before
anything harder is attempted.

#### Setup

- Domain [0, 2 pi]^2, periodic in x and y, time t in [0, 1], Re = 100.
- An array of counter-rotating vortices that decays in time. The exact solution:

```
u = -cos(x) sin(y) exp(-2t/Re)
v =  sin(x) cos(y) exp(-2t/Re)
p = -(1/4) (cos 2x + cos 2y) exp(-4t/Re)
```

- Only the initial condition (t = 0) and the equations are given to the network; the exact solution grades it.
- **Gate:** relative L2 error of the velocity below 1e-4.

#### How we trained it

- **Row G** (everything except SPINN): Fourier features (sigma 1), Modified MLP 6 x 256 with tanh and weight
  factorisation, causal weights, grad-norm weighting, RAD.
- **Stream-function formulation:** the network outputs (psi, p) with u = psi_y, v = -psi_x, so div u = 0 exactly.
- **Exact periodicity:** x and y enter through cos and sin of themselves, so the network is periodic by construction.
- 8,192 residual points and 2,048 initial-condition points per step; 50,000 Adam steps, then L-BFGS (stopped after
  1,200 iterations). RTX 5070 Ti, 0.14 s per step, about 2 hours.

#### Results

| quantity | result | gate |
|---|---|---|
| velocity, relative L2 | **4.05e-5** (u 4.17e-5, v 3.93e-5) | < 1e-4: **passed** |
| pressure, relative L2 | 2.59e-4 (after aligning the pressure's constant per time step) | |
| max divergence | 0 (exact, by the stream function) | |

**Why the pressure needs aligning:** in incompressible flow only pressure *differences* matter, so the network's
pressure is unique only up to a constant at each time. Before comparing, the mean is matched per time slice.

#### The ablation on this benchmark (rows A-G at 3,000 steps each)

| row | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| velocity error | 8.7e-3 | 2.25e-2 | 3.2e-2 | 2.18e-2 | 1.80e-2 | 1.71e-2 | 1.17e-2 |
| step time | 0.014 s | | | | | | 0.121 s |

- At only 3k steps the **plain network (row A) is the most accurate**: the extra machinery costs more steps before it
  pays off. Row G needed 50k steps to reach 4e-5.
- Rows A-C (velocity-pressure output) leave a divergence of 1.4e-2 to 7.7e-2; rows D-G (stream function) have exactly 0.
- Row G is ~9x slower per step (third derivatives of the stream function, plus the adaptive updates).

**Takeaway:** the physics code is right to 4e-5. Everything later builds on it.

#### In the viewer

Scene 1, "Taylor-Green 2D". Pigment in vertical bands is wound up by the vortices. The error view shows the
difference from the exact solution (tiny). The chart shows the error staying below the gate over the whole time window.

---

### 1b. Benchmark D, Taylor-Green 3D at Re = 1600 (failed)

#### Purpose

The standard 3D test used by the International Workshops on High-Order CFD Methods. The flow starts as smooth
vortices, then **turns turbulent around t = 8-9** (vortex stretching breaks them into ever smaller eddies), then
decays. It tests whether a PINN can capture turbulence transition, the hardest thing in the handbook.

#### Setup

- Domain [0, 2 pi]^3, periodic in all directions, t in [0, 20], Re = 1600.
- Initial condition: u = sin x cos y cos z, v = -cos x sin y cos z, w = 0, with the matching pressure.
- No exact solution. The reference is a direct numerical simulation (DNS): the kinetic-energy dissipation rate
  -dE/dt over time, digitised from DeBonis (2013, NASA), which **peaks at 0.01282 at t = 9.0**.
- **Gate:** the dissipation peak within 5% of the reference.

#### How we trained it

- **Row H: SPINN** (separable PINN). One small network per axis (t, x, y, z), each 4 layers x 64, combined as a
  rank-128 sum of products. Because the network separates, it can be evaluated and differentiated on a whole grid at
  once: 32^4 = about 1 million space-time points per step.
- **Vector-potential formulation:** u = curl A, so div u = 0 exactly, with a weak gauge term. Periodic features in x,
  y, z.
- Budget: **80k Adam + 1k L-BFGS** (planned 300k + 20k), grid **32^4** (the handbook's 64^4 did not fit: it needs
  one 11.1 GB buffer on a 16 GB card). 0.037 s per step, 74 minutes, 3.1 GB.

#### Results

| quantity | network | reference | |
|---|---|---|---|
| initial energy E_k(0) | 0.1246 | 0.125 | fitted |
| peak of -dE/dt | **0.0292 at t = 0** | 0.01282 at t = 9 | 127% off, at the wrong time |
| peak of 2 nu x enstrophy (dissipation by friction) | 4.9e-4 | 0.0128 | 96% off |
| energy spectrum slope (k = 4-16) | -3.96 at t = 0, -3.50 at t = 9 | -5/3 for turbulence | no turbulent cascade |

**Gate failed: no transition.**

#### Why it failed

1. **The network keeps the flow two-dimensional.** The vertical velocity w stays at about 0 (its loss terms reach
   1e-9 to 1e-15). Without w there is no vortex stretching, so enstrophy never grows and no small eddies form. The
   flow just decays smoothly from t = 0.
2. **Energy leaves through the equation error, not through friction.** The two ways of measuring dissipation (-dE/dt,
   and 2 nu times enstrophy) should agree; here they differ by a factor of ~60. The momentum residual (mean square
   2.5-7e-4) removes about as much energy as the physical dissipation should: the network "cheats" the energy
   balance instead of producing turbulence.
3. **Grad-norm weighting broke down.** It raised the weight of the w initial condition (target identically zero) to
   **6e9** and of the w equation to 2.6e3. Every term gets the same share of gradient whether it needs it or not.
   Same pathology as on the cavity and the cylinder.
4. **Resolution and budget.** 32 points per 2 pi cannot hold the small scales of Re = 1600 turbulence, and 80k steps
   is a quarter of the plan. Causal training's eps reached 10, never 100.

**Decision:** reaching the transition was not plausible within the budget; no rerun. (Suggested fix noted: keep
terms with a zero target out of adaptive weighting.)

**Takeaway:** PINNs prefer smooth solutions (spectral bias). A flow whose whole point is the breakdown into small
eddies is exactly what they resist. This matches the published experience that PINNs struggle with turbulence.

#### In the viewer

Scene 6, "Taylor-Green 3D": a 3D tank with pigment sheets carried by the network, and vortex tubes (Q-criterion). The
chart shows the network's dissipation (largest at t = 0, decaying) against the DNS curve (peaking at t = 9).

---

### Likely questions

- **Why is 2D so easy and 3D so hard?** 2D Taylor-Green stays laminar and smooth (it just decays). 3D at Re 1600 turns
  turbulent, which needs fine scales and vortex stretching that the network never develops.
- **Why didn't you use 64^4?** It does not fit in 16 GB (one 11.1 GB buffer). 32^4 trained at 3.1 GB.
- **Is the 4e-5 just luck?** No: the residual code is checked against exact solutions to 1e-10 in the test suite, and
  the stream function makes divergence exactly zero.
- **Why does the plain network win the 3k-step ablation?** Fewer things to learn; the advanced rows pay off only with
  longer training (row G reached 4e-5 at 50k).
- **What would you try next on 3D?** Exclude zero-target terms from grad-norm, use the full 64^4 grid and 300k steps on
  a larger GPU, and check whether w can be kicked out of zero (a small perturbation).

---

## Part 2. Ishwari: the lid-driven cavity (soft lid, hard lid) and PI-DeepONet

Read part 0 (shared basics) first. Numbers are from `docs/STATUS.md` (diagnoses D1-D3, D8-D12).

**One-line story:** the classic CFD test. With the lid as a soft constraint, the network matches an independent
high-resolution solution to 0.27%, yet misses the Ghia gate by 0.05 points because the gate is tighter than Ghia's
own data. Forcing the lid exactly breaks the physics in two corners and the whole vortex comes out 20% too weak. A
second network, PI-DeepONet, learns the cavity for every lid speed at once.

---

### 2a. Benchmark B, the lid-driven cavity

#### Purpose

A square box of fluid whose lid slides sideways, driving one large vortex plus smaller corner vortices. It is *the*
standard steady-flow validation case. It tests boundary conditions, steady solutions and a Reynolds-number curriculum.

#### Setup

- Domain [0, 1]^2, no-slip walls, the lid at y = 1 moving in +x.
- **Regularised lid** (removes the corner singularity of a plain moving lid):

```
u(x, 1) = 1 - cosh(r (x - 0.5)) / cosh(0.5 r),   r = 50,     v(x, 1) = 0
```

- Target Re = 1000, reached by a curriculum Re = 100 -> 400 -> 1000.
- **Reference:** Ghia, Ghia & Shin (1982) tables of u along the vertical centreline and v along the horizontal one.
- **Gate:** centreline errors below 2% at Re = 1000.
- We also wrote our own **finite-difference solver** (stream function-vorticity, up to 512 x 512 grid) for the exact
  same regularised-lid problem, as an independent reference.

#### How we trained it

- Velocity-pressure output, Modified MLP 6 x 256, Fourier features (sigma 10), 8,192 residual points per step.
- Curriculum: 20k steps at Re 100, 40k at 400, 140k at 1000 (Adam restarted per stage, weights kept), then 20k
  L-BFGS. About 2.3 hours per run, 0.025 s per step, 1.6 GB.
- Compared rows: **C** (soft boundary conditions, fixed weights), **D** (hard boundary conditions, fixed weights),
  **F** (hard boundary conditions + grad-norm), and soft-lid versions of D and F.

#### Results at Re = 1000

| run | lid | weights | Ghia u / v error | error vs our 512^2 solution | |
|---|---|---|---|---|---|
| **row C** (viewer: "Cavity, soft lid") | soft | fixed | **0.63% / 2.05%** | **0.27%** | gate missed by 0.05 points |
| row D, soft lid | soft (walls exact) | fixed | 1.04% / 2.20% | 1.9% | end of Adam |
| row F, soft lid | soft | grad-norm | 16.0% / 15.6% | 21.0% | |
| row D | **hard** | fixed | 21.3% / 19.8% | 28.3% | |
| **row F** (viewer: "Cavity, hard lid") | **hard** | grad-norm | **19.8% / 20.2%** | 26.1% | failed |

#### Why row C "misses" the gate: the gate is tighter than Ghia's data

- Our finite-difference solution converges with grid refinement (N = 128, 256, 512) to the spectral reference value of
  Botella & Peyret (1998), **not** to Ghia's.
- A fully converged solution of this exact benchmark scores **u 0.41% / v 2.29%** against Ghia's tables. Ghia's v
  values next to the right wall are slightly off. So *an exact solution misses the 2% v gate*.
- Row C is 0.27% from our 512^2 solution: below the discretisation error of a 256^2 finite-difference solve. It is
  essentially exact; the 2.05% is Ghia's error, not the network's.

#### Why the hard lid fails: an impossible boundary condition

- Forcing u = u_lid(x) exactly on the lid, together with u = v = 0 exactly on the side walls, makes **mass
  conservation impossible in the two top corners**: the lid profile is steep there (slope +/-50), the walls force
  v_y = 0, so div u = 0 would need u_x = 0, but the lid says otherwise.
- No smooth field satisfies both. The network ends with div u about 10 in two corner patches, a continuity floor of
  ~0.04 that never goes away, and the main vortex comes out too weak (speed error 10-30% across the core).
- Row D (hard lid, *fixed* weights) fails the same way, so **the hard lid is the cause, not the weighting**.
- The soft lid (exact no-slip on the three fixed walls and v = 0 on the lid, but the lid's u only as a loss term) lets
  the network trade a tiny lid error in the corners for correct physics.

#### Two more findings

- **Grad-norm undoes the soft lid:** in row F with the soft lid, the adaptive weighting raised the lid weight to ~465
  (making it hard again) and lowered the continuity weight to 0.21. Fixed weights did better.
- **L-BFGS overfits:** on the soft-lid row D, L-BFGS cut the loss 1000x (2.6e-3 -> 2.3e-6) on its fixed batch of points
  while the field error **tripled** (1.9% -> 6.7%). We report both, and dropped L-BFGS for later work.
- The short ablation (16k curriculum steps) was uninformative: every row ended 81-118% off because 1.6k steps at Re 100
  is too short. The comparison above uses the full 200k budget.

#### In the viewer

- Scene 2 "Cavity, soft lid" and scene 3 "Cavity, hard lid", with a Reynolds selector (100 / 400 / 1000).
- On the hard lid, **open the divergence view**: the two top corners light up. That is the incompatibility.
- The chart shows the centreline profiles against Ghia's points.

---

### 2b. PI-DeepONet, the cavity at any lid speed (passed)

#### Purpose

A normal PINN solves one problem: one Reynolds number. To change the lid speed you retrain for hours. A
**physics-informed DeepONet** learns the whole *family* of solutions: give it a Reynolds number and it returns that
flow in one forward pass. The handbook's test: can it predict a Reynolds number it never saw?

#### Setup

- Same cavity (soft lid), Re anywhere in [100, 1000].
- **No data at all:** trained on the equations only.
- **Hold-out band:** Re in [300, 530] is never sampled in training; **Re = 400 is the zero-shot query**.
- **Gate:** velocity error below 10% at Re = 400, graded against our finite-difference solution.

#### How it works

```
branch net(Re)  -> 128 numbers  b_k
trunk  net(x,y) -> 128 numbers  t_k              (u, v, p)(x, y; Re) = sum_k b_k t_k
```

Branch: 3 layers; trunk: 6 x 256 with Fourier features (sigma 10); fixed loss weights; soft lid.

#### The training journey (worth telling: it shows how we debugged)

| run | change | Re = 400 error |
|---|---|---|
| first run | Re up to 1000 from step 0 | 10.5% at Re 100 after 5k steps, then the loss spiked: **70-87%** at the end |
| + curriculum | widen the Re range 200 -> 500 -> 1000 | diverged again at peak learning rate 1e-3 (the cause was the rate, not the range) |
| + lower rate | peak lr 3e-4, gradient clipping at 1.0, 200k steps | 14.0% |
| continuations | 150k, 150k (slower decay), 100k (constant lr 1.5e-5) | 11.1% -> 10.27% -> 10.11% (plateau) |
| + bigger batch | 8 Re values x 8,192 points per step (was 4 x 4,096), lr 1e-5, 60k steps | **9.65%: passed** |

- The plateau at ~10% was gradient noise; doubling the Reynolds values and the points per step broke it.
- Total: **660k Adam steps**, three times the plan. Reported as such.
- Other Reynolds numbers: Re 100 2.06%, **Re 1000 30.0%**: it is weak at the high end, where the flow is hardest.

#### In the viewer

Scene 7 "Cavity, any lid": drag the lid-speed slider. Every position is one forward pass of one network. The hatched
band on the slider was never seen in training; Re 400 sits inside it.

---

### Likely questions

- **So did the cavity pass or fail?** Officially failed against the Ghia gate (2.05% vs 2%), which the team agreed to
  keep. Scientifically, row C is within 0.27% of a converged solution of the same problem, and that converged solution
  itself misses the gate.
- **Why not just use the hard lid, it's "exact"?** Because it contradicts mass conservation in the corners; exactness
  of a boundary condition does not help if the conditions are mutually incompatible.
- **Why does the DeepONet fail at Re 1000?** The highest Re has the thinnest boundary layers and strongest corner
  vortices; it is at the edge of the sampled range. 30% there, 2% at Re 100.
- **What is the DeepONet good for?** Instant answers for any parameter: design sweeps, sliders, real-time use.
- **How long did the cavity take?** About 2.3 hours per run on the RTX 5070 Ti; the finite-difference reference at
  512^2 took about 1 hour on the CPU.

---

## Part 3. Mantavya: flow past a cylinder, forward (failed) and inverse (passed)

Read part 0 (shared basics) first. Numbers are from `docs/STATUS.md` (diagnoses D4-D6, D9).

**One-line story:** the same kind of flow, two ways. Asked to *predict* the flow past a cylinder from the equations
alone, the network never makes the vortices shed and loses most of the fluid along the channel. Given scattered
velocity measurements instead, it recovers the fluid's viscosity to 0.065% and a pressure field nobody measured. That
second idea is exactly what the MRI blood-pressure work does next.

---

### 3a. Benchmark C, the DFG 2D-2 cylinder (forward solve, failed)

#### Purpose

The handbook calls it "the real test": unsteady flow with **vortex shedding** (the von Karman street). It is the
Schafer & Turek (1996) benchmark used by FEATFLOW, FEniCS and deal.II, so results are directly comparable to
published values. The handbook warns: "Benchmark C is where most projects stall."

#### Setup

- A channel [0, 2.2] x [0, 0.41] m with a cylinder of radius 0.05 at (0.2, 0.2). Fluid nu = 0.001, rho = 1.
- Parabolic inflow, peak 1.5 m/s (mean 1.0): **Re = 100** (mean speed x diameter / nu).
- No-slip on the walls and the cylinder; "do-nothing" outflow.
- **Reference (FEATFLOW):** Strouhal number St = 0.30 (shedding frequency), peak drag coefficient C_D = 3.22.
- **Gate:** St within 3%, C_D max within 5%.

Note: the handbook quotes St ~ 0.16-0.17. That is the *unconfined* cylinder; in this channel the published interval
is 0.295-0.305 (a documented deviation in README section 7).

#### How we trained it

- **Time marching:** the 8 seconds are split into windows of 0.5 s, trained one after another, each starting from the
  previous one's end.
- **Causal weights** inside each window (later times trained only once earlier ones are right).
- **Hard boundary conditions** through a distance function phi: u = g + phi N, so no-slip is exact (a bounded variant
  of the handbook's polynomial, which grows to ~10^4 downstream). Only the outflow is a loss term.
- Modified MLP 6 x 256, gelu, Fourier features (sigma 1), 16,384 residual points per step.
- **Budget:** the plan is 16 windows x 200k steps (about 80 GPU hours at the measured 0.089 s per step). Within the 12-hour window: **4 windows x 12k
  steps**, and a 2 x 6k stream-function run. Each window costs ~7 minutes of compilation alone.

#### Results

| run | what | best C_D max | mass reaching the outlet | shedding |
|---|---|---|---|---|
| row G, 16 x 4k (stopped) | as specified | 0.45 | | no |
| row G, impulsive start, 12k | | 0.32 | | no |
| row G, ramped inflow, 12k | fix for the start | 0.10 | **2%** (12% at x = 0.6) | no |
| stream function, 2 x 6k | exact div u = 0 | **1.54** | 75-77% | no |
| FEATFLOW reference | | **3.22** | 100% | St 0.30 |

**Gate failed: no shedding in any run.**

#### Why it failed: a chain of three diagnoses

1. **Too few steps per window (D4).** At 4k steps per window the drag was 0.3-0.45 against ~3.2: the boundary layer
   and wake were not resolved. Grad-norm put the outflow weight at 7,000-9,000 because that term's gradient is tiny.
2. **The impulsive start is impossible (D5).** Switching the full inflow on at t = 0 into fluid at rest demands that
   the whole channel starts moving instantly (incompressibility). The network instead advanced the inflow like a slow
   front: at t = 0.5 s only 0.008 of the 0.41 m^2/s inflow reached the outlet. **Fix:** ramp the inflow in smoothly
   with sin^2(pi t / 2) over the first second, so rest is a consistent starting state.
3. **Mass leaks through tiny errors (D6).** With the ramp, the residuals fell 20-80x and training looked healthy, but
   only 12% of the inflow crossed x = 0.6 and 2% reached the outlet. The network keeps the downstream fluid almost
   still and pays with a small divergence (~0.03) at every point: small locally, huge once added up over a channel 22
   diameters long. The mean-squared loss cannot see that.
4. **Stream function (D9):** making div u = 0 exact (u = psi_y, v = -psi_x) raised the flow reaching downstream to
   75-77% and C_D max to 1.54. The rest leaks through the walls, which are now only soft constraints. Shedding was
   still far away.

**Takeaway:** a forward PINN must *discover* the shedding instability on its own, from equations only, and the
optimiser prefers a smooth near-steady flow. Small pointwise errors that add up globally (here: lost mass) are a
recurring PINN failure; the MRI phantom showed the same pattern with pressure.

#### In the viewer

Scene 5 "Cylinder, forward". Pigment released upstream shows no vortex street. The flow-rate chart shows the bars that
should all reach the line (all mass leaving) and don't.

---

### 3b. The inverse problem, the cylinder wake from data (passed)

#### Purpose

The classic PINN demonstration (Raissi et al. 2019): given **sparse velocity measurements** of a flow, recover the
unknown physical coefficients *and* the pressure field, which was never measured. It tests the inverse mode, data
plus physics, the mode real measurements need.

#### Setup

- Data: a spectral-element (Nektar) simulation of the wake behind a cylinder at Re = 100 (diameter 1, U = 1,
  nu = 0.01), in a window of the wake with shedding.
- The network sees **5,000 scattered velocity samples** (u, v) over time. No pressure.
- The equations carry two unknown numbers the network must learn:

```
du/dt + lambda_1 (u . grad) u = -grad p + lambda_2 laplacian u        true values: lambda_1 = 1, lambda_2 = 0.01
```

- **Grading:** lambda_1, lambda_2, and the hidden pressure against the simulation (20 snapshots, 100k points).

#### How we trained it

- Stream-function output (psi, p): divergence-free by construction.
- 8 layers x 128, Fourier features (sigma 2); 2,048 data points and 4,096 residual points per step.
- 25k Adam + 1k L-BFGS (planned 100k + 20k). 0.055 s per step, **31 minutes**.

#### Results

| quantity | learned | true | error |
|---|---|---|---|
| lambda_1 (convection) | 0.99997 | 1 | **0.004%** |
| lambda_2 (viscosity) | 0.010006 | 0.01 | **0.065%** |
| pressure (never measured) | | | **2.44%** |
| velocity u / v | | | 0.54% / 2.69% |

**Passed.**

#### Why the inverse works when the forward fails

- In the forward problem nothing tells the network the wake must shed; it has to find the instability itself, and
  settles on a smooth flow.
- In the inverse problem the data already shows the shedding wake. The physics only has to supply two numbers and the
  pressure, and those are tightly fixed by Newton's law once the velocity is known.
- **This is the bridge to the next part:** an MRI scan gives velocity, never pressure. The blood-pressure models are
  this same inverse idea on real medical data.

#### In the viewer

Scene 4 "Wake from data". The "sensors" view shows the scattered measurement points; the pressure view shows the
recovered pressure, which never entered training. The rack shows the learned coefficients against the true ones.

---

### Likely questions

- **Why 4 windows x 12k when the plan was 16 x 200k?** The plan needs about 80 GPU hours for the cylinder alone (0.089 s per step x 3.2 million steps); the study
  had a 12-hour window. The diagnoses showed more steps alone would not have produced shedding.
- **Isn't 1.54 vs 3.22 close-ish?** No: drag that low means the flow around the cylinder is not developed, and without
  shedding there is no St at all.
- **Why the stream function only at the end?** It is three times more expensive per step (third derivatives) and makes
  the walls soft; it was the last experiment once the mass leak was diagnosed.
- **How can it find the viscosity to 0.065% from 5,000 points?** Viscosity changes how fast the wake's vortices
  diffuse; with velocity known in space and time, only one value of lambda_2 makes Newton's law balance everywhere.
- **Could the forward solve be fixed?** Ideas: much longer windows, exact walls with a stream function, a perturbation
  to trigger shedding, or a classical solver for the forward problem and PINNs for the inverse one.

---

## Part 4. Mehul: blood pressure from MRI (synthetic stenosis and the real phantom)

Part 0 (shared basics) has the toolkit. Numbers are from `docs/STATUS.md`, sections "Real-world use case"
and "Real data".

**One-line story:** the inverse idea from the cylinder wake, applied to medicine. An MRI scan measures blood velocity
but never pressure; the network infers the pressure from the velocity and Newton's law. On simulated arteries it beats
the clinical rule by a wide margin. On a real 3D-printed aorta scanned by a real MRI, it is not there yet, and we found
exactly why and how far each fix gets.

---

### 1. The clinical problem

- A **coarctation** (narrowing of the aorta) makes the heart push blood through a tight spot. Its severity is judged by
  the **pressure drop** across it.
- **Catheter:** the gold standard, a sensor threaded into the artery. Invasive, costly, risky.
- **Doppler ultrasound + simplified Bernoulli:** routine clinical estimate, dP [mmHg] = 4 v^2 (v = jet peak speed in
  m/s). Crude; off by 30% or more.
- **4D flow MRI:** measures the full 3D velocity in the vessel through an averaged heartbeat. Velocity only.
- **Goal:** pressure from the MRI alone, non-invasive, more accurate than Bernoulli.

Physics: rearranging Navier-Stokes, the pressure gradient follows from the velocity:

```
grad p = -rho ( du/dt + (u . grad) u - nu laplacian u )
dP = integral along the vessel of that, from the upstream to the downstream catheter position
```

---

### 2. Synthetic stenosis (passed, twice)

- **Data:** a 6 mm artery with a 60% narrowing, Re 150, 73 bpm. The truth from our own CFD solver (MAC projection, 50
  cells per diameter; reproduces Poiseuille to 0.6%). A simulated scan of it: 0.6 mm voxels, 24 phases, noise at SNR
  20, voxel and phase averaging.
- **Network:** 2D, periodic over the heartbeat, voxel-averaged data term (the network is averaged over each voxel and
  acquisition window, as the scanner does).
- **Key fix:** with soft walls the network learned almost no pressure (7e-4 mmHg). Exact walls via a stream function,
  psi = Q(t)(3 eta - eta^3)/4 + (1 - eta^2)^2 N with eta = y / h(x), plus a learned flow rate Q(t) and a pressure
  scale, fixed it.
- **No L-BFGS:** it fitted the scan's noise (loss far below the noise floor) and made the pressure worse.

| case | network peak-drop error | Bernoulli error |
|---|---|---|
| stenosis (Re 150) | **6.2%** (0.557 vs 0.594 mmHg) | +31% |
| fast jet (Re 300, 70% narrowing) | **4.1%** (3.96 vs 4.13 mmHg) | +21% |

---

### 3. The real phantom: Stanford 4D-FlowP coarctation phantom

- 3D-printed patient aortas with a coarctation, in a pulsing flow loop, scanned by a real Siemens MRI (1.8 mm voxels,
  13-17 frames per beat), with **catheter pressure recorded at 1 kHz**. Fluid: glycerol-water, 1.1 g/cm^3, 4.2 mPa s.
- **8 cases:** 2 patients (P1, P2) x baseline/severe narrowing x normal/low flow. Catheter peaks 7.2-24.7 mmHg.
- Test case: **P2 baseline, catheter 13.1 mmHg.**

#### Data problems we found and fixed

1. **Eddy-current offsets:** 1-2 cm/s on static tissue; subtract a fitted 2nd-order polynomial.
2. **Segmentation:** one case found the gel instead of the aorta; segment by flow speed after correction.
3. **Wrong velocity signs:** two of three velocity components had flipped signs in the export (the scan's own
   streamlines left the vessel). Chosen by the smallest divergence of the flow: mass must be conserved.
4. **Flow-meter check:** the scan's flow through the upstream plane matches the meter within ~8%.
5. **Clock offset:** the catheter's beat starts at a flow upstroke, the scan's at the MRI trigger: 121-147 ms apart,
   measured by matching the meter's flow to the scan's flow (correlation 0.97-1.00).

#### Baselines from the same scans (all 8 cases)

| method | slope vs catheter | mean error |
|---|---|---|
| Bernoulli 4 v^2 | 0.77 | 30%, always low |
| pressure Poisson (conventional 4D flow) | 0.52 (published: 0.63) | 46% |

**The bar: beat Bernoulli's 30%.**

#### The network: a 3D, periodic PINN on the scan

Input (t, x, y, z), output (u, v, w, p); voxel-averaged data term; momentum, continuity, no-slip walls; adaptive
weights; RAD tuned to the systolic jet (k = 2, c = 0, which lifts the jet's share of the data from 6% to 18%).

#### The journey on P2 baseline (catheter 13.1 mmHg)

| step | what we found | peak drop |
|---|---|---|
| first network | flow fits well (jet 9% slow), pressure almost flat | **0.30** |
| eddy viscosity, RAD, heavier physics weights | none helped; the learned eddy viscosity collapsed to zero | 0.06-0.39 |
| momentum balance along the vessel | the flow needs ~6 mmHg; the network leaves it as a tiny error at every point (small locally, adds up along 26 cm) | |
| **pressure fitted alone** to the frozen flow | the pressure *is* in the flow; joint training just failed to extract it; matches the catheter's waveform at correlation 0.96 | **7.26** |
| **separate pressure network** (12 heartbeat harmonics, multi-scale features), trained jointly | the fix works: the drop climbs during training | **5.28** (20k steps, still rising) |
| + **Smagorinsky eddy viscosity** nu_t = (C_s Delta)^2 abs(S) from the flow (does not collapse) | adds a little turbulent loss | **6.05** (20k steps) |

#### Why it is not 13.1 yet

- **Turbulent loss:** averaged over a beat, the catheter's drop is 3.68 mmHg of permanent loss; laminar physics gives
  only ~0.9. The jet's turbulence is below the 1.8 mm voxel size, which is what the eddy viscosity has to supply.
- **The ceiling of velocity-only data:** the scan's jet peaks at 1.42 m/s; 13.1 mmHg would need about 1.8 m/s. Every
  velocity-only method stalls at 7-8 mmHg here (Bernoulli 8.0, pressure Poisson 7.9, the stand-alone fit 7.3).
- **Next step:** the archive also has a 4D-FlowP scan that measures the **acceleration** directly. A first estimate
  from it gives 9.8-9.9 mmHg, above every velocity-only method. Adding it as a data term is novel: no
  physics-informed network using acceleration-encoded MRI was found in a literature search.

---

### Likely questions

- **Is the synthetic result really better than Doppler?** On simulated data, yes: 4-6% against Bernoulli's 21-31%.
  It is a synthetic study, not a clinical claim.
- **Why does the real phantom fail when the synthetic one works?** Real data has turbulence below the voxel size,
  noise, timing blur (60 ms frames) and scanner artefacts. The synthetic flows were laminar.
- **What was the single biggest fix?** The separate pressure network: from 0.3 to 5.3 mmHg at the same training
  budget, with the flow unchanged.
- **Why does a free eddy viscosity collapse?** Joint training could always explain the data without it, so the
  penalty that keeps it small drives it to zero. Computing it from the flow (Smagorinsky) avoids that.
- **What would make it clinical?** Beating Bernoulli on all 8 cases with one frozen setting (calibrate on one case,
  score the other seven), then real patient scans.

#### In the viewer

Tab "Real-world use case": the synthetic stenosis (scan, CFD truth, network pressure, the Bernoulli line). Tab "Real
MRI phantom": all 8 cases, the laminar network on the left and the eddy-viscosity network on the right, pigment
released at the upstream catheter plane; the bars compare every method with the catheter.
