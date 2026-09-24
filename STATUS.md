# Status log

Hardware: RTX 5070 Ti 16 GB (desktop, WSL2 Ubuntu, `~/pinn`), RTX 3060 Laptop 6 GB (dev, WSL2).
Handbook gates are in STEP 10 of `docs/PINN_Implementation_Handbook.md`. All runs use seed 42.

Every run lives in `runs/<name>/` (`config.json`, `metrics.csv`, `final_eval.json` from train.py,
`eval.json` from scripts/evaluate.py with the cost numbers, `cost.json`, `figures/`); the console output is
`runs/<name>.log` and nvidia-smi samples (every 15 s) are in `runs/<name>.gpu.csv`. Since 2026-09-24 runs are
started with `scripts/run_job.sh` (train, evaluate, figures, wall-clock). Failed and interrupted runs are
kept under their own names; nothing is deleted.

## Gates

| benchmark | gate | run | measured | status |
|---|---|---|---|---|
| A Taylor-Green 2D | rel L2 < 1e-4 | tgv2d_G (2026-09-22) | velocity 4.05e-5 (u 4.17e-5, v 3.93e-5), pressure 2.59e-4 after per-time gauge alignment, max div 0 | **passed** |
| B lid-driven cavity | Ghia u(0.5,y) and v(x,0.5) error < 2% at Re=1000 | cavity_F_r2 (row F) and cavity_C_r2 (row C), 2026-09-24 | F: u 19.8%, v 20.2%; C: u 0.63%, v 2.05% (0.27% from the N=512 FD solution of the same problem, which itself is 0.41% / 2.29% from Ghia, D2) | **failed** (F clearly, C by 0.05 points on v); diagnoses D1, D2; soft-lid F rerun queued |
| C DFG 2D-2 cylinder | St within 3% of 0.30, Cd_max within 5% of 3.23 | - | - | pending |
| D Taylor-Green 3D (SPINN) | dissipation peak within 5% of the reference (0.01282 at t=9.0, digitised) | - | - | pending |
| inverse wake | lambda_1, lambda_2 recovered; hidden pressure | - | - | pending |
| PI-DeepONet cavity | zero-shot error < 10% at Re=400 (Re in [300, 530] never sampled in training) | - | - | pending |

## Run log

Ghia errors are relative L2 over the Ghia points without the wall and lid points; "speed" is the relative L2
of |u| against the JAX-PI 128x128 reference field. Step time is the median over logged intervals (evaluation
steps included). Peak memory: JAX allocator peak of the training process / nvidia-smi peak minus the memory
already in use on the card before the run (the Windows desktop holds ~2.1 GB).

| date | run | configuration | result | wall-clock | step time | peak memory |
|---|---|---|---|---|---|---|
| 2026-09-22 | tgv2d_G | row G, 50k Adam + L-BFGS (stopped after 1.2k iterations by the old stall rule) | velocity 4.05e-5, pressure 2.59e-4 | ~2 h | 0.14 s | < 6 GB |
| 2026-09-22 | cavity_C | row C (soft BC, fixed weights), curriculum 20k/40k/140k + 20k L-BFGS, code before 24f0b1c (one lr schedule decaying across all stages) | Re=1000 checkpoint before L-BFGS: Ghia u 0.38%, v 2.33%, speed 4.3% | 5.4k s Adam (stages 616/1051/3704 s) | 0.026 s | - |
| 2026-09-23 | cavity_F | row F (hard BC, grad-norm), same budget and code | Ghia u 23.9%, v 24.7%, speed 30.1% | - | 0.025 s | - |
| 2026-09-24 | cavity_F_r2_killed | row F with the fixes of 24f0b1c | interrupted at step 63.7k: WSL shut the distro down when no terminal was attached (fixed with `instanceIdleTimeout=-1`, `vmIdleTimeout=-1` in `%USERPROFILE%\.wslconfig`); end of the Re=100 stage: Ghia u 3.7%, v 11.4% | - | 0.026 s | - |
| 2026-09-24 | cavity_F_r2 | row F with the fixes of 24f0b1c, 200k Adam + 20k L-BFGS | Re=1000 final: Ghia u 19.8%, v 20.2%, speed 24.5% (before L-BFGS 19.5% / 19.8% / 24.3%); Re=400: 17.9% / 24.8%; Re=100: 3.8% / 12.2%; velocity vs FD regularised-lid solution 26.1% (N=512) | 8278 s (Adam 4.9k s, L-BFGS 3.3k s) | 0.024 s (L-BFGS 0.17 s/iter) | 1.58 GB / 2.6 GB |
| 2026-09-24 | cavity_C_r2 | row C (soft BC, fixed weights) under the current code, same budget; L-BFGS stopped by the stall rule after ~9k iterations at loss 2.7e-7 | Re=1000 final: Ghia u 0.63%, v 2.05% (gate missed on v by 0.05 points), speed vs JAX-PI 4.2%, **velocity vs FD regularised-lid solution 0.27% (N=512; 0.44% vs N=256)**; Re=400: 3.1% / 5.5%; Re=100: 1.5% / 4.7% | 5721 s | 0.025 s | 1.59 GB / 2.6 GB |
| 2026-09-24 | cavity_D | row D (hard BC, fixed weights; row E is identical for a steady problem, causal weighting is off) | Re=1000 final: Ghia u 21.3%, v 19.8%, velocity vs FD (N=512) 28.3%; Re=400: 7.0% / 8.7% (FD 10.0%); Re=100: 3.6% / 8.8% (FD 7.9%) | 10023 s | 0.024 s | 1.59 GB / 2.6 GB |
| 2026-09-24 | tgv2d_G (re-evaluated) | cost numbers for the 2026-09-22 run | unchanged errors; inference 3.9e6 points/s (stream function, 65k batch) | - | 0.139 s | - |
| 2026-09-24 | probes (300-2100 steps, `runs/_probe_*`) | step time / memory before the long runs | tgv3d SPINN 32^4: 0.044 s/step, 3.1 GB, ~5 min compile; **64^4: out of memory** (one 11.1 GB buffer; the card has ~13.9 GB free); cylinder row G: 0.089 s/step with remat, 0.55 GB, plus ~10% for the grad-norm/RAD/causal updates every 1000 steps | - | - | - |

## Diagnoses and decisions

**D1 (2026-09-24) - cavity row F misses the gate with the fully hard constraint; the curriculum fixes of
24f0b1c did not change that.** In `runs/cavity_F_r2/metrics.csv` the momentum losses fall to ~7e-5 at
Re=1000 while the continuity loss stays between 1.5e-2 and 2.5e-1 for the whole run (0.045 at the end of Adam);
row C with soft boundary conditions reached 2e-8 on the same term. Grad-norm weighting reacts to the noisy,
large continuity gradient by settling at w/r_c = 0.27 against w/r_u = 7 and w/r_v = 16, so the interior
continuity is left at 1e-2 to 1e-1 (`figures/cavity_Re1000_final_residuals.png`), and the primary vortex comes
out too weak (speed error 0.1-0.3 across the core). Cause of the floor: the hard constraint imposes
u = u_lid(x) exactly on y = 1, and the regularised lid has u_lid'(0) = +50, u_lid'(1) = -50. On the side walls
u = v = 0 exactly, so v_y = 0 there, and continuity then requires u_x = 0 in the top corners. No smooth field
satisfies the boundary data and div u = 0 at both corners, so div u ~ 10 in two patches of ~1.5e-2 width, which
is exactly a mean-square floor of ~0.04. The lid extension exponent (y^8) and the per-stage lr restart do not
touch this. The soft constraint (row C) can trade a small lid error in the corners against continuity, and
JAX-PI's own cavity example also treats the lid softly and avoids sampling its corners.
Row D (hard lid, fixed weights, same code) fails the same way (Ghia 21.3% / 19.8%, 28.3% from the FD solution),
so the adaptive weighting is not the cause: the fully hard lid constraint is. Fix: `problem.lid_bc="soft"` (exact no-slip on the three fixed walls and exact v = 0 on the lid, lid
velocity as a loss term) removes the incompatibility while keeping the hard constraint where it is compatible.
It is queued as `cavity_F_softlid`.

**D2 (2026-09-24) - how far can a cavity PINN get from Ghia with this lid? An independent reference.**
`scripts/cavity_fd_reference.py` solves the steady cavity with finite differences (stream function -
vorticity, 2nd order, DST Poisson solve, pseudo-time to |d omega/dt| < 1e-7). At Re = 1000, N = 256:
unit lid psi_min = -0.11807 (Ghia -0.11793), Ghia error u 0.55%, v 2.12%; regularised lid (the problem the
PINNs solve) u 0.66%, v 1.77%. The JAX-PI reference field is itself 2.50% / 1.76% from Ghia (same as our FD
solution at N = 128, so it is a comparably coarse solution). Station by station, Ghia's v next to the right
wall (x >= 0.94) is about 0.01 smaller in magnitude than the converged FD solution; that alone is the 1.8-2.1%.
Row C tracks the FD solution to 1-2e-3 at every Ghia station (relative 0.42% in v, 0.47% in u, 0.44% over
the whole field) with a slight uniform overshoot of |v|, which is why its Ghia v error (2.05%) ends up just
above the regularised FD value (1.77%). Grid refinement makes the point sharper. N = 512 (converged in pseudo-time,
|d omega/dt| < 1e-7): unit lid psi_min = -0.11872; the N = 128/256/512 sequence (-0.11547, -0.11807, -0.11872)
extrapolates (Richardson, 2nd order) to -0.11894, the spectral value of Botella & Peyret (1998) (-0.1189366),
not Ghia's -0.11793. Against Ghia's tables the N = 512 solutions score u 0.89% / v 2.78% (unit lid) and
u 0.41% / v 2.29% (regularised lid): the better resolved the solution, the further it is from Ghia's v table.
**An exact solution of this benchmark (regularised lid) therefore misses the 2% v gate (~2.3%)**; the gate is
set tighter than the accuracy of the Ghia v data. 256 vs 512 FD velocity fields differ by 0.56% (regularised),
0.70% (unit). Row C is 0.27% from the N = 512 regularised solution (0.44% from N = 256), i.e. below the
discretisation error of the 256^2 finite-difference solve; row F is 26.1% from it. From now on the field error
against the finest FD solution of the same problem (`rel_l2_vel_vs_fd` in eval.json) is reported next to the
Ghia numbers. FD solutions also exist for Re = 400 and 100 (N = 256, regularised): Ghia errors u 0.23% / v 4.9%
and u 0.55% / v 3.5% (Ghia's low-Re v tables are coarser still), used for the curriculum stages.

## Code changes on 2026-09-24 (all committed)

- Cost accounting (handbook 7.6): `scripts/run_job.sh`; `mem/peak_gb` in every metrics row; elapsed time on
  L-BFGS rows; `evaluate.py` reports step time, peak memory, training wall-clock and inference throughput
  (65,536-point batches). Cavity evaluation/plots now also cover `latest.msgpack` (the `Re*.msgpack`
  checkpoints are written before L-BFGS).
- Causal weighting starts at the first eps of the schedule (1e-2) instead of `causal_tol` = 1.0; before, the
  first annealing step lowered eps and the logged `causal/eps` was not the one in use. tgv2d_G ran with the
  old behaviour.
- Benchmark D reference: digitised from DeBonis (2013) Fig. 4(a) (`scripts/digitize_tgv_reference.py`,
  `data/tgv3d_re1600/debonis2013_fig4a_ref_dissipation.csv`), peak 0.01282 at t = 9.00; check: eps(0) of the
  digitised curve 0.00044 vs analytic 3/(4 Re) = 0.00047. The HiOCFD pages publish no numeric curves.
- PI-DeepONet: Re in [300, 530] is never sampled in training, Re = 400 is the zero-shot query.
- Inverse problem: graded on 20 snapshots (100k points) with per-snapshot pressure gauge instead of one snapshot.
- Cylinder evaluation: gate errors, causal min weight per window, FEATFLOW overlay phase-aligned (the FEATFLOW
  series is already periodic at its t = 0: St 0.3002, Cd_max 3.2201, Cl_max 0.986 from `bdforces_q2_lv6_dt3`).
- Optional `problem.lid_bc="soft"` for the cavity (see D1); STEP 9 renders with a fixed colour range.

## Measured costs

- Benchmark A row G: 0.14 s/step on the 5070 Ti (about 1 s/step on the 3060 laptop), ~2 h total;
  XLA compile 5-15 min on first step; peak memory < 6 GB with `res_chunk=2048`.
- Benchmark B rows C/F: 0.024-0.026 s per Adam step, 0.17 s per L-BFGS iteration (full batch, line search),
  2.3 h per run including 20k L-BFGS iterations; 1.6 GB JAX peak; inference 1.1e7 points/s.
- Row A-C configs (VP, second order) are ~3x cheaper per step than rows D-G (stream function, third order).

## Open issues

- Cavity: D1 above (rows C, D and the soft-lid F rerun are queued in this order).
- Cylinder: start with `--windows 4 --steps 50000` to check the causal weights converge before the 16-window run.
- Not implemented from the handbook: OpenFOAM/FEniCSx ground truth (STEP 3.1), SOAP optimiser, browser demo (STEP 9.5).

## Ideas queued

- Interactive lid-speed (Re) slider on top of `deeponet_cavity`.
- "Place your own vortices" PI-DeepONet over vortex positions/strengths in the periodic box.
- 2D dye/particle tracers for the cylinder movie in `scripts/plot2d.py`.
