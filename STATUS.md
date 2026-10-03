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
| C DFG 2D-2 cylinder | St within 3% of 0.30, Cd_max within 5% of 3.23 | cylinder_* (all reduced, stopped early) | no shedding in any run; best C_D max 1.54 (stream function, 2 windows); VP runs lose 52-98% of the mass along the channel | **failed** (D4-D6, D9) |
| D Taylor-Green 3D (SPINN) | dissipation peak within 5% of the reference (0.01282 at t=9.0, digitised) | tgv3d_H (80k steps, reduced) | peak of -dE/dt 0.0292 at t=0 (127%); 2 nu zeta max 4.9e-4 (96%); no transition | **failed** (D7) |
| inverse wake | lambda_1, lambda_2 recovered; hidden pressure | cylinder_inverse (25k + 1k, reduced) | lambda_1 = 0.99997 (0.004%), lambda_2 = 0.010006 (0.065%); hidden pressure 2.44%; u 0.54%, v 2.69% (20 snapshots) | **passed** |
| PI-DeepONet cavity | zero-shot error < 10% at Re=400 (Re in [300, 530] never sampled in training) | deeponet_cavity (25k, reduced) | velocity vs FD at Re=400: 81% (Re=100 70%, Re=1000 87%) | **failed** (D8); curriculum rerun deeponet_cavity_curr running |

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
| 2026-09-24 | cavity_F_softlid | row F with `problem.lid_bc="soft"`, 200k Adam; L-BFGS interrupted (all jobs stopped on request), evaluated from the end-of-Adam checkpoint on 2026-09-27 | Re=1000: Ghia u 16.0%, v 15.6%, velocity vs FD 21.0%; Re=400: 10.0% / 11.7%; Re=100: 1.6% / 7.1%. Grad-norm drove w/u_lid to ~465 and w/r_c to 0.21: the soft lid became effectively hard again and continuity stayed at 2e-2..1e-1 | ~2.3 h Adam | 0.023 s | 1.59 GB |
| 2026-09-27 | cavity_D_softlid | row D (fixed weights) with the soft lid, 200k Adam + 20k L-BFGS; GPU shared with the KalaVision services | **end of Adam (Re1000.msgpack): Ghia u 1.04%, v 2.20%, velocity vs FD 1.9%**; after L-BFGS: u 4.96%, v 6.37%, FD 6.7% (L-BFGS loss 2.6e-3 -> 2.3e-6 on its fixed batch, D3); Re=400: 1.6% / 4.6% (FD 3.1%); Re=100: 0.7% / 2.1% (FD 4.8%) | 11761 s | 0.035 s (shared GPU) | 1.59 GB |
| 2026-09-27 | tgv3d_H | row H (SPINN, vector potential, 32^4 grid), 80k Adam + 1k L-BFGS (reduced from 300k + 20k); GPU shared until 19:24 UTC | **gate failed**: E_k(0) = 0.1246 (exact 0.125); -dE_k/dt peaks at t = 0 with 0.0292 (reference 0.01282 at t = 9.0, error 127%); 2 nu zeta never exceeds 4.9e-4 (error 96%); curve rel. L2 vs reference 1.52; spectrum slope k=4..16 -3.96 at t=0 and -3.50 at t=9 (`eval_spectrum_t9.json`; -5/3 expected), E_k(9) = 0.022 | 4458 s | 0.037 s | 3.07 GB |
| 2026-09-27 | cylinder_inverse | inverse wake, stream function, 25k Adam + 1k L-BFGS (reduced from 100k + 20k) | lambda_1 0.99997 (err 0.0035%), lambda_2 0.010006 (err 0.065%), hidden p 2.44%, u 0.54%, v 2.69% | 1868 s | 0.055 s | 1.63 GB |
| 2026-09-27 | deeponet_cavity | PI-DeepONet, soft lid, fixed weights, 25k Adam + 1k L-BFGS, Re log-uniform in [100,1000] minus [300,530] from step 0 | velocity vs FD: Re=100 70%, Re=400 81%, Re=1000 87%; after 5k steps it was at 10.5% (Re=100) before the loss spiked (D8) | 555 s | 0.013 s | 1.70 GB |
| 2026-09-27 | cavity_abl_{A,B,C,D,F,G} | ablation rows at 16k curriculum steps (1.6k/3.2k/11.2k), Adam only, rows D-G soft lid | velocity vs FD at Re=1000: A 91%, B 90%, C 81%, D 114%, F 118%, G 113%: none converged; 1.6k steps at Re=100 is too short before the jump to Re=400 (D9) | 255-587 s | 0.012-0.023 s | 0.3-0.6 GB |
| 2026-09-27 | tgv2d_abl_{A..G} | ablation rows at 3k Adam steps, warm-up 1k, no L-BFGS | velocity rel L2: A 8.7e-3, B 2.25e-2, C 3.2e-2, D 2.18e-2, E 1.80e-2, F 1.71e-2, G 1.17e-2; max div 1.4e-2/6.4e-2/7.7e-2 for A/B/C (VP), 0 for D-G (stream function) | 113-659 s | 0.014 (A) - 0.121 (G) s | 0.3-2.0 GB |
| 2026-09-28 | cylinder_sf_2x6k | stream function (div u = 0 exactly), soft Dirichlet data, row E (fixed weights, causal), inflow ramp, 2 windows x 6k, 8,192 residual points | flow-rate ratio at the end of window 1: 0.77 / 0.77 / 0.75 at x = 0.6 / 1.2 / 2.2 (VP: 0.12 / 0.04 / 0.02); C_D 0.87-1.54; no shedding | 3753 s | 0.359 s | 1.87 GB |
| 2026-09-27 | tgv3d_H movie | scripts/visualize.py, 96^3, 240 frames, grid-mode fields | runs/tgv3d_H/tgv3d.mp4 (3.3 MB) | ~6 min | - | - |
| 2026-10-03 | deeponet_cavity_curr2 | PI-DeepONet, soft lid, fixed weights, Re curriculum (Re_max 200/500/1000, 40k/40k/120k), peak lr 3e-4, gradient clipping 1.0, no L-BFGS | stable; velocity vs FD: Re=100 2.35%, **Re=400 (unseen) 14.0%** (Ghia u 11.3%, v 17.6%), Re=1000 40.5%; gate (<10%) missed; continued as deeponet_cavity_cont | 3068 s | 0.013 s | 0.34 GB |
| 2026-10-03 | deeponet_cavity_curr_unstable | same with peak lr 1e-3, stopped at ~30k steps | diverged in the first stage (D10) | 0.19 h | 0.021 s | - |
| 2026-10-03 | deeponet_cavity_cont | warm start from deeponet_cavity_curr2, full Re range (hold-out kept), 150k Adam, peak lr 3e-4 (warm-up 5k, x0.9 per 2k), clipping 1.0 | velocity vs FD: Re=100 1.99%, **Re=400 (unseen) 11.1%** (Ghia u 9.3%, v 13.4%), Re=1000 33.5%; flat from 105k steps on, where the lr had decayed below 1e-6; warm-up restart cost ~30k steps (31% at 15k) | 2379 s | 0.013 s | 0.34 GB |
| 2026-10-03 | deeponet_cavity_cont2 | warm start from deeponet_cavity_cont, 150k Adam, peak lr 2e-4, warm-up 2k, x0.9 per 6k, clipping 1.0 (a first attempt was killed at ~5k steps by an external WSL restart: `deeponet_cavity_cont2_killed`) | velocity vs FD: Re=100 2.31%, **Re=400 (unseen) 10.27%** (Ghia u 8.4%, v 12.3%), Re=1000 31.0%; still falling at the end (10.53% at 140k) | 2354 s | 0.013 s | 0.34 GB |
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

**Re-grading against the FD fields (2026-09-27, `scripts/eval_cavity_fd.py`, written to `runs/<run>/eval_fd.json`).**
Velocity relative L2 against the regularised-lid FD solution (N=512 at Re=1000, N=256 at 100/400), final model:
cavity_C (2026-09-22, old single lr schedule) 0.12% (end of Adam 0.10%); cavity_C_r2 0.27%; cavity_D_softlid 6.7%
(end of Adam 1.9%); cavity_F_softlid 21.0%; cavity_F_r2 26.1%; cavity_D 28.3%. cavity_F (2026-09-23) cannot be
re-graded: it was trained with the lid extension g = u_lid y, which the current code no longer builds (y^8), so
only its logged numbers (Ghia 23.9% / 24.7%) are valid.

**FD solver cost (CPU, 28 threads, machine shared with other workloads; `runs/cavity_fd*.log`).** Re=1000 to
|d omega/dt| < 1e-7: N=256 about 600 s (regularised) / 695 s (unit), N=512 3415 s / 3323 s; Re=400 N=256 236 s;
Re=100 N=256 418 s. It is an explicit pseudo-time code written for clarity, not speed.

**D3 (2026-09-27) - attribution for Benchmark B, and L-BFGS on a fixed batch overfits.** With the fully hard lid
both row D (fixed weights) and row F (grad-norm) fail (~20% Ghia, 26-28% from the FD solution). With the soft lid,
row D reaches Ghia u 1.04% / v 2.20% and 1.9% from the FD solution at the end of Adam, while row F stays at
16% / 16% (21% from FD) because grad-norm raises the lid weight until the constraint is hard again and lowers the
continuity weight. So on the cavity the hard lid constraint (rows D-G as specified) and the adaptive weighting
(rows F, G) each break training; soft boundary data with fixed weights (rows C and D-softlid) work. Decision: rows
D-G of the cavity ablation run with `lid_bc="soft"` from now on (the fully hard runs above are kept as evidence),
and PI-DeepONet uses the soft lid with fixed weights. Second finding: the Stage-2 L-BFGS runs full-batch on one
fixed draw of 8,192 collocation points; on cavity_D_softlid 20k iterations lowered that loss by three orders of
magnitude but tripled the field error (overfitting the fixed points); on cavity_C_r2 it stopped early and was
neutral, on cavity_F_r2 slightly harmful. Both pre- and post-L-BFGS numbers are reported from here on.

**D4 (2026-09-27) - cylinder at 4k steps per window: flow unresolved, budget moved to fewer, longer windows.**
`runs/cylinder_G_16x4k_stopped` (row G, 16 windows x 4k Adam steps, warm-up 1k, GPU shared) was stopped after two
windows: C_D = 0.32 at t = 0.25 s, 0.36 at 0.5 s and 0.45 at 0.75 s against ~3.2 for the developed flow, front-rear
pressure difference 0.19-0.29 against ~2.5; continuity loss ~2e-2, momentum ~6e-3; causal min weight 0.92-0.94 with
eps only at 0.1; grad-norm put the outflow-v weight at 7e3-9e3 (its gradient is tiny). The drag integral is covered by
analytic tests (tests/test_metrics.py), so the low drag is an unresolved boundary layer / wake, not a metric bug. With
14 more windows the unresolved state would only have been propagated, and shedding (needed for St) was not going to
appear, so the same GPU time was moved to 4 windows x 12k steps for row G (`cylinder_G_4x12k`) and, if time allows at
the end of the queue, row E with fixed weights (`cylinder_E_4x12k`) to test on the cylinder itself whether the
grad-norm outflow weights hurt, as they did on the cavity. Each window costs ~7 min of XLA compilation (the first
step, the first grad-norm update and the first RAD resampling each compile; the RAD residual is re-jitted at every
resampling because it closes over the current parameters), which is why fewer windows are also cheaper.

**D5 (2026-09-27) - the impulsive start of 2D-2 is incompatible data in time; fix: ramp the inflow.** With 12k steps per
window (`runs/cylinder_G_4x12k_impulsive`, stopped in window 1) the drag stayed at C_D = 0.31 (t = 0.25 s) and 0.32
(0.5 s), continuity at 2-3e-2, and the grad-norm weight of the outflow-v term grew to 2.4e5. The flow rate through
cross-sections of the trained window 0 shows why: the hard inflow constraint gives 0.41 m^2/s at x = 0 from t = 0,
but at t = 0.5 s the network carries 0.24 at x = 0.1, 0.14 at x = 0.6, 0.03 at x = 1.2 and 0.008 at the outlet. It
advances the inflow like a slow front instead of the instantaneous channel-wide start that incompressibility
demands when full inflow meets a fluid at rest, i.e. it trades continuity for temporal smoothness, just as the exact
cavity lid traded continuity at the corners (D1). More steps cannot fix that. Fix (problem setup, not method):
`problem.inflow_ramp = 1.0` ramps the 2D-2 inflow amplitude with sin^2(pi t / 2) over the first second, so rest is a
compatible initial state; the periodic 2D-2 state is unaffected. It is now the cylinder default; runs before it had
an impulsive start (`inflow_ramp` absent = 0). Rerun: `cylinder_G_ramp_4x12k` (4 windows x 12k steps).

**D6 (2026-09-27) - with the ramp the residuals drop 20-80x, but the velocity-pressure network still does not
conserve mass.** `runs/cylinder_G_ramp_4x12k` window 0 (12k steps, stopped afterwards): continuity loss 3e-4..1e-3
(impulsive: 2-3e-2), momentum 2e-5..2e-4, causal eps reached 10 (min weight 0.67-0.98), so training behaves. But
at t = 0.5 s the inlet carries 0.205 m^2/s while only 0.025 cross x = 0.6 and 0.005 leave through the outlet
(flow-rate ratios at the window end: 0.12 / 0.04 / 0.02 at x = 0.6 / 1.2 / 2.2); the pressure drop along the channel
is 0.18 Pa where accelerating the channel flow needs ~3.5 Pa. The network satisfies momentum locally by keeping the
downstream fluid almost still and pays with a small continuity residual everywhere (|div u| ~ 0.03, small per point,
large once integrated over the 22-diameter channel). C_D 0.08-0.10 (normalised with the nominal U = 1).
Partial evaluations of all stopped cylinder runs are in `runs/<run>/eval_partial.json` (St values there are
artefacts of 0.5-1 s records without shedding). Consequence: the cylinder rows as implemented (VP output, exact
Dirichlet data but divergence only as a loss) cannot represent mass conservation over the channel at this budget.
A stream-function formulation (`problem.formulation="streamfunction"`, soft Dirichlet data, exact div u = 0 so
the flow rate is fixed by psi on the walls) is queued last as `cylinder_sf_2x6k` (2 windows x 6k, 8,192 residual
points, fixed weights, causal on). The cylinder gate is failed either way within this time budget.

**D7 (2026-09-27) - Benchmark D: no transition, energy leaks through the residual; grad-norm weights explode.**
tgv3d_H (80k Adam, 32^4) fits the initial condition (E_k(0) = 0.1246) and then decays smoothly: E_k falls to ~0.01 by
t = 20, -dE_k/dt is largest at t = 0 (0.029) and 2 nu zeta stays at its initial value (~4.7e-4) throughout, so the two
dissipation estimates differ by a factor ~60 and the energy identity is violated: energy leaves through the momentum
residual (mean square 2.5-7e-4 in the last 40k steps, i.e. <u.r> ~ 5e-3, the size of the physical dissipation), not
through viscosity. w stays ~0 (r_w ~ 1e-9, ic_w ~ 1e-15): the network keeps the flow two-dimensional, so no vortex
stretching and no enstrophy growth. Grad-norm raised the weight of ic_w (target identically 0) to 6e9 and of r_w to
2.6e3; every term, satisfied or not, gets the same share of the gradient. Causal eps reached 10 at step 15k and
never 100 (min weight 0.8-0.92). This is the same weighting pathology as the cavity lid (w ~ 465, D3) and the
cylinder outflow (w ~ 2.4e5, D5): terms whose gradient is tiny receive enormous weights. A fix would at least put
terms with an identically-zero target (ic_w) into `weighting.fixed_terms`, but reaching the Re = 1600 transition is
not plausible at this budget; not rerun within the 12-hour window. The STEP 9 movie is rendered from this model.

**D8 (2026-10-03) - PI-DeepONet without a Reynolds curriculum is unstable.** deeponet_cavity sampled Re up to 1000
from the first step. Its error against the FD field at Re=100 was 10.5% after 5k steps, then the loss spiked
(continuity 2.6 at 10k) and the run ended at 70-87%. Handbook 6.3 forbids starting the cavity at Re=1000; the
parametric model needs the same. Fix: `problem.curriculum_Re_max = (200, 500, 1000)` widens the sampled range
stage by stage (40k/40k/120k steps, Adam restarted per stage, hold-out band kept; the sampler now clips the band to
the stage range). Rerun as deeponet_cavity_curr (200k Adam, soft lid, fixed weights, no L-BFGS after D3).

**D9 (2026-10-03) - short cavity ablation is uninformative; cylinder with exact divergence.** At 16k curriculum
steps every cavity row ends 81-118% from the FD field: 1.6k steps at Re=100 is too short before the jump to Re=400,
and all rows collapse there. The ablation of Benchmark B is therefore taken at the full budget (200k Adam steps,
compared at the end of Adam): C, D, F (hard lid) and D, F (soft lid) exist; A, B and G (soft lid) are queued.
The stream-function cylinder (exact div u = 0, soft Dirichlet data) raises the flow-rate ratio from 0.02-0.12 to
0.75-0.77 and C_D max from 0.10 to 1.54 in two windows; the remaining 23-25% loss is leakage through the soft
wall and inflow conditions (psi not exactly constant on the walls). Shedding and the C gate remain far away.

**D10 (2026-10-03) - the DeepONet instability is the learning rate, not the Reynolds range.** With the curriculum
(deeponet_cavity_curr, stage 1: Re in [100, 200]) the run again reached 7.8% from the FD field at Re=100 after 5k
steps and then diverged (loss 9.9 at 10k, 13.6 at 20k): the blow-up starts when the warm-up reaches the peak rate
1e-3, inside the easiest stage. Stopped at ~30k steps (`runs/deeponet_cavity_curr_unstable`). Rerun
deeponet_cavity_curr2 with peak lr 3e-4 and global gradient-norm clipping at 1.0 (`optim.clip_grad_norm`).

**D11 (2026-10-03) - PI-DeepONet budget.** The curriculum run (200k) ended at 14.0% at the unseen Re = 400 and the first
continuation (150k, same schedule) at 11.1%, flat over its last 45k steps because the exponential decay had brought
the learning rate to ~5e-7. A second continuation (deeponet_cavity_cont2, 150k) uses a slower decay (x0.9 every 6k
steps, peak 2e-4, warm-up 2k) so that the rate stays useful to the end. This is extra budget beyond the planned
200k + 20k (now 350k + 150k Adam steps in total), reported as such. Note: WSL was restarted externally at 08:34:45 UTC,
killing cavity_G_softlid one minute into its compile; it was requeued.

**Plan change (2026-09-27, requested): finish everything within 12 hours.** The specified budgets (cylinder 16 x 200k
steps, SPINN 300k, full ablation on A-C) need ~150-160 GPU hours. On request the remaining runs use reduced
budgets, stated per run: cylinder row G 4 windows x 12k Adam steps with the inflow ramp (warm-up 1k, no L-BFGS; see D4, D5); SPINN row H 80k + 1k
L-BFGS at 32^4 (64^4 does not fit, see probes); inverse 25k + 1k; PI-DeepONet 25k + 1k; ablation rows A-G on
Benchmark B at 16k curriculum steps and on Benchmark A at 3k steps, Adam only (warm-up 1k, same seed and
collocation budget within each benchmark; L-BFGS dropped after D3); the cylinder ablation is not run. The GPU is shared with two
KalaVision services (run.py + ffmpeg decoders, ~70% utilisation, ~7.5 GB) until 2026-09-27 19:24 UTC, when they
were stopped on request (scheduled tasks \Kala\KalaVision-Attendance and -Bag disabled; re-enable with
Enable-ScheduledTask -TaskPath "\Kala\\" ...); runs up to and including the start of tgv3d_H were shared; step times measured while sharing are ~1.3-1.5x the exclusive ones, and the nvidia-smi "above idle" memory
is not meaningful then (the JAX peak is).

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
