# Status log

Hardware: RTX 5070 Ti 16 GB (desktop, WSL2 Ubuntu, `~/pinn`), RTX 3060 Laptop 6 GB (dev, WSL2).
Handbook gates are in STEP 10 of `docs/PINN_Implementation_Handbook.md`. All runs use seed 42.

Every run lives in `runs/<name>/` (`config.json`, `metrics.csv`, `final_eval.json` from train.py,
`eval.json` from scripts/evaluate.py with the cost numbers, `cost.json`, `figures/`); the console output is
`runs/<name>.log` and nvidia-smi samples (every 15 s) are in `runs/<name>.gpu.csv`. Since 2026-09-24 runs are
started with `scripts/machines/run_job.sh` (train, evaluate, figures, wall-clock). Failed and interrupted runs are
kept under their own names; nothing is deleted.

## Gates

| benchmark | gate | run | measured | status |
|---|---|---|---|---|
| A Taylor-Green 2D | rel L2 < 1e-4 | tgv2d_G (2026-09-22) | velocity 4.05e-5 (u 4.17e-5, v 3.93e-5), pressure 2.59e-4 after per-time gauge alignment, max div 0 | **passed** |
| B lid-driven cavity | Ghia u(0.5,y) and v(x,0.5) error < 2% at Re=1000 | cavity_F_r2 (row F) and cavity_C_r2 (row C), 2026-09-24 | F: u 19.8%, v 20.2%; C: u 0.63%, v 2.05% (0.27% from the N=512 FD solution of the same problem, which itself is 0.41% / 2.29% from Ghia, D2) | **failed** (F clearly, C by 0.05 points on v); diagnoses D1, D2; soft-lid F rerun queued |
| C DFG 2D-2 cylinder | St within 3% of 0.30, Cd_max within 5% of 3.23 | cylinder_* (all reduced, stopped early) | no shedding in any run; best C_D max 1.54 (stream function, 2 windows); VP runs lose 52-98% of the mass along the channel | **failed** (D4-D6, D9) |
| D Taylor-Green 3D (SPINN) | dissipation peak within 5% of the reference (0.01282 at t=9.0, digitised) | tgv3d_H (80k steps, reduced) | peak of -dE/dt 0.0292 at t=0 (127%); 2 nu zeta max 4.9e-4 (96%); no transition | **failed** (D7) |
| inverse wake | lambda_1, lambda_2 recovered; hidden pressure | cylinder_inverse (25k + 1k, reduced) | lambda_1 = 0.99997 (0.004%), lambda_2 = 0.010006 (0.065%); hidden pressure 2.44%; u 0.54%, v 2.69% (20 snapshots) | **passed** |
| PI-DeepONet cavity | zero-shot error < 10% at Re=400 (Re in [300, 530] never sampled in training) | deeponet_cavity_curr2 + 4 continuations (660k Adam steps in total, lr 3e-4 with clipping, last 60k with a 4x larger batch; planned 200k + 20k) | velocity vs FD at Re=400: **9.65%** (14.0% after the curriculum, then 11.1%, 10.27%, 10.11%, 9.65%); Re=100 2.1%, Re=1000 30.0% | **passed** (D8, D10-D12) |
| the same, fast jet (`stenosis_mri_fast`, data/stenosis_fast) | peak drop within 10% of the truth (4.13 mmHg) | stenosis_mri_fast (2026-10-20, RTX 5070 Ti, 6 x 192, 80k Adam steps, 1.6 h; published to the model repository) | peak dp 3.96 mmHg (**4.1%**), cycle RMS 3.6%, mean 1.84 against 1.95; hidden pressure 9.0%, u 7.2% (v 104%: the cross-flow is small); simplified Bernoulli 4.99 mmHg (+20.7%) | **passed** |
| real data: coarctation phantom, P2 baseline | network peak drop closer to the catheter than Bernoulli (30% mean error) | coa_P2_baseline (laminar, 60k, prep v4); coa_P2_baseline_eddy (RAD, stopped at 37.4k) | 0.30 and 0.10 mmHg against 13.1 (97.7%, 99.2%); Bernoulli 8.0, pressure Poisson 7.9 | **failing** (momentum residual carries the systolic acceleration; "Real data") |
| real-world use case: stenosis from MRI (synthetic) | peak trans-stenotic pressure drop within 10% of the CFD truth (0.594 mmHg) | stenosis_mri_G (2026-10-20, RTX 3060 Laptop, 6 x 128, 40k Adam steps, 1.6 h; served checkpoint = end of Adam) | peak dp 0.557 mmHg (**6.2%**), cycle RMS 2.9%; hidden pressure 3.9%, u 4.1% (v 62%: the cross-flow is small); simplified Bernoulli on the same scan 0.781 mmHg (+31.5%) | **passed** |

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
| 2026-09-27 | tgv3d_H movie | scripts/results/visualize.py, 96^3, 240 frames, grid-mode fields | runs/tgv3d_H/tgv3d.mp4 (3.3 MB) | ~6 min | - | - |
| 2026-10-03 | deeponet_cavity_curr2 | PI-DeepONet, soft lid, fixed weights, Re curriculum (Re_max 200/500/1000, 40k/40k/120k), peak lr 3e-4, gradient clipping 1.0, no L-BFGS | stable; velocity vs FD: Re=100 2.35%, **Re=400 (unseen) 14.0%** (Ghia u 11.3%, v 17.6%), Re=1000 40.5%; gate (<10%) missed; continued as deeponet_cavity_cont | 3068 s | 0.013 s | 0.34 GB |
| 2026-10-03 | deeponet_cavity_curr_unstable | same with peak lr 1e-3, stopped at ~30k steps | diverged in the first stage (D10) | 0.19 h | 0.021 s | - |
| 2026-10-03 | deeponet_cavity_cont | warm start from deeponet_cavity_curr2, full Re range (hold-out kept), 150k Adam, peak lr 3e-4 (warm-up 5k, x0.9 per 2k), clipping 1.0 | velocity vs FD: Re=100 1.99%, **Re=400 (unseen) 11.1%** (Ghia u 9.3%, v 13.4%), Re=1000 33.5%; flat from 105k steps on, where the lr had decayed below 1e-6; warm-up restart cost ~30k steps (31% at 15k) | 2379 s | 0.013 s | 0.34 GB |
| 2026-10-03 | deeponet_cavity_cont2 | warm start from deeponet_cavity_cont, 150k Adam, peak lr 2e-4, warm-up 2k, x0.9 per 6k, clipping 1.0 (a first attempt was killed at ~5k steps by an external WSL restart: `deeponet_cavity_cont2_killed`) | velocity vs FD: Re=100 2.31%, **Re=400 (unseen) 10.27%** (Ghia u 8.4%, v 12.3%), Re=1000 31.0%; still falling at the end (10.53% at 140k) | 2354 s | 0.013 s | 0.34 GB |
| 2026-10-03 | deeponet_cavity_cont3 | warm start from deeponet_cavity_cont2, 100k Adam at constant lr 1.5e-5 (no warm-up), clipping 1.0 | velocity vs FD: Re=100 2.21%, **Re=400 (unseen) 10.11%** (Ghia u 8.1%, v 12.5%), Re=1000 30.6%; plateau 9.82-10.11% over the last 60k steps (9.82% at 40k); gate (<10%) missed by 0.11 points | 1558 s | 0.013 s | 0.34 GB |
| 2026-10-03 | deeponet_cavity_cont4 | warm start from deeponet_cavity_cont3, 60k Adam at constant lr 1e-5, **8 Re values x 8,192 points per step** (was 4 x 4,096), clipping 1.0 | velocity vs FD: Re=100 2.06%, **Re=400 (unseen) 9.65%** (Ghia u 7.9%, v 11.5%), Re=1000 30.0%; between 9.55% and 9.74% at every evaluation from 10k steps on: gate passed | 1731 s | 0.025 s | 0.33 GB |
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
`scripts/data/cavity_fd_reference.py` solves the steady cavity with finite differences (stream function -
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

**Re-grading against the FD fields (2026-09-27, `scripts/results/eval_cavity_fd.py`, written to `runs/<run>/eval_fd.json`).**
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

**Totals (2026-10-03).** GPU time of the whole study: 30.5 h (30.0 h before the last DeepONet leg) (`claude helper/diagnostics/gpu_hours.py`: job logs plus the three runs
before the job wrapper). Tests: 57/57 pass (`pytest`, CPU, 2026-10-03). Gates: A passed, inverse passed; B failed
(row C 0.63% / 2.05%, an exact solution of the regularised problem gives 0.41% / 2.29%); C failed (no shedding);
D failed (no transition); PI-DeepONet passed after a larger-batch continuation (9.65% at the unseen Re = 400, D12). External WSL restarts on
2026-10-03 (08:34, ~08:41, 09:51 UTC) killed cavity_G_softlid (requeued, then dropped for time) and the first
deeponet_cavity_cont2 attempt.

**D12 (2026-10-03) - PI-DeepONet passes with a larger batch.** At constant lr 1.5e-5 the error at Re = 400 had
plateaued at 9.82-10.11% (gradient noise of 4 Re x 4,096 points per step). With 8 Re x 8,192 points and lr 1e-5 it
dropped to 9.74% within 10k steps and stayed between 9.55% and 9.74% to the end of the 60k-step leg (final 9.65%).
Total budget 660k Adam steps, three times the plan; reported as such. Decision (user, 2026-10-03): Benchmark B stays
graded against the Ghia gate (failed), and no further GPU time goes to C and D (failed). Study total: 30.5 GPU hours.

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

- Cost accounting (handbook 7.6): `scripts/machines/run_job.sh`; `mem/peak_gb` in every metrics row; elapsed time on
  L-BFGS rows; `evaluate.py` reports step time, peak memory, training wall-clock and inference throughput
  (65,536-point batches). Cavity evaluation/plots now also cover `latest.msgpack` (the `Re*.msgpack`
  checkpoints are written before L-BFGS).
- Causal weighting starts at the first eps of the schedule (1e-2) instead of `causal_tol` = 1.0; before, the
  first annealing step lowered eps and the logged `causal/eps` was not the one in use. tgv2d_G ran with the
  old behaviour.
- Benchmark D reference: digitised from DeBonis (2013) Fig. 4(a) (`scripts/data/digitize_tgv_reference.py`,
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

## Real-world use case: blood pressure from MRI-like flow data (2026-10-20)

- **Dataset** (`scripts/data/make_stenosis_data.py`, 596 s on the laptop CPU).
  - The vessel: a 6 mm artery with a 60% narrowing, Re 150, 73 bpm.
  - The truth: the MAC projection solver at 50 cells per diameter, three cycles. The last two cycles agree to
    float32; max |div u| is 1e-13.
  - The scan: 0.6 mm voxels, 24 phases, SNR 20, giving 840 lumen voxels x 24 phases = 20,160 samples
    (VENC 0.51 m/s, sigma 1.2 cm/s).
  - The solver reproduces Poiseuille's pressure gradient to 0.6% (tests/test_vessel.py).
- **Cost on the RTX 3060 Laptop (6 GB, shared with the live server).**
  - Compiling the training step: 6.3 min.
  - At the default size (6 x 192, 4096 collocation points): 820 ms per step. 80k steps would take about 18 h,
    so that size is for the desktop.
  - The overnight laptop run uses 6 x 128 with 2048 points.
  - Evaluation: 50 s the first time (compile), then 0.3 s.
- **Run stenosis_mri_G (2026-10-20 01:5x to 04:55 IST, laptop).**
  - A first run with soft no-slip walls (kept as `runs/stenosis_mri_G_softwall_5k`) learned almost no pressure: the
    peak dp was 7e-4 mmHg at step 5k, and the walls slipped by 0.24 RMS. The fix was exact walls (stream function
    built from eta = y/h(x), with a learned Fourier flow rate Q(t)) and the pressure output scaled by 10.
  - With exact walls, the peak dp error went from 30% at step 5k to 12% at 10k, 8.4% at 15k, 7.7% at 25k and
    6.2% at 40k.
  - **L-BFGS overfits the scan.** 2000 iterations on one fixed batch drove the loss from 1.6 to 1.4e-4, far
    below the noise floor. Afterwards the pressure error was 8.0% and the peak dp error 12.9%, with u at 22%.
  - The served `latest.msgpack` is therefore `ckpt_40000` (end of Adam). The L-BFGS result is kept as
    `latest_lbfgs.msgpack` and `final_eval_after_lbfgs.json`, and `lbfgs_steps` now defaults to 0 for this benchmark.
- **Viewer.** The "Real-world use case" tab shows the scan and the CFD truth before training. The trained network
  plugs in on the next server restart (`scripts/machines/train_local_wsl.sh` restarts it when training ends).

## Real data: coarctation phantom, 4D flow MRI vs catheter (2026-10-20, in progress)

- **Data.** The public 4D-FlowP aortic coarctation phantom (Nair et al., Stanford Digital Repository,
  doi:10.25740/ms528wr9373, CC BY-NC 4.0).
  - 8 cases: 2 patient geometries x baseline/severe x 2 flow rates.
  - Siemens 4D flow at 1.8 mm, 13-17 frames; catheter AAo and DAo pressure at 1 kHz.
  - Fluid: glycerol-water 40/60, 1.1 g/cm^3, 4.2 mPa s.
  - Fetched selectively (`scripts/data/remote_zip.py`, 1.4 GB of the 18 GB archive); read by `pinnflow/coa_phantom.py`.
- **Catheter peak drops (location 3, beat-averaged):**

  | case | baseline | severe | baseline, low flow | severe, low flow |
  |---|---|---|---|---|
  | P1 | 16.2 | 24.7 | 9.0 | 13.4 |
  | P2 | 13.1 | 20.8 | 7.2 | 12.0 |

  - P2 changes by under 1.4 mmHg across the 5 downstream catheter positions.
  - P1 spans 11.7-23.2 (baseline) and 24.3-32.0 (severe), so P2 is the cleaner test.
- **The scan agrees with the phantom's flow meter.**
  - Ascending aorta: 248 mL/s peak against 245 mL/s, 67 mL/s mean against 67.7 mL/s.
  - Near the arch the measured flux falls to 60%, from signal loss in complex flow. The upstream pressure plane then
    moves down the ascending aorta to where the flux agrees (P2 baseline: 25 mm, 62 against 67.7 mL/s).
- **Background phase correction (prep v3).** Eddy currents put a 1-2 cm/s offset on the static gel. P2_severe's
  first segmentation found the gel instead of the aorta (plane flux 2% of the meter). Subtracting a 2nd-order
  polynomial fitted to static tissue fixes it.
  - All 8 cases segment to 108-198 mL.
  - Every upstream plane carries 92-109% of the meter's flow.
- **Velocity component signs (prep v4).** The export's stated encoding directions fix each component's axis but
  not its sign. With the wrong signs the scan's own streamlines left the vessel within a few centimetres (the
  viewer's pigment stuck at the release plane). `fix_component_signs` picks the signs with the smallest divergence of
  the peak frame; all 8 cases chose row -1, column +1, slice -1. Every network trained on prep v3 data is invalid.
- **Scan-based baselines on all 8 cases, prep v4** (`scripts/results/coa_report.py`, peak drop in mmHg):

  | case | catheter | pressure Poisson | Bernoulli 4 v^2 |
  |---|---|---|---|
  | P1 baseline | 16.2 | 6.5 | 10.3 |
  | P1 severe | 24.7 | 10.2 | 22.0 |
  | P2 baseline | 13.1 | 7.9 | 8.0 |
  | P2 severe | 20.8 | 14.5 | 16.7 |
  | P1 baseline, low flow | 8.9 | 5.3 | 5.1 |
  | P1 severe, low flow | 13.4 | 6.7 | 12.0 |
  | P2 baseline, low flow | 7.2 | 4.9 | 3.9 |
  | P2 severe, low flow | 12.0 | 5.5 | 7.4 |

  - **Bernoulli:** slope 0.77, R^2 0.88, mean error 30%, always low. The 1.8 mm voxels blur the jet's core; the
    peak is taken through a 3x3x3 median at the 99.5th percentile, because a single noisy or aliased voxel had read
    4.0 m/s at a 2.75 m/s VENC.
  - **Pressure Poisson:** slope 0.52, R^2 0.62, mean error 46%. Before the sign fix it read slope 0.1; the source
    publication reports conventional 4D flow at slope 0.63, so the corrected data now reproduces the published
    method. It takes the jet's measured decay at face value, losses included.
  - **The bar for the networks:** beat Bernoulli's 30%.
- **First network on prep v4 data: `coa_P2_baseline` (laminar), 60k steps, no RAD.** 0.30 mmHg against 13.1
  (97.7% error); velocity fit 35% over the whole beat. Diagnosis (2026-10-23, on its checkpoint):
  - **The jet is there.** Relative error by region: whole beat 35%, diastole 58% (slow flow, noise-dominated),
    systole 27%, systolic jet voxels 21%, scan > 1 m/s 19%. The network's jet is 9% slower than the scan (1.09
    against 1.19 m/s); its single fastest voxel (2.6 m/s in the scan) reads 0.15 m/s.
  - **The pressure drops and recovers.** At peak flow the network's pressure falls by about 4 mmHg through the
    narrowing and recovers to 0.2 mmHg at the downstream plane. That is laminar flow at the fluid's own viscosity:
    a smooth, nearly frictionless flow converts the jet's speed back into pressure. The catheter measures the
    turbulent loss, which laminar physics has no term for. The 9% slow jet costs about 17% of the drop (pressure goes
    with speed squared); the recovery costs the rest.
  - **What follows.** The eddy-viscosity network is the test of the method (its only run, on prep v3 data, learned
    nu_t ~ 0). If nu_t collapses again, lower `problem.nu_t_reg_weight` (0.01). The systolic jet is 6% of the
    voxel-frames, so RAD is now on for `coa_mri` (next item).
- **RAD for the phantom (2026-10-23).** It was off: `get_config` skips the ablation rows for the real-data benchmarks,
  and the problem had no `residual_magnitude_fn`, so `--set training.rad_every=...` silently resampled nothing.
  Now every 1000 steps the collocation points are redrawn by the momentum + continuity residual, and the scan's
  voxel-frames by the network's misfit (`training.rad_data`), p ~ err^k / mean(err^k) + c. With the handbook's k=1,
  c=1 the systolic jet's share of the data batch rises only from 6% to 7.4%; with k=2, c=0 (the `coa_mri` default
  now; `rad_uniform_frac` keeps 20% uniform) to 18%. One update takes about 7 s on the laptop CPU. Not trained yet.
- **The eddy test with RAD, and why the pressure stays flat (desktop, 2026-10-23; training stopped on request).**
  - **`coa_P2_baseline_eddy`** (prep v4, RAD k=2 c=0, `nu_t_reg_weight` 0.01), stopped at step 37.4k of 60k:

    | step | peak drop (mmHg) | nu_t / nu, mean | nu_t / nu, p99 | velocity fit |
    |---|---|---|---|---|
    | 5k | 0.39 | 2.5 | 12.3 | 0.41 |
    | 10k | 0.16 | 0.06 | 0.69 | 0.36 |
    | 20k | 0.17 | 0.003 | 0.03 | 0.35 |
    | 30k | 0.10 | 0.002 | 0.002 | 0.34 |

  - **`test_eddy_reg0001`** (the same with `nu_t_reg_weight` 0.001), stopped at 19.4k: nu_t/nu mean 10.6 -> 1.6 -> 0.35
    (p99 24 -> 16 -> 5) at 5k/10k/15k, peak drop 0.34 -> 0.14 -> 0.10 mmHg. A 10x weaker penalty only slows the
    collapse; and while nu_t was still 10x the fluid's viscosity the drop was no larger. The penalty is not the cause.
  - **Two laminar tests, P2 baseline, collocation RAD only (k=1, c=1), stopped at 25k:** `test_rad` 0.23 -> 0.06 mmHg
    over 5k-25k; `test_rad_phys` (the momentum and continuity weights fixed at 10, data terms adaptive) 0.15 -> 0.11
    mmHg, with r_u 5x lower than without (6e-5 against 3e-4). Neither run's momentum balance improved along the path
    below.
  - **The momentum balance along the vessel** (`claude helper/diagnostics/coa_momentum_path.py`, laminar `coa_P2_baseline`, prep v4, no
    RAD). Path integrals over the 259 mm lumen path from the AAo to the DAo plane centre, in mmHg:

    | t (x 10 ms) | int u_t | int conv | int visc | required p(DAo) - p(AAo) | network |
    |---|---|---|---|---|---|
    | 12.5 | +4.55 | -0.13 | -0.11 | -4.53 | +0.05 |
    | 18.5 | +3.73 | +2.31 | -0.17 | -6.21 | -0.12 |
    | 24.5 (peak flow) | -0.51 | +5.89 | -0.15 | -5.52 | -0.34 |
    | 49.0 | -3.72 | -0.89 | -0.17 | +4.44 | +0.10 |

    - The network's own velocity field needs a 6.2 mmHg peak drop from the laminar terms alone (pressure Poisson on
      the scan: 7.9; catheter 13.1). Its pressure supplies 0.1-0.3. The difference sits in the momentum residual.
    - The unbalanced part is mostly the time derivative: the column of fluid speeds up early in systole (+4.6 mmHg at
      t = 12.5) and slows after it (-3.7 at t = 49) with no pressure difference behind either.
    - The residual is small on average (mean square 4e-4) but coherent in systole: RMS |r_mom| 0.06 in systole
      against 0.012 in diastole, twice that in the 200 fastest voxels. 0.02 per length unit along the flow over
      26 units is the whole missing drop.
    - `test_rad_phys` at 20k shows the same pattern (t = 12.5: +4.94 u_t, network -0.03), so neither RAD nor heavier
      physics weights close it. With the residual already carrying the imbalance, nu_t has nothing to explain, which
      is why it collapses.
  - **Not yet known:** why the optimiser prefers an unbalanced time derivative to a pressure gradient. Candidates to
    check (CPU, no training): whether the scan's own frame-to-frame acceleration is consistent with a rigid,
    incompressible vessel (phantom wall compliance, the 60 ms acquisition window against the ~60 ms frame spacing,
    the frame times against the catheter phase), and the per-time pressure anchor at the DAo centre.
- **The pressure is in the velocity field; joint training does not extract it (laptop, 2026-10-23).**
  - `claude helper/diagnostics/coa_pressure_fit.py` freezes the trained velocity of laminar `coa_P2_baseline` (prep v4), computes
    F = u_t + (u . grad) u - nu lap u at 262,144 points over the lumen and the beat, and fits a separate pressure network
    alone: min mean |F + grad p|^2 (8000 Adam steps, CPU, 28 min). That is the pressure Poisson projection of the
    network's own flow, with the no-flux wall condition arising naturally.

    | | peak drop (mmHg) | mean over the beat | mean square momentum residual |
    |---|---|---|---|
    | the joint network's own pressure | 0.30 | 0.00 | 1.17e-3 |
    | pressure fitted to the same velocity | **7.26** | 0.85 | **7.7e-4** (34% lower) |
    | catheter | 13.1 | 3.68 | |

  - The fitted drop follows the catheter's waveform over the beat with correlation **0.96** once the catheter's clock is
    shifted by -118 ms (next item): up to +5.5 mmHg while the flow accelerates, -3.0 while it decelerates. The flat
    joint pressure is therefore not forced by the data: a pressure exists that lowers the very loss joint training
    minimises, and the shared output never found it. Earlier entries' "laminar physics recovers the pressure
    downstream" is wrong: the laminar terms of the network's flow need about 7 mmHg.
  - **The remaining gap to 13.1.** Over a whole beat the time-derivative term integrates to zero, so the beat-mean drop
    is friction loss plus a small kinetic-energy difference between the planes: 0.85 laminar against the catheter's
    3.68. The difference (about 2.8 mmHg on average, 5.7 at the peak) is the turbulent loss the eddy viscosity is for;
    the rest is the scan's blur in time (60 ms frames) and space (1.8 mm voxels), which caps every MRI method here.
- **The catheter's clock (prep v5).** The catheter's beat starts at the flow meter's upstroke (`beat_average`); the
  scan's frames count from the MRI trigger. `catheter_clock_shift` matches the scan's upstream-plane flux per frame to
  the meter's flow (smoothed over one acquisition window): catheter phase = scan time + shift, with shifts of -121 to
  -147 ms and correlations 0.97-1.00 on all 8 cases (P2 baseline -147 ms; the physics fit above gave -118). Peak and
  mean, the scores, do not depend on it; any comparison over the beat does. Stored as `cath_shift_s`; the problem
  exposes the catheter on the scan clock as `cath_t`, `cath_dp_scan`.
- **A separate pressure network (`problem.pressure_net`, the `coa_mri` default now).** The pressure no longer comes from
  the velocity network's fourth output but from its own network: 12 time harmonics of the beat (exactly periodic),
  isotropic Fourier features at sigma 1, 4 and 12 in space, 4 x 128 tanh, trained jointly with the velocity. It is the
  design of the stand-alone fit above. Run configs without the key keep the shared output, so older checkpoints load as
  trained. **Under test**: `claude helper/diagnostics/coa_pressure_net_test.py` warm-starts the velocity from `coa_P2_baseline` with a
  fresh pressure network. On the laptop GPU: peak drop 2.04 (random start) -> 0.34 mmHg at step 500 while the momentum
  residual fell from 2.9e-2 to 1.6e-3; not decided yet. If the drop stays flat, a separate network is not enough and
  the loss has to check the drop along the vessel (an integral form of the momentum balance).
  - **Desktop test, 2026-10-23: passed.** P2 baseline, prep v5 (`cath_shift_ms` -147, `cath_shift_corr` 0.974), RTX
    5070 Ti capped at 250 W, laminar, RAD on, not published (`runs/_test_pnet_warm`, `runs/_test_pnet_cold`).
    Peak and mean drop in mmHg against the catheter's 13.1 and 3.68:

    | step | warm: peak | mean | velocity fit | r_u |
    |---|---|---|---|---|
    | 0 | 2.04 (fresh pressure network) | +1.08 | 0.346 | 2.9e-2 |
    | 500 | 0.34 | -0.11 | 0.344 | 1.6e-3 |
    | 1000 | 0.53 | +0.26 | 0.345 | 8.7e-4 |
    | 1500 | 0.76 | +0.46 | 0.340 | 6.4e-4 |
    | 2000 | 1.00 | +0.58 | 0.348 | 5.5e-4 |
    | 2500 | 1.37 | +0.63 | 0.348 | 4.8e-4 |
    | 3000 | 1.55 | +0.63 | 0.358 | 4.2e-4 |
    | 3500 | 1.93 | +0.68 | 0.351 | 4.0e-4 |
    | 4000 | 2.16 | +0.74 | 0.357 | 4.1e-4 |
    | 4500 | 2.46 | +0.77 | 0.361 | 3.7e-4 |
    | 5000 | 2.81 | +0.89 | 0.359 | 2.9e-4 |
    | 5500 | 3.14 | +0.86 | 0.360 | 3.8e-4 |
    | 6000 | **3.36** | +0.88 | 0.358 | 3.1e-4 |

    | step | cold (from scratch, `train.py --steps 20000`): peak | mean | velocity fit | r_u | old shared output: peak |
    |---|---|---|---|---|---|
    | 0 | 1.20 | +0.42 | 5.154 | 1.6e+1 | |
    | 5000 | 1.99 | +0.38 | 0.404 | 9.0e-4 | 0.24 |
    | 10000 | 3.41 | +0.72 | 0.358 | 4.0e-4 | |
    | 15000 | 4.60 | +0.90 | 0.348 | 2.5e-4 | |
    | 20000 | **5.28** | +0.86 | 0.342 | 1.4e-4 | 0.25 |

    - Both climb while the momentum residual falls: the separate network puts the inertia into the pressure. The
      warm run gains about 0.3 mmHg per 500 steps to the end; the cold one is still rising at 20k (+0.68 over the
      last 5k), 21x the shared output's drop at the same step, at an unchanged velocity fit.
    - The mean settles at 0.86-0.88 mmHg, the laminar limit the stand-alone fit gave (0.85): over a beat the
      time-derivative term integrates to zero and laminar friction is all that is left. The peak's expected laminar
      ceiling is the stand-alone 7.26 (Bernoulli 8.0 on this case); the rest of the catheter's 13.1 needs the
      turbulent loss, i.e. the eddy-viscosity network, which can now act on the pressure.
  - **The eddy viscosity with the pressure network (desktop, 2026-10-23): nu_t still collapses.** Cold, 20k steps,
    P2 baseline, prep v5, not published (`runs/_test_pnet_cold_eddy`, `runs/_test_pnet_cold_eddy_reg0001`):

    | step | nu_t_reg 0.01: peak | mean | nu_t/nu mean | p99 | nu_t_reg 0.001: peak | mean | nu_t/nu mean | p99 |
    |---|---|---|---|---|---|---|---|---|
    | 5000 | 1.78 | 0.29 | 0.88 | 5.1 | 2.11 | 0.49 | 5.5 | 16.4 |
    | 10000 | 4.26 | 0.68 | 0.037 | 0.45 | 4.09 | 0.72 | 0.71 | 5.9 |
    | 15000 | 5.00 | 0.98 | 0.007 | 0.11 | 4.66 | 0.79 | 0.16 | 2.4 |
    | 20000 | **5.48** | 0.86 | 0.002 | 0.03 | **5.63** | 0.95 | 0.036 | 0.54 |

    - Velocity fit 0.342-0.343 at 20k in both, as laminar (0.342). Laminar at 20k: 5.28 and 0.86.
    - Both end where the laminar network does: the peak within 0.35 mmHg of it, the mean drop 0.86-0.95 against the
      catheter's 3.68. The weaker penalty only slows the collapse (nu_t/nu drops 4-7x per 5k steps either way), as it
      did with the shared output, and the remaining nu_t adds no measurable loss.
    - So the pressure network removes the old blocker (the drop now climbs; 5.3-5.6 mmHg at 20k, still rising) but
      not the eddy viscosity's: joint training gains nothing from a free nu_t field and removes it. Reaching the
      turbulent part of the catheter's drop needs a different turbulence term, for example nu_t tied to the resolved
      flow (a mixing-length or Smagorinsky form from the velocity gradients) instead of a free output. Not decided; the
      desktop queue stays stopped.
- **Smagorinsky eddy viscosity (`problem.smagorinsky_cs`, desktop, 2026-10-23).** nu_t = (C_s Delta)^2 |S| from the
  resolved strain rate, Delta = the voxel (1.80 mm), in the same stress divergence as the learned field; its gradient
  by forward mode through |S|. Checked on the CPU before training: grad nu_t against central differences 1e-3,
  C_s = 1e-6 reproduces the laminar residual exactly, all losses finite. Off by default.
  - **Cold tests, C_s 0.17 (Lilly), not tuned,** P2 baseline, prep v5, pressure network, RAD, 20k steps, not published.
    The scan blur averaged over 4 points per voxel-frame (`runs/_test_smag_sub4`) or 16 (`runs/_test_smag_sub16`):

    | step | sub 4: peak | mean | nu_t/nu mean | p99 | fit | sub 16: peak | mean | nu_t/nu mean | p99 | fit |
    |---|---|---|---|---|---|---|---|---|---|---|
    | 5000 | 1.91 | 0.41 | 1.29 | 5.2 | 0.405 | 2.28 | 0.59 | 1.35 | 5.4 | 0.403 |
    | 10000 | 4.91 | 0.99 | 1.27 | 7.7 | 0.359 | 3.31 | 0.76 | 1.35 | 8.2 | 0.360 |
    | 15000 | 5.17 | 0.96 | 1.27 | 8.4 | 0.350 | 4.27 | 0.97 | 1.34 | 9.0 | 0.354 |
    | 20000 | **6.05** | 0.92 | 1.18 | 8.8 | 0.352 | **5.30** | 1.04 | 1.22 | 9.6 | 0.345 |

    - At 20k, all runs on the pressure network: laminar 5.28 / 0.86, learned nu_t 5.48 / 0.86 and 5.63 / 0.95,
      Smagorinsky 6.05 / 0.92 and 5.30 / 1.04 (peak / mean, mmHg; catheter 13.1 / 3.68; Bernoulli 8.0). The
      Smagorinsky viscosity does not collapse (nu_t/nu ~1.2 mean, 9-10 at p99) and adds a little loss: +0.06-0.18 mmHg
      on the mean, in line with an a priori estimate on the laminar run's flow (+0.29 mean, at most +2.1 just after peak
      flow). The 16-point blur did not sharpen the jet (velocity fit as the 4-point run); the order of the two runs
      changed between evaluations, so their 0.75 mmHg difference at 20k is within the spread of single runs.
    - The resolved jet sets nu_t: its shear is that of a voxel-smoothed jet, so nu_t stays near the fluid's own
      viscosity. C_s = 0.17 assumes resolved small-scale turbulence; at 1.8 mm it is likely too low, but raising it is
      a calibration: do it on one case, freeze it, and score the other seven.
- **The ceiling of velocity-only data (2026-10-23).** The scan's jet peaks at 1.42 m/s (the 99.5th percentile
  through a 3x3x3 median); the catheter's 13.1 mmHg would need about 1.8 m/s in 4 v^2. Bernoulli 8.0, pressure Poisson
  7.9 and the stand-alone pressure fit 7.3 agree, and the source publication reports conventional 4D flow at slope
  0.63. The networks (5.3-6.1 at 20k, still rising) are below that ceiling, so longer training should close part of
  the gap. **Correction (same day):** this is a ceiling of velocity-only data, not of the resolution. The same
  publication's 4D-FlowP acquisition, at the same 1.7-1.8 mm and 60 ms, encodes the acceleration as well and reaches
  slope 0.95 (next item).
  - **The 2D phase-contrast series do not resolve the jet.** P2 baseline has three single-slice through-plane series,
    INLET, OUTLET (13 cm lower) and BRANCHES (head vessels): 28 frames per beat (37.5 ms) against the 4D flow's 17,
    but 1.875 mm pixels and 6 mm slices, near-axial planes for flow measurement. They would add timing and the flow
    split as another voxel-averaged data term on the same network, not the jet's peak speed. Not built.
- **4D-FlowP: the measured acceleration (desktop, 2026-10-23).** Every case has a second acquisition in the archive,
  `Imaging/<case>/4DFlowP/data.npz` (about 2.1 GB per case; fetch with `scripts/data/remote_zip.py ... --get
  'Imaging/<case>/4DFlowP/*'`), not used so far. It holds `velocity` (cm/s) and `acceleration` (m/s^2), both
  (3, x, y, z, 17) with x = FH, y = RL, z = AP at 1.7 x 1.8 x 1.8 mm and 60 ms (venc 250 cm/s, aenc 300 m/s^2), an
  approximate vessel mask (38,860 voxels on P2 baseline) and the complex time-averaged images. No scanner geometry:
  registering it to the conventional 4D flow (and the catheter planes) is a separate step, not done yet.
  - **Velocity:** p99 1.43 m/s at peak flow (frame 4), as the conventional scan (1.42). **Acceleration:** p99 about
    80 m/s^2 in systole (p99.9 up to 177), 14 in diastole; per-component noise about 3.0 m/s^2 (diastolic RMS).
  - **What it measures** (`claude helper/diagnostics/flowp_material_accel.py`): smoothing velocity and measured acceleration alike
    (Gaussian, sigma 1-1.5 voxels), the measured field follows the material acceleration dv/dt + (v . grad) v
    derived from the scan's own velocity (correlation +0.46-0.53, slope +0.40-0.58 in the systolic jet) and not the
    local dv/dt alone (correlation ~0). So it is the fluid's material acceleration, rho Du/Dt in the momentum
    equation, and its signs match the velocity's (the velocity's own divergence also picks the stated signs).
  - **First estimate** (`claude helper/diagnostics/flowp_checks.py`): rho times the measured acceleration (3x3x3 mean in the mask)
    integrated along a lumen path from the ascending to the descending aorta at matched heights 30, 50 and 70 mm below
    the arch (path 147-230 mm): peak **9.8-9.9 mmHg**, mean 1.4-1.8, nearly independent of the height. The same
    integral of the velocity-derived acceleration gives 19 / 3.9, but differentiation triples its noise (RMS 28-41
    against 12-14 m/s^2) and a single path through a noisy field is path-dependent, so that number is not reliable.
    One centre line, no plane averages, no catheter positions: a sanity check only. It is above every velocity-only
    estimate (7.3-8.0) and below the catheter (13.1).
  - No physics-informed network using acceleration-encoded MRI was found in a literature search (abstract level,
    2026-10-23).
  - **Registration onto the conventional scan** (`claude helper/diagnostics/flowp_register.py --case P2_baseline`, CPU,
    24 s; output `data/stanford_coa/prepared/P2_baseline_flowp.npz`, 26 MB, not committed). The 4D-FlowP array axes
    are the patient axes, each increasing along LPS (FH -> +z, RL -> +x, AP -> +y), with the stored component signs.
    The translation comes from an FFT mask cross-correlation and is refined on the systolic velocity: voxel (0,0,0) is
    at LPS (-118.3, -38.7, -244.1) mm, uncertain to about 1.5 mm. Rotation is not needed (a fitted 0.5 degree did not
    help).
    - Mask Dice 0.856 against 0.593 for the next sign combination.
    - Systolic velocity at the 31,614 shared voxels (93% of the scan's 33,980): correlation 0.84 / 0.94 / 0.66 per
      local component (main flow 0.94), 0.87 overall, slope 0.80.
    - Upstream-plane flux per frame: 4D-FlowP 59.8 mL/s mean against the scan's 62.0 (meter 67.7), correlation 0.982.
    - Integer frame offset 0. The data agree better with the 4D-FlowP frames shifted +0.4 to +0.5 frame (about 30 ms),
      so both are stored: `acc` (as registered) and `acc_frac` (half a frame, linear in time). Acceleration noise
      2.85 m/s^2 per component (calm frames).
  - **The acceleration term** (`problem.accel_data`, off by default; `problem.accel_key` = `acc` or `acc_frac`). Terms
    acc_u, acc_v, acc_w match the network's material acceleration u_t + (u . grad) u, averaged over each voxel and
    acquisition window like the velocity, to the registered measurement at the shared voxel-frames (537,438 on P2
    baseline). Checked on the CPU: Du/Dt by forward-mode directions against central differences 1.3e-3; the losses are
    finite on the real file; a default config's loss terms are unchanged.
  - **Running (desktop, 2026-10-23):** P2 baseline, Smagorinsky 0.17 + pressure network + the acceleration term,
    20k steps, `acc` (`runs/_test_accel_smag`) and `acc_frac` (`runs/_test_accel_frac_smag`), not published yet.
- **Networks.** `coa_mri`: 3D, periodic over the beat, voxel-averaged data. It runs laminar, or with a learned eddy
  viscosity field (`problem.eddy_viscosity`). Training runs on the desktop (`claude helper/DESKTOP_TRAINING.md`); results go to
  `runs/coa_report/report.md`.

## Open issues

- Coarctation phantom: every network so far gives 0.06-0.39 mmHg against 13.1 on P2 baseline (laminar, eddy, RAD,
  heavier physics weights, a 10x weaker nu_t penalty). The momentum residual, not the pressure, carries the flow's
  acceleration in systole ("Real data", 2026-10-23). A pressure fitted alone to the same velocity gives 7.26 mmHg
  (correlation 0.96 with the catheter's waveform), so the cause is the joint training. The separate pressure network
  passed its desktop test (5.28 mmHg at 20k from scratch, still rising; "Real data"). With it the free eddy viscosity
  still collapses to ~0 (nu_t_reg 0.01 and 0.001; 5.48 and 5.63 mmHg, mean 0.86-0.95 against 3.68), so the turbulent
  loss is not captured. A Smagorinsky term computed from the flow does not collapse but is weak at this resolution
  (6.05 and 5.30 mmHg at 20k, mean 0.92-1.04). The remaining gap is mostly the scan's resolution ("The ceiling").
  Open choices: a 60k run of the Smagorinsky network, a C_s calibrated on one case and scored on seven, the 2D series
  as a timing term. The desktop queue stays stopped until one is chosen (prep v5 retrains all 16, about 17 h).
- Desktop stability: Windows restarted without a clean shutdown (Kernel-Power 41, no bugcheck) on 2026-10-20 (twice),
  10-21 and 10-22 (twice); four of the five during GPU training. Suspect the power supply or the GPU's 16-pin cable under
  load spikes. `nvidia-smi -pl 250` (the card's minimum, admin) would cap them. A restart kills the run in progress;
  the driver is resumable but not restarted automatically.

- Cavity: D1 above (rows C, D and the soft-lid F rerun are queued in this order).
- Cylinder: start with `--windows 4 --steps 50000` to check the causal weights converge before the 16-window run.
- Not implemented from the handbook: OpenFOAM/FEniCSx ground truth (STEP 3.1), SOAP optimiser, browser demo (STEP 9.5).

## Ideas queued

- Interactive lid-speed (Re) slider on top of `deeponet_cavity`.
- "Place your own vortices" PI-DeepONet over vortex positions/strengths in the periodic box.
- 2D dye/particle tracers for the cylinder movie in `scripts/results/plot2d.py`.
