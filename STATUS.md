# Status log

Hardware: RTX 5070 Ti 16 GB (desktop, WSL2 Ubuntu, `~/pinn`), RTX 3060 Laptop 6 GB (dev, WSL2).
Handbook gates are in STEP 10 of `docs/PINN_Implementation_Handbook.md`.

| benchmark | row | run | result | gate | status |
|---|---|---|---|---|---|
| A Taylor-Green 2D | G | 50k Adam + 1.2k L-BFGS, 0.14 s/step on 5070 Ti | rel L2 velocity 4.0e-5, pressure ~4e-4 (per-time gauge), div = 0 | rel L2 < 1e-4 | **passed** (2026-09-22) |
| B lid-driven cavity | F | curriculum 20k/40k/140k + L-BFGS | Ghia error 24% at Re=1000, loss 0.14 before L-BFGS | < 2% at Re=1000 | **failed** (2026-09-23); fixes in commit 24f0b1c (lr reset per stage, anchor excluded from weighting, L-BFGS stall rule, lid extension y^8); rerun pending, row C comparison pending |
| C DFG cylinder | G | not run | | St within 3%, Cd within 5% | pending |
| D Taylor-Green 3D (SPINN) | H | not run | | dissipation peak within 5% | pending |
| inverse wake | - | not run | | lambda_1, lambda_2 recovered | pending |
| PI-DeepONet cavity | - | not run | | zero-shot error < 10% at unseen Re | pending |

## Measured costs

- Benchmark A row G: 0.14 s/step on the 5070 Ti (about 1 s/step on the 3060 laptop), ~2 h total;
  XLA compile 5-15 min on first step; peak memory < 6 GB with `res_chunk=2048`.
- Row A-C configs (VP, second order) are ~3x cheaper per step than rows D-G (stream function, third order).

## Open issues

- Cavity row F: rerun with the fixes above; if it still misses while row C passes, the hard-constraint
  formulation (`u = g + phi N`) is the culprit on this problem and the distance-function scaling needs work.
- Cylinder: start with `--windows 4 --steps 50000` to check the causal weights converge before the 16-window run.
- Benchmark D reference curves (HiOCFD dissipation/enstrophy) still need digitising from the PDFs in `data/tgv3d_re1600/`.
- Not implemented from the handbook: OpenFOAM/FEniCSx ground truth (STEP 3.1), SOAP optimiser, browser demo (STEP 9.5).

## Ideas queued

- Interactive lid-speed (Re) slider on top of `deeponet_cavity`.
- "Place your own vortices" PI-DeepONet over vortex positions/strengths in the periodic box.
- 2D dye/particle tracers for the cylinder movie in `scripts/plot2d.py`.
