# Product

<!-- impeccable:product-schema 1 -->

## Platform

desktop

A native desktop window (Windows), not a web page and not a mobile app.

## Stack

Python desktop application: Qt (PySide6) for the window and controls, PyVista embedded through pyvistaqt for every viewport. It runs in the native Windows `pinn` conda environment and asks `scripts/serve.py` for predictions over HTTP; the server runs the trained networks on the GPU inside WSL (`scripts/serve_local_wsl.sh` prints its address). Chosen by the user over a browser app.

## Users

Primary: people the author shows this work to (reviewers, professors, supervisors, teammates), watching on the author's laptop in person, on a projector, or over a screen share. They know what a fluid simulation is, but not what a physics-informed neural network is or what each benchmark tests. The author drives; the audience watches and asks questions.

Secondary: the author, rehearsing and answering "what happens if..." questions live.

## Product Purpose

An interactive viewer for the seven trained PINN models of this research project. Each scene shows one trained network's flow next to an independent reference, with the pass/fail verdict and the measured number, and lets the presenter move time, lid speed (Reynolds number) or the camera while every frame is computed live by the network. Success: a viewer who has never heard of PINNs can say, after a few minutes, what was trained, how close each model is to the truth, and why the failed ones failed.

## Positioning

Every pixel and every particle on screen is a live neural-network prediction computed on the GPU at the moment it is shown, not playback of a stored simulation. Particles move by asking the network for the velocity at their current positions (70-90 updates per second for 5,000 particles on the RTX 3060 laptop). Next to it sits an independent reference: the exact solution, a converged finite-difference solution, measured spectral-element data, FEATFLOW drag curves, or a DNS dissipation curve.

## Operating Context

- Machine: Windows laptop with an RTX 3060 Laptop GPU (6 GB); the networks run in WSL2, the window runs on Windows.
- Presented live, often to a group, sometimes on a projector or a shared screen, so text and contrast must survive compression and distance.
- First request for a new grid size compiles for 20-30 s; after that, full 2D fields take 18-215 ms per 128x128 frame depending on the scene, a 48^3 3D field ~80 ms, particle velocity queries ~11-14 ms.
- The viewer must reach the server at the WSL address, not localhost (Windows' localhost forwarding adds 35-50 ms per request).

## Capabilities and Constraints

Scenes, verdicts and numbers come from `STATUS.md` and the server's `/models` registry:

| scene | verdict | reference | controls |
|---|---|---|---|
| Taylor-Green vortex 2D | passed (velocity 4.0e-5) | exact solution | time |
| Lid-driven cavity, soft lid (row C) | field passed (0.27% from FD); Ghia v gate missed by 0.05 points | finite-difference solution N=512, Ghia 1982 | Re 100 / 400 / 1000 |
| Lid-driven cavity, hard lid (row F) | failed (Ghia ~20%) | finite-difference solution | Re 100 / 400 / 1000 |
| Cylinder wake, inverse problem | passed (viscosity 0.065%, hidden pressure 2.4%) | Nektar data (Raissi 2019) + 5,000 sensor points | time |
| Flow past a cylinder, forward | failed (no shedding, 25% mass loss) | FEATFLOW drag/lift | time |
| Taylor-Green vortex 3D (SPINN) | failed (no transition at t=9) | DNS dissipation curve | time |
| Cavity for any lid speed (PI-DeepONet) | passed (9.65% at unseen Re=400) | FD solutions at Re 100/400/1000 | Re 100-1000, held-out band 300-530 |

Server endpoints: `/models`, `/field`, `/curve`, `/points`, `/velocity`. It is not a general simulator: no stirring, no new obstacles, no new geometries; each network solves the one flow (or flow family) it was trained on.

## Evidence on Hand

All numbers in `STATUS.md`, the paper (`paper/main.pdf`), the trained checkpoints under `runs/`, and the reference data under `data/`. Do not invent comparisons, speed-ups or accuracy claims. In particular, do not claim the PINN is faster or cheaper than OpenFOAM: training took 30.5 GPU hours and classical CFD wins single forward solves; the advantages are instant re-query, parametric families and the inverse problem.

## Product Principles

1. Live means live: nothing on screen is precomputed playback unless it is labelled as a reference.
2. Prediction and reference are always comparable: same instant, same grid, same colour scale, side by side.
3. Failures get the same care as passes, and each failure is shown with the picture that explains it.
4. The verdict and its number are always visible; the audience never has to ask "is this one good?".
5. A presenter must be able to drive it confidently: no state where the screen is blank, frozen or ambiguous while the GPU compiles.
