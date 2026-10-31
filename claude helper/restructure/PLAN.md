# Restructure plan (for approval; nothing has moved yet)

Branch `restructure` off `main` at the commit you approve. Numerics, configs, seeds, schedules, sampling and every
checkpoint stay as they are; this moves code, it does not rewrite it. Line counts below are estimates from today's
files (`wc -l`), rounded.

## 0. Decisions I need from you

| # | question | my recommendation |
|---|---|---|
| D1 | `core/` with your 6 files, or 8 smaller ones? With 6, `training.py` absorbs the loss weighting, the `Problem` base class, generic sampling and the config defaults (~760 lines). | **8 files**: add `core/losses.py` (weighting schemes) and `core/config.py` (base config + ablation rows A-H). Every core file then stays under ~500 lines, and each answers one viva question. |
| D2 | `core/io.py` shadows Python's standard `io` module: running any file inside `core/` directly would make `import numpy` load ours and crash. | Name it **`core/files.py`** (checkpoints, CSV log, data loaders). |
| D3 | How runnable files find `core/`: `python models/cavity.py` puts `models/` on the import path, not the repo root. | Two lines at the top of each runnable file add the repo root to `sys.path` (as `scripts/` does today). Do **not** `pip install` top-level packages called `core`, `models`, `tools`: those generic names can collide with other installed packages. |
| D4 | The current uncommitted fixes on `main` (cylinder "not trained" bug in `scripts/serve.py`, the heartbeat chart and pressure scale in `viewer/`, the presentation in `claude helper/`). | Commit them to `main` first (they are fixes, independent of the restructure), then branch. Otherwise they ride along unreviewed on `restructure`. |
| D5 | The desktop PC keeps committing to `main` (today: `coa_inverse.py`, `configs.py`, diagnostics). Every such change after the branch point must be ported by hand into `models/coa_mri.py` etc. | Freeze model code on `main` while the restructure runs (diagnostics in `claude helper/` are fine), or accept porting PC commits at merge time; I list each ported commit. |
| D6 | Viewer charts in the browser: hand-drawn SVG, or a charting library from a CDN. | **Hand-drawn SVG** (no extra dependency; the charts are simple: lines, bars, one log axis). three.js only for the 3D tanks, pinned to `three@0.160.0` (jsDelivr ES modules, plus its `OrbitControls`). |
| D7 | Timing. The viva/presentation is now; this is several sessions of work, and the web viewer is most of it. | Keep `main` (and its PySide6 viewer) as the demo; merge `restructure` only after the equivalence table and viewer parity are done. |

## 1. New tree (expected lines)

```
pinn/
  README.md, CLAUDE.md, INDEX.md (new, ~450), pyproject.toml, requirements.txt, environment.yml
  core/                                   shared code only (8 files with D1; 6 without)
    __init__.py                      ~15   configure_jax (from pinnflow/__init__.py)
    physics.py                      ~300   NS residuals (VP, stream function, vector potential), directional/hvp AD,
                                            grid residuals for SPINN, vorticity/Q/lambda2, chunked_vmap
    networks.py                     ~480   Dense, PeriodEmbs, FourierEmbs, Embedding, Mlp, ModifiedMlp, PirateNet,
                                            pirate_pi_init, SPINN, DeepONet, create_arch  (class names unchanged)
    constraints.py                   ~70   generic hard-BC tools: r_and/r_or/r_and_smooth, hard_dirichlet, fold_rwf_params
    losses.py                       ~190   mse, causal weights + annealer, grad-norm, NTK, SA masks, RAD resampling,
                                            pressure anchor, pressure-gauge alignment
    training.py                     ~470   Problem base class, TrainState, make_lr_schedule, make_adam, Trainer,
                                            run_lbfgs, train_curriculum, train_time_windows, generic samplers
                                            (uniform_box, uniform_time, spinn_axes, linspace_axes), the shared
                                            command line (train / evaluate / plot, --set overrides)
    config.py                       ~160   base config, ABLATION rows A-H, apply_ablation (generic part), get_config
                                            dispatching to models/*.config()
    metrics.py                      ~230   relative L2, divergence, drag/lift, Strouhal, cycle stats, energy,
                                            enstrophy, dissipation, spectrum, throughput, step time, cost summary
    files.py                        ~380   save/load params, CSVLogger, Timer, meshgrid_points, device memory,
                                            data paths and generic loaders (JAX-PI cavity, CausalPINNs NS, FEATFLOW,
                                            HiOCFD), VTK export, particle advection, frame rendering, video encode,
                                            2D field-plot and movie helpers
  models/                                 one file per trained model; each runs on its own
    __init__.py                      ~30   MODELS registry: benchmark name -> module (replaces PROBLEMS)
    tgv2d.py                        ~330
    cavity.py                       ~560
    cylinder.py                     ~720
    tgv3d.py                        ~470
    cylinder_inverse.py             ~270
    deeponet_cavity.py              ~290   imports the cavity's geometry and references from models/cavity.py
    stenosis_mri.py                 ~470
    coa_mri.py                      ~640
  viewer/
    backend/
      server.py                     ~260   FastAPI app and the routes /health /models /field /curve /points
                                            /velocity /bundle (unchanged URLs and payloads), static frontend
      inference.py                  ~330   Scene: load a run with its model module, restore params, generic field
                                            and velocity evaluation, binary packing; calls each model's viewer hooks
      catalog.py                    ~220   the scene list: titles, summaries, gates, controls, run per scene
                                            (today's REGISTRY in scripts/serve.py)
    frontend/
      index.html                    ~140
      style.css                     ~450   tokens and layout from claude helper/design/DESIGN.md
      app.js                        ~450   state, scene selection, tabs/rail, controls, stamp, wiring
      scenes.js                     ~160   per-scene view lists, emitters, pigments (today's viewer/scenes.py)
      particles.js                  ~260   RK2 advection through /velocity, emitters, retirement in the lumen
      bath.js                       ~300   2D field + pigment rendering on canvas, colour maps, legends
      tank.js                       ~330   three.js: Taylor-Green box with Q tubes; the phantom's two linked aortas
      charts.js                     ~420   SVG charts: error vs time, centreline vs Ghia, flow-rate bars,
                                            dissipation, stenosis drop, phantom heartbeat + bars, coefficients
    (viewer/*.py, the PySide6 app, stays until parity, then is deleted in its own commit)
  tools/
    data/       download_data.sh, cavity_fd_reference.py (~155), digitize_tgv_reference.py (~135),
                stenosis_data.py (~400: the MAC CFD solver + simulated scan + make command),
                coa_data.py (~520: DICOM/catheter reading, segmentation, sign fix, clock, prepare command),
                coa_baselines.py (~200: pressure Poisson, Bernoulli), remote_zip.py (~140)
    results/    coa_report.py (~110), paper_figures.py (~200), paper_tables.py (~105)
    machines/   setup_linux_gpu.sh, setup_wsl_gpu.sh, check_env.py, train_local_wsl.sh, serve_local_wsl.sh,
                run_job.sh, desktop_train_all.sh, publish_models.sh, fetch_models.py   (paths updated only)
  tests/        conftest.py + the 9 current test files (imports updated) + test_index.py (new, ~90)
  data/ runs/ paper/ docs/ external/ "claude helper/"   unchanged
```

Each `models/<name>.py` has the same skeleton, in this order:

1. plain-English docstring: the flow, what the network outputs, the tricks it uses, what it is graded against,
   and its result from docs/STATUS.md;
2. the benchmark itself: domain, boundary/initial conditions, exact or reference solution;
3. `config(row)`: its defaults and the ablation rows that apply to it;
4. the problem class (unchanged name): network wrapper, sampling, loss terms;
5. `train(cfg, workdir)`, `evaluate(workdir)`, `plot(workdir)`;
6. viewer hooks: `viewer_field(scene, t, Re, n)` and `viewer_curve(scene)`;
7. `if __name__ == "__main__": core.training.cli(this module)`.

That gives `python models/cavity.py train [--ablation C] [--steps N] [--set key=value]`, plus `evaluate --workdir ...`
and `plot --workdir ...`.

## 2. Where everything goes

### pinnflow/ (the library)

| current | function / class | new home |
|---|---|---|
| `__init__.py` | configure_jax | core/__init__.py |
| `physics.py` | everything | core/physics.py |
| `archs.py` | everything (class names kept, so parameter trees are identical) | core/networks.py |
| `constraints.py` | r_and, r_or, r_and_smooth, hard_dirichlet, fold_rwf_params | core/constraints.py |
| | phi_unit_square, phi_unit_square_open_top, lid_extension | models/cavity.py (deeponet_cavity imports them) |
| | cylinder_signed_distance_sq, phi_channel_cylinder, phi_channel_cylinder_bounded, cylinder_inflow_extension | models/cylinder.py |
| `losses.py` | everything | core/losses.py (core/training.py if D1 = 6 files) |
| `training.py` | everything | core/training.py |
| `problems/base.py` | Problem | core/training.py |
| `problems/__init__.py` | PROBLEMS | models/__init__.py: MODELS |
| `sampling.py` | uniform_box, uniform_time, spinn_axes, linspace_axes | core/training.py |
| | square_boundary | models/cavity.py |
| | channel_boundaries, channel_interior, channel_initial | models/cylinder.py |
| `metrics.py` | everything | core/metrics.py |
| `utils.py` | chunked_vmap | core/physics.py |
| | save_params, load_params, CSVLogger, Timer, meshgrid_points, device_peak_gb, count_params | core/files.py |
| `data.py` | PROJECT_ROOT, DATA_DIR, EXTERNAL_DIR, load_causal_ns, load_jaxpi_cavity, _read_numeric_table, list_featflow_files, load_featflow_series, load_hiocfd_reference | core/files.py |
| | cavity_centerlines | models/cavity.py |
| | load_cylinder_wake, sample_sparse | models/cylinder_inverse.py |
| `viz.py` | everything | core/files.py (used by tgv3d plot and evaluate) |
| `configs.py` | _base, ABLATION, generic part of _apply_ablation, get_config | core/config.py |
| | each `_benchmark_defaults` branch, each benchmark branch of `_apply_ablation` | the model file's `config(row)` |
| `benchmarks.py` | TaylorGreen2D | models/tgv2d.py |
| | ghia tables, ghia_tables, LidDrivenCavity | models/cavity.py |
| | DFGCylinder | models/cylinder.py |
| | CylinderWakeInverse | models/cylinder_inverse.py |
| | TaylorGreen3D | models/tgv3d.py |
| | BENCHMARKS | dropped (the registry is models/__init__.py) |
| `problems/tgv2d.py` | TaylorGreen2DPINN | models/tgv2d.py |
| `problems/cavity.py` | CavityPINN | models/cavity.py |
| `problems/cylinder.py` | DFGCylinderPINN | models/cylinder.py |
| `problems/tgv3d.py` | TaylorGreen3DSPINN | models/tgv3d.py |
| `problems/cylinder_inverse.py` | CylinderWakeInversePINN | models/cylinder_inverse.py |
| `problems/deeponet.py` | ParametricCavityDeepONet | models/deeponet_cavity.py |
| `problems/stenosis_inverse.py` | StenosisMRIInversePINN | models/stenosis_mri.py |
| `problems/coa_inverse.py` | PressureMLP, CoarctationMRIInversePINN, L_REF/U_REF/T_REF | models/coa_mri.py |
| `vessel.py` | StenosisVessel, MMHG, load_dataset, section_mean, truth_pressure_drop, bernoulli_pressure_drop | models/stenosis_mri.py |
| | _pad_odd, _upwind3, solve_cfd, periodicity_error, mri_from_cfd, mri_samples, save_dataset | tools/data/stenosis_data.py |
| `coa_phantom.py` | CASES, PREP_VERSION, FLUID, COA_DIR | models/coa_mri.py (the data tools import them) |
| | load_catheter, beat_average, catheter_clock_shift, catheter_pressure_drop, _series_dir, _frames_of, load_4dflow, fix_component_signs, _shading_corrected, correct_background, segment_lumen, wall_faces, prepare_case | tools/data/coa_data.py |
| | ppe_pressure_drop, _plane_voxels, poisson_plane_drop, BERNOULLI_VERSION, bernoulli_peak | tools/data/coa_baselines.py |

### scripts/

| current | function | new home |
|---|---|---|
| `train.py` | parse_overrides, main | core/training.py: cli (shared) |
| | cavity curriculum branch / cylinder windows branch / DeepONet curriculum branch / default branch | `train()` in models/cavity.py, cylinder.py, deeponet_cavity.py, and the shared default in core/training.py |
| `evaluate.py` | load_cfg, restore | core/files.py |
| | eval_cylinder | models/cylinder.py: evaluate |
| | eval_tgv3d | models/tgv3d.py: evaluate |
| | step_time, cost_summary | core/metrics.py |
| | main's cavity / DeepONet branches | models/cavity.py, models/deeponet_cavity.py: evaluate |
| `serve.py` | REGISTRY | viewer/backend/catalog.py |
| | load_cfg, read_json, pack_fields, grid2d, Scene (load, _restore, field, _field_fn, _eval_pts, points, velocity, describe), build_bundle | viewer/backend/inference.py |
| | build_app, main, the route functions | viewer/backend/server.py |
| | _field_stenosis, _stenosis_ranges, _curve_stenosis | models/stenosis_mri.py: viewer hooks |
| | _field_coa, _curve_coa | models/coa_mri.py: viewer hooks |
| | _field_tgv3d, _curve_tgv3d | models/tgv3d.py |
| | _cavity_fd, _curve_cavity | models/cavity.py |
| | _curve_deeponet | models/deeponet_cavity.py |
| | _curve_cylinder | models/cylinder.py |
| | _nearest_snapshot, _wake_ref, _curve_inverse | models/cylinder_inverse.py |
| | _curve_tgv2d | models/tgv2d.py |
| `results/plot2d.py` | load_cfg, restore, fields_2d, one, _save_movie | core/files.py |
| | plot_tgv2d / plot_cavity / plot_cylinder (+ window_for, snapshot) | `plot()` in the three model files |
| `results/visualize.py` | main | models/tgv3d.py: plot |
| `results/eval_cavity_fd.py` | main | models/cavity.py: evaluate (`--fd`) |
| `results/coa_report.py`, `paper_figures.py`, `paper_tables.py` | all | tools/results/ (imports updated) |
| `data/*` | all | tools/data/ (make_stenosis_data.py merges into stenosis_data.py; prepare_coa.py into coa_data.py) |
| `machines/*` | all | tools/machines/ (paths updated: `models/<name>.py train` replaces `scripts/train.py --benchmark <name>`) |

### viewer/ (PySide6) -> viewer/frontend/

| current | becomes |
|---|---|
| scenes.py | frontend/scenes.js |
| workers.py: ParticleEngine, PigmentRaster | frontend/particles.js, bath.js |
| workers.py: FieldWorker, AuxWorker, Warmer | fetch calls in app.js (the warm-up becomes a server-side option) |
| bath.py, render.py, materials.py | frontend/bath.js + style.css |
| tank.py | frontend/tank.js (three.js) |
| rack.py | frontend/charts.js |
| controls.py, panel.py, theme.py, app.py | frontend/app.js, index.html, style.css |
| client.py | the browser's fetch; WSL start-up moves to tools/machines/serve_local_wsl.sh |

## 3. What does not move

- `data/`, `runs/` and their layouts: every run folder keeps its name and files; `config.json` contents are untouched.
- `external/` (the JAX-PI package is installed from there), `paper/` sources, `docs/PINN_Implementation_Handbook.md`,
  `claude helper/`, `CLAUDE.md` at the root.
- Server URLs and payloads (`/models`, `/field`, `/curve`, `/points`, `/velocity`, `/bundle`, `/health`), so the
  desktop's publish/fetch loop and any old client keep working.
- Class names of every network and problem (`ModifiedMlp`, `SPINN`, `PressureMLP`, `CavityPINN`, ...): Flax keys
  parameter trees by module class names, so keeping the names keeps every checkpoint loadable without a key-mapping
  loader. A mapping loader is written only if a test shows a tree changed.

## 4. Checkpoints and the equivalence table

1. **Before any move**, on `main`: `claude helper/restructure/snapshot.py` evaluates every run the viewer serves (25 scene
   slots, of which 11 hold trained runs today: tgv2d_G, cavity_C_r2, cavity_F_r2, cylinder_sf_2x6k, cylinder_inverse,
   tgv3d_H, deeponet_cavity_cont4, stenosis_mri_G, stenosis_mri_fast, coa_P2_baseline, coa_P2_baseline_eddy) with the old code, on the CPU (`JAX_PLATFORMS=cpu`, float32, highest precision,
   fixed seeds), and stores the `final_eval` numbers plus a fingerprint of each parameter tree (key paths, shapes, sum).
2. After each step, the same script runs against the new code and prints the table: run, quantity, old, new, relative
   difference. Target 1e-6; any larger difference is explained in the commit.
3. Comparing old-code-now against new-code-now isolates the restructure. Comparing against the stored
   `final_eval.json` files (written by older code versions) is shown too, in a separate column.

Slow evaluations (the cylinder curve took ~7 min on the laptop GPU) run once per side.

## 5. Order of work (one commit each, tests after each)

1. `main`: commit the current fixes (D4), take the old-code snapshot, branch `restructure`.
2. core/ (moved files + a temporary `pinnflow/` that re-exports from core, so nothing breaks mid-way).
3. models/tgv2d.py, cavity.py, deeponet_cavity.py, cylinder.py, cylinder_inverse.py, tgv3d.py, stenosis_mri.py,
   coa_mri.py: one commit each, with the equivalence rows for that model.
4. tools/: data, results, machines; remove the `pinnflow/` shim and `scripts/`.
5. viewer/backend (the PySide6 viewer talks to it unchanged: same endpoints).
6. viewer/frontend, scene by scene, against a parity checklist (prediction beside reference at the same instant and
   scale, verdict and number visible, time and Re controls, particles from the network's velocity, the 3D tanks).
7. Delete the PySide6 viewer (its own commit).
8. INDEX.md + tests/test_index.py, README.md (file map and which file writes what in runs/), CLAUDE.md, a dated
   docs/STATUS.md entry. INDEX.md is then updated in every later commit that moves or renames something.

## 6. INDEX.md and its test

Sections in your order: find by question (~30), benchmarks, outputs (each file a run produces and the function that
writes it), files (every file, one line), concepts (15, one sentence plus the place in code), data (every data file;
for `.npz`, array names and shapes read from the files), libraries, commands.

`tests/test_index.py` fails when:
- a tracked `.py`, `.html`, `.css` or `.js` file is not named in INDEX.md;
- a path named in INDEX.md does not exist;
- a `path: function` reference names a function, class or constant that the file does not define (checked by parsing,
  without importing JAX).

## 7. What I expect not to keep identical (each will be stated in its commit)

- **Import paths and command lines** change by design: `pinnflow.*` becomes `core.*`/`models.*`, and
  `scripts/train.py --benchmark X` becomes `models/X.py train`. The desktop driver and docs are updated to match.
- **The viewer's look:** the browser version follows DESIGN.md but will not be pixel-identical to the PySide6
  painting (paper textures, the ink stamp's uneven density). The behaviour list in constraint 6 is kept.
- **Floating point:** moving code does not change arithmetic, but jit-compiled programs can differ in the last bits
  if a function's traced structure changes (e.g. a helper inlined differently). The CPU equivalence run will show it.
- **Desktop commits after the branch point** need porting (D5).

## 8. Effort

Core and models: about 2 sessions including the equivalence runs. Backend: about 1. Frontend to parity: about 3-4
(it replaces ~3,700 lines of PySide6). Docs, INDEX.md and its test: about 1.
