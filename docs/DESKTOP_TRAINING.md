# Training on the desktop, results to the laptop

The desktop (RTX 5070 Ti, WSL2 Ubuntu, repository at `~/pinn`, conda env `pinn` from miniforge) trains; the laptop
shows the results in the live viewer. Finished runs travel through a private GitHub repository,
`VoyagerToko/pinn-models`, as one release per run:

    desktop: scripts/desktop_train_all.sh  -> trains, then scripts/publish_models.sh <run>  -> release in VoyagerToko/pinn-models
    laptop:  scripts/fetch_models.py (running in the background)  -> downloads new releases into runs/, restarts the server

## What is trained, in this order

| run | what | why |
|---|---|---|
| `stenosis_mri_fast` | synthetic stenosis, 70% narrowing, Re 300, a ~1.1 m/s jet, 4.2 mmHg peak drop | closer to the fast jets the clinical Doppler rule (4 v^2) was made for |
| `coa_P2_baseline`, `coa_P2_baseline_eddy` | real 4D flow MRI of a coarctation phantom, scored against catheter pressure | laminar network vs network with a learned eddy viscosity |
| `coa_P2_severe`, `coa_P2_severe_eddy` | the same for the severe P2 phantom | |
| `coa_<case>`, `coa_<case>_eddy` for each of the other six cases, in pairs | P1 cases and the low-flow cases | the full comparison |

`runs/coa_report/report.md` (republished after every real-data run) tables every case: catheter, conventional 4D-flow
pressure Poisson estimate, simplified Bernoulli, and both networks, with the slope and R^2 against the catheter.

## Things the desktop must know

- **Data is not in git.** The driver creates it.
  - `data/stenosis_fast`: CFD, about 30 min of CPU.
  - `data/stanford_coa`: about 1.5 GB fetched selectively out of the 18 GB public archive by `scripts/remote_zip.py`.
- **Real-data preparation (`scripts/prepare_coa.py`) checks the scan against the phantom's flow meter.**
  - It writes `data/stanford_coa/prepared/<case>_qa.png` and `_qa.json`.
  - A "CHECK THE SEGMENTATION" line means the scan's flux through the upstream plane disagrees with the meter by
    more than 20% even after the automatic plane shift (see the docstring of `prepare_case`). Look at the QA image
    before trusting that case.
- **Findings so far (2026-10-23; details in STATUS.md, "Real data").**
  - Since the sign fix (prep v4) the conventional pressure Poisson estimate reads slope 0.52 against the catheter,
    near the published 0.63. The data is sound.
  - `coa_P2_baseline` (laminar, prep v4): 0.30 against 13.1 mmHg. Its jet is there, 9% too slow, but its pressure
    falls about 4 mmHg through the narrowing and recovers downstream: laminar physics has no term for the
    turbulent loss the catheter measures. More laminar runs will not fix this; the eddy-viscosity network is the test.
  - If an eddy run ends with `nu_t_mean_over_nu` near 0 again, rerun it with `--set problem.nu_t_reg_weight=0.001`.
  - Desktop, later on 2026-10-23: the eddy run with RAD collapsed too (nu_t/nu 2.5 at 5k, 0.002 at 30k; 0.10 mmHg),
    and so did a run with `nu_t_reg_weight=0.001` (10.6 -> 0.35 by 15k; 0.10 mmHg). Along the vessel the momentum
    residual, not the pressure, carries the flow's acceleration (`scripts/coa_momentum_path.py`; STATUS.md). **The
    queue is stopped**: do not restart the driver for the phantom runs until that is resolved.
- **RAD for `coa_mri` needs this repository at or after the commit "Phantom: RAD of collocation points and data".**
  - Before it, `--set training.rad_every=1000` was accepted but did nothing for this problem: no residual function,
    so the resampling was skipped silently. A run started on older code is uniform sampling whatever its log says.
    Check with `grep -n residual_magnitude_fn pinnflow/problems/coa_inverse.py`; empty output means no RAD.
  - Now `coa_mri` defaults to RAD every 1000 steps with k=2, c=0. That redraws the collocation points by residual
    and the scan's voxel-frames by misfit, which concentrates on the systolic jet (6% of the data uniformly, 18%
    with RAD). No `--set` needed.
  - To retrain a run already marked done on prep v4 (for example `coa_P2_baseline`) with RAD, move its
    `runs/<name>` folder aside first; the driver skips runs whose `final_eval.json` is current.
- **No L-BFGS for these benchmarks.** It overfits the scan's noise (STATUS.md, 2026-10-20).
- **GPU memory:** `training.res_chunk` sets the peak; 16 GB is plenty at the defaults.
- **Licence:** the phantom data is CC BY-NC 4.0 (Nair et al., Stanford Digital Repository, doi:10.25740/ms528wr9373).
  Cite it in anything written up. Keep the data and the models out of the public repository.

## One-time setup on the desktop

```bash
cd ~/pinn && git pull                     # if line-ending noise blocks the pull: git checkout -- . && git pull
source ~/miniforge3/bin/activate pinn
gh --version || conda install -y -c conda-forge gh
gh auth login --hostname github.com --git-protocol https --web   # sign in as VoyagerToko (one-time code in the browser)
gh auth setup-git
gh repo view VoyagerToko/pinn-models      # must succeed: the private models repo
```

## Running

```bash
SMOKE=1 bash scripts/desktop_train_all.sh                       # quick check of every kind of run (300 steps each)
nohup setsid bash scripts/desktop_train_all.sh > /dev/null 2>&1 < /dev/null &   # the real thing, detached
tail -f runs/desktop_train_all.log
```

The driver is resumable. It skips any run that already has `runs/<run>/final_eval.json`, and publishes each run once,
marking it with `.published`.
