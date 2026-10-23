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
  - `coa_P2_baseline` (laminar, prep v4): 0.30 against 13.1 mmHg. Its jet is there, 9% too slow. (An earlier note
    here said its pressure recovers downstream under laminar physics; that was wrong, see the next items.)
  - Desktop, later on 2026-10-23: the eddy run with RAD collapsed too (nu_t/nu 2.5 at 5k, 0.002 at 30k; 0.10 mmHg),
    and so did a run with `nu_t_reg_weight=0.001` (10.6 -> 0.35 by 15k; 0.10 mmHg). Along the vessel the momentum
    residual, not the pressure, carries the flow's acceleration (`scripts/coa_momentum_path.py`; STATUS.md). **The
    queue is stopped**: do not restart the driver for the phantom runs until that is resolved.
  - **The P2 baseline viewer slots hold 20k-step tests (2026-10-23, prep v5, published):** `runs/coa_P2_baseline` is
    `_test_pnet_cold` (laminar, pressure network, 5.28 mmHg) and `runs/coa_P2_baseline_eddy` is `_test_smag_sub4`
    (Smagorinsky C_s 0.17, 6.05 mmHg); each has a `SOURCE.txt`. The driver counts them as done; move them aside before
    a full 60k run. The earlier runs are kept as `coa_P2_baseline_v4_shared` and `coa_P2_baseline_eddy_killed_v5`.
  - Laptop, 2026-10-23: a pressure fitted alone to `coa_P2_baseline`'s frozen velocity gives 7.26 mmHg with a third
    less momentum residual than the joint network's (`scripts/coa_pressure_fit.py`), and follows the catheter's
    waveform at correlation 0.96. The pressure is in the flow; joint training fails to extract it.
- **The pressure-network test (do this before any phantom run; the queue stays stopped until it passes).**
  - The repository now has `problem.pressure_net` (the `coa_mri` default): the pressure from its own, periodic network.
    It is untested in joint training. Prep v5 also adds the catheter's clock offset, so every prepared case and every
    earlier phantom run counts as stale: **do not start `desktop_train_all.sh`**, it would move all phantom runs aside
    and retrain all 16 with the untested change.
  - Re-prepare only what the test needs: `python -u scripts/prepare_coa.py --case P2_baseline`. Expect
    `cath_shift_ms` near -147 and `cath_shift_corr` near 0.97.
  - Warm test (velocity from the trained `runs/coa_P2_baseline`, a fresh pressure network):
    `python -u scripts/coa_pressure_net_test.py --steps 6000 > runs/_test_pnet_warm.log 2>&1`.
  - Cold test (everything from scratch, as a real run):
    `python -u scripts/train.py --benchmark coa_mri --set "problem.case='P2_baseline'" --steps 20000 --workdir runs/_test_pnet_cold > runs/_test_pnet_cold.log 2>&1`.
    For comparison, the old network had 0.24 mmHg at 5k and 0.25 at 20k.
  - **Pass:** `eval/dp_peak_pinn_mmHg` climbs well above 1 mmHg and keeps rising (the stand-alone fit reached 7.26).
    **Fail:** it stays near 0.3 while the momentum residual falls. Report the eval lines (every 5000 steps for the cold
    run, every 500 for the warm one), and on fail run `python scripts/coa_momentum_path.py _test_pnet_cold`.
  - If an eddy run ends with `nu_t_mean_over_nu` near 0 again, rerun it with `--set problem.nu_t_reg_weight=0.001`.
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
