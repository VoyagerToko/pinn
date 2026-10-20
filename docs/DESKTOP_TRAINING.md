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
| `coa_<other six cases>_eddy`, then `coa_<other six>` | P1 cases and the low-flow cases | the full comparison |

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
- **The conventional pressure Poisson estimate is expected to come out far below the catheter.** Measured on P2
  baseline: 0.7 mmHg against 13.1 mmHg. Laminar physics at 1.8 mm voxels recovers the jet's pressure downstream; the
  catheter's drop is turbulent loss. The eddy-viscosity network exists to test whether that loss can be inferred.
  Report what comes out, whichever way it goes.
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
