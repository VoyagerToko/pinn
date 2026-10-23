#!/usr/bin/env bash
# The training machine's whole job, resumable: data, the real-data preparation, every training run in priority
# order, and after each run its publication to the laptop (scripts/machines/publish_models.sh -> scripts/machines/fetch_models.py).
#
#   bash scripts/machines/desktop_train_all.sh            # everything still missing
#   SMOKE=1 bash scripts/machines/desktop_train_all.sh    # 300-step check of each kind of run, nothing published
#   ONLY="coa_P2_baseline coa_P2_baseline_eddy" bash scripts/machines/desktop_train_all.sh
#
# A run whose runs/<name>/final_eval.json exists is not retrained; one that is published carries runs/<name>/.published.
# Logs: runs/<name>.log per run, runs/desktop_train_all.log for the whole job.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for c in "$HOME/miniforge3/bin/activate" "$HOME/miniconda3/bin/activate" "$HOME/anaconda3/bin/activate"; do
  [ -f "$c" ] && source "$c" pinn && break
done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Python downloads (remote_zip.py) verify TLS against the conda env's CA bundle, which lacks the root of
# stacks.stanford.edu's chain (emSign Root CA - G1); the system store has it. Use the system store when present.
if [ -z "${SSL_CERT_FILE:-}" ] && [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
fi
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=2}"
mkdir -p runs
exec > >(tee -a runs/desktop_train_all.log) 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "checks"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { say "no GPU visible"; exit 1; }
python -c "import pydicom" 2>/dev/null || pip install -q pydicom
if [ "${SMOKE:-0}" != "1" ]; then
  gh auth status >/dev/null 2>&1 || { say "GitHub CLI is not logged in: run 'gh auth login' (and 'gh auth setup-git') first"; exit 1; }
fi

say "data: the fast-jet synthetic stenosis"
[ -f data/stenosis_fast/mri.npz ] || python -u scripts/data/make_stenosis_data.py --Re 300 --severity 0.7 --bpm 73 --N 60 --length 18 --fov 0.5 11.5 --out data/stenosis_fast

say "data: the 4D-FlowP coarctation phantom (Stanford Digital Repository, CC BY-NC 4.0)"
D=data/stanford_coa
mkdir -p "$D"
B=https://stacks.stanford.edu/file/druid:ms528wr9373
for f in README.txt CathPressure.zip STL.zip; do [ -f "$D/$f" ] || curl -sSL -o "$D/$f" "$B/$f"; done
[ -d "$D/CathPressure" ] || (cd "$D" && python -c "import zipfile; [zipfile.ZipFile(z).extractall('.') for z in ('CathPressure.zip', 'STL.zip')]")
python -u scripts/data/remote_zip.py "$B/Imaging.zip" --out "$D" --workers 8 \
  --get 'Imaging/*/4DFlow_highRes/*' 'Imaging/*/2DPC_*/*' 'Imaging/*/3D_GRE_highRes/*' 'Imaging/*/PhoenixZIPReport/*' 'Imaging/*/4DFlowP/readme.txt'
python -u scripts/data/prepare_coa.py --all
python -u scripts/results/coa_report.py --baselines-only

publish() {
  local name="$1"
  [ "${SMOKE:-0}" = "1" ] && return 0
  [ -f "runs/$name/.published" ] && return 0
  bash scripts/machines/publish_models.sh "$name" && touch "runs/$name/.published"
}

run() {  # run <name> <train.py args...>
  local name="$1"; shift
  if [ -n "${ONLY:-}" ] && ! [[ " $ONLY " == *" $name "* ]]; then return 0; fi
  if [ "${SMOKE:-0}" = "1" ]; then
    say "smoke: $name"
    python -u scripts/train.py "$@" --steps 300 --lbfgs 0 --workdir "runs/_smoke/$name" --set logging.log_every_steps=50 logging.eval_every_steps=300 2>&1 | grep -E "step=|final|Error|error" | tail -4
    return 0
  fi
  if [[ "$name" == coa_* ]] && [ -f "runs/$name/final_eval.json" ]; then
    # trained on data from an older preparation (before the background phase correction): keep it aside, retrain
    v=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('data_prep_version', 0))" "runs/$name/final_eval.json")
    cur=$(python -c "from pinnflow.coa_phantom import PREP_VERSION; print(PREP_VERSION)")
    if [ "$v" -lt "$cur" ]; then say "stale data in $name (prep v$v): moving it to ${name}_stale"; rm -rf "runs/${name}_stale"; mv "runs/$name" "runs/${name}_stale"; fi
  fi
  if [ ! -f "runs/$name/final_eval.json" ]; then
    say "train: $name"
    python -u scripts/train.py "$@" --workdir "runs/$name" > "runs/$name.log" 2>&1 || { say "FAILED: $name (see runs/$name.log)"; return 1; }
    say "done: $name: $(tr -d '\n' < "runs/$name/final_eval.json" | head -c 400)"
  fi
  publish "$name"
  if [[ "$name" == coa_* ]]; then
    python -u scripts/results/coa_report.py > /dev/null && rm -f runs/coa_report/.published && publish coa_report
  fi
}

# 1. the synthetic stenosis in the Doppler regime (default network size; L-BFGS is off for this benchmark)
run stenosis_mri_fast --benchmark stenosis_mri --set "problem.data_dir='data/stenosis_fast'" problem.p_scale=40.0
# 2. real data, the two P2 cases first (their catheter drop is least sensitive to the exact downstream position)
for c in P2_baseline P2_severe; do
  run "coa_${c}" --benchmark coa_mri --set "problem.case='$c'"
  run "coa_${c}_eddy" --benchmark coa_mri --set "problem.case='$c'" problem.eddy_viscosity=True
done
# 3. the other six cases, each as a pair: the laminar network, then the eddy-viscosity one (every finished case has both)
for c in P1_baseline P1_severe P2_baseline_lowflow P2_severe_lowflow P1_baseline_lowflow P1_severe_lowflow; do
  run "coa_${c}" --benchmark coa_mri --set "problem.case='$c'"
  run "coa_${c}_eddy" --benchmark coa_mri --set "problem.case='$c'" problem.eddy_viscosity=True
done
say "all runs finished"
[ "${SMOKE:-0}" = "1" ] || cat runs/coa_report/report.md
