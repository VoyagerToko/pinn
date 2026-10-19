#!/usr/bin/env bash
# Train on this machine's WSL GPU environment, in the background, and restart the live server on the result.
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/<you>/pinn/scripts/train_local_wsl.sh stenosis_mri            # the full run
#   wsl -d Ubuntu -- bash /mnt/c/Users/<you>/pinn/scripts/train_local_wsl.sh stenosis_mri --steps 2000 --lbfgs 0 --workdir runs_smoke/x
#
# Arguments after the benchmark name go to scripts/train.py. The log is runs/train_<benchmark>.log; when training
# ends the server is restarted (scripts/serve_local_wsl.sh), so the viewer picks the new network up on its next start.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${1:?usage: train_local_wsl.sh <benchmark> [train.py args...]}"
shift
for c in "$HOME/miniforge3/bin/activate" "$HOME/miniconda3/bin/activate" "$HOME/anaconda3/bin/activate"; do
  [ -f "$c" ] && source "$c" pinn && break
done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=2}"
cd "$ROOT"
mkdir -p runs
LOG="runs/train_${BENCH}.log"
if [ "${FOREGROUND:-0}" = "1" ]; then
  python scripts/train.py --benchmark "$BENCH" "$@" 2>&1 | grep -v "Warning" | tee "$LOG"
  exit "${PIPESTATUS[0]}"
fi
# setsid: a session of its own, so the job survives the wsl.exe call that started it (nohup alone does not)
setsid nohup bash -c "python -u scripts/train.py --benchmark '$BENCH' $* > '$LOG' 2>&1; echo \"[train] exit code \$?\" >> '$LOG'; bash scripts/serve_local_wsl.sh >> '$LOG' 2>&1" >/dev/null 2>&1 < /dev/null &
sleep 3
if pgrep -f "scripts/train.py --benchmark $BENCH" >/dev/null; then
  echo "training $BENCH in the background; follow it with: tail -f $ROOT/$LOG"
else
  echo "training did not start; see $ROOT/$LOG"; exit 1
fi
