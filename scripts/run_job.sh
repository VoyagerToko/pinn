#!/usr/bin/env bash
# One benchmark run end to end: train -> evaluate -> figures, with the cost numbers of handbook 7.6.
#
#   bash scripts/run_job.sh NAME BENCHMARK "PLOT_ARGS" [train.py args...]
#   bash scripts/run_job.sh cavity_F cavity "" --ablation F
#   bash scripts/run_job.sh cylinder_G_w4 cylinder "--animate 200" --ablation G --windows 4 --steps 50000
#
# Writes runs/NAME.log (all output), runs/NAME.gpu.csv (nvidia-smi samples every 15 s:
# unix time, memory.used MiB, utilisation %), runs/NAME/cost.json (training wall-clock, exit code,
# memory used on the card before the run started). PLOT_ARGS "none" skips the figures;
# for tgv3d the figures come from scripts/visualize.py and are run separately.
set -u
name=$1
bench=$2
plot=$3
shift 3
cd "$(dirname "$0")/.."
source ~/miniforge3/bin/activate pinn
export XLA_PYTHON_CLIENT_PREALLOCATE=false
wd=runs/$name
log=runs/$name.log
mkdir -p "$wd"
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
(while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)"; sleep 15; done) >> "runs/$name.gpu.csv" &
mon=$!
echo "[run_job] $(date -Is) start $name: train.py --benchmark $bench --workdir $wd $*" >> "$log"
t0=$(date +%s)
python scripts/train.py --benchmark "$bench" --workdir "$wd" "$@" >> "$log" 2>&1
rc=$?
t1=$(date +%s)
kill "$mon" 2>/dev/null
echo "{\"train_wall_s\": $((t1 - t0)), \"exit\": $rc, \"gpu_baseline_mib\": $base, \"start_unix\": $t0, \"end_unix\": $t1}" > "$wd/cost.json"
echo "[run_job] $(date -Is) train exit=$rc wall=$((t1 - t0)) s" >> "$log"
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
python scripts/evaluate.py --benchmark "$bench" --workdir "$wd" >> "$log" 2>&1
echo "[run_job] $(date -Is) evaluate exit=$?" >> "$log"
if [ "$plot" != "none" ] && [ "$bench" != "tgv3d" ] && [ "$bench" != "cylinder_inverse" ] && [ "$bench" != "deeponet_cavity" ]; then
  # shellcheck disable=SC2086
  python scripts/plot2d.py --benchmark "$bench" --workdir "$wd" $plot >> "$log" 2>&1
  echo "[run_job] $(date -Is) plot2d exit=$?" >> "$log"
fi
echo "[run_job] $(date -Is) done $name" >> "$log"
