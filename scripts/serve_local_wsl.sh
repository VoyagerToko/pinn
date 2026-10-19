#!/usr/bin/env bash
# Start scripts/serve.py on this machine's WSL GPU environment against the scenes in runs/, and print the
# address a Windows-side viewer should use. Connect to the WSL address, not localhost: Windows' localhost
# forwarding into WSL adds ~35-50 ms to every request that carries a body (measured), which caps particle
# updates at ~15/s instead of ~70-90/s.
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/<you>/pinn/scripts/serve_local_wsl.sh
#
# Get the scenes first: on the training machine `curl -o pinn_bundle.tar.gz http://<host>:8000/bundle`,
# then `tar -xzf pinn_bundle.tar.gz` at the repository root here.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for c in "$HOME/miniforge3/bin/activate" "$HOME/miniconda3/bin/activate" "$HOME/anaconda3/bin/activate"; do
  [ -f "$c" ] && source "$c" pinn && break
done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$ROOT"
pkill -f scripts/serve.py 2>/dev/null
nohup python scripts/serve.py --port 8000 --no-warm > runs/serve_local.log 2>&1 &
for i in $(seq 1 60); do
  curl -s http://localhost:8000/health >/dev/null 2>&1 && break
  sleep 2
done
curl -s http://localhost:8000/health; echo
grep "\[serve\]" runs/serve_local.log
echo "viewer address: http://$(hostname -I | awk '{print $1}'):8000"
