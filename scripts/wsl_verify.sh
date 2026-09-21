#!/usr/bin/env bash
# Verify the WSL GPU environment: env report + full pytest on the GPU.
#     wsl -d Ubuntu -- bash /mnt/c/Users/mehul/pinn/scripts/wsl_verify.sh
set -uo pipefail
source "$HOME/miniconda3/bin/activate" pinn
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /mnt/c/Users/mehul/pinn
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
python scripts/check_env.py 2>&1 | grep -v "WARNING:absl"
echo "===== PYTEST (GPU) ====="
python -m pytest tests -q -p no:cacheprovider 2>&1 | grep -v "WARNING:absl" | tail -25
