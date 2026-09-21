#!/usr/bin/env bash
# Handbook STEP 10, week 1 gate: run the JAX-PI lid-driven-cavity example *unmodified* and check
# that it reproduces the repository's reported error. Run inside WSL Ubuntu (GPU):
#     wsl -d Ubuntu -- bash /mnt/c/Users/mehul/pinn/scripts/run_jaxpi_ldc_example.sh
# W&B is forced offline so no login is required (logs land in external/jaxpi/examples/ldc/wandb/).
set -euo pipefail
source "$HOME/miniconda3/bin/activate" pinn
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /mnt/c/Users/mehul/pinn/external/jaxpi/examples/ldc
python main.py --config=configs/default.py --workdir=. "$@"
