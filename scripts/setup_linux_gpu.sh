#!/usr/bin/env bash
# Generic GPU environment for Linux or WSL2 Ubuntu (JAX ships CUDA wheels for Linux only).
# Installs Miniforge (user-level, no sudo) if no conda is found, then builds the `pinn` env.
#
#     bash scripts/setup_linux_gpu.sh
#
# Afterwards:  source ~/miniforge3/bin/activate pinn   (or your conda's activate)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONDA=""
for c in "$(command -v conda || true)" "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
  [ -n "$c" ] && [ -x "$c" ] && CONDA="$c" && break
done
if [ -z "$CONDA" ]; then
  echo "no conda found - installing Miniforge to $HOME/miniforge3"
  curl -L -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  CONDA="$HOME/miniforge3/bin/conda"
fi
echo "using conda: $CONDA"

ENV_PREFIX="$("$CONDA" info --base)/envs/pinn"
[ -x "$ENV_PREFIX/bin/python" ] || "$CONDA" create -y -n pinn python=3.11
PY="$ENV_PREFIX/bin/python"

$PY -m pip install --upgrade pip
# CUDA JAX first (bundled CUDA libraries; only the NVIDIA driver is needed on the host)
$PY -m pip install --upgrade "jax[cuda12]>=0.7"
$PY -m pip install -r "$ROOT/requirements.txt"
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # jaxpi.samplers imports torch
$PY -m pip install -e "$ROOT/external/jaxpi" -e "$ROOT"

echo "=== GPU CHECK ==="
$PY -c "import jax, flax, optax; print('jax', jax.__version__, 'flax', flax.__version__, 'optax', optax.__version__); print('devices', jax.devices())"
echo
echo "activate with:  source $(dirname "$(dirname "$ENV_PREFIX")")/bin/activate pinn"
echo "then:           export XLA_PYTHON_CLIENT_PREALLOCATE=false && python scripts/check_env.py && pytest"
