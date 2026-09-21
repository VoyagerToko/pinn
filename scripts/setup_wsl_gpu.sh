#!/usr/bin/env bash
# Build the GPU training environment inside WSL2 Ubuntu (JAX has CUDA wheels for Linux only).
# Usage (from Windows):  wsl -d Ubuntu -- bash /mnt/c/Users/mehul/pinn/scripts/setup_wsl_gpu.sh
set -uo pipefail
CONDA="$HOME/miniconda3/bin/conda"
PROJ="/mnt/c/Users/mehul/pinn"
ENV_PY="$HOME/miniconda3/envs/pinn/bin/python"

if [ ! -x "$ENV_PY" ]; then
  "$CONDA" create -y -n pinn python=3.11 || exit 1
fi
PIP="$ENV_PY -m pip"
$PIP install --upgrade pip
# JAX with CUDA 12 (bundled CUDA libs via pip; needs only the Windows NVIDIA driver)
$PIP install --upgrade "jax[cuda12]" || exit 1
$PIP install flax optax ml_collections wandb hydra-core absl-py scipy matplotlib h5py tabulate pytest tqdm pyyaml meshio imageio imageio-ffmpeg || exit 1
# jaxpi.samplers imports torch; CPU-only build keeps the install small
$PIP install torch --index-url https://download.pytorch.org/whl/cpu || exit 1
# JAX-PI (main branch) as an editable install so the unmodified examples run
$PIP install -e "$PROJ/external/jaxpi" || exit 1
# this project's package
$PIP install -e "$PROJ" 2>/dev/null || echo "pinnflow not yet installable (pyproject missing) - rerun later"
echo "=== GPU CHECK ==="
"$ENV_PY" -c "import jax; print('jax', jax.__version__); print('devices', jax.devices()); import jax.numpy as jnp; print((jnp.ones((2048,2048))@jnp.ones((2048,2048))).sum())"
