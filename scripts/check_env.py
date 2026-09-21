"""Environment and data check (run this first in each environment).

    python scripts/check_env.py            # report + 20-step smoke training of Benchmark A
    python scripts/check_env.py --no-smoke
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = ["jax", "jaxlib", "flax", "optax", "ml_collections", "numpy", "scipy", "matplotlib", "h5py"]
OPTIONAL = ["pyvista", "vtk", "meshio", "imageio_ffmpeg", "wandb", "torch", "jaxpi", "pytest"]
DATA = [
    "data/cylinder_wake/cylinder_nektar_wake.mat",
    "data/dfg_benchmark/dfg2d2_draglift_q2_cn_lv3-6_dt1-4",
    "data/dfg_benchmark/dfg2d2_pressure_q2_cn_lv3-6_dt1-4",
    "data/dfg_benchmark/dfg2d3_draglift_q2_cn_lv1-6_dt4",
    "data/dfg_benchmark/dfg2d3_pressure_q2_cn_lv1-6_dt4",
    "data/tgv3d_re1600/DeBonis2013_NASA_TGV.pdf",
    "data/tgv3d_re1600/HiOCFD4_BS1_TaylorGreenVortexRe1600.pdf",
    "external/CausalPINNs/data/NS.npy",
    "external/jaxpi/examples/ldc/data/ldc_Re1000.mat",
    "external/jaxpi-pirate/jaxpi/archs.py",
    "external/SPINN/navier_stokes4d.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    for m in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(m)
            print(f"  [ok]      {m:14s} {getattr(mod, '__version__', '')}")
        except Exception as e:
            tag = "[MISSING]" if m in REQUIRED else "[optional]"
            print(f"  {tag} {m:14s} {type(e).__name__}")

    import jax

    print(f"\njax backend: {jax.default_backend()}  devices: {jax.devices()}")
    if jax.default_backend() == "cpu":
        print("  note: CPU only. GPU JAX on Windows runs inside WSL2 -> see README (scripts/setup_wsl_gpu.sh).")

    print("\ndata / external:")
    for rel in DATA:
        p = ROOT / rel
        print(f"  [{'ok' if p.exists() else 'MISSING'}] {rel}")

    if args.no_smoke:
        return
    print("\nsmoke test: Benchmark A, 20 Adam steps, tiny network ...")
    import pinnflow

    pinnflow.configure_jax()
    from pinnflow.configs import get_config
    from pinnflow.problems import TaylorGreen2DPINN
    from pinnflow.training import Trainer

    cfg = get_config("tgv2d", "G", {"arch.num_layers": 2, "arch.hidden_dim": 32, "arch.fourier_emb.embed_dim": 32, "training.res_batch_size": 256, "training.ic_batch_size": 64, "optim.warmup_steps": 5})
    t0 = time.time()
    tr = Trainer(TaylorGreen2DPINN(cfg), cfg, ROOT / "runs" / "_smoke", jax.random.PRNGKey(0))
    tr.train(20, log_every=10, eval_every=20, ckpt_every=None)
    print(f"smoke test passed in {time.time() - t0:.1f}s -> runs/_smoke/metrics.csv")


if __name__ == "__main__":
    main()
