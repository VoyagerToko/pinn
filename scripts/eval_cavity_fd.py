"""Re-grade finished cavity runs against the finite-difference regularised-lid references (CPU is fine).

    JAX_PLATFORMS=cpu python scripts/eval_cavity_fd.py runs/cavity_C_r2 runs/cavity_D ...

Writes <run>/eval_fd.json with CavityPINN.evaluate() for every Re*.msgpack (end of each Adam stage) and for
latest.msgpack (after L-BFGS if it ran), without touching eval.json (whose cost numbers were measured on
the GPU during the run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402


def main():
    for wd in map(Path, sys.argv[1:]):
        cfg = ml_collections.ConfigDict(json.loads((wd / "config.json").read_text()))
        problem = PROBLEMS["cavity"](cfg)
        template = problem.init_params(jax.random.PRNGKey(0))
        out = {}
        for ck in sorted(wd.glob("Re*.msgpack")):
            Re = int(ck.stem[2:])
            problem.set_Re(Re)
            params, _ = load_params(ck, template=template)
            out.update({f"Re{Re}/{k}": v for k, v in problem.evaluate(params).items()})
        Re_final = int(list(cfg.problem.curriculum_Re)[-1])
        problem.set_Re(Re_final)
        params, _ = load_params(wd / "latest.msgpack", template=template)
        out.update({f"final_Re{Re_final}/{k}": v for k, v in problem.evaluate(params).items()})
        (wd / "eval_fd.json").write_text(json.dumps(out, indent=2))
        print(wd.name, {k: round(v, 4) for k, v in out.items() if "vel_vs_fd" in k or "ghia" in k})


if __name__ == "__main__":
    main()
