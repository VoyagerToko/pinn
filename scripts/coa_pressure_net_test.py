"""Does joint training with the separate pressure network (problem.pressure_net) put the pressure drop into the pressure?

Warm start: the velocity network comes from a trained run made with the shared pressure output (its fourth output
column is dropped), the pressure network starts fresh, and both train together with the coa_mri defaults (RAD on).
The question is only whether the peak drop climbs from ~0.3 mmHg towards the ~7 mmHg that a pressure fitted alone
to the same velocity reaches (scripts/coa_pressure_fit.py); `evaluate` runs every 500 steps.

    python scripts/coa_pressure_net_test.py                          # from runs/coa_P2_baseline, 3000 steps
    python scripts/coa_pressure_net_test.py --from coa_P2_baseline --steps 6000

The warm-started run goes to runs/_test_pnet_warm (metrics.csv, final line printed). For a test from scratch use
train.py: python scripts/train.py --benchmark coa_mri --set "problem.case='P2_baseline'" --steps 20000 \\
--workdir runs/_test_pnet_cold
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow.configs import get_config  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.training import Trainer  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default="coa_P2_baseline", help="trained run (shared pressure output) under runs/")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workdir", default="runs/_test_pnet_warm")
    args = ap.parse_args()

    old_dir = ROOT / "runs" / args.src
    old_cfg = ml_collections.ConfigDict(json.loads((old_dir / "config.json").read_text()))
    if old_cfg.problem.get("pressure_net", False):
        raise SystemExit(f"{args.src} already has a separate pressure network; warm-start from a run made without one")
    old_pr = PROBLEMS["coa_mri"](old_cfg)
    old_params, _ = load_params(old_dir / "latest.msgpack", template=old_pr.init_params(jax.random.PRNGKey(0)))

    case = str(old_cfg.problem.case)
    cfg = get_config("coa_mri", "G", {"problem.case": case, "problem.eddy_viscosity": bool(old_cfg.problem.get("eddy_viscosity", False)),
                                      "logging.log_every_steps": 100, "logging.eval_every_steps": 500})
    pr = PROBLEMS["coa_mri"](cfg)
    tr = Trainer(pr, cfg, ROOT / args.workdir, jax.random.PRNGKey(0))
    new = tr.state.params

    def take(n, o):  # the same layers; the output layer drops the old pressure column
        return o if n.shape == o.shape else o[..., : n.shape[-1]]

    if old_pr.eddy:  # the old output layer is (u, v, w, p, nu_t); the new velocity network's is (u, v, w, nu_t)
        def take(n, o):  # noqa: F811
            if n.shape == o.shape:
                return o
            return jax.numpy.concatenate([o[..., :3], o[..., 4:5]], -1)

    vel = jax.tree_util.tree_map(take, new["vel"], old_params)
    tr.state = tr.state.replace(params={"vel": vel, "p": new["p"]})
    ev0 = pr.evaluate(tr.state.params)
    print(f"[test] {case}: start: velocity fit {ev0['velocity_fit_rel']:.3f} (the trained flow), peak drop {ev0['dp_peak_pinn_mmHg']:.2f} mmHg "
          f"(fresh pressure network); catheter {ev0['dp_peak_cath_mmHg']:.1f}", flush=True)
    t0 = time.time()
    tr.train(args.steps, log_every=100, eval_every=500, ckpt_every=None, rad_every=cfg.training.rad_every)
    tr.save()
    ev = pr.evaluate(tr.state.params)
    print(f"[test] after {args.steps} steps ({time.time() - t0:.0f} s): peak drop {ev['dp_peak_pinn_mmHg']:.2f} mmHg, mean {ev['dp_mean_pinn_mmHg']:.2f} "
          f"(catheter {ev['dp_peak_cath_mmHg']:.1f}, {ev['dp_mean_cath_mmHg']:.2f}); velocity fit {ev['velocity_fit_rel']:.3f}", flush=True)
    (ROOT / args.workdir / "final_eval.json").write_text(json.dumps(ev, indent=1))


if __name__ == "__main__":
    main()
