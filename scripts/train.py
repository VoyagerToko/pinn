"""Train one benchmark at one ablation row (handbook STEP 6 / STEP 8).

Examples
    python scripts/train.py --benchmark tgv2d   --ablation G
    python scripts/train.py --benchmark cavity  --ablation F --steps 60000          # curriculum 100->400->1000
    python scripts/train.py --benchmark cylinder --ablation G --windows 4 --steps 50000
    python scripts/train.py --benchmark tgv3d   --ablation H --set training.n_per_axis="(16,24,24,24)"
    python scripts/train.py --benchmark cylinder_inverse
    python scripts/train.py --benchmark deeponet_cavity
    python scripts/train.py --benchmark tgv2d --ablation A --steps 2000 --lbfgs 0 --workdir runs/smoke

--set key=value overrides any config entry (values parsed as python literals).
Metrics go to <workdir>/metrics.csv, checkpoints to <workdir>/*.msgpack, the final evaluation to
<workdir>/final_eval.json. Add --wandb to mirror the logs to Weights & Biases.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402

from pinnflow.configs import get_config  # noqa: E402
from pinnflow.problems import PROBLEMS, CavityPINN, DFGCylinderPINN  # noqa: E402
from pinnflow.training import Trainer, train_curriculum, train_time_windows  # noqa: E402


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--ablation", default="G", help="ablation row A-H (STEP 8)")
    ap.add_argument("--steps", type=int, default=None, help="Adam steps (per window for cylinder; total for the cavity curriculum)")
    ap.add_argument("--lbfgs", type=int, default=None, help="L-BFGS iterations after Adam (0 disables)")
    ap.add_argument("--windows", type=int, default=None, help="number of time windows (cylinder)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()

    cfg = get_config(args.benchmark, args.ablation, parse_overrides(args.set))
    if args.seed is not None:
        cfg.seed = args.seed
    if args.steps is not None:
        cfg.training.max_steps = args.steps
    if args.lbfgs is not None:
        cfg.training.lbfgs_steps = args.lbfgs
    workdir = Path(args.workdir or ROOT / "runs" / cfg.name)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "config.json").write_text(cfg.to_json_best_effort(indent=2))
    print(f"[train] devices: {jax.devices()}")
    print(f"[train] {cfg.name}: arch={cfg.arch.arch_name} scheme={cfg.weighting.scheme} causal={cfg.weighting.use_causal} rad_every={cfg.training.rad_every}")

    key = jax.random.PRNGKey(int(cfg.seed))
    log_kw = dict(log_every=cfg.logging.log_every_steps, eval_every=cfg.logging.eval_every_steps, ckpt_every=cfg.logging.ckpt_every_steps, rad_every=cfg.training.rad_every)

    if args.benchmark == "cavity":
        problem = CavityPINN(cfg)
        trainer = Trainer(problem, cfg, workdir, key, use_wandb=args.wandb)
        Re_list = list(cfg.problem.curriculum_Re)
        steps = list(cfg.problem.curriculum_steps)
        if args.steps is not None:  # rescale the curriculum to the requested total
            tot = sum(steps)
            steps = [max(1, int(round(args.steps * s / tot))) for s in steps]
        train_curriculum(trainer, Re_list, steps, **log_kw)
        if cfg.training.lbfgs_steps:
            trainer.lbfgs(int(cfg.training.lbfgs_steps))
        final = problem.evaluate(trainer.state.params)

    elif args.benchmark == "cylinder":
        dt = float(cfg.problem.window_dt)
        n_windows = args.windows or int(cfg.problem.num_time_windows)

        def make_trainer(idx, ic_fn):
            problem = DFGCylinderPINN(cfg, t0=idx * dt, t1=(idx + 1) * dt, ic_fn=ic_fn)
            return Trainer(problem, cfg, workdir / f"window_{idx:02d}", jax.random.fold_in(key, idx), use_wandb=args.wandb)

        trainers = train_time_windows(make_trainer, n_windows, int(cfg.training.max_steps), **log_kw)
        if cfg.training.lbfgs_steps:
            trainers[-1].lbfgs(int(cfg.training.lbfgs_steps))
        final = {f"window_{i:02d}/{k}": v for i, tr in enumerate(trainers) for k, v in tr.problem.evaluate(tr.state.params).items()}

    else:
        problem = PROBLEMS[args.benchmark](cfg)
        trainer = Trainer(problem, cfg, workdir, key, use_wandb=args.wandb)
        trainer.train(int(cfg.training.max_steps), **log_kw)
        if cfg.training.lbfgs_steps:
            trainer.lbfgs(int(cfg.training.lbfgs_steps))
        final = problem.evaluate(trainer.state.params)

    (workdir / "final_eval.json").write_text(json.dumps(final, indent=2))
    print("[train] final evaluation:")
    for k, v in final.items():
        print(f"    {k:>32s} = {v:.4e}" if isinstance(v, float) else f"    {k:>32s} = {v}")


if __name__ == "__main__":
    main()
