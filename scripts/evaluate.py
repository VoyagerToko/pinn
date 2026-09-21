"""STEP 7 validation of a trained run.

    python scripts/evaluate.py --benchmark cylinder --workdir runs/cylinder_G
    python scripts/evaluate.py --benchmark tgv3d   --workdir runs/tgv3d_H --spectrum-n 64
    python scripts/evaluate.py --benchmark tgv2d   --workdir runs/tgv2d_G

Writes <workdir>/eval.json (+ CSV/PNG for time series) and prints a comparison against the
reference values stored in :mod:`pinnflow.benchmarks` / downloaded under data/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow import metrics as M  # noqa: E402
from pinnflow.data import load_featflow_series  # noqa: E402
from pinnflow.problems import PROBLEMS, DFGCylinderPINN  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402


def load_cfg(workdir: Path) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(json.loads((workdir / "config.json").read_text()))


def restore(problem, path: Path):
    template = problem.init_params(jax.random.PRNGKey(0))
    params, extra = load_params(path, template=template)
    return params


def eval_cylinder(cfg, workdir: Path, dt_eval: float = 0.005):
    dt = float(cfg.problem.window_dt)
    windows = sorted(workdir.glob("window_*"))
    series = {"t": [], "Cd": [], "Cl": [], "dP": []}
    for w in windows:
        idx = int(w.name.split("_")[1])
        problem = DFGCylinderPINN(cfg, t0=idx * dt, t1=(idx + 1) * dt)
        params = restore(problem, w / "latest.msgpack")
        times = np.arange(idx * dt, (idx + 1) * dt, dt_eval)
        s = M.drag_lift_series(problem.velocity_dim_fn(params), times, problem.bench)
        for k in series:
            series[k].append(s[k])
    series = {k: np.concatenate(v) for k, v in series.items()}
    np.savetxt(workdir / "drag_lift.csv", np.stack([series[k] for k in ("t", "Cd", "Cl", "dP")], 1), delimiter=",", header="t,Cd,Cl,dP", comments="")
    out = {}
    bench = DFGCylinderPINN(cfg).bench
    if len(series["t"]) > 20:
        St, f = M.strouhal(series["t"], series["Cl"], D=bench.diameter, U=bench.U_ref)
        out.update({"St": St, "f": f, **M.periodic_cycle_stats(series["t"], series["Cd"], series["Cl"])})
    for k, (lo, hi) in bench.reference.items():
        if k.startswith(bench.variant):
            out[f"ref/{k}"] = [lo, hi]
    try:
        ref = load_featflow_series("draglift", bench.variant)
        out["featflow/Cd_max"] = float(ref["Cd"][len(ref["Cd"]) // 2 :].max())
        out["featflow/Cl_max"] = float(ref["Cl"][len(ref["Cl"]) // 2 :].max())
        out["featflow/file"] = ref["file"]
    except FileNotFoundError:
        pass
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(series["t"], series["Cd"], label="PINN")
        ax[1].plot(series["t"], series["Cl"], label="PINN")
        try:
            ax[0].plot(ref["t"], ref["Cd"], "k--", lw=0.8, label="FEATFLOW")
            ax[1].plot(ref["t"], ref["Cl"], "k--", lw=0.8, label="FEATFLOW")
        except Exception:
            pass
        ax[0].set_ylabel("C_D"), ax[1].set_ylabel("C_L"), ax[1].set_xlabel("t [s]"), ax[0].legend()
        fig.tight_layout(), fig.savefig(workdir / "drag_lift.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    return out


def eval_tgv3d(cfg, workdir: Path, n_hist: int = 32, spectrum_n: int = 64, n_times: int = 201):
    problem = PROBLEMS["tgv3d"](cfg)
    params = restore(problem, workdir / "latest.msgpack")
    times = np.linspace(0.0, problem.bench.T, n_times)
    hist = problem.energy_history(params, times, n=n_hist)
    eps_E = M.dissipation_from_energy(hist["t"], hist["Ek"])
    eps_Z = M.dissipation_from_enstrophy(hist["enstrophy"], problem.bench.nu)
    np.savetxt(workdir / "energy_history.csv", np.stack([hist["t"], hist["Ek"], hist["enstrophy"], eps_E, eps_Z], 1), delimiter=",", header="t,Ek,enstrophy,eps_from_energy,eps_from_enstrophy", comments="")
    i = int(np.argmax(eps_E))
    out = {
        "Ek0": float(hist["Ek"][0]), "Ek0_exact": problem.bench.initial_energy(),
        "dissipation_peak": float(eps_E[i]), "dissipation_peak_time": float(hist["t"][i]),
        "dissipation_peak_from_enstrophy": float(eps_Z.max()),
        "ref/dissipation_peak": problem.bench.dissipation_peak_value, "ref/dissipation_peak_time": problem.bench.dissipation_peak_time,
    }
    # 7.5 spectrum at the dissipation peak time
    from pinnflow.sampling import linspace_axes

    axes = linspace_axes(problem.dom[1:], (spectrum_n,) * 3)
    t_peak = float(hist["t"][i])
    u = np.asarray(problem.velocity_grid(params, [jnp.asarray([t_peak]), *axes])[:3, 0])
    k, E = M.energy_spectrum(u)
    np.savetxt(workdir / f"spectrum_t{t_peak:.1f}.csv", np.stack([k, E], 1), delimiter=",", header="k,E", comments="")
    out["spectrum_slope_k4_16"] = M.inertial_range_slope(k, E)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--checkpoint", default="latest.msgpack")
    ap.add_argument("--spectrum-n", type=int, default=64)
    args = ap.parse_args()
    workdir = Path(args.workdir)
    cfg = load_cfg(workdir)
    if args.benchmark == "cylinder":
        out = eval_cylinder(cfg, workdir)
    elif args.benchmark == "tgv3d":
        out = eval_tgv3d(cfg, workdir, spectrum_n=args.spectrum_n)
    else:
        problem = PROBLEMS[args.benchmark](cfg)
        if args.benchmark == "cavity":
            out = {}
            for ck in sorted(workdir.glob("Re*.msgpack")):
                Re = int(ck.stem[2:])
                problem.set_Re(Re)
                params = restore(problem, ck)
                out.update({f"Re{Re}/{k}": v for k, v in problem.evaluate(params).items()})
            if not out:
                out = problem.evaluate(restore(problem, workdir / args.checkpoint))
        else:
            out = problem.evaluate(restore(problem, workdir / args.checkpoint))
    (workdir / "eval.json").write_text(json.dumps(out, indent=2, default=str))
    for k, v in out.items():
        print(f"{k:>40s} : {v}")


if __name__ == "__main__":
    main()
