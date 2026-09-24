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
from typing import Dict

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
    # reference dissipation curve digitised from DeBonis (2013) Fig. 4(a) (scripts/digitize_tgv_reference.py)
    ref_path = ROOT / "data" / "tgv3d_re1600" / "debonis2013_fig4a_ref_dissipation.csv"
    ref = None
    if ref_path.exists():
        ref = np.loadtxt(ref_path, delimiter=",", skiprows=1)
        j = int(np.argmax(ref[:, 1]))
        out["ref/dissipation_peak"] = float(ref[j, 1])
        out["ref/dissipation_peak_time"] = float(ref[j, 0])
        out["ref/source"] = "digitised from DeBonis 2013 NASA/TM-2013-217850 Fig. 4(a)"
        out["dissipation_peak_rel_err"] = float(abs(out["dissipation_peak"] - ref[j, 1]) / ref[j, 1])
        out["dissipation_peak_from_enstrophy_rel_err"] = float(abs(out["dissipation_peak_from_enstrophy"] - ref[j, 1]) / ref[j, 1])
        eps_ref = np.interp(hist["t"], ref[:, 0], ref[:, 1])
        out["dissipation_rel_l2_vs_ref"] = float(np.linalg.norm(eps_E - eps_ref) / np.linalg.norm(eps_ref))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ax[0].plot(hist["t"], hist["Ek"], label="PINN")
        ax[0].set_xlabel("t"), ax[0].set_ylabel("E_k"), ax[0].set_title("kinetic energy")
        ax[1].plot(hist["t"], eps_E, label="PINN  -dE_k/dt")
        ax[1].plot(hist["t"], eps_Z, "--", label="PINN  2 nu zeta")
        if ref is not None:
            ax[1].plot(ref[:, 0], ref[:, 1], "k-", lw=1.0, label="reference (digitised, DeBonis 2013)")
        ax[1].set_xlabel("t"), ax[1].set_ylabel("epsilon"), ax[1].set_title("dissipation rate"), ax[1].legend(fontsize=8)
        kk = k[1:]
        ax[2].loglog(kk, E[1:], label=f"PINN, t = {t_peak:.1f}")
        ax[2].loglog(kk[3:17], E[4] * (kk[3:17] / kk[3]) ** (-5.0 / 3.0), "k:", label="k^-5/3")
        ax[2].set_xlabel("k"), ax[2].set_ylabel("E(k)"), ax[2].set_title("energy spectrum"), ax[2].legend(fontsize=8)
        fig.tight_layout(), fig.savefig(workdir / "energy_dissipation_spectrum.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    return out


def step_time(workdir: Path) -> Dict[str, float]:
    """Median seconds per Adam step from metrics.csv (``time`` restarts at every stage/window, so
    only consecutive rows of the same stage are differenced; the first interval holds the compile)."""
    import csv

    files = sorted(workdir.glob("window_*/metrics.csv")) or [workdir / "metrics.csv"]
    rates = []
    for f in files:
        if not f.exists():
            continue
        rows = [r for r in csv.DictReader(open(f)) if not r.get("loss/lbfgs")]
        for a, b in zip(rows, rows[1:]):
            ta, tb, sa, sb = float(a["time"]), float(b["time"]), float(a["step"]), float(b["step"])
            if tb > ta and sb > sa:  # same stage (time restarts per stage/window)
                rates.append((tb - ta) / (sb - sa))
    out = {}
    if rates:
        out["cost/step_time_s"] = float(np.median(rates))
    peak = []
    for f in files:
        if f.exists():
            peak += [float(r["mem/peak_gb"]) for r in csv.DictReader(open(f)) if r.get("mem/peak_gb") not in (None, "", "nan")]
    if peak:
        out["cost/jax_peak_gb"] = float(max(peak))
    return out


def cost_summary(workdir: Path, throughput_fn=None, n_points: int = 1 << 18, dim_box=None) -> Dict[str, float]:
    """Handbook 7.6: training wall-clock (cost.json written by scripts/run_job.sh), step time, peak memory
    (JAX allocator and nvidia-smi samples), inference throughput in query points per second."""
    out = step_time(workdir)
    cj = workdir / "cost.json"
    if cj.exists():
        c = json.loads(cj.read_text())
        out["cost/train_wall_s"] = float(c.get("train_wall_s", float("nan")))
        base = float(c.get("gpu_baseline_mib", 0.0))
        gpu = workdir.parent / f"{workdir.name}.gpu.csv"
        if gpu.exists():
            used = [float(l.split(",")[1]) for l in gpu.read_text().splitlines() if l.count(",") >= 2]
            if used:
                out["cost/nvidia_smi_peak_mib"] = max(used)
                out["cost/nvidia_smi_peak_minus_baseline_mib"] = max(used) - base
    if throughput_fn is not None and dim_box is not None:
        box = np.asarray(dim_box, dtype=np.float32)
        pts = jnp.asarray(np.random.default_rng(0).uniform(box[:, 0], box[:, 1], (n_points, box.shape[0])).astype(np.float32))
        out["cost/inference_pts_per_s"] = float(M.inference_throughput(throughput_fn, pts))
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
    thr_fn, box = None, None
    if args.benchmark == "cylinder":
        out = eval_cylinder(cfg, workdir)
        last = sorted(workdir.glob("window_*"))[-1]
        idx = int(last.name.split("_")[1])
        dt = float(cfg.problem.window_dt)
        problem = DFGCylinderPINN(cfg, t0=idx * dt, t1=(idx + 1) * dt)
        params = restore(problem, last / "latest.msgpack")
        thr_fn = jax.vmap(problem.velocity_dim_fn(params))
        box = [[idx * dt, (idx + 1) * dt], [0.0, problem.bench.length], [0.0, problem.bench.height]]
    elif args.benchmark == "tgv3d":
        out = eval_tgv3d(cfg, workdir, spectrum_n=args.spectrum_n)
        problem = PROBLEMS["tgv3d"](cfg)
        params = restore(problem, workdir / args.checkpoint)
        vel = problem.velocity_fn(params)
        thr_fn = jax.vmap(lambda z: vel(z)[:3])
        box = np.asarray(problem.dom)
        # separable evaluation: 64^3 spatial grid at one time instant = n^3 points per call
        from pinnflow.sampling import linspace_axes

        axes = linspace_axes(problem.dom[1:], (64, 64, 64))
        grid = jax.jit(lambda t: problem.velocity_grid(params, [t, *axes]))
        t1 = jnp.asarray([9.0])
        grid(t1).block_until_ready()
        import time as _time

        s = _time.perf_counter()
        for _ in range(3):
            grid(t1).block_until_ready()
        out["cost/inference_grid_pts_per_s"] = float(3 * 64**3 / (_time.perf_counter() - s))
    else:
        problem = PROBLEMS[args.benchmark](cfg)
        if args.benchmark == "cavity":
            out = {}
            for ck in sorted(workdir.glob("Re*.msgpack")):
                Re = int(ck.stem[2:])
                problem.set_Re(Re)
                params = restore(problem, ck)
                out.update({f"Re{Re}/{k}": v for k, v in problem.evaluate(params).items()})
            # Re*.msgpack are written after each Adam stage; latest.msgpack also includes L-BFGS
            Re_final = int(list(cfg.problem.curriculum_Re)[-1])
            problem.set_Re(Re_final)
            params = restore(problem, workdir / args.checkpoint)
            out.update({f"final_Re{Re_final}/{k}": v for k, v in problem.evaluate(params).items()})
            thr_fn = jax.vmap(problem.velocity_fn(params))
            box = np.asarray(problem.dom)
        else:
            params = restore(problem, workdir / args.checkpoint)
            out = problem.evaluate(params)
            if args.benchmark == "deeponet_cavity":
                Re_eval = float(cfg.problem.get("zero_shot_Re", 1000))
                thr_fn = jax.vmap(problem.net(params, Re_eval))
            else:
                thr_fn = jax.vmap(problem.velocity_fn(params))
            box = np.asarray(problem.dom)
    out.update(cost_summary(workdir, thr_fn, dim_box=box))
    (workdir / "eval.json").write_text(json.dumps(out, indent=2, default=str))
    for k, v in out.items():
        print(f"{k:>40s} : {v}")


if __name__ == "__main__":
    main()
