"""Figures and animations for the 2D benchmarks (A, B, C). The trained network is a continuous
function, so every frame is sampled directly at the requested resolution - no interpolation.

    python scripts/plot2d.py --benchmark tgv2d    --workdir runs/tgv2d_G                 # fields vs exact + error maps
    python scripts/plot2d.py --benchmark tgv2d    --workdir runs/tgv2d_G --animate 120   # vorticity movie (mp4 or gif)
    python scripts/plot2d.py --benchmark cavity   --workdir runs/cavity_F                # streamlines + Ghia profiles per Re
    python scripts/plot2d.py --benchmark cylinder --workdir runs/cylinder_G --animate 200 # vortex street snapshots + movie

Outputs go to <workdir>/figures/.
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow.benchmarks import ghia_tables  # noqa: E402
from pinnflow.losses import align_pressure_gauge  # noqa: E402
from pinnflow.physics import velocity_gradient, vorticity_from_grad  # noqa: E402
from pinnflow.problems import PROBLEMS, DFGCylinderPINN  # noqa: E402
from pinnflow.utils import chunked_vmap, load_params  # noqa: E402


# --------------------------------------------------------------------------------------
def load_cfg(workdir: Path) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(json.loads((workdir / "config.json").read_text()))


def restore(problem, path: Path):
    params, _ = load_params(path, template=problem.init_params(jax.random.PRNGKey(0)))
    return params


def fields_2d(vel_fn, pts: jnp.ndarray, unsteady: bool = True):
    """(u, v, p, vorticity) on points (N, d) for a point-wise (u, v, p) function."""
    g = velocity_gradient(vel_fn, dim=2, unsteady=unsteady)

    def one(z):
        uvp = vel_fn(z)
        _, grad_u = g(z)
        return uvp, vorticity_from_grad(grad_u)

    uvp, w = chunked_vmap(one, pts, chunk=16384)
    return np.asarray(uvp[:, 0]), np.asarray(uvp[:, 1]), np.asarray(uvp[:, 2]), np.asarray(w)


def _save_movie(fig, update, n_frames: int, fps: int, path: Path):
    """Write an mp4 via imageio-ffmpeg's binary, falling back to a gif."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(str(path.with_suffix(".mp4")), writer=FFMpegWriter(fps=fps, bitrate=4000))
        return path.with_suffix(".mp4")
    except Exception as e:  # no ffmpeg available
        print("mp4 failed (", e, ") - writing gif")
        anim.save(str(path.with_suffix(".gif")), writer=PillowWriter(fps=min(fps, 20)))
        return path.with_suffix(".gif")


# --------------------------------------------------------------------------------------
# Benchmark A
# --------------------------------------------------------------------------------------
def plot_tgv2d(cfg, workdir: Path, ckpt: str, n: int, times, animate: int, fps: int):
    problem = PROBLEMS["tgv2d"](cfg)
    params = restore(problem, workdir / ckpt)
    vel = problem.velocity_fn(params)
    b = problem.bench
    out = workdir / "figures"
    out.mkdir(exist_ok=True)
    g = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(g, g, indexing="ij")
    times = [float(t) for t in (times or np.linspace(0, b.T, 3))]

    for t in times:
        pts = jnp.stack([jnp.full(X.size, t), jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], -1)
        u, v, p, w = fields_2d(vel, pts)
        ue, ve, pe = (np.asarray(a) for a in b.exact(t, X.ravel(), Y.ravel()))
        we = np.asarray(2 * np.sin(X.ravel()) * np.sin(Y.ravel()) * np.exp(-2 * t / b.Re))  # exact vorticity v_x - u_y
        p = np.asarray(align_pressure_gauge(jnp.asarray(p), jnp.asarray(pe)))
        rows = [("speed", np.hypot(u, v), np.hypot(ue, ve)), ("vorticity", w, we), ("pressure", p, pe)]
        fig, ax = plt.subplots(3, 3, figsize=(12, 11))
        for i, (name, pred, ref) in enumerate(rows):
            vmin, vmax = ref.min(), ref.max()
            for j, (arr, title) in enumerate([(pred, f"PINN {name}"), (ref, f"exact {name}")]):
                im = ax[i, j].pcolormesh(X, Y, arr.reshape(n, n), cmap="RdBu_r" if name != "speed" else "viridis", vmin=vmin, vmax=vmax, shading="auto")
                ax[i, j].set_title(title)
                fig.colorbar(im, ax=ax[i, j], shrink=0.8)
            err = np.abs(pred - ref).reshape(n, n)
            im = ax[i, 2].pcolormesh(X, Y, err, cmap="magma", shading="auto")
            ax[i, 2].set_title(f"|error|  max={err.max():.2e}  relL2={np.linalg.norm(pred - ref) / np.linalg.norm(ref):.2e}")
            fig.colorbar(im, ax=ax[i, 2], shrink=0.8)
            for a in ax[i]:
                a.set_aspect("equal")
        fig.suptitle(f"Taylor-Green 2D, Re={b.Re:g}, t={t:.3f}")
        fig.tight_layout()
        f = out / f"tgv2d_t{t:.3f}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print("saved", f)

    if animate:
        ts = np.linspace(0, b.T, animate)
        pts0 = jnp.stack([jnp.zeros(X.size), jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], -1)
        frames = []
        for t in ts:
            _, _, _, w = fields_2d(vel, pts0.at[:, 0].set(t))
            frames.append(w.reshape(n, n))
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.pcolormesh(X, Y, frames[0], cmap="RdBu_r", vmin=-2, vmax=2, shading="auto")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="vorticity")
        title = ax.set_title("t = 0.000")

        def update(i):
            im.set_array(frames[i].ravel())
            title.set_text(f"t = {ts[i]:.3f}")
            return im, title

        f = _save_movie(fig, update, len(frames), fps, out / "tgv2d_vorticity")
        plt.close(fig)
        print("saved", f)


# --------------------------------------------------------------------------------------
# Benchmark B
# --------------------------------------------------------------------------------------
def plot_cavity(cfg, workdir: Path, ckpt: str, n: int):
    problem = PROBLEMS["cavity"](cfg)
    out = workdir / "figures"
    out.mkdir(exist_ok=True)
    ckpts = sorted(workdir.glob("Re*.msgpack")) or [workdir / ckpt]
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    for ck in ckpts:
        Re = int(ck.stem[2:]) if ck.stem.startswith("Re") else int(round(float(cfg.problem.Re)))
        problem.set_Re(Re)
        params = restore(problem, ck)
        vel = problem.velocity_fn(params)
        pts = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], -1)
        u, v, p, w = fields_2d(vel, pts, unsteady=False)
        U, V, W = u.reshape(n, n), v.reshape(n, n), w.reshape(n, n)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        sp = ax[0].pcolormesh(X, Y, np.hypot(U, V), cmap="viridis", shading="auto")
        ax[0].streamplot(g, g, U.T, V.T, color="w", density=1.4, linewidth=0.6)
        ax[0].set_title(f"speed + streamlines, Re={Re}")
        fig.colorbar(sp, ax=ax[0])
        vo = ax[1].pcolormesh(X, Y, np.clip(W, -10, 10), cmap="RdBu_r", shading="auto")
        ax[1].set_title("vorticity (clipped to +-10)")
        fig.colorbar(vo, ax=ax[1])
        for a in ax[:2]:
            a.set_aspect("equal")
        # centreline profiles against Ghia et al. (unit lid; our lid is regularised near the corners)
        try:
            tab = ghia_tables(Re)
            yq = np.linspace(0, 1, 201)
            uc = np.asarray(chunked_vmap(vel, jnp.stack([jnp.full(201, 0.5), jnp.asarray(yq)], -1))[:, 0])
            vc = np.asarray(chunked_vmap(vel, jnp.stack([jnp.asarray(yq), jnp.full(201, 0.5)], -1))[:, 1])
            ax[2].plot(uc, yq, "b-", label="PINN u(0.5, y)")
            ax[2].plot(tab["u"], tab["y"], "bo", ms=4, label="Ghia u")
            ax[2].plot(yq, vc, "r-", label="PINN v(x, 0.5)")
            ax[2].plot(tab["x"], tab["v"], "rs", ms=4, label="Ghia v")
            ax[2].set_xlabel("u  or  x"), ax[2].set_ylabel("y  or  v"), ax[2].legend(fontsize=8), ax[2].grid(alpha=0.3)
            ax[2].set_title("centreline profiles vs Ghia (1982)")
        except KeyError:
            ax[2].axis("off")
        fig.tight_layout()
        f = out / f"cavity_Re{Re}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print("saved", f)


# --------------------------------------------------------------------------------------
# Benchmark C
# --------------------------------------------------------------------------------------
def plot_cylinder(cfg, workdir: Path, ckpt: str, n: int, times, animate: int, fps: int):
    out = workdir / "figures"
    out.mkdir(exist_ok=True)
    dt = float(cfg.problem.window_dt)
    windows = sorted(workdir.glob("window_*"))
    if not windows:
        raise FileNotFoundError("no window_* directories in workdir")
    bench = DFGCylinderPINN(cfg).bench
    nx, ny = n, max(8, int(n * bench.height / bench.length))
    x = np.linspace(0, bench.length, nx)
    y = np.linspace(0, bench.height, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = ~np.asarray(bench.in_fluid(X, Y))

    def window_for(t):
        idx = min(int(t // dt), len(windows) - 1)
        w = windows[idx]
        problem = DFGCylinderPINN(cfg, t0=idx * dt, t1=(idx + 1) * dt)
        return problem, restore(problem, w / ckpt)

    def snapshot(t):
        problem, params = window_for(t)
        vel = problem.velocity_dim_fn(params)
        pts = jnp.stack([jnp.full(X.size, t), jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], -1)
        u, v, p, w = fields_2d(vel, pts)
        return [np.ma.array(a.reshape(nx, ny), mask=mask) for a in (np.hypot(u, v), w, p)]

    t_end = len(windows) * dt
    times = [float(t) for t in (times or np.linspace(0.25 * t_end, t_end - 1e-6, 3))]
    for t in times:
        speed, w, p = snapshot(t)
        fig, ax = plt.subplots(3, 1, figsize=(14, 8))
        for a, arr, name, cmap, lim in zip(ax, (speed, w, p), ("speed [m/s]", "vorticity [1/s]", "pressure"), ("viridis", "RdBu_r", "coolwarm"), (None, 40, None)):
            kw = {"vmin": -lim, "vmax": lim} if lim else {}
            im = a.pcolormesh(X, Y, arr, cmap=cmap, shading="auto", **kw)
            a.add_patch(plt.Circle(bench.center, bench.radius, color="k"))
            a.set_aspect("equal"), a.set_title(f"{name}, t = {t:.3f} s")
            fig.colorbar(im, ax=a, shrink=0.9)
        fig.tight_layout()
        f = out / f"cylinder_t{t:.3f}.png"
        fig.savefig(f, dpi=130)
        plt.close(fig)
        print("saved", f)

    if animate:
        ts = np.linspace(0, t_end - 1e-6, animate)
        frames = [snapshot(t)[1] for t in ts]
        fig, ax = plt.subplots(figsize=(14, 3.2))
        im = ax.pcolormesh(X, Y, frames[0], cmap="RdBu_r", vmin=-40, vmax=40, shading="auto")
        ax.add_patch(plt.Circle(bench.center, bench.radius, color="k"))
        ax.set_aspect("equal")
        title = ax.set_title("t = 0.000 s")

        def update(i):
            im.set_array(frames[i].ravel())
            title.set_text(f"vorticity, t = {ts[i]:.3f} s")
            return im, title

        f = _save_movie(fig, update, len(frames), fps, out / "cylinder_vorticity")
        plt.close(fig)
        print("saved", f)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=["tgv2d", "cavity", "cylinder"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--checkpoint", default="latest.msgpack")
    ap.add_argument("--n", type=int, default=256, help="grid points per axis (x axis for the cylinder)")
    ap.add_argument("--times", default=None, help="comma-separated snapshot times, e.g. 0,0.5,1")
    ap.add_argument("--animate", type=int, default=0, help="number of movie frames (0 = no movie)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    workdir = Path(args.workdir)
    cfg = load_cfg(workdir)
    times = [float(t) for t in args.times.split(",")] if args.times else None
    if args.benchmark == "tgv2d":
        plot_tgv2d(cfg, workdir, args.checkpoint, args.n, times, args.animate, args.fps)
    elif args.benchmark == "cavity":
        plot_cavity(cfg, workdir, args.checkpoint, args.n)
    else:
        plot_cylinder(cfg, workdir, args.checkpoint, args.n, times, args.animate, args.fps)


if __name__ == "__main__":
    main()
