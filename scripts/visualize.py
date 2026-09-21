"""STEP 9 - sample a trained 3D model on a grid, extract vortex tubes, render, encode.

    python scripts/visualize.py --workdir runs/tgv3d_H --n 128 --frames 600 --fps 60
    python scripts/visualize.py --workdir runs/tgv3d_H --n 64 --frames 60 --no-render      # .vti/.vtp only (ParaView)
    python scripts/visualize.py --workdir runs/tgv3d_H --particles 5000                    # 9.3 particle advection -> npz

Requires the ``viz`` extras (pyvista, vtk, imageio-ffmpeg).
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

from pinnflow import viz  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--checkpoint", default="latest.msgpack")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--particles", type=int, default=0)
    args = ap.parse_args()
    workdir = Path(args.workdir)
    cfg = ml_collections.ConfigDict(json.loads((workdir / "config.json").read_text()))
    problem = PROBLEMS["tgv3d"](cfg)
    params, _ = load_params(workdir / args.checkpoint, template=problem.init_params(jax.random.PRNGKey(0)))
    vel_p = problem.velocity_fn(params)
    T = float(problem.bench.T)
    times = np.linspace(0.0, T, args.frames)

    if args.particles > 0:
        seeds = np.random.default_rng(0).uniform(0, 2 * np.pi, (args.particles, 3))
        batched = lambda t, X: jax.vmap(lambda x: vel_p(jnp.concatenate([jnp.array([t]), x]))[:3])(X)
        traj = viz.advect_particles(batched, seeds, (0.0, T), n_frames=args.frames)
        np.savez_compressed(workdir / "particles.npz", t=times, xyz=traj)
        print("particle trajectories ->", workdir / "particles.npz")

    out = workdir / "frames"
    viz.export_frames(vel_p, out, n=args.n, times=times)
    print(f"{args.frames} .vti frames ->", out)
    if not args.no_render:
        import pyvista as pv

        render_dir = workdir / "render"
        for frame in range(args.frames):
            grid = pv.read(str(out / f"frame_{frame:04d}.vti"))
            tubes = pv.read(str(out / f"q_{frame:04d}.vtp"))
            viz.render_frame(grid, tubes, render_dir / f"{frame:04d}.png", frame=frame)
        viz.encode_video(render_dir, workdir / "tgv3d.mp4", fps=args.fps)
        print("video ->", workdir / "tgv3d.mp4")


if __name__ == "__main__":
    main()
