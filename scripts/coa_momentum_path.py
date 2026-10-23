"""Momentum balance of a trained coarctation network along a lumen path from the AAo plane to the DAo plane.

For a network that satisfies the momentum equation, the pressure difference between the two planes equals
minus the path integral of (u_t + u . grad u - nu lap u). This script prints each term's path integral per scan
frame in mmHg, so an imbalance (the residual carrying the flow's inertia instead of the pressure) is visible
directly. It also compares the fastest scan voxels at peak flow with the network.

    python scripts/coa_momentum_path.py coa_P2_baseline                  # runs/<name>/latest.msgpack
    python scripts/coa_momentum_path.py coa_P2_baseline_eddy --ckpt ckpt_30000.msgpack
    JAX_PLATFORMS=cpu python scripts/coa_momentum_path.py ...            # alongside a GPU run

The path is the shortest route through the lumen voxels (26-neighbour graph), weighted to stay away from the wall.
Only the laminar terms are integrated (an eddy run's nu_t is printed by evaluate; its stress term is not included).
"""
from __future__ import annotations

import argparse
import heapq
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

from pinnflow import physics  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.problems.coa_inverse import T_REF  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402


def lumen_path(prob):
    """Voxel centres from the AAo plane centre to the DAo plane centre, avoiding the wall."""
    pos = np.asarray(prob.lumen)
    vox = prob.voxel
    ijk = np.round((pos - pos.min(0)) / vox).astype(int)
    index = {tuple(v): n for n, v in enumerate(ijk)}
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
    missing = np.array([sum((tuple(v + o) not in index) for o in offs) for v in ijk])
    src = int(np.argmin(np.linalg.norm(pos - prob.aao.mean(0), axis=1)))
    dst = int(np.argmin(np.linalg.norm(pos - prob.dao.mean(0), axis=1)))
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        dd, n = heapq.heappop(pq)
        if n == dst:
            break
        if dd > dist.get(n, np.inf):
            continue
        for o in offs:
            m = index.get(tuple(ijk[n] + o))
            if m is None:
                continue
            w = np.linalg.norm(o * vox) * (1.0 + 2.0 * missing[m])
            if dd + w < dist.get(m, np.inf):
                dist[m], prev[m] = dd + w, n
                heapq.heappush(pq, (dd + w, m))
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return pos[path[::-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name under runs/ (its config.json selects the case)")
    ap.add_argument("--ckpt", default="latest.msgpack", help="checkpoint file in runs/<run>/")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    cfg = ml_collections.ConfigDict(json.loads((run / "config.json").read_text()))
    prob = PROBLEMS["coa_mri"](cfg)
    params, _ = load_params(run / args.ckpt, prob.init_params(jax.random.PRNGKey(0)))
    f = prob.net(params)
    k = prob.mmHg_per_unit

    path = lumen_path(prob)
    seg = np.diff(path, axis=0)
    mid = 0.5 * (path[1:] + path[:-1])
    print(f"{prob.case}: path of {len(path)} voxels, {np.linalg.norm(seg, axis=1).sum() / (1e-3 / 0.01):.0f} mm, AAo -> DAo")

    def terms(z):
        out = f(z)
        u = out[:3]
        u_t = physics.directional(f, z, 0)[:3]
        conv, lap, gp = jnp.zeros(3), jnp.zeros(3), []
        for j, s in enumerate((1, 2, 3)):
            f_s, f_ss = physics.directional(f, z, s, second=True)
            conv = conv + u[j] * f_s[:3]
            lap = lap + f_ss[:3]
            gp.append(f_s[3])
        return u_t, conv, lap / prob.Re, jnp.stack(gp)

    T = jax.jit(jax.vmap(terms))
    times = np.asarray(prob.d["t_s"]) / T_REF
    print("path integrals in mmHg; balance requires  int grad p = -(int u_t + int conv - int visc)")
    print("     t   int u_t   int conv   int visc   int grad p   required   p(DAo)-p(AAo) eval")
    dp_eval = prob.pressure_drop(params, times)
    for ti, de in zip(times, dp_eval):
        z = jnp.asarray(np.concatenate([np.full((len(mid), 1), ti, np.float32), mid], 1))
        ut, cv, vs, gp = (float((np.asarray(a) * seg).sum()) * k for a in T(z))
        print(f"{ti:6.2f}  {ut:+8.2f}  {cv:+9.2f}  {vs:+9.2f}  {gp:+11.2f}  {-(ut + cv - vs):+9.2f}   {-de:+8.2f}")

    vel = np.asarray(prob.d["vel"])
    sp = np.linalg.norm(vel, axis=2)
    F = int(np.argmax(np.percentile(sp, 99.5, axis=1)))
    pos = np.asarray(prob.lumen)
    un = np.linalg.norm(prob.predict(params, np.concatenate([np.full((len(pos), 1), times[F], np.float32), pos], 1))[:, :3], axis=1)
    order = np.argsort(sp[F])[::-1]
    print(f"jet at the peak frame ({F}, t = {times[F]:.2f}): mean speed of the fastest scan voxels, scan vs network (m/s)")
    for n in (20, 100, 500, 2000):
        print(f"  fastest {n:5d}: {sp[F][order[:n]].mean():.2f} vs {un[order[:n]].mean():.2f}")


if __name__ == "__main__":
    main()
