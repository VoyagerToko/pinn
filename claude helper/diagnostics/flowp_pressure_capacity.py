"""Why does the pressure network give less drop than a grid solve of the same measured acceleration? (CPU)

Separates the two candidates on the registered 4D-FlowP acceleration (prepared/<case>_flowp.npz):
    1. resampling: the least-squares pressure Poisson solve (as flowp_ppe_reproduce.py) on the conventional scan's own
       lumen voxels and the registered acceleration (compare with 7.72 mmHg on the native 4D-FlowP grid);
    2. capacity: pressure networks of increasing size fitted to the same field by least squares, min |a + grad p|^2
       (the coa_mri default is 4 x 128 tanh with Fourier sigma 1, 4, 12; 5.33 mmHg).
Every variant is read at the scan's catheter planes (prepared aao_mm / dao_mm), max over the frames.

    python "claude helper/diagnostics/flowp_pressure_capacity.py" --case P2_baseline
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pinnflow  # noqa: E402

pinnflow.configure_jax()
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

RHO, MMHG = 1100.0, 133.322


class PNet(nn.Module):
    w: float
    centre: jnp.ndarray
    scale: float
    sigmas: tuple
    nfeat: int = 32
    harmonics: int = 12
    width: int = 128
    depth: int = 4

    @nn.compact
    def __call__(self, z):
        k = jnp.arange(1, self.harmonics + 1)
        x = (z[1:] - self.centre) / self.scale
        B = self.variable("consts", "B", lambda: jnp.concatenate(
            [jax.random.normal(jax.random.PRNGKey(s), (3, self.nfeat)) * sg for s, sg in enumerate(self.sigmas)], 1)).value
        h = jnp.concatenate([jnp.cos(k * self.w * z[0]), jnp.sin(k * self.w * z[0]), x, jnp.cos(x @ B), jnp.sin(x @ B)])
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="P2_baseline")
    ap.add_argument("--key", default="acc_frac")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--grid-only", action="store_true", help="only the grid solve (part 1)")
    args = ap.parse_args()
    prep = np.load(ROOT / "data/stanford_coa/prepared" / f"{args.case}.npz")
    reg = np.load(ROOT / "data/stanford_coa/prepared" / f"{args.case}_flowp.npz")
    pos = prep["pos_mm"].astype(np.float64)  # (N, 3) local (a, b, c) mm
    vox = prep["voxel_mm"].astype(np.float64)
    acc = reg[args.key].astype(np.float64)  # (T, N, 3) m/s^2 local frame
    inside = reg["inside"].astype(bool)
    T, N = acc.shape[:2]
    t_nd = prep["t_s"] / 0.01
    period_nd = float(prep["period_s"]) / 0.01

    # ---- 1. grid solve on the scan's voxels (the inside ones)
    ijk = np.round(pos / vox).astype(int)
    sel = np.nonzero(inside)[0]
    grid = -np.ones(ijk.max(0) + 2, np.int64)
    grid[tuple(ijk[sel].T)] = np.arange(len(sel))
    rows, cols, vals, pairs, r = [], [], [], [], 0
    for ax in range(3):
        e = np.zeros(3, int)
        e[ax] = 1
        nb = grid[tuple((ijk[sel] + e).T)]
        ok = nb >= 0
        ia, ib = np.arange(len(sel))[ok], nb[ok]
        rr = np.arange(r, r + len(ia))
        h = vox[ax] * 1e-3
        rows += [rr, rr]
        cols += [ia, ib]
        vals += [np.full(len(ia), -1 / h), np.full(len(ia), 1 / h)]
        pairs.append((ax, ia, ib))
        r += len(ia)
    G = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, len(sel)))

    def cells(plane_mm):
        g = grid[tuple(np.clip(np.round(plane_mm / vox).astype(int), 0, np.array(grid.shape) - 1).T)]
        return g[g >= 0]

    cu, cd = cells(prep["aao_mm"]), cells(prep["dao_mm"])
    dp = []
    for t in range(T):
        a_t = acc[t, sel]
        rhs = np.concatenate([-RHO * 0.5 * (a_t[ia, ax] + a_t[ib, ax]) for ax, ia, ib in pairs])
        P = lsmr(G, rhs, atol=1e-8, btol=1e-8, maxiter=4000)[0]
        dp.append((P[cu].mean() - P[cd].mean()) / MMHG)
    dp = np.array(dp)
    print(f"1. grid solve on the scan's voxels ({len(sel)} inside, planes {len(cu)}/{len(cd)} cells): peak {dp.max():.2f}, mean {dp.mean():+.2f} mmHg")

    if args.grid_only:
        return
    # ---- 2. pressure networks of increasing capacity, least squares to the same field (non-dimensional units)
    fi, ni = np.nonzero(np.broadcast_to(inside[None], (T, N)))
    Z = np.concatenate([t_nd[fi, None], pos[ni] * 0.1], 1).astype(np.float32)  # (t, x) in 10 ms / 10 mm units
    A = (acc[fi, ni] * 0.01).astype(np.float32)  # a L / U^2
    lo, hi = pos.min(0) * 0.1, pos.max(0) * 0.1
    variants = [("default 4x128, sigma 1/4/12", dict(sigmas=(1.0, 4.0, 12.0), width=128, depth=4)),
                ("wider 4x256, sigma 1/4/12/24", dict(sigmas=(1.0, 4.0, 12.0, 24.0), width=256, depth=4)),
                ("6x256, sigma 1/4/12/24/40", dict(sigmas=(1.0, 4.0, 12.0, 24.0, 40.0), width=256, depth=6))]
    times = np.linspace(0.0, period_nd, 64, endpoint=False)
    for name, kw in variants:
        net = PNet(w=2 * np.pi / period_nd, centre=jnp.asarray(0.5 * (lo + hi)), scale=float(0.5 * (hi - lo).max()), **kw)
        var = net.init(jax.random.PRNGKey(2), jnp.asarray(Z[0]))
        consts, th = var["consts"], var["params"]
        pf = lambda p, z: net.apply({"params": p, "consts": consts}, z)  # noqa: E731
        gx = jax.vmap(lambda p, z: jax.grad(pf, argnums=1)(p, z)[1:], in_axes=(None, 0))
        opt = optax.adam(optax.cosine_decay_schedule(2e-3, args.steps, alpha=0.02))
        st = opt.init(th)

        @jax.jit
        def step(th, st, z, ab):
            val, g = jax.value_and_grad(lambda p: jnp.mean(jnp.sum((ab + gx(p, z)) ** 2, 1)))(th)
            up, st = opt.update(g, st, th)
            return optax.apply_updates(th, up), st, val

        Zj, Aj = jnp.asarray(Z), jnp.asarray(A)
        rng = np.random.default_rng(0)
        t0 = time.time()
        for i in range(args.steps):
            idx = rng.integers(0, len(Z), 8192)
            th, st, val = step(th, st, Zj[idx], Aj[idx])
        pj = jax.jit(jax.vmap(lambda z: pf(th, z)))

        def plane(pl, ti):
            z = np.concatenate([np.full((len(pl), 1), ti, np.float32), pl * 0.1], 1).astype(np.float32)
            return float(np.asarray(pj(jnp.asarray(z))).mean())

        d = np.array([plane(prep["aao_mm"], ti) - plane(prep["dao_mm"], ti) for ti in times]) * RHO / MMHG
        print(f"2. {name:30s}: loss {float(val):.3e}, peak {d.max():.2f}, mean {d.mean():+.2f} mmHg  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
