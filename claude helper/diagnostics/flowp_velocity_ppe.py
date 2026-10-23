"""Pressure drop from velocity-derived acceleration on the scan's voxels: conventional velocity vs the registered 4D-FlowP
velocity (CPU). Answers whether 4D-FlowP's velocity carries more drop than the conventional scan's (10.8 against 7.9 mmHg
on their own grids) once both sit on the same voxels, and how much of it survives smoothing (a noise check).

a = dv/dt + (v . grad) v by central differences on the scan's lumen grid (periodic in time over the 17 frames), then the
least-squares pressure Poisson solve of flowp_pressure_capacity.py, read at the scan's catheter planes.

    python "claude helper/diagnostics/flowp_velocity_ppe.py" --case P2_baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsmr

ROOT = Path(__file__).resolve().parents[2]
RHO, MMHG = 1100.0, 133.322


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="P2_baseline")
    args = ap.parse_args()
    prep = np.load(ROOT / "data/stanford_coa/prepared" / f"{args.case}.npz")
    reg = np.load(ROOT / "data/stanford_coa/prepared" / f"{args.case}_flowp.npz")
    pos, vox = prep["pos_mm"].astype(np.float64), prep["voxel_mm"].astype(np.float64)
    T = prep["vel"].shape[0]
    dt = float(np.median(np.diff(prep["t_s"])))
    ijk = np.round(pos / vox).astype(int)
    shape = tuple(ijk.max(0) + 1)
    lumen = np.zeros(shape, bool)
    lumen[tuple(ijk.T)] = True
    inside = reg["inside"].astype(bool)
    sel = np.nonzero(inside)[0]
    grid = -np.ones(shape, np.int64)
    grid[tuple(ijk[sel].T)] = np.arange(len(sel))
    rows, cols, vals, pairs, r = [], [], [], [], 0
    for ax in range(3):
        e = np.zeros(3, int)
        e[ax] = 1
        q = ijk[sel] + e
        ok = np.all(q < np.array(shape), 1)
        nb = -np.ones(len(sel), np.int64)
        nb[ok] = grid[tuple(q[ok].T)]
        good = nb >= 0
        ia, ib = np.arange(len(sel))[good], nb[good]
        rr = np.arange(r, r + len(ia))
        h = vox[ax] * 1e-3
        rows += [rr, rr]
        cols += [ia, ib]
        vals += [np.full(len(ia), -1 / h), np.full(len(ia), 1 / h)]
        pairs.append((ax, ia, ib))
        r += len(ia)
    G = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, len(sel)))

    def cells(plane_mm):
        g = grid[tuple(np.clip(np.round(plane_mm / vox).astype(int), 0, np.array(shape) - 1).T)]
        return g[g >= 0]

    cu, cd = cells(prep["aao_mm"]), cells(prep["dao_mm"])

    def accel(vel, sigma):
        """vel (T, N, 3) on the lumen voxels -> material acceleration (T, N, 3) by central differences on the grid."""
        V = np.zeros((T,) + shape + (3,))
        V[:, ijk[:, 0], ijk[:, 1], ijk[:, 2], :] = vel
        if sigma > 0:  # Gaussian smoothing inside the lumen (normalised), per frame and component
            w = ndimage.gaussian_filter(lumen.astype(float), sigma)
            for t in range(T):
                for c in range(3):
                    V[t, ..., c] = np.where(lumen, ndimage.gaussian_filter(V[t, ..., c] * lumen, sigma) / np.maximum(w, 1e-6), 0)
        dvdt = (np.roll(V, -1, 0) - np.roll(V, 1, 0)) / (2 * dt)
        conv = np.zeros_like(V)
        for c in range(3):
            g = [np.gradient(V[..., c], vox[ax] * 1e-3, axis=ax + 1) for ax in range(3)]  # grid axes carry local a, b, c
            conv[..., c] = sum(V[..., ax] * g[ax] for ax in range(3))
        A = dvdt + conv
        return A[:, ijk[:, 0], ijk[:, 1], ijk[:, 2], :]

    def drop(acc):
        dp = []
        for t in range(T):
            a_t = acc[t, sel]
            rhs = np.concatenate([-RHO * 0.5 * (a_t[ia, ax] + a_t[ib, ax]) for ax, ia, ib in pairs])
            P = lsmr(G, rhs, atol=1e-8, btol=1e-8, maxiter=4000)[0]
            dp.append((P[cu].mean() - P[cd].mean()) / MMHG)
        return np.array(dp)

    print(f"{args.case}: {len(sel)} shared voxels, planes {len(cu)}/{len(cd)} cells; catheter {float(prep['cath_dp_peak_mmHg']):.2f} mmHg")
    for name, vel in (("conventional velocity", prep["vel"].astype(np.float64)), ("4D-FlowP velocity (registered)", reg["vel_flowp"].astype(np.float64))):
        for sigma in (0.0, 0.7, 1.0):
            dp = drop(accel(vel, sigma))
            print(f"  {name:32s} smoothing sigma {sigma:3.1f} voxel: peak {dp.max():6.2f}, mean {dp.mean():+5.2f} mmHg")
    dp = drop(reg["acc"].astype(np.float64))
    print(f"  {'measured acceleration (acc)':32s}                     : peak {dp.max():6.2f}, mean {dp.mean():+5.2f} mmHg")


if __name__ == "__main__":
    main()
