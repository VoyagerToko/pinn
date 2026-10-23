"""Reproduce the source publication's 4D-FlowP pressure drop on the released data, and read it at several slice positions.

The publication (PMC13252443) solves laplacian P = -rho div a over the segmented fluid mask with first-order finite
differences and LSMR, viscous and gravity terms neglected, no regularisation, and reports the maximum over the beat of
the AAo-DAo difference of cross-section means at the catheter slices (AAo proximal to the brachiocephalic trunk; DAo two
vertebral spaces above the diaphragm, both located by gel markers). Here, per frame on the native 4D-FlowP grid:

    min_P  sum over neighbour pairs (i, j) inside the mask of  ((P_j - P_i) / h + rho (a_i + a_j) / 2 . e)^2

solved with scipy's LSMR: the least-squares form of that Poisson problem, its natural (Neumann) wall condition included.
The drop is read at:
    - our catheter planes (the 2D-PC inlet/outlet cells of prepared/<case>.npz), mapped through the registration;
    - head-feet slices of the ascending limb just below the arch (the BCT origin) and of the descending limb at a range of
      heights below the coarctation (no gel markers are located here: a sweep shows how much the position matters).
The same solve with the acceleration derived from 4D-FlowP's own velocity, dv/dt + (v . grad) v, is the "conventional
4D Flow" counterpart (publication: 6.4 mmHg on P2 baseline at the higher flow; 4D-FlowP 15.9; catheter 13.2).

    python "claude helper/diagnostics/flowp_ppe_reproduce.py" --case P2_baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsmr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RHO = 1100.0
MMHG = 133.322
H = np.array([1.7, 1.8, 1.8]) * 1e-3  # m: grid spacing along (FH, RL, AP) = array axes 0, 1, 2
DT = 0.060


def gradient_system(mask):
    """Sparse first-order difference operator over neighbour pairs inside the mask: rows (pair, axis), cols voxels."""
    idx = -np.ones(mask.shape, np.int64)
    pts = np.argwhere(mask)
    idx[tuple(pts.T)] = np.arange(len(pts))
    rows, cols, vals, pairs = [], [], [], []
    r = 0
    for ax in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        both = mask[tuple(sl_a)] & mask[tuple(sl_b)]
        ia = idx[tuple(sl_a)][both]
        ib = idx[tuple(sl_b)][both]
        n = len(ia)
        rr = np.arange(r, r + n)
        rows += [rr, rr]
        cols += [ia, ib]
        vals += [np.full(n, -1.0 / H[ax]), np.full(n, 1.0 / H[ax])]
        pairs.append((ax, ia, ib))
        r += n
    G = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, len(pts)))
    return G, pairs, pts, idx


def solve_pressure(G, pairs, acc_vox):
    """acc_vox (3, n) m/s^2 at the mask voxels -> P (n,) Pa with grad P ~ -rho a (least squares)."""
    rhs = np.concatenate([-RHO * 0.5 * (acc_vox[ax, ia] + acc_vox[ax, ib]) for ax, ia, ib in pairs])
    return lsmr(G, rhs, atol=1e-8, btol=1e-8, maxiter=4000)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="P2_baseline")
    args = ap.parse_args()
    z = np.load(ROOT / "data/stanford_coa/Imaging" / args.case / "4DFlowP/data.npz", allow_pickle=False)
    mask = z["approximate_vessel_mask"].astype(bool)
    lab, n = ndimage.label(mask)
    mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)  # the connected vessel
    v = (z["velocity"] / 100.0).astype(np.float32)
    a = z["acceleration"].astype(np.float32)
    T = v.shape[-1]
    G, pairs, pts, idx = gradient_system(mask)
    print(f"{args.case}: {len(pts)} mask voxels, {G.shape[0]} neighbour pairs, {T} frames")

    # the velocity-derived acceleration ("conventional"), on the same grid
    dvdt = (np.roll(v, -1, -1) - np.roll(v, 1, -1)) / (2 * DT)
    conv = np.zeros_like(v)
    for i in range(3):
        g = [np.gradient(v[i], H[j], axis=j) for j in range(3)]
        conv[i] = sum(v[j] * g[j] for j in range(3))
    a_vel = dvdt + conv

    P_meas = np.stack([solve_pressure(G, pairs, a[:, pts[:, 0], pts[:, 1], pts[:, 2], t]) for t in range(T)])  # (T, n)
    P_vel = np.stack([solve_pressure(G, pairs, a_vel[:, pts[:, 0], pts[:, 1], pts[:, 2], t]) for t in range(T)])

    def section_drop(P, sec_up, sec_dn):
        iu = idx[tuple(sec_up.T)]
        idn = idx[tuple(sec_dn.T)]
        iu, idn = iu[iu >= 0], idn[idn >= 0]
        dp = (P[:, iu].mean(1) - P[:, idn].mean(1)) / MMHG
        return dp

    # ---- our catheter planes, mapped through the registration
    from pinnflow.coa_phantom import load_4dflow

    reg = np.load(ROOT / "data/stanford_coa/prepared" / f"{args.case}_flowp.npz")
    prep = np.load(ROOT / "data/stanford_coa/prepared" / f"{args.case}.npz")
    d = load_4dflow(args.case)
    origin = d["xyz_mm"][0, 0, 0]
    E = np.stack([d["row_dir"], d["col_dir"], d["normal"]])  # local (a, b, c) -> LPS rows
    Ainv = np.linalg.inv(reg["affine_lps"])

    def to_flowp(local_mm):
        lps = origin + local_mm @ E
        ijk = (np.concatenate([lps, np.ones((len(lps), 1))], 1) @ Ainv.T)[:, :3]
        return np.unique(np.round(ijk).astype(int), axis=0)

    up, dn = to_flowp(prep["aao_mm"]), to_flowp(prep["dao_mm"])
    for name, P in (("measured a (4D-FlowP)", P_meas), ("velocity-derived a", P_vel)):
        dp = section_drop(P, up, dn)
        print(f"our planes ({len(up)} / {len(dn)} cells), {name:22s}: peak {dp.max():+6.2f}, mean {dp.mean():+5.2f} mmHg | per frame "
              + " ".join(f"{x:+.1f}" for x in dp))

    # ---- head-feet slices: ascending limb below the arch, descending limb at a range of heights
    area = mask.sum((1, 2))
    k_arch = int(np.argmax(area))
    side = 1 if area[k_arch + 5:].sum() > area[:k_arch - 5].sum() else -1

    def limb_section(k, ascending):
        lab, n = ndimage.label(mask[k])
        best, best_q = None, None
        for L in range(1, n + 1):
            sec = lab == L
            if sec.sum() < 20:
                continue
            q = float(v[0, k][sec].mean(0)[2:7].mean())  # systolic mean FH velocity: ascending flows head-wards (+FH)
            if best is None or (q > best_q if ascending else q < best_q):
                best, best_q = sec, q
        if best is None:
            return None
        jj, kk = np.nonzero(best)
        return np.stack([np.full(len(jj), k), jj, kk], 1)

    up_depths = (10, 20, 30, 40, 60)
    dn_depths = (60, 80, 100, 120, 140, 160, 180, 200)
    for name, P in (("measured a (4D-FlowP)", P_meas), ("velocity-derived a", P_vel)):
        print(f"\npeak drop (mmHg), {name}: rows = AAo slice depth below the arch, columns = DAo slice depth below the arch")
        print("        " + " ".join(f"{dd:6d}" for dd in dn_depths))
        for ud in up_depths:
            su = limb_section(k_arch + side * int(round(ud / 1.7)), True)
            row = []
            for dd in dn_depths:
                sd = limb_section(k_arch + side * int(round(dd / 1.7)), False)
                row.append(section_drop(P, su, sd).max() if (su is not None and sd is not None) else np.nan)
            print(f"  {ud:4d}  " + " ".join(f"{x:6.1f}" for x in row))
    cath = float(prep["cath_dp_peak_mmHg"])
    print(f"\ncatheter peak {cath:.2f} mmHg (publication, this case: 4D Flow 6.4, 4D-FlowP 15.9, catheter 13.2 at the higher flow)")


if __name__ == "__main__":
    main()
