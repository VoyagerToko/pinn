"""Finite-difference reference for the lid-driven cavity (Benchmark B), independent of any PINN.

    python scripts/cavity_fd_reference.py --Re 1000 --N 512 --lid regularised
    python scripts/cavity_fd_reference.py --Re 1000 --N 512 --lid unit

Why: the PINN benchmark uses the regularised lid u = 1 - cosh(r(x - 1/2))/cosh(r/2), r = 50, while
Ghia et al. (1982) and the JAX-PI fields use a unit lid. This solver gives (a) a check of the Ghia
comparison with a unit lid and (b) a ground truth for the problem the PINNs actually solve, so the
part of the Ghia error that comes from the lid regularisation can be separated from PINN error.

Method: stream function - vorticity form on a uniform (N+1)^2 grid,
    lap psi = -omega,   omega_t + u omega_x + v omega_y = lap omega / Re,   u = psi_y, v = -psi_x,
second-order central differences, Thom's formula for the wall vorticity, the Poisson equation
solved exactly for the 5-point Laplacian with a type-I sine transform, explicit time stepping to
steady state (pseudo-time). Output: data/cavity_fd/cavity_fd_Re<Re>_<lid>_N<N>.npz with x, y, psi,
omega, u, v (arrays indexed [i, j] = (x_i, y_j), the convention of pinnflow.data.load_jaxpi_cavity).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

ROOT = Path(__file__).resolve().parents[1]


def lid_profile(x, kind: str, r: float = 50.0):
    if kind == "unit":
        u = np.ones_like(x)
        u[0] = u[-1] = 0.0  # corner nodes belong to the side walls
        return u
    return 1.0 - np.cosh(r * (x - 0.5)) / np.cosh(0.5 * r)


def solve(Re: float, N: int, lid: str, tol: float = 1e-7, max_time: float = 400.0, cfl: float = 0.4, log_every: int = 2000):
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    ulid = lid_profile(x, lid)
    # exact inverse of the 5-point Laplacian with homogeneous Dirichlet data (DST-I diagonalises it)
    k = np.arange(1, N)
    lam = (2.0 * np.cos(np.pi * k / N) - 2.0) / h**2
    lam2 = lam[:, None] + lam[None, :]

    def poisson(rhs):  # solve lap psi = rhs on the interior, psi = 0 on the boundary
        return idstn(dstn(rhs, type=1, workers=-1) / lam2, type=1, workers=-1)

    psi = np.zeros((N + 1, N + 1))
    om = np.zeros((N + 1, N + 1))
    dt = min(cfl * h, 0.2 * Re * h * h)  # convective (|u| <= 1) and viscous limits
    t, it, t0 = 0.0, 0, time.perf_counter()

    def wall_vorticity(psi, om):
        om[:, -1] = -2.0 * (psi[:, -2] + h * ulid) / h**2  # lid y = 1 (psi_y = u_lid)
        om[:, 0] = -2.0 * psi[:, 1] / h**2  # bottom
        om[0, :] = -2.0 * psi[1, :] / h**2  # left
        om[-1, :] = -2.0 * psi[-2, :] / h**2  # right
        return om

    def rhs(psi, om):
        u = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        v = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        om_x = (om[2:, 1:-1] - om[:-2, 1:-1]) / (2 * h)
        om_y = (om[1:-1, 2:] - om[1:-1, :-2]) / (2 * h)
        lap = (om[2:, 1:-1] + om[:-2, 1:-1] + om[1:-1, 2:] + om[1:-1, :-2] - 4 * om[1:-1, 1:-1]) / h**2
        return -(u * om_x + v * om_y) + lap / Re

    while t < max_time:
        # Heun (RK2) step for the interior vorticity; psi and wall vorticity follow each stage
        om = wall_vorticity(psi, om)
        k1 = rhs(psi, om)
        om1 = om.copy()
        om1[1:-1, 1:-1] += dt * k1
        psi1 = np.zeros_like(psi)
        psi1[1:-1, 1:-1] = poisson(-om1[1:-1, 1:-1])
        om1 = wall_vorticity(psi1, om1)
        k2 = rhs(psi1, om1)
        om_new = om.copy()
        om_new[1:-1, 1:-1] += 0.5 * dt * (k1 + k2)
        psi[1:-1, 1:-1] = poisson(-om_new[1:-1, 1:-1])
        change = np.abs(om_new[1:-1, 1:-1] - om[1:-1, 1:-1]).max() / dt
        om = om_new
        t += dt
        it += 1
        if it % log_every == 0:
            print(f"[fd] Re={Re:g} N={N} lid={lid} t={t:7.2f} it={it} max|d omega/dt|={change:.3e} psi_min={psi.min():.6f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        if change < tol:
            break
    om = wall_vorticity(psi, om)
    # velocities: central differences inside, boundary data on the walls
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * h)
    v[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2 * h)
    u[:, -1] = ulid
    u[:, 0] = 0.0
    u[0, :] = u[-1, :] = 0.0
    u[0, -1] = u[-1, -1] = 0.0
    v[:, 0] = v[:, -1] = 0.0
    return {"x": x, "y": x.copy(), "psi": psi, "omega": om, "u": u, "v": v, "t_final": t, "iterations": it, "residual": change}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Re", type=float, default=1000.0)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--lid", choices=["regularised", "unit"], default="regularised")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max-time", type=float, default=400.0)
    a = ap.parse_args()
    sol = solve(a.Re, a.N, a.lid, tol=a.tol, max_time=a.max_time)
    out = ROOT / "data" / "cavity_fd" / f"cavity_fd_Re{int(a.Re)}_{a.lid}_N{a.N}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v, dtype=np.float64) for k, v in sol.items()}, Re=a.Re, N=a.N)
    i_min = np.unravel_index(np.argmin(sol["psi"]), sol["psi"].shape)
    print(f"[fd] done: t={sol['t_final']:.1f}, {sol['iterations']} steps, residual {sol['residual']:.2e}; "
          f"primary vortex psi_min={sol['psi'].min():.6f} at ({sol['x'][i_min[0]]:.4f}, {sol['y'][i_min[1]]:.4f}) -> {out}")
    for k, v in ghia_errors(sol, int(a.Re)).items():
        print(f"[fd] {k} = {v:.5f}")


def load(Re: int, lid: str, N: int):
    d = np.load(ROOT / "data" / "cavity_fd" / f"cavity_fd_Re{int(Re)}_{lid}_N{N}.npz")
    return {k: d[k] for k in d.files}


def centerlines(sol, y_query, x_query):
    """u(0.5, y) and v(x, 0.5) by cubic interpolation of the FD field."""
    from scipy.interpolate import RegularGridInterpolator

    fu = RegularGridInterpolator((sol["x"], sol["y"]), sol["u"], method="cubic")
    fv = RegularGridInterpolator((sol["x"], sol["y"]), sol["v"], method="cubic")
    return fu(np.stack([np.full_like(y_query, 0.5), y_query], -1)), fv(np.stack([x_query, np.full_like(x_query, 0.5)], -1))


def ghia_errors(sol, Re: int):
    """Same metric as CavityPINN.evaluate: relative L2 over the Ghia points without the wall/lid points."""
    import sys

    sys.path.insert(0, str(ROOT))
    from pinnflow.benchmarks import ghia_tables

    tab = ghia_tables(Re)
    yq, xq = tab["y"][1:-1], tab["x"][1:-1]
    u_c, v_c = centerlines(sol, yq, xq)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return {"ghia_u_rel_err": rel(u_c, tab["u"][1:-1]), "ghia_v_rel_err": rel(v_c, tab["v"][1:-1])}


if __name__ == "__main__":
    main()
