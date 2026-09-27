"""Figures for paper/ from the trained checkpoints (CPU is enough).

    JAX_PLATFORMS=cpu python scripts/paper_figures.py cavity        # centreline profiles + divergence maps
    JAX_PLATFORMS=cpu python scripts/paper_figures.py ablation      # bar chart of the ablation rows
    JAX_PLATFORMS=cpu python scripts/paper_figures.py all

Outputs go to paper/figures/*.pdf. Every curve is computed from a checkpoint in runs/ or from the
finite-difference / Ghia references; nothing is copied from other publications.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow.benchmarks import ghia_tables  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.utils import chunked_vmap, load_params  # noqa: E402

FIG = ROOT / "paper" / "figures"
RUNS = ROOT / "runs"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150})


def cavity_model(run: str, ckpt: str, Re: int = 1000):
    cfg = ml_collections.ConfigDict(json.loads((RUNS / run / "config.json").read_text()))
    pb = PROBLEMS["cavity"](cfg)
    pb.set_Re(Re)
    params, _ = load_params(RUNS / run / ckpt, template=pb.init_params(jax.random.PRNGKey(0)))
    return pb, params


def fig_cavity():
    import cavity_fd_reference as fd

    ref = fd.load(1000, "regularised", 512)
    tab = ghia_tables(1000)
    s = np.linspace(0.0, 1.0, 201)
    uf, vf = fd.centerlines(ref, s, s)
    runs = [
        ("cavity_C_r2", "latest.msgpack", "row C (soft BC)", "C0", "-"),
        ("cavity_D_softlid", "Re1000.msgpack", "row D, soft lid (Adam)", "C2", "--"),
        ("cavity_F_r2", "latest.msgpack", "row F (hard lid)", "C3", "-."),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].plot(uf, s, "k-", lw=2.2, alpha=0.35, label="FD $512^2$, same lid")
    ax[1].plot(s, vf, "k-", lw=2.2, alpha=0.35, label="FD $512^2$, same lid")
    for run, ck, label, c, ls in runs:
        pb, params = cavity_model(run, ck)
        vel = pb.velocity_fn(params)
        u = np.asarray(chunked_vmap(vel, jnp.stack([jnp.full_like(jnp.asarray(s), 0.5), jnp.asarray(s)], -1)))[:, 0]
        v = np.asarray(chunked_vmap(vel, jnp.stack([jnp.asarray(s), jnp.full_like(jnp.asarray(s), 0.5)], -1)))[:, 1]
        ax[0].plot(u, s, color=c, ls=ls, lw=1.0, label=label)
        ax[1].plot(s, v, color=c, ls=ls, lw=1.0, label=label)
    ax[0].plot(tab["u"], tab["y"], "ko", ms=3, mfc="none", label="Ghia et al.")
    ax[1].plot(tab["x"], tab["v"], "ko", ms=3, mfc="none", label="Ghia et al.")
    ax[0].set_xlabel("$u(0.5,y)$"), ax[0].set_ylabel("$y$"), ax[1].set_xlabel("$x$"), ax[1].set_ylabel("$v(x,0.5)$")
    ax[0].legend(loc="lower right"), ax[0].grid(alpha=0.3), ax[1].grid(alpha=0.3)
    ax[0].set_title("vertical centreline, Re = 1000"), ax[1].set_title("horizontal centreline, Re = 1000")
    fig.tight_layout()
    fig.savefig(FIG / "cavity_profiles.pdf")
    plt.close(fig)

    # divergence maps: hard lid (corner floor) vs soft BC
    n = 201
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], -1))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for a, (run, ck, title) in zip(ax, [("cavity_F_r2", "latest.msgpack", "row F, hard lid"), ("cavity_C_r2", "latest.msgpack", "row C, soft boundary data")]):
        pb, params = cavity_model(run, ck)
        _, div = chunked_vmap(pb.residual_fn(params, 1000.0), pts, chunk=16384)
        im = a.pcolormesh(X, Y, np.log10(np.abs(np.asarray(div)).reshape(n, n) + 1e-12), cmap="magma", vmin=-6, vmax=1.5, shading="auto", rasterized=True)
        a.set_aspect("equal"), a.set_title(f"{title}: $\\log_{{10}}|\\nabla\\cdot u|$"), a.set_xlabel("$x$"), a.set_ylabel("$y$")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(FIG / "cavity_divergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("cavity figures ->", FIG)


def fig_tgv2d():
    """Benchmark A at t = T: PINN vorticity and the pointwise velocity error (log scale)."""
    from pinnflow.physics import velocity_gradient, vorticity_from_grad

    run = "tgv2d_G"
    cfg = ml_collections.ConfigDict(json.loads((RUNS / run / "config.json").read_text()))
    pb = PROBLEMS["tgv2d"](cfg)
    params, _ = load_params(RUNS / run / "latest.msgpack", template=pb.init_params(jax.random.PRNGKey(0)))
    vel = pb.velocity_fn(params)
    g = velocity_gradient(vel, dim=2, unsteady=True)
    n, t = 128, float(pb.bench.T)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    pts = jnp.stack([jnp.full(X.size, t), jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], -1)

    def one(z):
        _, gu = g(z)
        return vel(z), vorticity_from_grad(gu)

    uvp, w = chunked_vmap(one, pts, chunk=4096)
    ue, ve, _ = (np.asarray(a) for a in pb.bench.exact(t, X.ravel(), Y.ravel()))
    err = np.hypot(np.asarray(uvp[:, 0]) - ue, np.asarray(uvp[:, 1]) - ve).reshape(n, n)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    im0 = ax[0].pcolormesh(X, Y, np.asarray(w).reshape(n, n), cmap="RdBu_r", shading="auto", rasterized=True)
    ax[0].set_title(f"PINN vorticity, t = {t:g}")
    fig.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].pcolormesh(X, Y, np.log10(err + 1e-12), cmap="magma", shading="auto", rasterized=True)
    ax[1].set_title("$\\log_{10}|u_{PINN}-u_{exact}|$")
    fig.colorbar(im1, ax=ax[1], shrink=0.85)
    for a in ax:
        a.set_aspect("equal"), a.set_xlabel("$x$"), a.set_ylabel("$y$")
    fig.tight_layout()
    fig.savefig(FIG / "tgv2d.pdf")
    plt.close(fig)
    print("tgv2d figure ->", FIG)


def fig_tgv3d():
    """Benchmark D: kinetic energy, both dissipation estimates against the digitised reference, spectrum."""
    run = RUNS / "tgv3d_H"
    h = np.loadtxt(run / "energy_history.csv", delimiter=",", skiprows=1)
    ref = np.loadtxt(ROOT / "data" / "tgv3d_re1600" / "debonis2013_fig4a_ref_dissipation.csv", delimiter=",", skiprows=1)
    spec = [run / "spectrum_t9.0.csv"] if (run / "spectrum_t9.0.csv").exists() else sorted(run.glob("spectrum_t*.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax[0].plot(h[:, 0], h[:, 1], "C0")
    ax[0].set_xlabel("$t$"), ax[0].set_ylabel("$E_k$"), ax[0].set_title("kinetic energy")
    ax[1].plot(ref[:, 0], ref[:, 1], "k-", lw=1.2, label="reference (digitised)")
    ax[1].plot(h[:, 0], h[:, 3], "C0-", lw=1.0, label="PINN $-dE_k/dt$")
    ax[1].plot(h[:, 0], h[:, 4], "C1--", lw=1.0, label="PINN $2\\nu\\zeta$")
    ax[1].set_xlabel("$t$"), ax[1].set_ylabel("$\\varepsilon$"), ax[1].set_title("dissipation rate"), ax[1].legend(fontsize=6)
    if spec:
        s = np.loadtxt(spec[0], delimiter=",", skiprows=1)
        k, E = s[1:, 0], s[1:, 1]
        ax[2].loglog(k, E, "C0", label="PINN, " + spec[0].stem.replace("spectrum_t", "t = "))
        kk = k[(k >= 4) & (k <= 16)]
        ax[2].loglog(kk, E[k == 4][0] * (kk / 4) ** (-5 / 3), "k:", label="$k^{-5/3}$")
        ax[2].set_xlabel("$k$"), ax[2].set_ylabel("$E(k)$"), ax[2].set_title("energy spectrum"), ax[2].legend(fontsize=6)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "tgv3d_energy.pdf")
    plt.close(fig)
    print("tgv3d figure ->", FIG)


def _eval(run: str):
    p = RUNS / run / "eval.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_ablation():
    rows = "ABCDEFG"
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    # Benchmark A: velocity relative L2 at the end of the reduced budget
    vals = []
    for r in rows:
        e = _eval(f"tgv2d_abl_{r}")
        vals.append(e["rel_l2_vel"] if e else np.nan)
    ax[0].bar(list(rows), vals, color="C0")
    ax[0].set_yscale("log"), ax[0].set_title("A: Taylor-Green 2D, velocity rel. $L^2$"), ax[0].set_xlabel("row")
    # Benchmark B: velocity error against the FD field at Re = 1000 (row E = row D for a steady problem)
    vals = []
    for r in rows:
        e = _eval(f"cavity_abl_{'D' if r == 'E' else r}")
        vals.append(e.get("final_Re1000/rel_l2_vel_vs_fd", np.nan) if e else np.nan)
    ax[1].bar(list(rows), vals, color="C1")
    ax[1].set_yscale("log"), ax[1].set_title("B: cavity Re = 1000, velocity rel. $L^2$ vs FD"), ax[1].set_xlabel("row")
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "ablation.pdf")
    plt.close(fig)
    print("ablation figure ->", FIG)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("tgv2d", "all"):
        fig_tgv2d()
    if what in ("tgv3d", "all"):
        fig_tgv3d()
    if what in ("cavity", "all"):
        fig_cavity()
    if what in ("ablation", "all"):
        fig_ablation()
