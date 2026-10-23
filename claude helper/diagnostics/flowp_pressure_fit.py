"""What pressure drop does the measured 4D-FlowP acceleration alone imply at the catheter planes?

Fits a fresh pressure network p(t, x) (the design of coa_pressure_fit.py) by least squares to the registered
acceleration only, min mean |a_measured + grad p|^2 over the shared voxel-frames (claude helper/diagnostics/
flowp_register.py -> prepared/<case>_flowp.npz). That is the 4D-FlowP pressure estimate's principle (grad p = -rho Du/Dt,
viscous terms left out) without any velocity network: the gradient part of the measured field, the no-flux wall
condition arising naturally. It then reports the section-averaged drop between the scan's catheter planes over the beat,
next to a centre-line path integral of the same field (flowp_checks.py), which also keeps the field's curl part.

    python "claude helper/diagnostics/flowp_pressure_fit.py"                         # P2_baseline, acc_frac
    python "claude helper/diagnostics/flowp_pressure_fit.py" --key acc --steps 4000
    JAX_PLATFORMS=cpu python "claude helper/diagnostics/flowp_pressure_fit.py" ...   # alongside a GPU run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from coa_pressure_fit import PressureNet  # noqa: E402  (the same pressure network)
from pinnflow.configs import get_config  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="P2_baseline")
    ap.add_argument("--key", default="acc_frac", help="acc (as registered) or acc_frac (half a frame later)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    cfg = get_config("coa_mri", "G", {"problem.case": args.case, "problem.accel_data": True, "problem.accel_key": args.key})
    prob = PROBLEMS["coa_mri"](cfg)
    k_mmHg = prob.mmHg_per_unit
    Z = np.asarray(prob.train_c)[np.asarray(prob.acc_idx)]          # (t, a, b, c) of the shared voxel-frames
    A = np.asarray(prob.train_acc)[np.asarray(prob.acc_idx)]        # measured acceleration, non-dimensional
    print(f"{prob.case} ({args.key}): {len(Z)} voxel-frames, measured |a| RMS {np.sqrt(np.mean(np.sum(A**2, 1))) / (0.01):.1f} m/s^2")

    pos = np.asarray(prob.lumen)
    lo, hi = pos.min(0), pos.max(0)
    net = PressureNet(w=2 * np.pi / prob.T, centre=jnp.asarray(0.5 * (lo + hi)), scale=float(0.5 * (hi - lo).max()))
    variables = net.init(jax.random.PRNGKey(2), jnp.asarray(Z[0]))
    consts, theta = variables["consts"], variables["params"]
    p_fn = lambda th, z: net.apply({"params": th, "consts": consts}, z)  # noqa: E731
    grad_x = jax.vmap(lambda th, z: jax.grad(p_fn, argnums=1)(th, z)[1:], in_axes=(None, 0))

    def loss(th, z, ab):
        return jnp.mean(jnp.sum((ab + grad_x(th, z)) ** 2, axis=1))

    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.02))
    st = opt.init(theta)

    @jax.jit
    def step(th, st, z, ab):
        val, g = jax.value_and_grad(loss)(th, z, ab)
        up, st = opt.update(g, st, th)
        return optax.apply_updates(th, up), st, val

    Zj, Aj = jnp.asarray(Z), jnp.asarray(A)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in range(args.steps):
        idx = rng.integers(0, len(Z), args.batch)
        theta, st, val = step(theta, st, Zj[idx], Aj[idx])
        if i % 1000 == 0 or i == args.steps - 1:
            print(f"  step {i:5d}  mean |a + grad p|^2 {float(val):.3e}  ({time.time() - t0:.0f} s)")
    sel = np.linspace(0, len(Z) - 1, min(65536, len(Z))).astype(int)
    G = np.concatenate([np.asarray(grad_x(theta, Zj[jnp.asarray(c)])) for c in np.array_split(sel, max(1, len(sel) // 8192))])
    print(f"fit: relative |a + grad p| / |a| = {np.linalg.norm(A[sel] + G) / np.linalg.norm(A[sel]):.3f} (noise alone would leave about 0.5)")

    times = np.linspace(0.0, prob.T, 64, endpoint=False)
    pf = jax.jit(jax.vmap(lambda z: p_fn(theta, z)))

    def plane_mean(plane, ti):
        z = np.concatenate([np.full((len(plane), 1), ti, np.float32), plane], 1)
        return float(np.asarray(pf(jnp.asarray(z))).mean())

    dp = np.array([plane_mean(prob.aao, ti) - plane_mean(prob.dao, ti) for ti in times]) * k_mmHg
    print(f"pressure fitted to the measured acceleration alone: peak drop {dp.max():.2f} mmHg, mean {dp.mean():.2f} "
          f"(catheter {prob.cath_peak:.2f} / {float(prob.cath_dp.mean()):.2f})")
    print("  t (ms)  drop")
    for i in range(0, 64, 4):
        print(f"  {times[i] * 10:6.0f}  {dp[i]:+6.2f}")


if __name__ == "__main__":
    main()
