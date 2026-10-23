"""Is the pressure drop in a trained coarctation network's velocity field? Fit a pressure to it and see.

The joint network leaves the flow's systolic acceleration in its momentum residual instead of its pressure
(claude helper/diagnostics/coa_momentum_path.py). This script freezes the trained velocity, computes the laminar momentum terms

    F = u_t + (u . grad) u - nu lap u

at points spread over the lumen and the beat, and fits a fresh, separate pressure network p(t, x) by least squares,
min mean |F + grad p|^2. That minimum is the pressure the velocity field implies: the gradient part of F, with the
no-flux wall condition arising naturally (the same projection the pressure Poisson baseline makes on the scan).

    a fitted drop near the catheter's (or the path integral's ~6 mmHg)  -> the information is in the velocity; joint
                                                                         training fails to put it into the pressure
    a fitted drop near the joint network's 0.1-0.3 mmHg                 -> F has no such gradient part: the imbalance
                                                                         is not expressible as a pressure at all

It also prints the mean momentum residual of both pressures against the same F: the gap is the loss joint training
left unclaimed.

    python "claude helper/diagnostics/coa_pressure_fit.py" coa_P2_baseline
    python "claude helper/diagnostics/coa_pressure_fit.py" coa_P2_baseline --points 262144 --steps 8000
    JAX_PLATFORMS=cpu python "claude helper/diagnostics/coa_pressure_fit.py" ...   # alongside a GPU run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pinnflow  # noqa: E402

pinnflow.configure_jax()

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_collections  # noqa: E402
import optax  # noqa: E402

from pinnflow import physics  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402


class PressureNet(nn.Module):
    """p(t, x): exact beat periodicity through time harmonics, multi-scale Fourier features in space."""

    w: float              # 2 pi / period
    centre: jnp.ndarray   # (3,)
    scale: float          # one length for all three axes (isotropic features)
    harmonics: int = 12
    width: int = 128
    depth: int = 4

    @nn.compact
    def __call__(self, z):
        k = jnp.arange(1, self.harmonics + 1)
        t = z[0]
        x = (z[1:] - self.centre) / self.scale
        B = self.variable("consts", "B", lambda: jnp.concatenate([
            jax.random.normal(jax.random.PRNGKey(s), (3, 32)) * sigma for s, sigma in enumerate((1.0, 4.0, 12.0))], 1)).value
        h = jnp.concatenate([jnp.cos(k * self.w * t), jnp.sin(k * self.w * t), x, jnp.cos(x @ B), jnp.sin(x @ B)])
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name under runs/ (its config.json selects the case)")
    ap.add_argument("--ckpt", default="latest.msgpack")
    ap.add_argument("--points", type=int, default=262144, help="fixed points over the lumen and the beat")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    cfg = ml_collections.ConfigDict(json.loads((run / "config.json").read_text()))
    prob = PROBLEMS["coa_mri"](cfg)
    params, _ = load_params(run / args.ckpt, prob.init_params(jax.random.PRNGKey(0)))
    f = prob.net(params)
    k_mmHg = prob.mmHg_per_unit

    # ---- F = u_t + conv - visc of the frozen velocity, and the joint network's own grad p, at fixed points
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
        return u_t + conv - lap / prob.Re, jnp.stack(gp)

    T = jax.jit(jax.vmap(terms))
    t0 = time.time()
    Z = np.asarray(prob.uniform_collocation(jax.random.PRNGKey(1), args.points))
    Fs, Gs = [], []
    for s in range(0, len(Z), 8192):
        a, b = T(jnp.asarray(Z[s : s + 8192]))
        Fs.append(np.asarray(a))
        Gs.append(np.asarray(b))
    F, G_joint = np.concatenate(Fs), np.concatenate(Gs)
    print(f"{prob.case}: momentum terms at {len(Z)} points in {time.time() - t0:.0f} s")
    flow = np.linalg.norm(np.asarray(prob.train_uvw).reshape(len(prob.d["t_s"]), -1, 3), axis=2).mean(1)
    print(f"  mean |F| {np.linalg.norm(F, axis=1).mean():.4f}; joint network's mean |F + grad p| "
          f"{np.linalg.norm(F + G_joint, axis=1).mean():.4f}, mean square {np.mean(np.sum((F + G_joint) ** 2, 1)):.2e}")

    # ---- fit p alone: min mean |F + grad p|^2
    pos = np.asarray(prob.lumen)
    lo, hi = pos.min(0), pos.max(0)
    net = PressureNet(w=2 * np.pi / prob.T, centre=jnp.asarray(0.5 * (lo + hi)), scale=float(0.5 * (hi - lo).max()))
    variables = net.init(jax.random.PRNGKey(2), jnp.asarray(Z[0]))
    consts, theta = variables["consts"], variables["params"]
    p_fn = lambda th, z: net.apply({"params": th, "consts": consts}, z)  # noqa: E731
    grad_x = jax.vmap(lambda th, z: jax.grad(p_fn, argnums=1)(th, z)[1:], in_axes=(None, 0))

    def loss(th, z, Fb):
        return jnp.mean(jnp.sum((Fb + grad_x(th, z)) ** 2, axis=1))

    sched = optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.02)
    opt = optax.adam(sched)
    st = opt.init(theta)

    @jax.jit
    def step(th, st, z, Fb):
        val, g = jax.value_and_grad(loss)(th, z, Fb)
        up, st = opt.update(g, st, th)
        return optax.apply_updates(th, up), st, val

    Zj, Fj = jnp.asarray(Z), jnp.asarray(F)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in range(args.steps):
        idx = rng.integers(0, len(Z), args.batch)
        theta, st, val = step(theta, st, Zj[idx], Fj[idx])
        if i % 1000 == 0 or i == args.steps - 1:
            print(f"  step {i:5d}  mean |F + grad p|^2 {float(val):.3e}  ({time.time() - t0:.0f} s)")
    full = float(np.mean([loss(theta, Zj[s : s + 16384], Fj[s : s + 16384]) for s in range(0, len(Z), 16384)]))
    print(f"fitted pressure: mean square residual {full:.2e} (joint network {np.mean(np.sum((F + G_joint) ** 2, 1)):.2e})")

    # ---- the drop between the catheter planes over the beat, fitted vs joint
    times = np.linspace(0.0, prob.T, 64, endpoint=False)
    pf = jax.jit(jax.vmap(lambda z: p_fn(theta, z)))

    def plane_mean(plane, ti):
        z = np.concatenate([np.full((len(plane), 1), ti, np.float32), plane], 1)
        return float(np.asarray(pf(jnp.asarray(z))).mean())

    dp_fit = np.array([plane_mean(prob.aao, ti) - plane_mean(prob.dao, ti) for ti in times]) * k_mmHg
    dp_joint = prob.pressure_drop(params, times)
    print(f"peak drop AAo -> DAo: fitted {dp_fit.max():.2f} mmHg, joint network {dp_joint.max():.2f}, catheter {prob.cath_peak:.2f}")
    print(f"mean drop over the beat: fitted {dp_fit.mean():.2f}, joint {dp_joint.mean():.2f}, catheter {float(prob.cath_dp.mean()):.2f}")
    print("  t (ms)  fitted  joint")
    for i in range(0, 64, 4):
        print(f"  {times[i] * 10:6.0f}  {dp_fit[i]:+6.2f}  {dp_joint[i]:+6.2f}")
    out = run / "pressure_fit.json"
    out.write_text(json.dumps({"t_ms": (times * 10).tolist(), "dp_fit_mmHg": dp_fit.tolist(), "dp_joint_mmHg": dp_joint.tolist(),
                               "residual_ms_fit": full, "residual_ms_joint": float(np.mean(np.sum((F + G_joint) ** 2, 1))),
                               "flow_per_frame": flow.tolist()}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
