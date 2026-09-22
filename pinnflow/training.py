"""STEP 6 - training procedure.

6.1  Stage 1 Adam (warm-up + exponential decay) and Stage 2 L-BFGS (strong-Wolfe zoom line search)
6.3  Reynolds-number curriculum (warm start from the previous Re)
6.4  time marching (sequence-to-sequence windows)
plus checkpointing and CSV/stdout logging. Weights & Biases is optional.

The :class:`Trainer` is problem-agnostic. A *problem* object (see :mod:`pinnflow.problems`) provides

    problem.arch                      flax module
    problem.init_params(key)          -> params
    problem.sample_batch(key)         -> batch (pytree of arrays)
    problem.losses(params, batch)     -> dict[str, scalar]      (unweighted terms)
    problem.init_weights              dict[str, float]
    problem.evaluate(params)          -> dict[str, float]       (validation metrics, STEP 7)
    problem.ntk_diags(params, batch)  -> dict[str, (n,)]        (optional, for scheme="ntk")
    problem.causal_min_weight(params, batch) -> scalar          (optional, for causal annealing)
    problem.update_collocation(key, params)                     (optional hook, e.g. RAD)
    problem.set_causal_eps(eps)                                 (optional)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state

from . import losses as L
from .utils import CSVLogger, count_params, load_params, save_params


# --------------------------------------------------------------------------------------
# 6.1 optimiser
# --------------------------------------------------------------------------------------
def make_lr_schedule(learning_rate: float = 1e-3, warmup_steps: int = 5000, decay_rate: float = 0.9, decay_steps: int = 2000, staircase: bool = False):
    """Linear warm-up 0 -> lr over ``warmup_steps`` then exponential decay x``decay_rate`` every ``decay_steps``."""
    if warmup_steps > 0:
        return optax.warmup_exponential_decay_schedule(
            init_value=0.0, peak_value=learning_rate, warmup_steps=warmup_steps, transition_steps=decay_steps, decay_rate=decay_rate, staircase=staircase
        )
    return optax.exponential_decay(init_value=learning_rate, transition_steps=decay_steps, decay_rate=decay_rate, staircase=staircase)


def make_adam(optim_cfg) -> optax.GradientTransformation:
    lr = make_lr_schedule(
        optim_cfg.get("learning_rate", 1e-3),
        optim_cfg.get("warmup_steps", 5000),
        optim_cfg.get("decay_rate", 0.9),
        optim_cfg.get("decay_steps", 2000),
        optim_cfg.get("staircase", False),
    )
    tx = optax.adam(lr, b1=optim_cfg.get("beta1", 0.9), b2=optim_cfg.get("beta2", 0.999), eps=optim_cfg.get("eps", 1e-8))
    clip = optim_cfg.get("clip_grad_norm", None)
    if clip:
        tx = optax.chain(optax.clip_by_global_norm(clip), tx)
    k = optim_cfg.get("grad_accum_steps", 0)
    if k and k > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=k)
    return tx


class TrainState(train_state.TrainState):
    weights: Dict[str, Any] = struct.field(pytree_node=True)
    momentum: float = struct.field(pytree_node=False, default=0.9)

    def apply_weights(self, new_weights):
        return self.replace(weights=L.ema_update(self.weights, new_weights, self.momentum))


# --------------------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, problem, config, workdir: os.PathLike, key: jax.Array, use_wandb: bool = False):
        self.problem = problem
        self.config = config
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.key, k_init = jax.random.split(self.key)
        params = problem.init_params(k_init)
        weights = {k: jnp.asarray(float(v)) for k, v in problem.init_weights.items()}
        self.state = TrainState.create(
            apply_fn=problem.arch.apply, params=params, tx=make_adam(config.optim), weights=weights, momentum=config.weighting.get("momentum", 0.9)
        )
        self.scheme = config.weighting.get("scheme", "none")
        self.causal = bool(config.weighting.get("use_causal", False))
        self.annealer = L.CausalAnnealer(tuple(config.weighting.get("causal_eps_schedule", (1e-2, 1e-1, 1.0, 10.0, 100.0)))) if self.causal else None
        if self.causal and hasattr(problem, "set_causal_eps"):
            problem.set_causal_eps(config.weighting.get("causal_tol", self.annealer.eps))
        self.logger = CSVLogger(self.workdir / "metrics.csv")
        self.wandb = None
        if use_wandb:
            import wandb  # optional dependency

            wandb.init(project=config.get("project", "pinnflow"), name=config.get("name", self.workdir.name), config=config.to_dict() if hasattr(config, "to_dict") else dict(config))
            self.wandb = wandb
        self.step_offset = 0
        print(f"[trainer] {problem.__class__.__name__}: {count_params(params):,} parameters, workdir={self.workdir}")

        # jit-compiled kernels (problem is closed over, so re-instantiate Trainer when the problem changes shape)
        def total_loss(params, weights, batch):
            terms = problem.losses(params, batch)
            return L.weighted_total(terms, weights), terms

        @jax.jit
        def _step(state: TrainState, batch):
            (loss, terms), grads = jax.value_and_grad(total_loss, has_aux=True)(state.params, state.weights, batch)
            state = state.apply_gradients(grads=grads)
            return state, loss, terms

        @jax.jit
        def _update_weights(state: TrainState, batch):
            if self.scheme == "grad_norm":
                w = L.grad_norm_weights(problem.losses, state.params, batch, reference=config.weighting.get("grad_norm_reference", None))
            elif self.scheme == "ntk":
                w = L.ntk_weights(problem.ntk_diags(state.params, batch))
            else:
                return state
            fixed = tuple(config.weighting.get("fixed_terms", ("p_anchor", "gauge")))
            w = {k: (state.weights[k] if k in fixed else w.get(k, state.weights[k])) for k in state.weights}
            return state.apply_weights(w)

        self._step = _step
        self._update_weights = _update_weights
        self._causal_min = jax.jit(problem.causal_min_weight) if self.causal and hasattr(problem, "causal_min_weight") else None

    # ------------------------------------------------------------------
    def next_key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def train(self, max_steps: int, log_every: int = 100, eval_every: int = 5000, ckpt_every: Optional[int] = 10000, rad_every: Optional[int] = None, tag: str = "") -> TrainState:
        cfg = self.config
        update_every = cfg.weighting.get("update_every_steps", 1000)
        t0 = time.perf_counter()
        print(f"[trainer] compiling and training {max_steps} steps {tag}...")
        for step in range(max_steps):
            batch = self.problem.sample_batch(self.next_key())
            self.state, loss, terms = self._step(self.state, batch)

            if self.scheme in ("grad_norm", "ntk") and step % update_every == 0:
                self.state = self._update_weights(self.state, batch)

            if self.causal and self._causal_min is not None and step % update_every == 0 and step > 0:
                mw = float(self._causal_min(self.state.params, batch))
                if self.annealer.update(mw) and hasattr(self.problem, "set_causal_eps"):
                    self.problem.set_causal_eps(self.annealer.eps)
                    print(f"[trainer] causal eps -> {self.annealer.eps} (min weight {mw:.3f})")

            if rad_every and step > 0 and step % rad_every == 0 and hasattr(self.problem, "update_collocation"):
                self.problem.update_collocation(self.next_key(), self.state.params)

            gstep = step + self.step_offset
            if step % log_every == 0 or step == max_steps - 1:
                row = {"step": gstep, "loss": float(loss), "time": time.perf_counter() - t0}
                row.update({f"loss/{k}": float(v) for k, v in terms.items()})
                row.update({f"w/{k}": float(v) for k, v in self.state.weights.items()})
                if self.causal and self._causal_min is not None:
                    row["causal/min_w"] = float(self._causal_min(self.state.params, batch))
                    row["causal/eps"] = self.annealer.eps
                if eval_every and (step % eval_every == 0 or step == max_steps - 1) and hasattr(self.problem, "evaluate"):
                    row.update({f"eval/{k}": float(v) for k, v in self.problem.evaluate(self.state.params).items()})
                self._log(row)

            if ckpt_every and (step + 1) % ckpt_every == 0:
                self.save(f"ckpt_{gstep + 1}.msgpack")

        self.step_offset += max_steps
        self.save("latest.msgpack")
        return self.state

    def _log(self, row: Dict):
        self.logger.log(row)
        if self.wandb is not None:
            self.wandb.log(row, step=int(row["step"]))
        keys = [k for k in row if k.startswith("loss") or k.startswith("eval") or k.startswith("causal")]
        msg = " ".join(f"{k}={row[k]:.3e}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in ["step", *keys])
        print(f"[{row['time']:8.1f}s] {msg}")

    def save(self, name: str = "latest.msgpack"):
        extra = {"step": int(self.step_offset), "weights": {k: float(v) for k, v in self.state.weights.items()}}
        save_params(self.state.params, self.workdir / name, extra)

    def load(self, path: os.PathLike):
        params, extra = load_params(path, template=self.state.params)
        self.state = self.state.replace(params=params)
        return extra

    # ------------------------------------------------------------------
    # 6.1 Stage 2 - L-BFGS
    # ------------------------------------------------------------------
    def lbfgs(self, max_iters: int = 20000, batch=None, memory_size: int = 20, log_every: int = 100, tol: float = 1e-12) -> TrainState:
        """Full-batch L-BFGS with a strong-Wolfe zoom line search on a fixed collocation batch."""
        batch = self.problem.sample_batch(self.next_key()) if batch is None else batch
        weights = self.state.weights

        def loss_fn(params):
            return L.weighted_total(self.problem.losses(params, batch), weights)

        params = run_lbfgs(loss_fn, self.state.params, max_iters, memory_size=memory_size, log_every=log_every, tol=tol, logger=lambda i, v: self._log({"step": self.step_offset + i, "loss": v, "time": 0.0, "loss/lbfgs": v}))
        self.state = self.state.replace(params=params)
        self.step_offset += max_iters
        self.save("latest_lbfgs.msgpack")
        self.save("latest.msgpack")  # evaluate.py / visualize.py default to latest.msgpack
        return self.state


def run_lbfgs(loss_fn: Callable, params, max_iters: int, memory_size: int = 20, log_every: int = 100, tol: float = 1e-12, logger: Optional[Callable] = None, patience: int = 50):
    """Optax L-BFGS loop (``optax.lbfgs`` uses the zoom line search satisfying the strong Wolfe conditions)."""
    opt = optax.lbfgs(memory_size=memory_size)
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(params, state):
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=loss_fn)
        params = optax.apply_updates(params, updates)
        return params, state, value

    state = opt.init(params)
    prev, stalls = np.inf, 0
    for i in range(max_iters):
        params, state, value = step(params, state)
        v = float(value)
        if logger is not None and (i % log_every == 0 or i == max_iters - 1):
            logger(i, v)
        if not np.isfinite(v):
            break
        stalls = stalls + 1 if abs(prev - v) < tol * max(1.0, abs(v)) else 0
        if stalls >= patience:
            break
        prev = v
    return params


# --------------------------------------------------------------------------------------
# 6.3 curriculum and 6.4 time marching
# --------------------------------------------------------------------------------------
def train_curriculum(trainer: Trainer, Re_list, steps_list, **train_kwargs):
    """for Re in [100, 200, 400, 1000]: warm-start from the previous Re, train, checkpoint."""
    assert len(Re_list) == len(steps_list)
    for i, (Re, steps) in enumerate(zip(Re_list, steps_list)):
        trainer.problem.set_Re(Re)
        if i > 0:  # warm start the weights but restart Adam and its warm-up/decay schedule for the new Re
            trainer.state = trainer.state.replace(opt_state=trainer.state.tx.init(trainer.state.params))
        trainer.train(int(steps), tag=f"Re={Re}", **train_kwargs)
        trainer.save(f"Re{int(Re)}.msgpack")
    return trainer.state


def train_time_windows(make_trainer: Callable[[int, Optional[Callable]], Trainer], num_windows: int, steps_per_window: int, **train_kwargs):
    """6.4 sequence-to-sequence: window n uses the terminal state of window n-1 as its initial condition.

    ``make_trainer(window_idx, ic_fn)`` must build a Trainer whose problem covers window ``idx``
    and uses ``ic_fn(x, y) -> (u, v, p)`` as the initial condition (``None`` for the first window).
    Returns the list of trained trainers (one per window).
    """
    trainers = []
    ic_fn = None
    for n in range(num_windows):
        tr = make_trainer(n, ic_fn)
        if trainers:  # warm start: same architecture, continue from the previous window's weights
            tr.state = tr.state.replace(params=trainers[-1].state.params, weights=trainers[-1].state.weights)
        tr.train(int(steps_per_window), tag=f"window {n}", **train_kwargs)
        ic_fn = tr.problem.terminal_state_fn(tr.state.params)
        trainers.append(tr)
    return trainers
