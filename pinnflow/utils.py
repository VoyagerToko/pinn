"""Small utilities: chunked evaluation, parameter (de)serialisation, CSV logging, timing."""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization


def chunked_vmap(fn: Callable, pts: jnp.ndarray, chunk: int = 8192, **vmap_kwargs) -> jnp.ndarray:
    """Apply ``vmap(fn)`` to ``pts`` (N, ...) in chunks so second-order AD fits in GPU memory."""
    vfn = jax.jit(jax.vmap(fn, **vmap_kwargs))
    n = pts.shape[0]
    outs = []
    for s in range(0, n, chunk):
        outs.append(vfn(pts[s : s + chunk]))
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outs) if len(outs) > 1 else outs[0]


def device_peak_gb() -> float:
    """Peak bytes held by this process's allocator on the first device, in GB (nan on CPU)."""
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
        return float(stats.get("peak_bytes_in_use", float("nan"))) / 1e9
    except Exception:
        return float("nan")


def count_params(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def save_params(params, path: os.PathLike, extra: Optional[Dict] = None) -> None:
    """Serialise a param pytree (plus optional small metadata dict) with flax msgpack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": serialization.to_state_dict(jax.device_get(params)), "extra": extra or {}}
    path.write_bytes(serialization.msgpack_serialize(payload))


def load_params(path: os.PathLike, template=None):
    """Load params saved by :func:`save_params`.

    With ``template`` (e.g. freshly initialised params of the same arch) the exact pytree structure,
    including the ``(g, V)`` tuples of factorised kernels, is restored; without it you get the raw
    state dict (tuples appear as ``{"0": g, "1": V}``).
    """
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    params = payload["params"]
    if template is not None:
        params = serialization.from_state_dict(template, params)
    return params, payload.get("extra", {})


class CSVLogger:
    """Append-only CSV metrics log; columns are the union of keys seen (header written once)."""

    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = None

    def log(self, row: Dict):
        row = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in row.items()}
        new = not self.path.exists()
        if self._fields is None:
            if not new:
                with open(self.path, newline="") as f:
                    self._fields = next(csv.reader(f))
            else:
                self._fields = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self._fields})


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        t = time.perf_counter()
        dt, self.t0 = t - self.t0, t
        return dt


def meshgrid_points(*axes) -> jnp.ndarray:
    """Cartesian product of 1-D axes -> (prod n_i, d) points in 'ij' order."""
    grids = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([g.ravel() for g in grids], axis=-1)
