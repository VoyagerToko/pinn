"""STEP 4 - network architectures.

All point-wise networks take a single input vector ``x`` of shape ``(d,)`` and return a
vector of shape ``(out_dim,)``; batch them with ``jax.vmap``. This is the JAX-PI convention
and is what makes ``jax.grad`` / ``jax.jacfwd`` with respect to individual coordinates clean.

Components (handbook numbering):
    4.2  FourierEmbs      random Fourier features, multi-scale, separate temporal scale
    4.3  ModifiedMlp      gated architecture (Wang, Teng & Perdikaris 2021)
    4.4  Dense(reparam)   random weight factorisation  W = diag(exp(s)) V
    4.5  PirateNet        adaptive residual blocks, identity at init (alpha = 0)
    4.7  SPINN            separable PINN, u = sum_j prod_i f_j^i(x_i)
    4.8  DeepONet         PI-DeepONet branch/trunk for parametric (Re) generalisation
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze
from jax import random
from jax.nn.initializers import constant, glorot_normal, normal, zeros

ACTIVATIONS: Dict[str, Callable] = {
    "tanh": jnp.tanh,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "relu": nn.relu,
    "sin": jnp.sin,
}


def get_activation(name: str) -> Callable:
    if name not in ACTIVATIONS:
        raise NotImplementedError(f"Activation {name!r} not supported; choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]


# --------------------------------------------------------------------------------------
# 4.4 Random weight factorisation
# --------------------------------------------------------------------------------------
def _weight_fact(init_fn: Callable, mean: float, stddev: float) -> Callable:
    """W = g * V with g = exp(s), s ~ N(mean, stddev^2). Returns the pair (g, V) as one param."""

    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = jnp.exp(mean + normal(stddev)(key2, (shape[-1],)))
        return g, w / g

    return init


class Dense(nn.Module):
    """Linear layer with optional random weight factorisation (``reparam={'type': 'weight_fact', ...}``)."""

    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        if not self.reparam:
            kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        elif self.reparam["type"] in ("weight_fact", "rwf"):
            g, v = self.param(
                "kernel",
                _weight_fact(self.kernel_init, self.reparam.get("mean", 1.0), self.reparam.get("stddev", 0.1)),
                (x.shape[-1], self.features),
            )
            kernel = g * v
        else:
            raise NotImplementedError(f"reparam type {self.reparam['type']!r} not supported")
        bias = self.param("bias", self.bias_init, (self.features,))
        return jnp.dot(x, kernel) + bias


# --------------------------------------------------------------------------------------
# Input embeddings
# --------------------------------------------------------------------------------------
class PeriodEmbs(nn.Module):
    """Replace coordinate ``x_i`` by ``(cos(w_i x_i), sin(w_i x_i))`` on the given axes.

    Makes the network exactly periodic (Benchmarks A and D). ``period`` holds the angular
    frequency ``w_i = 2*pi / L_i`` per listed axis.
    """

    period: Sequence[float]
    axis: Sequence[int]
    trainable: Sequence[bool] = ()

    def setup(self):
        trainable = tuple(self.trainable) if self.trainable else (False,) * len(self.axis)
        params = {}
        for idx, is_trainable in enumerate(trainable):
            if is_trainable:
                params[f"period_{idx}"] = self.param(f"period_{idx}", constant(self.period[idx]), ())
            else:
                params[f"period_{idx}"] = self.period[idx]
        self.period_params = freeze(params)

    @nn.compact
    def __call__(self, x):
        y = []
        axis = tuple(self.axis)
        for i in range(x.shape[-1]):
            xi = x[..., i]
            if i in axis:
                w = self.period_params[f"period_{axis.index(i)}"]
                y.extend([jnp.cos(w * xi), jnp.sin(w * xi)])
            else:
                y.append(xi)
        return jnp.stack(y, axis=-1)


class FourierEmbs(nn.Module):
    """4.2 Random Fourier features ``[cos(Bx), sin(Bx)]`` with ``B_ij ~ N(0, sigma^2)``.

    * ``embed_scale`` may be a single sigma or a tuple for the multi-scale variant
      (e.g. ``(1.0, 10.0)``); the ``embed_dim`` budget is split evenly between scales.
    * ``time_axis``/``time_scale`` give the temporal input its own (smaller) sigma.

    Convention follows JAX-PI (no 2*pi factor inside the cosine), so the handbook's
    ``embed_scale: 10.0`` maps one-to-one onto ``FourierEmbs(embed_scale=10.0)``.
    """

    embed_scale: Union[float, Sequence[float]] = 10.0
    embed_dim: int = 256
    time_axis: Optional[int] = None
    time_scale: Optional[float] = None

    @nn.compact
    def __call__(self, x):
        scales = tuple(self.embed_scale) if isinstance(self.embed_scale, (list, tuple)) else (float(self.embed_scale),)
        d = x.shape[-1]
        m = max(1, self.embed_dim // (2 * len(scales)))
        feats = []
        for i, sigma in enumerate(scales):
            col_scale = jnp.full((d,), sigma)
            if self.time_axis is not None and self.time_scale is not None:
                col_scale = col_scale.at[self.time_axis].set(self.time_scale)
            kernel = self.param(f"kernel_{i}", normal(1.0), (d, m)) * col_scale[:, None]
            z = jnp.dot(x, kernel)
            feats += [jnp.cos(z), jnp.sin(z)]
        return jnp.concatenate(feats, axis=-1)


class Embedding(nn.Module):
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
        return x


# --------------------------------------------------------------------------------------
# 4.1 / 4.3 / 4.5 point-wise networks
# --------------------------------------------------------------------------------------
class Mlp(nn.Module):
    """4.1 Baseline MLP (tanh, Glorot normal). With ``fourier_emb``/``reparam`` it becomes ablation row B."""

    arch_name: str = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, x, return_features: bool = False):
        act = get_activation(self.activation)
        x = Embedding(self.periodicity, self.fourier_emb)(x)
        for _ in range(self.num_layers):
            x = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        y = Dense(self.out_dim, reparam=self.reparam)(x)
        return (x, y) if return_features else y


class ModifiedMlp(nn.Module):
    """4.3 Modified MLP: ``U, V`` computed once from the embedding and injected at every layer."""

    arch_name: str = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, x, return_features: bool = False):
        act = get_activation(self.activation)
        x = Embedding(self.periodicity, self.fourier_emb)(x)
        u = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        v = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        for _ in range(self.num_layers):
            z = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
            x = z * u + (1 - z) * v
        y = Dense(self.out_dim, reparam=self.reparam)(x)
        return (x, y) if return_features else y


class PIModifiedBottleneck(nn.Module):
    """4.5 PirateNet block: three gated dense layers and an adaptive residual ``alpha*h + (1-alpha)*x``."""

    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Optional[Dict]

    @nn.compact
    def __call__(self, x, u, v):
        act = get_activation(self.activation)
        identity = x
        f = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        z1 = f * u + (1 - f) * v
        g = act(Dense(self.hidden_dim, reparam=self.reparam)(z1))
        z2 = g * u + (1 - g) * v
        h = act(Dense(self.output_dim, reparam=self.reparam)(z2))
        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        return alpha * h + (1 - alpha) * identity


class PirateNet(nn.Module):
    """4.5 PirateNet (Wang et al. 2024). ``nonlinearity`` is the initial alpha; 0 gives identity blocks.

    The Fourier embedding width must equal ``hidden_dim`` (residual path); the constructor
    therefore forces ``fourier_emb['embed_dim'] = hidden_dim`` if they disagree.
    ``pi_init`` (optional, shape ``(hidden_dim, out_dim)``) is the physics-informed
    least-squares initialisation of the output layer; see :func:`pirate_pi_init`.
    """

    arch_name: str = "PirateNet"
    num_layers: int = 3
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None
    reparam: Optional[Dict] = None
    pi_init: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self, x, return_features: bool = False):
        act = get_activation(self.activation)
        fourier_emb = dict(self.fourier_emb) if self.fourier_emb else {"embed_scale": 2.0}
        fourier_emb["embed_dim"] = self.hidden_dim
        x = Embedding(self.periodicity, fourier_emb)(x)
        u = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        v = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x, u, v)
        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)
        else:
            y = Dense(self.out_dim, reparam=self.reparam)(x)
        return (x, y) if return_features else y


def pirate_pi_init(arch: nn.Module, params, x_data: jnp.ndarray, y_data: jnp.ndarray, reg: float = 1e-6) -> jnp.ndarray:
    """Least-squares initialisation of the PirateNet output layer from (sparse) data.

    Solves ``min_W ||Phi(x) W - y||^2`` where ``Phi`` are the penultimate features at init.
    Rebuild the arch with ``pi_init=W`` afterwards.
    """
    feats, _ = jax.vmap(lambda x: arch.apply(params, x, return_features=True))(x_data)
    gram = feats.T @ feats + reg * jnp.eye(feats.shape[-1])
    return jnp.linalg.solve(gram, feats.T @ y_data)


# --------------------------------------------------------------------------------------
# 4.7 Separable PINN
# --------------------------------------------------------------------------------------
class _AxisMlp(nn.Module):
    """One body network ``R^k -> R^(rank*out_dim)`` for a single coordinate axis."""

    num_layers: int
    hidden_dim: int
    out_features: int
    activation: str
    mlp: str
    reparam: Optional[Dict]

    @nn.compact
    def __call__(self, c):
        act = get_activation(self.activation)
        if self.mlp == "modified_mlp":
            u = act(Dense(self.hidden_dim, reparam=self.reparam)(c))
            v = act(Dense(self.hidden_dim, reparam=self.reparam)(c))
            h = act(Dense(self.hidden_dim, reparam=self.reparam)(c))
            for _ in range(self.num_layers - 1):
                z = act(Dense(self.hidden_dim, reparam=self.reparam)(h))
                h = (1 - z) * u + z * v
        else:
            h = c
            for _ in range(self.num_layers):
                h = act(Dense(self.hidden_dim, reparam=self.reparam)(h))
        return Dense(self.out_features, reparam=self.reparam)(h)


class SPINN(nn.Module):
    """4.7 Separable PINN (Cho et al. 2023).

    ``u_o(x_1..x_d) = sum_{j=1}^{rank} prod_{i=1}^{d} f^{(i)}_{o,j}(x_i)`` with one small MLP per axis.

    Two evaluation modes:
      * ``mode="grid"``:   ``coords`` is a list of ``d`` 1-D arrays (n_1,), ..., (n_d,);
        returns an array of shape ``(out_dim, n_1, ..., n_d)`` - the separable evaluation
        that gives ``prod n_i`` collocation points for ``sum n_i`` network evaluations.
        Derivatives on this grid are taken with forward-mode ``jax.jvp`` along each 1-D axis
        (see :mod:`pinnflow.physics`).
      * ``mode="points"``: ``coords`` is an ``(N, d)`` matrix of scattered points; returns
        ``(N, out_dim)`` - used for validation on held-out points and visualisation.

    ``periodicity={"axes": (1, 2, 3), "period": 2*pi, "num_freqs": 4}`` makes the listed
    axes exactly periodic (Benchmark D). Input order is ``(t, x, y, z)``.
    """

    arch_name: str = "SPINN"
    in_dim: int = 4
    num_layers: int = 4
    hidden_dim: int = 64
    rank: int = 128
    out_dim: int = 4
    activation: str = "tanh"
    mlp: str = "modified_mlp"
    periodicity: Optional[Dict] = None
    reparam: Optional[Dict] = None

    def setup(self):
        self.axis_nets = [
            _AxisMlp(self.num_layers, self.hidden_dim, self.rank * self.out_dim, self.activation, self.mlp, self.reparam)
            for _ in range(self.in_dim)
        ]

    def _embed_axis(self, i: int, c: jnp.ndarray) -> jnp.ndarray:
        c = c.reshape(-1, 1)
        if self.periodicity and i in tuple(self.periodicity["axes"]):
            k = jnp.arange(1, int(self.periodicity.get("num_freqs", 4)) + 1) * (2 * jnp.pi / self.periodicity["period"])
            c = jnp.concatenate([jnp.cos(c * k), jnp.sin(c * k)], axis=-1)
        return c

    def __call__(self, coords, mode: str = "grid"):
        if mode == "grid":
            feats = [self.axis_nets[i](self._embed_axis(i, c)) for i, c in enumerate(coords)]  # (n_i, r*out)
            outs = []
            for o in range(self.out_dim):
                sl = slice(o * self.rank, (o + 1) * self.rank)
                pred = feats[0][:, sl].T  # (r, n_1)
                for i in range(1, len(feats) - 1):
                    pred = jnp.einsum("r...,jr->r...j", pred, feats[i][:, sl])
                if len(feats) > 1:
                    pred = jnp.einsum("r...,jr->...j", pred, feats[-1][:, sl])  # contract rank last: no (n^d * r) tensor
                else:
                    pred = pred.sum(0)
                outs.append(pred)
            return jnp.stack(outs, axis=0)
        elif mode == "points":
            pts = jnp.asarray(coords)
            single = pts.ndim == 1
            pts = pts.reshape(-1, self.in_dim)
            feats = [self.axis_nets[i](self._embed_axis(i, pts[:, i])) for i in range(self.in_dim)]  # (N, r*out)
            prod = feats[0]
            for f in feats[1:]:
                prod = prod * f
            y = prod.reshape(pts.shape[0], self.out_dim, self.rank).sum(-1)  # (N, out)
            return y[0] if single else y
        raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------------------
# 4.8 PI-DeepONet
# --------------------------------------------------------------------------------------
class MlpBlock(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str
    reparam: Optional[Dict]

    @nn.compact
    def __call__(self, x):
        act = get_activation(self.activation)
        for _ in range(self.num_layers):
            x = act(Dense(self.hidden_dim, reparam=self.reparam)(x))
        return Dense(self.out_dim, reparam=self.reparam)(x)


class DeepONet(nn.Module):
    """4.8 PI-DeepONet: ``G(a)(y) = sum_k b_k(a) t_k(y) + b0`` per output, with ``p`` modes.

    ``branch_input`` is the parameter vector (e.g. ``[Re]`` or an inflow profile), ``x`` the query
    point. The trunk is a Modified MLP with Fourier features so it inherits Components 1-3.
    """

    arch_name: str = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4
    hidden_dim: int = 256
    p: int = 128
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Optional[Dict] = None
    fourier_emb: Optional[Dict] = None
    reparam: Optional[Dict] = None

    @nn.compact
    def __call__(self, branch_input, x):
        b = MlpBlock(self.num_branch_layers, self.hidden_dim, self.p * self.out_dim, self.activation, self.reparam)(
            jnp.atleast_1d(branch_input)
        )
        t = ModifiedMlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.p * self.out_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)
        y = (b * t).reshape(self.out_dim, self.p).sum(-1)
        bias = self.param("bias", zeros, (self.out_dim,))
        return y + bias


# --------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------
_ARCHS = {
    "Mlp": Mlp,
    "ModifiedMlp": ModifiedMlp,
    "PirateNet": PirateNet,
    "SPINN": SPINN,
    "DeepONet": DeepONet,
}


def _plain(obj):
    """Recursively convert ml_collections ConfigDicts / FrozenDicts to plain python containers."""
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return tuple(_plain(v) for v in obj)
    return obj


def create_arch(config) -> nn.Module:
    """Build an architecture from a config mapping with an ``arch_name`` key (STEP 6.6 style)."""
    cfg = _plain(config)
    name = cfg.pop("arch_name")
    cfg = {k: v for k, v in cfg.items() if v is not None and v is not False}
    if name not in _ARCHS:
        raise NotImplementedError(f"arch {name!r} not supported; choose from {list(_ARCHS)}")
    return _ARCHS[name](arch_name=name, **cfg)
