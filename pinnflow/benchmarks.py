"""STEP 2 - the four benchmarks, defined exactly, plus the inverse cylinder-wake problem.

Each benchmark object carries: domain, exact solution / initial condition / boundary data,
reference values and the acceptance gate from handbook STEP 10. Nothing here is trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
TWO_PI = 2.0 * np.pi


# ======================================================================================
# 2.1 Benchmark A - 2D Taylor-Green vortex (analytic)
# ======================================================================================
@dataclass(frozen=True)
class TaylorGreen2D:
    """Domain [0, 2pi]^2, periodic, t in [0, T]. Exact solution used as ground truth.

    Acceptance gate: relative L2 < 1e-4 (STEP 10, week 2).
    """

    Re: float = 100.0
    T: float = 1.0
    gate_rel_l2: float = 1e-4

    @property
    def domain(self) -> np.ndarray:
        # rows: (t, x, y)
        return np.array([[0.0, self.T], [0.0, TWO_PI], [0.0, TWO_PI]])

    def exact(self, t: Array, x: Array, y: Array) -> Tuple[Array, Array, Array]:
        decay = jnp.exp(-2.0 * t / self.Re)
        u = -jnp.cos(x) * jnp.sin(y) * decay
        v = jnp.sin(x) * jnp.cos(y) * decay
        p = -0.25 * (jnp.cos(2 * x) + jnp.cos(2 * y)) * jnp.exp(-4.0 * t / self.Re)
        return u, v, p

    def exact_streamfunction(self, t: Array, x: Array, y: Array) -> Array:
        """psi with u = psi_y, v = -psi_x:  psi = cos(x) cos(y) exp(-2t/Re)
        (psi_y = -cos x sin y = u,  -psi_x = sin x cos y = v)."""
        return jnp.cos(x) * jnp.cos(y) * jnp.exp(-2.0 * t / self.Re)

    def exact_vector(self, z: Array) -> Array:
        """Point-wise (u, v, p) for z = (t, x, y) - plugs straight into physics residuals."""
        u, v, p = self.exact(z[0], z[1], z[2])
        return jnp.stack([u, v, p])

    def exact_psi_p(self, z: Array) -> Array:
        return jnp.stack([self.exact_streamfunction(z[0], z[1], z[2]), self.exact(z[0], z[1], z[2])[2]])


# ======================================================================================
# 2.2 Benchmark B - lid-driven cavity (steady)
# ======================================================================================
_GHIA_Y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0])
_GHIA_U = {
    100: [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0],
    400: [1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0],
    1000: [1.0, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0],
    3200: [1.0, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791, 0.07156, -0.04272, -0.08663, -0.24427, -0.34323, -0.41933, -0.37827, -0.35344, -0.32407, 0.0],
    5000: [1.0, 0.48223, 0.46120, 0.45992, 0.46036, 0.33556, 0.20087, 0.08183, -0.03039, -0.07404, -0.22855, -0.33050, -0.40435, -0.43643, -0.42901, -0.41165, 0.0],
}
_GHIA_X = np.array([1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0])
_GHIA_V = {
    100: [0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.0],
    400: [0.0, -0.12146, -0.15663, -0.19254, -0.22847, -0.33827, -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.0],
    1000: [0.0, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.0],
    3200: [0.0, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307, -0.37401, -0.31184, 0.00999, 0.28188, 0.29030, 0.37119, 0.42768, 0.41906, 0.40917, 0.39560, 0.0],
    5000: [0.0, -0.49774, -0.55069, -0.55408, -0.52876, -0.41442, -0.36214, -0.30018, 0.00945, 0.27280, 0.28066, 0.35368, 0.42951, 0.43648, 0.43329, 0.42447, 0.0],
}


def ghia_tables(Re: int) -> Dict[str, np.ndarray]:
    """Ghia, Ghia & Shin (1982) Tables I/II: u(y) on x=0.5 and v(x) on y=0.5 for the *unit lid*.

    Transcribed from the widely reproduced tables; ``tests/test_benchmarks.py`` cross-checks them
    against the JAX-PI reference fields shipped in ``external/jaxpi/examples/ldc/data``.
    Note: the regularised lid used for training (below) differs from Ghia's unit lid near the
    corners, so compare against Ghia only away from y = 1.
    """
    if Re not in _GHIA_U:
        raise KeyError(f"Ghia tables available for Re in {sorted(_GHIA_U)}")
    return {"y": _GHIA_Y.copy(), "u": np.array(_GHIA_U[Re]), "x": _GHIA_X.copy(), "v": np.array(_GHIA_V[Re])}


@dataclass(frozen=True)
class LidDrivenCavity:
    """Domain [0,1]^2, no-slip walls, lid at y=1 moving in +x with the regularised profile

        u(x, 1) = 1 - cosh(r (x - 0.5)) / cosh(0.5 r),   r = 50,   v(x, 1) = 0.

    Curriculum: Re = 100 -> 400 -> 1000 (handbook 6.3). Gate: Re=1000 within 2% of Ghia.
    """

    Re: float = 100.0
    r: float = 50.0
    curriculum: Tuple[int, ...] = (100, 400, 1000)
    gate_rel_error: float = 0.02

    @property
    def domain(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])  # rows: (x, y)

    def lid_profile(self, x: Array) -> Array:
        return 1.0 - jnp.cosh(self.r * (x - 0.5)) / jnp.cosh(0.5 * self.r)

    def boundary_velocity(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """(u, v) on the boundary: lid profile on y=1, zero elsewhere."""
        on_lid = jnp.isclose(y, 1.0)
        u = jnp.where(on_lid, self.lid_profile(x), 0.0)
        return u, jnp.zeros_like(u)

    def reference(self) -> Dict[str, np.ndarray]:
        return ghia_tables(int(self.Re))


# ======================================================================================
# 2.3 Benchmark C - DFG flow past a cylinder (Schaefer & Turek 1996)
# ======================================================================================
@dataclass(frozen=True)
class DFGCylinder:
    """Channel [0, 2.2] x [0, 0.41], cylinder centre (0.2, 0.2), radius 0.05, nu = 0.001, rho = 1.

    ``variant``: "2D-2" steady inflow amplitude (Re = 100, periodic shedding) or
                 "2D-3" inflow modulated by sin(pi t / 8), t in (0, 8).
    Non-dimensionalisation: U = U_mean = 1.0 (2/3 * 1.5), L = D = 0.1  =>  Re = 100.

    Reference intervals (Schaefer & Turek 1996; FEATFLOW benchmark pages):
        2D-2:  Cd_max in [3.22, 3.24], Cl_max in [0.99, 1.01], St in [0.295, 0.305], dP in [2.46, 2.50]
        2D-3:  Cd_max in [2.93, 2.97] (t~3.93), Cl_max in [0.47, 0.49] (t~5.69), dP(t=8) in [-0.115, -0.105]
    Note: the unconfined-cylinder value St ~ 0.164 quoted in handbook 7.3 applies to the
    Raissi wake data set (Benchmark C inverse problem), *not* to this confined DFG channel.
    """

    variant: str = "2D-2"
    U_max: float = 1.5
    nu: float = 1e-3
    rho: float = 1.0
    length: float = 2.2
    height: float = 0.41
    center: Tuple[float, float] = (0.2, 0.2)
    radius: float = 0.05
    T: float = 8.0
    # nondimensionalisation
    U_ref: float = 1.0  # mean inflow velocity
    L_ref: float = 0.1  # cylinder diameter
    reference: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "2D-2/Cd_max": (3.22, 3.24),
            "2D-2/Cl_max": (0.99, 1.01),
            "2D-2/St": (0.295, 0.305),
            "2D-2/dP": (2.46, 2.50),
            "2D-3/Cd_max": (2.93, 2.97),
            "2D-3/Cl_max": (0.47, 0.49),
            "2D-3/dP_t8": (-0.115, -0.105),
        }
    )

    @property
    def Re(self) -> float:
        return self.U_ref * self.L_ref / self.nu  # 100

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def domain(self) -> np.ndarray:
        """Dimensional (t, x, y) box; the cylinder is excluded by rejection sampling."""
        return np.array([[0.0, self.T], [0.0, self.length], [0.0, self.height]])

    def inflow(self, t: Array, y: Array) -> Array:
        """u_in(t, y) = 4 U y (0.41 - y) / 0.41^2, times sin(pi t / 8) for 2D-3."""
        amp = self.U_max
        if self.variant == "2D-3":
            amp = amp * jnp.sin(jnp.pi * t / 8.0)
        return 4.0 * amp * y * (self.height - y) / self.height**2

    def in_fluid(self, x: Array, y: Array) -> Array:
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 > self.radius**2

    def cylinder_surface(self, n: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Points, outward normals (n_x, n_y) and arc-length weights on the cylinder for STEP 7.2."""
        theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        ds = np.full(n, TWO_PI * self.radius / n)
        return np.stack([x, y], -1), np.cos(theta), np.sin(theta), ds

    def pressure_probe_points(self) -> np.ndarray:
        """a1 = (0.15, 0.2) front, a2 = (0.25, 0.2) rear; dP = p(a1) - p(a2)."""
        return np.array([[0.15, 0.2], [0.25, 0.2]])

    def nondim(self, t, x, y):
        """Dimensional -> non-dimensional coordinates (t U/L, x/L, y/L)."""
        return t * self.U_ref / self.L_ref, x / self.L_ref, y / self.L_ref

    def redim_velocity(self, u):
        return u * self.U_ref

    def redim_pressure(self, p):
        return p * self.rho * self.U_ref**2


# ======================================================================================
# 2.3 (inverse) - Raissi et al. 2019 cylinder wake data set
# ======================================================================================
@dataclass(frozen=True)
class CylinderWakeInverse:
    """Nektar spectral-element wake at Re = 100 (cylinder diameter 1, U = 1, nu = 0.01).

    Window x in [1, 8], y in [-2, 2], t in [0, 19.9] (200 snapshots, 5000 points).
    Inverse problem: learn lambda_1 (convection) and lambda_2 (= nu) from sparse (u, v) data.
    Targets: lambda_1 = 1.0, lambda_2 = 0.01, Strouhal ~ 0.164 (unconfined cylinder).
    """

    path: str = "data/cylinder_wake/cylinder_nektar_wake.mat"
    nu_true: float = 0.01
    lambda1_true: float = 1.0
    St_ref: Tuple[float, float] = (0.16, 0.17)
    n_train: int = 5000

    @property
    def domain(self) -> np.ndarray:
        return np.array([[0.0, 19.9], [1.0, 8.0], [-2.0, 2.0]])


# ======================================================================================
# 2.4 Benchmark D - 3D Taylor-Green vortex at Re = 1600
# ======================================================================================
@dataclass(frozen=True)
class TaylorGreen3D:
    """Domain [0, 2pi]^3, triply periodic, t in [0, 20], Re = 1600 (nu = 1/1600, U = L = 1).

    Initial condition:
        u =  sin x cos y cos z,  v = -cos x sin y cos z,  w = 0,
        p = (1/16)(cos 2x + cos 2y)(cos 2z + 2)
    Reference: HiOCFD4/5 dissipation and enstrophy curves; the dissipation peak near t ~ 9
    (approximately 1.28e-2 for the 512^3 spectral DNS) is the acceptance criterion (within 5%).
    """

    Re: float = 1600.0
    T: float = 20.0
    dissipation_peak_time: float = 9.0
    dissipation_peak_value: float = 0.0128  # approximate; compare against the downloaded HiOCFD data
    gate_rel_error: float = 0.05

    @property
    def nu(self) -> float:
        return 1.0 / self.Re

    @property
    def domain(self) -> np.ndarray:
        return np.array([[0.0, self.T], [0.0, TWO_PI], [0.0, TWO_PI], [0.0, TWO_PI]])

    def initial_condition(self, x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array, Array]:
        u = jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
        v = -jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
        w = jnp.zeros_like(u)
        p = (1.0 / 16.0) * (jnp.cos(2 * x) + jnp.cos(2 * y)) * (jnp.cos(2 * z) + 2.0)
        return u, v, w, p

    def initial_vector_potential(self, x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
        """A with curl A = u0:  A = (0, 0, sin x sin y cos z)  (check: A3_y = sin x cos y cos z = u, -A3_x = -cos x sin y cos z = v)."""
        zero = jnp.zeros_like(x)
        return zero, zero, jnp.sin(x) * jnp.sin(y) * jnp.cos(z)

    def initial_energy(self) -> float:
        """E_k(0) = (1/|Omega|) int 1/2 |u0|^2 = 1/8 (analytic)."""
        return 0.125


BENCHMARKS = {
    "tgv2d": TaylorGreen2D,
    "cavity": LidDrivenCavity,
    "cylinder": DFGCylinder,
    "cylinder_inverse": CylinderWakeInverse,
    "tgv3d": TaylorGreen3D,
}
