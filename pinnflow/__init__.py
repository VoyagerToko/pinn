"""pinnflow: physics-informed neural networks for incompressible Navier-Stokes.

Layout (one module per handbook step):
    physics      STEP 1  non-dimensional NS residuals (VP, stream function, vector potential), forward-mode helpers
    benchmarks   STEP 2  Taylor-Green 2D/3D, lid-driven cavity, DFG cylinder, cylinder-wake inverse problem
    data         STEP 3  dataset loaders (cylinder wake .mat, CausalPINNs NS, JAX-PI cavity fields, Ghia tables)
    archs        STEP 4  Fourier features, Modified MLP, RWF, PirateNet, SPINN, PI-DeepONet
    constraints  STEP 4.6 hard Dirichlet constraints, approximate distance functions, RWF fold-back
    losses       STEP 5  loss terms, grad-norm / NTK / causal weighting, RAD resampling, pressure anchoring
    training     STEP 6  Adam schedule, L-BFGS stage, curriculum, time marching, checkpoints
    metrics      STEP 7  relative L2, divergence, drag/lift, Strouhal, energy/dissipation/enstrophy, spectra
    viz          STEP 9  field sampling to VTK, Q-criterion, particle advection, rendering, encoding
    problems/    STEP 5-6 concrete PINN problem classes (loss dictionaries + samplers) for each benchmark
    configs      STEP 6.6/8 reference configs and the ablation ladder A-H
"""

import jax

__version__ = "0.1.0"


def configure_jax(x64: bool = False, matmul_precision: str = "highest") -> None:
    """Handbook STEP 6.2: always run with the highest matmul precision.

    float32 with ``jax_default_matmul_precision="highest"`` is the reproducible default;
    pass ``x64=True`` only for the machine-precision residual checks in the tests.
    """
    jax.config.update("jax_default_matmul_precision", matmul_precision)
    if x64:
        jax.config.update("jax_enable_x64", True)
