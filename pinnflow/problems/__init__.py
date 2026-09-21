"""Concrete PINN problems. Each class wires a benchmark (STEP 2), an architecture (STEP 4), the
residuals (STEP 1) and the loss terms (STEP 5) into the interface consumed by
:class:`pinnflow.training.Trainer`.
"""
from .base import Problem
from .cavity import CavityPINN
from .cylinder import DFGCylinderPINN
from .cylinder_inverse import CylinderWakeInversePINN
from .deeponet import ParametricCavityDeepONet
from .tgv2d import TaylorGreen2DPINN
from .tgv3d import TaylorGreen3DSPINN

PROBLEMS = {
    "tgv2d": TaylorGreen2DPINN,
    "cavity": CavityPINN,
    "cylinder": DFGCylinderPINN,
    "tgv3d": TaylorGreen3DSPINN,
    "cylinder_inverse": CylinderWakeInversePINN,
    "deeponet_cavity": ParametricCavityDeepONet,
}

__all__ = ["Problem", "PROBLEMS", *[c.__name__ for c in PROBLEMS.values()]]
