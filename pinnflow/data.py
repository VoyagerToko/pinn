"""STEP 3 - dataset loaders. Forward solves use these only for grading; the inverse problem
(cylinder wake) feeds sparse data into the loss on purpose. Always state which mode you ran.

Paths default to the repository layout created by ``scripts/setup``:
    data/cylinder_wake/cylinder_nektar_wake.mat        Raissi et al. 2019 (Nektar, Re=100)
    data/dfg_benchmark/dfg2d{2,3}_*/                    FEATFLOW reference Cd, Cl, dP series
    data/tgv3d_re1600/                                  HiOCFD / NASA references (pdf + any data files)
    external/CausalPINNs/data/NS.npy                    2D NS on a torus (vorticity form)
    external/jaxpi/examples/ldc/data/ldc_Re*.mat        JAX-PI cavity reference fields (128x128)
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = PROJECT_ROOT / "external"


# --------------------------------------------------------------------------------------
# cylinder wake (inverse problem)
# --------------------------------------------------------------------------------------
def load_cylinder_wake(path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Returns the raw arrays plus flattened (t, x, y, u, v, p) columns of length N*T.

    X_star (N,2), t (T,), U_star (N,2,T), p_star (N,T)  with N = 5000, T = 200.
    """
    from scipy.io import loadmat

    d = loadmat(str(path or DATA_DIR / "cylinder_wake" / "cylinder_nektar_wake.mat"))
    X, t, U, P = d["X_star"], d["t"].ravel(), d["U_star"], d["p_star"]
    N, T = X.shape[0], t.shape[0]
    TT = np.tile(t[None, :], (N, 1))
    XX = np.tile(X[:, 0:1], (1, T))
    YY = np.tile(X[:, 1:2], (1, T))
    return {
        "X_star": X, "t": t, "U_star": U, "p_star": P,
        "flat": {"t": TT.ravel(), "x": XX.ravel(), "y": YY.ravel(), "u": U[:, 0, :].ravel(), "v": U[:, 1, :].ravel(), "p": P.ravel()},
    }


def sample_sparse(flat: Dict[str, np.ndarray], n: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat["t"].shape[0], n, replace=False)
    return {k: v[idx] for k, v in flat.items()}


# --------------------------------------------------------------------------------------
# CausalPINNs / JAX-PI NS on a torus (vorticity form) - direct comparability with SPINN & causal paper
# --------------------------------------------------------------------------------------
def load_causal_ns(path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """sol (200,128,128) vorticity, u0/v0/w0 (128,128), x/y (128,), t (200,), viscosity 0.01."""
    d = np.load(str(path or EXTERNAL_DIR / "CausalPINNs" / "data" / "NS.npy"), allow_pickle=True).item()
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}


# --------------------------------------------------------------------------------------
# JAX-PI lid-driven cavity reference fields
# --------------------------------------------------------------------------------------
def load_jaxpi_cavity(Re: int) -> Dict[str, np.ndarray]:
    """u, v (128,128) with u[i, j] = u(x_i, y_j); x, y (128,); nu. Available Re: 100, 400, 1000, 3200, 5000."""
    from scipy.io import loadmat

    d = loadmat(str(EXTERNAL_DIR / "jaxpi" / "examples" / "ldc" / "data" / f"ldc_Re{Re}.mat"))
    return {"u": d["u"], "v": d["v"], "x": d["x"].ravel(), "y": d["y"].ravel(), "nu": float(np.asarray(d["nu"]).ravel()[0])}


def cavity_centerlines(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, y_query: np.ndarray, x_query: np.ndarray) -> Dict[str, np.ndarray]:
    """Interpolate u along x=0.5 at ``y_query`` and v along y=0.5 at ``x_query`` from a field with u[i,j]=u(x_i,y_j)."""
    from scipy.interpolate import RegularGridInterpolator

    fu = RegularGridInterpolator((x, y), u, bounds_error=False, fill_value=None)
    fv = RegularGridInterpolator((x, y), v, bounds_error=False, fill_value=None)
    u_c = fu(np.stack([np.full_like(y_query, 0.5), y_query], -1))
    v_c = fv(np.stack([x_query, np.full_like(x_query, 0.5)], -1))
    return {"u_centerline": u_c, "v_centerline": v_c}


# --------------------------------------------------------------------------------------
# FEATFLOW DFG reference series
# --------------------------------------------------------------------------------------
def _read_numeric_table(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#%!":
            continue
        try:
            rows.append([float(v) for v in re.split(r"[\s,;]+", s)])
        except ValueError:
            continue
    width = max(len(r) for r in rows)
    return np.array([r for r in rows if len(r) == width])


def list_featflow_files(variant: str = "2D-2") -> Dict[str, list]:
    tag = "dfg2d2" if variant == "2D-2" else "dfg2d3"
    out = {}
    for d in sorted((DATA_DIR / "dfg_benchmark").glob(f"{tag}_*")):
        if d.is_dir():
            out[d.name] = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    return out


def load_featflow_series(kind: str = "draglift", variant: str = "2D-2", level: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load a FEATFLOW reference time series (finest level / smallest dt by default).

    File layouts (from the headers):
        bdforces_*    : ``timestep time bdc horiz vert``          -> t = col 1, Cd = col 3, Cl = col 4
        pointvalues_* : ``timestep time x y type deriv value ...`` -> p(a1) = col 6, p(a2) = col 11
    Returns t, Cd, Cl for ``kind="draglift"`` and t, p_front, p_rear, dP for ``kind="pressure"``.
    """
    tag = "dfg2d2" if variant == "2D-2" else "dfg2d3"
    dirs = [d for d in (DATA_DIR / "dfg_benchmark").glob(f"{tag}_{kind}_*") if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no FEATFLOW {kind} directory for {variant} under {DATA_DIR / 'dfg_benchmark'}")
    prefix = "bdforces" if kind == "draglift" else "pointvalues"
    files = sorted(p for p in dirs[0].rglob(f"{prefix}*") if p.is_file())
    if level is not None:
        files = [f for f in files if level in f.name]
    if not files:
        raise FileNotFoundError("no matching files")
    f = files[-1]
    table = _read_numeric_table(f.read_text(errors="ignore"))
    out = {"file": str(f), "columns": table, "t": table[:, 1]}
    if kind == "draglift":
        out.update({"Cd": table[:, 3], "Cl": table[:, 4]})
    else:
        out.update({"p_front": table[:, 6], "p_rear": table[:, 11], "dP": table[:, 6] - table[:, 11]})
    return out


# --------------------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------------------
def load_hiocfd_reference(path: Path) -> np.ndarray:
    """Read a whitespace/comma separated reference table (t, value...) from the HiOCFD downloads."""
    return _read_numeric_table(Path(path).read_text(errors="ignore"))
