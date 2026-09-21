import numpy as np
import pytest

from pinnflow import data as D
from pinnflow.benchmarks import DFGCylinder


def test_cylinder_wake_dataset():
    if not (D.DATA_DIR / "cylinder_wake" / "cylinder_nektar_wake.mat").exists():
        pytest.skip("cylinder wake data not downloaded")
    d = D.load_cylinder_wake()
    assert d["X_star"].shape == (5000, 2) and d["U_star"].shape == (5000, 2, 200) and d["p_star"].shape == (5000, 200)
    assert d["flat"]["u"].shape == (5000 * 200,)
    sp = D.sample_sparse(d["flat"], 100, seed=1)
    assert sp["t"].shape == (100,)


def test_featflow_reference_series_match_published_intervals():
    try:
        dl = D.load_featflow_series("draglift", "2D-2")
        pr = D.load_featflow_series("pressure", "2D-2")
    except FileNotFoundError:
        pytest.skip("FEATFLOW reference not downloaded")
    b = DFGCylinder("2D-2")
    n = len(dl["t"])
    cd_max, cl_max = dl["Cd"][n // 2 :].max(), dl["Cl"][n // 2 :].max()
    lo, hi = b.reference["2D-2/Cd_max"]
    assert lo - 0.01 <= cd_max <= hi + 0.01, cd_max
    lo, hi = b.reference["2D-2/Cl_max"]
    assert lo - 0.02 <= cl_max <= hi + 0.02, cl_max
    assert pr["dP"].shape == pr["t"].shape and 2.3 < pr["dP"][n // 2 :].max() < 2.6


def test_featflow_2d3_series():
    try:
        dl = D.load_featflow_series("draglift", "2D-3")
    except FileNotFoundError:
        pytest.skip("FEATFLOW 2D-3 reference not downloaded")
    b = DFGCylinder("2D-3")
    lo, hi = b.reference["2D-3/Cd_max"]
    assert lo - 0.02 <= dl["Cd"].max() <= hi + 0.02
    assert abs(dl["t"][np.argmax(dl["Cd"])] - 3.93) < 0.1


def test_causal_ns_and_jaxpi_cavity():
    if not (D.EXTERNAL_DIR / "CausalPINNs" / "data" / "NS.npy").exists():
        pytest.skip("CausalPINNs not cloned")
    d = D.load_causal_ns()
    assert d["sol"].shape == (200, 128, 128) and abs(d["viscosity"] - 0.01) < 1e-12
    ref = D.load_jaxpi_cavity(1000)
    assert ref["u"].shape == (128, 128) and abs(ref["nu"] - 1e-3) < 1e-12
