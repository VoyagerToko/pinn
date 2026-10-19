"""The seven scenes: what each one shows, how its pigment is laid down, and which evidence sits on the rack.

Verdict numbers are the measured values from STATUS.md (product truth); the viewer never recomputes them.
Live, per-frame numbers are computed separately and shown as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from viewer import theme

CAPACITY = 4096  # particles per scene: fixed, so the server compiles one velocity kernel per scene


@dataclass
class SceneSpec:
    sid: str
    name: str
    short: str
    verdict: str            # passed | failed | gate missed
    headline: str           # the measured number, fixed precision
    headline_label: str     # what that number is
    views: List[str]
    field_n: int
    dye_rate: float         # simulated time units the pigment advances per wall-clock second
    play_rate: float = 0.0  # flow time per wall-clock second while playing (0: steady scene)
    coupled: bool = False   # True: playing moves the flow time and the pigment together (true pathlines)
    layout: str = "none"    # initial pigment: vbands | hbands | slabs | none
    emitter: Optional[Dict] = None
    periodic: bool = False
    re_mode: Optional[str] = None  # discrete | continuous
    chart: str = "none"
    evidence: Optional[str] = None  # the view that shows why a failure failed
    dims: int = 2
    start_at_end: bool = False  # open at the end of the time window (flows that start from rest)
    tab: str = "bench"          # bench: the seven benchmarks | real: the real-world use case
    trained: bool = True        # False until the scene's network exists (the server says so)

    @property
    def pigment(self) -> str:
        return theme.PIGMENTS[theme.SCENE_PIGMENT[self.sid]]


SCENES: List[SceneSpec] = [
    SceneSpec("tgv2d", "Taylor-Green vortex", "Taylor-Green 2D", "passed", "4.0e-5", "velocity error against the exact solution; gate below 1e-4",
              ["pigment", "speed", "vorticity", "pressure", "error"], 128, dye_rate=0.8, play_rate=0.1, layout="vbands", periodic=True, chart="error_time"),
    SceneSpec("cavity_ok", "Cavity, soft lid", "Cavity, soft lid", "gate missed", "0.27 %", "field error against converged finite differences; Ghia v 2.05 %, gate 2 %",
              ["pigment", "speed", "vorticity", "divergence", "error"], 128, dye_rate=0.35, layout="hbands", re_mode="discrete", chart="centreline"),
    SceneSpec("cavity_hard", "Cavity, hard lid", "Cavity, hard lid", "failed", "19.8 %", "Ghia centreline error at Re 1000; gate below 2 %",
              ["pigment", "speed", "vorticity", "divergence", "error"], 128, dye_rate=0.35, layout="hbands", re_mode="discrete", chart="centreline", evidence="divergence"),
    SceneSpec("inverse", "Wake, from sparse data", "Wake from data", "passed", "0.065 %", "viscosity error, learned from 5,000 velocity samples; hidden pressure 2.4 %",
              ["pigment", "sensors", "speed", "vorticity", "pressure", "error"], 128, dye_rate=1.0, play_rate=1.0, coupled=True,
              emitter={"x": 1.08, "rows": 16, "per_step": 16}, chart="lambdas"),
    SceneSpec("cylinder_fwd", "Cylinder, forward solve", "Cylinder, forward", "failed", "1.54", "peak drag coefficient, FEATFLOW gives 3.22; no shedding",
              ["pigment", "speed", "vorticity", "pressure"], 256, dye_rate=0.12, play_rate=0.1,
              emitter={"x": 0.02, "rows": 8, "per_step": 8}, chart="flowrate", evidence="flowrate", start_at_end=True),
    SceneSpec("tgv3d", "Taylor-Green vortex, 3D", "Taylor-Green 3D", "failed", "t = 0", "where dissipation peaks; the DNS peaks at t = 9",
              ["pigment", "speed"], 48, dye_rate=1.0, play_rate=1.0, coupled=True, layout="slabs", periodic=True, chart="dissipation", evidence="dissipation", dims=3),
    SceneSpec("deeponet", "Cavity, any lid speed", "Cavity, any lid", "passed", "9.65 %", "error at Re 400, a lid speed it never saw; gate below 10 %",
              ["pigment", "speed", "vorticity", "error"], 128, dye_rate=0.35, layout="hbands", re_mode="continuous", chart="centreline_live"),
    # the real-world use case: verdict and number are filled in from the server once the network is trained
    SceneSpec("stenosis", "Blood pressure from an MRI scan", "Stenosis, from MRI", "not trained", "-",
              "peak pressure-drop error against the CFD truth; gate below 10 %",
              ["pigment", "mri", "speed", "pressure", "error"], 360, dye_rate=1.0, play_rate=1.5, coupled=True,
              emitter={"x": 0.55, "rows": 24, "per_step": 24}, chart="pressure_drop", evidence="pressure", tab="real"),
]
TABS = [("bench", "The seven benchmarks"), ("real", "Real-world use case")]


def scenes_in(tab: str) -> List[SceneSpec]:
    return [s for s in SCENES if s.tab == tab]
BY_ID = {s.sid: s for s in SCENES}


def band_pigments(spec: SceneSpec) -> List[str]:
    """Pigments laid in alternating bands when a scene opens: the scene's own pigment and ink."""
    return [spec.pigment, theme.INK]


def initial_dye(spec: SceneSpec, domain: List[float], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions (N, d), pigment index (N,), active (N,)) for the opening of a scene."""
    n, d = CAPACITY, spec.dims
    lo = np.asarray(domain[0::2], np.float32)
    hi = np.asarray(domain[1::2], np.float32)
    pos = lo + rng.random((n, d), dtype=np.float32) * (hi - lo)
    pig = np.zeros(n, np.int16)
    active = np.ones(n, bool)
    bands = 10
    if spec.layout == "vbands":
        k = np.floor((pos[:, 0] - lo[0]) / (hi[0] - lo[0]) * bands).astype(int)
        pig = (k % 2).astype(np.int16)
    elif spec.layout == "hbands":
        k = np.floor((pos[:, 1] - lo[1]) / (hi[1] - lo[1]) * bands).astype(int)
        pig = (k % 2).astype(np.int16)
    elif spec.layout == "slabs":
        # three thin sheets of pigment across the box, alternating colours
        z = (np.arange(n) % 3 + 0.5) / 3 * (hi[2] - lo[2]) + lo[2]
        pos[:, 2] = z + rng.normal(0, 0.06, n).astype(np.float32)
        pig = ((np.arange(n) % 3) % 2).astype(np.int16)
    else:
        active[:] = False
    return pos, pig, active


def rake(spec: SceneSpec, domain: List[float], step: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """New pigment released by the dye rake upstream: positions (k, 2) and pigment index (k,)."""
    e = spec.emitter
    rows = e["rows"]
    y0, y1 = domain[2], domain[3]
    pad = 0.04 * (y1 - y0)
    y = np.linspace(y0 + pad, y1 - pad, rows) + rng.normal(0, 0.002 * (y1 - y0), rows)
    x = np.full(rows, e["x"]) + rng.random(rows) * 0.004 * (domain[1] - domain[0])
    pig = ((np.arange(rows) // 2) % 2).astype(np.int16)  # pairs of rows alternate pigment
    return np.stack([x, y], -1).astype(np.float32), pig
