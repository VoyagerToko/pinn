"""LaTeX rows for the paper's ablation and cost tables, read from runs/*/eval.json, cost.json and metrics.csv.

    python scripts/paper_tables.py ablation     # -> paper/sections/table_ablation.tex
    python scripts/paper_tables.py cost         # -> paper/sections/table_cost.tex

Steps-to-target (handbook STEP 8) is the first evaluation step at which the target is met:
Benchmark A: velocity relative L2 < 1e-2; Benchmark B: velocity relative L2 against the FD field at
Re = 1000 < 5% (evaluated only in the Re = 1000 stage). "--" = never reached within the budget.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "sections"


def load(run, name="eval.json"):
    p = RUNS / run / name
    return json.loads(p.read_text()) if p.exists() else {}


def sci(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    e = int(math.floor(math.log10(abs(x)))) if x else 0
    m = x / 10**e
    return f"${m:.{digits - 1}f}\\times10^{{{e}}}$"


def pct(x, digits=2):
    return "--" if x is None else f"{100 * x:.{digits}g}\\%"


def steps_to(run, col, target, files=None):
    files = files or [RUNS / run / "metrics.csv"]
    for f in files:
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            v = r.get(col)
            if v not in (None, "") and float(v) < target:
                return int(float(r["step"]))
    return None


def ablation():
    lines = []
    for row in "ABCDEFG":
        a_run = f"tgv2d_abl_{row}"
        b_run = f"cavity_abl_{'D' if row == 'E' else row}"
        ea, eb = load(a_run), load(b_run)
        a_err = ea.get("rel_l2_vel")
        a_steps = steps_to(a_run, "eval/rel_l2_vel", 1e-2)
        b_fd = eb.get("final_Re1000/rel_l2_vel_vs_fd")
        b_u, b_v = eb.get("final_Re1000/ghia_u_rel_err"), eb.get("final_Re1000/ghia_v_rel_err")
        b_steps = steps_to(b_run, "eval/rel_l2_vel_vs_fd", 0.05)
        note = " (= D)" if row == "E" else ""
        lines.append(
            f"{row}{note} & {sci(a_err)} & {a_steps if a_steps is not None else '--'} & "
            f"{pct(b_fd)} & {pct(b_u)} / {pct(b_v)} & {b_steps if b_steps is not None else '--'}\\\\"
        )
    (OUT / "table_ablation_rows.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


COST_RUNS = [
    ("A row G", "tgv2d_G"),
    ("B row C", "cavity_C_r2"),
    ("B row D, soft lid", "cavity_D_softlid"),
    ("B row F", "cavity_F_r2"),
    ("C row G (reduced)", "cylinder_G"),
    ("D row H (reduced)", "tgv3d_H"),
    ("inverse (reduced)", "cylinder_inverse"),
    ("PI-DeepONet (reduced)", "deeponet_cavity"),
]


def cost():
    lines = []
    for label, run in COST_RUNS:
        e, c = load(run), load(run, "cost.json")
        wall = c.get("train_wall_s")
        wall_s = "--" if wall is None else (f"{wall / 3600:.2f} h" if wall >= 3600 else f"{wall / 60:.0f} min")
        st = e.get("cost/step_time_s")
        mem = e.get("cost/jax_peak_gb")
        thr = e.get("cost/inference_pts_per_s")
        lines.append(f"{label} & {wall_s} & {('%.3f' % st) if st else '--'} & {('%.2f' % mem) if mem else '--'} & {sci(thr) if thr else '--'}\\\\")
    (OUT / "table_cost_rows.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ablation", "all"):
        ablation()
    if what in ("cost", "all"):
        cost()
