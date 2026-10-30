"""How does a trained phantom network's peak drop depend on where the downstream plane sits? (CPU, read-only)

The catheter recorded the descending-aorta pressure at 5 slice locations; on the P1 geometry its peak drop changes a lot
between them (P1 baseline 11.7 to 23.2 mmHg), on P2 hardly at all. The scan's downstream plane is the 2D-PC outlet plane,
not located at the catheter's marker. This reads a trained network's peak drop against every head-feet slice of the
descending limb (the lumen voxels of one image row, in the connected piece that flows feet-wards in systole), from just
below the arch to the bottom of the scan, next to the catheter's values at its 5 locations.

    python "claude helper/diagnostics/coa_dao_sweep.py" coa_P1_baseline_lam_60k coa_P1_baseline_accel_60k
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pinnflow  # noqa: E402

pinnflow.configure_jax()
import jax  # noqa: E402
import ml_collections  # noqa: E402
from scipy import ndimage  # noqa: E402

from pinnflow.coa_phantom import catheter_pressure_drop  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402

for run in sys.argv[1:]:
    wd = ROOT / "runs" / run
    cfg = ml_collections.ConfigDict(json.loads((wd / "config.json").read_text()))
    with cfg.unlocked():
        cfg.problem.accel_data = False  # evaluation only
    pr = PROBLEMS["coa_mri"](cfg)
    params, _ = load_params(wd / "latest.msgpack", pr.init_params(jax.random.PRNGKey(0)))
    d = pr.d
    pos_mm = np.asarray(d["pos_mm"], float)
    vox = np.asarray(d["voxel_mm"], float)
    ijk = np.round(pos_mm / vox).astype(int)  # local (a, b, c) indices; b runs along the image rows (head -> feet)
    vel = np.asarray(d["vel"])  # (T, N, 3), component 1 along b
    sys_frames = np.argsort(np.linalg.norm(vel, axis=2).mean(1))[-4:]
    vb = vel[sys_frames, :, 1].mean(0)  # systolic velocity along b (+ = feet-wards)
    times = np.linspace(0.0, pr.T, 48, endpoint=False)

    def plane_p(cells_mm):
        T_, P_ = np.meshgrid(times.astype(np.float32), np.arange(len(cells_mm)), indexing="ij")
        pts = np.concatenate([T_.reshape(-1, 1), (cells_mm[P_.ravel()] * 0.1).astype(np.float32)], 1)
        return pr.predict(params, pts)[:, 3].reshape(len(times), len(cells_mm)).mean(1)

    p_up = plane_p(np.asarray(d["aao_mm"], float))
    b_dao = float(np.asarray(d["dao_mm"])[:, 1].mean() / vox[1])
    rows = []
    shape = ijk.max(0) + 1
    for b in range(ijk[:, 1].min(), ijk[:, 1].max() + 1):
        sel = ijk[:, 1] == b
        if sel.sum() < 15:
            continue
        sl = np.zeros((shape[0], shape[2]), int)
        sl[ijk[sel, 0], ijk[sel, 2]] = np.nonzero(sel)[0] + 1
        lab, n = ndimage.label(sl > 0)
        best, best_v = None, 0.0
        for L in range(1, n + 1):
            idx = sl[lab == L] - 1
            if len(idx) < 10:
                continue
            v = float(vb[idx].mean())
            if v > best_v:  # the feet-wards (descending) piece
                best, best_v = idx, v
        if best is None:
            continue
        dp = (p_up - plane_p(pos_mm[best])) * pr.mmHg_per_unit
        rows.append((b, float(dp.max()), float(dp.mean()), len(best), best_v))
    cath = []
    for k in range(1, 6):
        try:
            cath.append(catheter_pressure_drop(pr.case, k)["dp_peak_mmHg"])
        except Exception:  # noqa: BLE001
            cath.append(np.nan)
    print(f"== {run} ({pr.case}): catheter peak at locations 1-5: " + " ".join(f"{c:.1f}" for c in cath)
          + f"; scored plane (2D-PC outlet) at row b = {b_dao:.0f}, network peak there {json.loads((wd / 'final_eval.json').read_text())['dp_peak_pinn_mmHg']:.2f}")
    print("  row b  (mm down)   network peak / mean drop (mmHg)   cells   systolic v (m/s)")
    for b, pk, mn, nc, v in rows[:: max(1, len(rows) // 25)]:
        mark = "  <- scored plane" if abs(b - b_dao) < 1 else ""
        print(f"  {b:4d}   {b * vox[1]:6.0f}         {pk:6.2f} / {mn:+5.2f}               {nc:4d}    {v:.2f}{mark}")
    pk = np.array([r[1] for r in rows])
    print(f"  network peak drop over the descending limb: {pk.min():.1f} to {pk.max():.1f} mmHg; catheter over its 5 locations: "
          f"{np.nanmin(cath):.1f} to {np.nanmax(cath):.1f}")
