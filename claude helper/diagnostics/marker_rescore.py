"""Re-read trained phantom networks at the catheter's own locations, found from the gel markers (CPU, read-only).

The P1 cases' catheter drop changes a lot between the 5 descending-aorta (DAo) locations (P1 baseline 11.7 -> 23.2
mmHg), and the networks were scored at the 2D-PC outlet plane, not at a catheter marker. In the 3D GRE anatomy scan the
markers are dark bars in the gel (clear in slices ~62-66 of the P1 scans): 5 stacked beside the DAo, 1 beside the
ascending aorta (AAo). This finds each bar's row from the dark fraction in a column band (head-feet profile, peaks),
converts it to a patient (LPS) height, and reads the network's pressure as the cross-section mean of the lumen voxels of
the right limb at that height (descending: flows feet-wards in systole; ascending: head-wards). DAo locations are
numbered from the top (proximal) down. Reported: peak and mean drop at AAo marker -> DAo location 3 (and the other four),
next to the catheter at each location and the scores at the original planes.

    python "claude helper/diagnostics/marker_rescore.py" P1_baseline coa_P1_baseline_lam_60k coa_P1_baseline_accel_60k
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pinnflow  # noqa: E402

pinnflow.configure_jax()
import jax  # noqa: E402
import ml_collections  # noqa: E402

from pinnflow.coa_phantom import catheter_pressure_drop, load_4dflow  # noqa: E402
from pinnflow.problems import PROBLEMS  # noqa: E402
from pinnflow.utils import load_params  # noqa: E402

case, runs = sys.argv[1], sys.argv[2:]
# per geometry: the GRE slices where the markers show, the column bands beside the descending / ascending limb, and the
# row range below the arch (above it the bands cross the vessels' own edges)
CFG = {"P1": dict(slab=(62, 67), dao_cols=(246, 270), aao_cols=(112, 140), rows=(160, 300)),
       "P2_baseline": dict(slab=(54, 59), dao_cols=(255, 282), aao_cols=(65, 95), rows=(130, 300)),
       "P2_baseline_lowflow": dict(slab=(61, 66), dao_cols=(255, 282), aao_cols=(68, 98), rows=(130, 300)),
       "P2_severe": dict(slab=(45, 51), dao_cols=(258, 285), aao_cols=(60, 100), rows=(120, 300)),
       "P2_severe_lowflow": dict(slab=(54, 59), dao_cols=(255, 282), aao_cols=(55, 90), rows=(110, 300))}
cfg_c = CFG.get(case, CFG["P1"])
SLAB, DAO_COLS, AAO_COLS, ROWS = cfg_c["slab"], cfg_c["dao_cols"], cfg_c["aao_cols"], cfg_c["rows"]
OVERLAY = next((a.split("=", 1)[1] for a in runs if a.startswith("--overlay=")), None)
runs = [a for a in runs if not a.startswith("--")]

# ---- markers in the GRE scan
f = [p for p in sorted((ROOT / "data/stanford_coa/Imaging" / case / "3D_GRE_highRes").iterdir())
     if "ORIGINAL" in list(pydicom.dcmread(p, stop_before_pixels=True).ImageType)][0]
ds = pydicom.dcmread(f)
v = ds.pixel_array.astype(np.float32)
pf = ds.PerFrameFunctionalGroupsSequence
sh = ds.SharedFunctionalGroupsSequence[0]
pos = np.array([np.array(p.PlanePositionSequence[0].ImagePositionPatient, float) for p in pf])
ori = np.array((pf[0].get("PlaneOrientationSequence") or sh.PlaneOrientationSequence)[0].ImageOrientationPatient, float)
sp = np.array((pf[0].get("PixelMeasuresSequence") or sh.PixelMeasuresSequence)[0].PixelSpacing, float)
cdir = ori[3:]
slab = v[SLAB[0]:SLAB[1]].min(0)
k_mid = 0.5 * (SLAB[0] + SLAB[1] - 1)


def row_to_z(r):
    p0 = pos[int(np.floor(k_mid))]
    return float((p0 + r * sp[0] * cdir)[2])


def bar_rows(cols, n):
    prof = (slab[:, cols[0]:cols[1]] < 80).mean(1)
    prof[: ROWS[0]] = 0
    prof[ROWS[1]:] = 0
    pk, props = find_peaks(prof, distance=10, prominence=0.15)
    best = pk[np.argsort(props["prominences"])[::-1][:n]]
    return np.sort(best), prof


# rows checked by eye on the slab image (1 slanted bar at the top, then 4 straight ones); the automatic peak search
# misses the slanted bar and can take an air bubble, so these override it (location 3 is the middle bar either way)
CHECKED_DAO_ROWS = {"P1_baseline": [176, 209, 224, 244, 260], "P1_baseline_lowflow": [161, 192, 210, 227, 241],
                    "P1_severe": [150, 173, 193, 210, 227], "P1_severe_lowflow": [150, 183, 198, 218, 234]}
dao_rows, dao_prof = bar_rows(DAO_COLS, 5)
if case in CHECKED_DAO_ROWS:
    print(f"  automatic DAo rows {list(dao_rows)} replaced by the checked rows {CHECKED_DAO_ROWS[case]}")
    dao_rows = np.array(CHECKED_DAO_ROWS[case])
aao_rows, aao_prof = bar_rows(AAO_COLS, 1)
CHECKED_AAO_ROWS = {"P2_baseline": 143, "P2_severe_lowflow": 130}  # checked by eye (the band picks other dark spots)
if case in CHECKED_AAO_ROWS:
    print(f"  automatic AAo row {list(aao_rows)} replaced by the checked row {CHECKED_AAO_ROWS[case]}")
    aao_rows = np.array([CHECKED_AAO_ROWS[case]])
if OVERLAY:  # the detections drawn on the slab, for a check by eye
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 9), dpi=90)
    ax.imshow(slab, cmap="gray", vmin=0, vmax=170)
    for r in dao_rows:
        ax.plot(DAO_COLS, [r, r], "r-", lw=2)
    for r in aao_rows:
        ax.plot(AAO_COLS, [r, r], "y-", lw=2)
    ax.set_xlim(40, 320)
    ax.set_ylim(320, 90)
    ax.set_title(f"{case}: DAo {list(map(int, dao_rows))}, AAo {list(map(int, aao_rows))}")
    fig.savefig(OVERLAY)
dao_z = [row_to_z(r) for r in dao_rows]
aao_z = row_to_z(aao_rows[0])
print(f"{case}: GRE markers (rows in slices {SLAB[0]}-{SLAB[1] - 1}): AAo row {aao_rows[0]} (z {aao_z:.1f} mm); "
      f"DAo rows {list(dao_rows)} (z " + ", ".join(f"{z:.1f}" for z in dao_z) + " mm, top -> bottom = locations 1 -> 5)")

# ---- the scan's lumen voxels in patient coordinates, systolic flow direction along z
d = load_4dflow(case)
prep = np.load(ROOT / "data/stanford_coa/prepared" / f"{case}.npz")
E = np.stack([d["row_dir"], d["col_dir"], d["normal"]])
origin = d["xyz_mm"][0, 0, 0]
pos_mm = prep["pos_mm"].astype(float)
lps_vox = origin + pos_mm @ E
vel = prep["vel"].astype(float)  # (T, N, 3) local
sys_f = np.argsort(np.linalg.norm(vel, axis=2).mean(1))[-4:]
vz = (vel[sys_f].mean(0) @ E)[:, 2]  # + head-wards
dz = float(np.max(np.abs(E[:, 2]) * prep["voxel_mm"]))  # the voxel spacing along z (the axis closest to head-feet)


def section(z, ascending):
    near = np.abs(lps_vox[:, 2] - z) <= 0.6 * dz
    cells = np.nonzero(near & ((vz > 0.05) if ascending else (vz < -0.05)))[0]
    return pos_mm[cells]


up_m = section(aao_z, True)
dn_m = [section(z, False) for z in dao_z]
print(f"  cells: AAo marker {len(up_m)}, DAo markers " + ", ".join(str(len(c)) for c in dn_m)
      + f"; original planes: AAo {len(prep['aao_mm'])} cells at z {np.mean((origin + prep['aao_mm'] @ E)[:, 2]):.1f}, "
      f"DAo {len(prep['dao_mm'])} cells at z {np.mean((origin + prep['dao_mm'] @ E)[:, 2]):.1f}")
cath = [catheter_pressure_drop(case, k)["dp_peak_mmHg"] for k in range(1, 6)]
print("  catheter peak at locations 1-5: " + " ".join(f"{c:.1f}" for c in cath))

for run in runs:
    wd = ROOT / "runs" / run
    cfg = ml_collections.ConfigDict(json.loads((wd / "config.json").read_text()))
    with cfg.unlocked():
        cfg.problem.accel_data = False
    pr = PROBLEMS["coa_mri"](cfg)
    params, _ = load_params(wd / "latest.msgpack", pr.init_params(jax.random.PRNGKey(0)))
    times = np.linspace(0.0, pr.T, 64, endpoint=False)

    def plane_p(cells_mm):
        T_, P_ = np.meshgrid(times.astype(np.float32), np.arange(len(cells_mm)), indexing="ij")
        pts = np.concatenate([T_.reshape(-1, 1), (cells_mm[P_.ravel()] * 0.1).astype(np.float32)], 1)
        return pr.predict(params, pts)[:, 3].reshape(len(times), len(cells_mm)).mean(1)

    k = pr.mmHg_per_unit
    p_up_m, p_up_o = plane_p(up_m), plane_p(prep["aao_mm"].astype(float))
    p_dn_o = plane_p(prep["dao_mm"].astype(float))
    orig = (p_up_o - p_dn_o) * k
    print(f"  {run}: original planes peak {orig.max():.2f} (error {100 * abs(orig.max() - cath[2]) / cath[2]:.0f}%)")
    for i, cells in enumerate(dn_m):
        dp = (p_up_m - plane_p(cells)) * k
        tag = "  <- location 3" if i == 2 else ""
        print(f"     AAo marker -> DAo location {i + 1}: peak {dp.max():6.2f}, mean {dp.mean():+5.2f} mmHg; catheter {cath[i]:.1f} "
              f"(error {100 * abs(dp.max() - cath[i]) / cath[i]:.0f}%){tag}")
