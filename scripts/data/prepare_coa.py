"""Prepare cases of the public 4D-FlowP coarctation phantom for training (see pinnflow/coa_phantom.py).

    python scripts/data/prepare_coa.py --case P2_baseline          # -> data/stanford_coa/prepared/P2_baseline.npz (+ _qa.png)
    python scripts/data/prepare_coa.py --all

For each case it also writes a QA image (the lumen mask and the two catheter planes, the time-mean speed) and checks the
scan against the phantom's flow meters: the 4D-flow flux through the upstream plane over the beat against the "Total
flow" meter. A segmentation that misses the vessel, or a sign error, shows up as a mismatch here.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pinnflow.coa_phantom import CASES, COA_DIR, PREP_VERSION, beat_average, load_4dflow, load_catheter, prepare_case, segment_lumen  # noqa: E402


def qa(case: str, out_png: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_4dflow(case)
    mask = segment_lumen(d)
    prep = dict(np.load(COA_DIR / "prepared" / f"{case}.npz"))
    sp = np.linalg.norm(d["velocity"], axis=-1)
    i_a = int(round(prep["aao_mm"][:, 1].mean() / d["voxel_mm"][0]))
    i_d = int(round(prep["dao_mm"][:, 1].mean() / d["voxel_mm"][0]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mask.max(0), cmap="gray")
    ax[0].axhline(i_a, color="r", lw=1)
    ax[0].axhline(i_d, color="c", lw=1)
    ax[0].set_title(f"{case}: lumen mask (MIP); red AAo plane, cyan DAo plane")
    ax[1].imshow(np.where(mask[None], sp, 0).mean(0).max(0), cmap="viridis")
    ax[1].set_title("time-mean speed in the lumen (MIP)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=70)
    plt.close(fig)
    # flux through the AAo plane cells (local velocity component along the image rows, b) vs the total-flow meter
    vel = prep["vel"]  # (F, N, 3)
    pos = prep["pos_mm"]
    a_cells = {tuple(np.round(p, 3)) for p in prep["aao_mm"]}
    sel = np.array([tuple(np.round(p, 3)) in a_cells for p in pos])
    # flux through the slab of voxels within half a voxel of the plane: sum(v . n) * voxel volume / voxel size
    vox = np.asarray(d["voxel_mm"]) * 1e-3
    q_mri = (vel[:, sel, :] @ prep["aao_normal"]).sum(1) * np.prod(vox) / vox.mean() * 1e6  # mL/s, head-wards positive
    rec = load_catheter(case)["locations"][3]
    t, q = rec["Total flow"]
    _, qm, _, _, _ = beat_average(t, q, q)
    q_meter = qm * 1000 / 60
    return {"aao_flux_mean_mri_mLs": float(q_mri.mean()), "aao_flux_peak_mri_mLs": float(q_mri.max()),
            "total_flow_mean_meter_mLs": float(q_meter.mean()), "total_flow_peak_meter_mLs": float(q_meter.max()),
            "lumen_mL": float(mask.sum() * np.prod(d["voxel_mm"]) / 1000)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="P2_baseline")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-prepare cases that already have an npz")
    args = ap.parse_args()
    out_dir = COA_DIR / "prepared"
    for case in (CASES if args.all else [args.case]):
        npz = out_dir / f"{case}.npz"
        try:
            stale = npz.exists() and int(np.load(npz).get("prep_version", 0)) < PREP_VERSION  # an older pipeline wrote it
            if args.force or stale or not npz.exists():
                info = prepare_case(case, npz)
                print(case, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items()}, flush=True)
            check = qa(case, out_dir / f"{case}_qa.png")
            (out_dir / f"{case}_qa.json").write_text(json.dumps(check, indent=2))
            ratio = check["aao_flux_mean_mri_mLs"] / max(check["total_flow_mean_meter_mLs"], 1e-9)
            flag = "OK" if 0.8 < ratio < 1.2 else "CHECK THE SEGMENTATION"
            print(f"  QA {case}: AAo flux mean {check['aao_flux_mean_mri_mLs']:.1f} mL/s vs meter {check['total_flow_mean_meter_mLs']:.1f} "
                  f"(ratio {ratio:.2f}, {flag}); lumen {check['lumen_mL']:.0f} mL", flush=True)
        except FileNotFoundError as e:
            print(case, "skipped:", e, flush=True)


if __name__ == "__main__":
    main()
