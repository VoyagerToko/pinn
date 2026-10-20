"""Create the real-world use-case dataset: CFD truth for a pulsatile flow through a stenosed artery, and the
4D-flow-MRI-like velocity data a scan of it would give. Writes data/stenosis/{truth.npz, mri.npz, vessel.json}.

    python scripts/make_stenosis_data.py                    # defaults: 60% stenosis, Re 150, 0.6 mm voxels, SNR 20
    python scripts/make_stenosis_data.py --snr 10 --out data/stenosis_snr10
    python scripts/make_stenosis_data.py --voxel 0.2        # 1.2 mm voxels (must be a multiple of 1/N)
    python scripts/make_stenosis_data.py --Re 300 --severity 0.7 --bpm 73 --N 60 --length 18 --fov 0.5 11.5 --out data/stenosis_fast
                                                            # the Doppler regime: a ~1.6 m/s jet, ~10 mmHg

About 8 minutes on a laptop CPU at the default grid (N = 50 cells per diameter, three cardiac cycles; the last is
kept). The training problem reads the directory given by problem.data_dir (default data/stenosis).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pinnflow.vessel import (  # noqa: E402
    StenosisVessel, bernoulli_pressure_drop, mri_from_cfd, mri_samples, periodicity_error, save_dataset, solve_cfd, truth_pressure_drop,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "stenosis"))
    ap.add_argument("--severity", type=float, default=0.6, help="diameter reduction at the throat")
    ap.add_argument("--Re", type=float, default=150.0)
    ap.add_argument("--N", type=int, default=50, help="CFD cells per diameter (the voxel must be a multiple of 1/N)")
    ap.add_argument("--bpm", type=float, default=None, help="heart rate; sets the non-dimensional period from the physical scale (default: period 12)")
    ap.add_argument("--length", type=float, default=14.0, help="vessel length in diameters (outlet far downstream of the field of view)")
    ap.add_argument("--fov", type=float, nargs=2, default=(0.5, 9.5), help="scanned field of view along the vessel, in diameters")
    ap.add_argument("--periods", type=int, default=3, help="cardiac cycles from rest; the last one is kept")
    ap.add_argument("--voxel", type=float, default=0.1, help="MRI voxel size in diameters (0.1 = 0.6 mm)")
    ap.add_argument("--phases", type=int, default=24, help="MRI cardiac phases per beat")
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v = StenosisVessel(severity=args.severity, Re=args.Re, length=args.length, fov=tuple(args.fov))
    if args.bpm:  # the same heart rate at any Re: period = (60 / bpm) * U / D
        U = v.Re * v.nu_m2s / v.D_m
        v = StenosisVessel(severity=args.severity, Re=args.Re, length=args.length, fov=tuple(args.fov), period=60.0 / args.bpm * U / v.D_m)
    phys = v.physical()
    print(f"[stenosis] vessel D = {phys['D_mm']:.1f} mm, mean inflow {phys['U_ms']:.3f} m/s, Re {v.Re:g}, "
          f"beat {phys['period_s']:.2f} s ({phys['heart_rate_bpm']:.0f} bpm), {100 * v.severity:.0f}% stenosis")
    t0 = time.time()
    truth = solve_cfd(v, N=args.N, n_periods=args.periods, frames=4 * args.phases)
    per = periodicity_error(truth)
    print(f"[stenosis] CFD done in {time.time() - t0:.0f} s; cycle-to-cycle change of the pressure drop {100 * per:.2f}%")
    mri = mri_from_cfd(truth, v, voxel=args.voxel, phases=args.phases, snr=args.snr, seed=args.seed)
    smp = mri_samples(mri)
    k = phys["mmHg_per_unit"]
    dp = truth_pressure_drop(truth, v) * k
    dpb = bernoulli_pressure_drop(mri, v)
    print(f"[stenosis] MRI: {mri['lumen'].sum()} lumen voxels x {args.phases} phases = {len(smp['t'])} samples, "
          f"VENC {mri['venc'] * phys['U_ms']:.2f} m/s, noise sigma {mri['sigma'] * phys['U_ms'] * 100:.1f} cm/s")
    print(f"[stenosis] peak pressure drop (CFD truth) {dp.max():.2f} mmHg; simplified Bernoulli from the scan {dpb.max():.2f} mmHg")
    meta = {"cfd": {"N": args.N, "periods": args.periods, "periodicity_error": per, "seconds": time.time() - t0},
            "mri": {"voxel": args.voxel, "phases": args.phases, "snr": args.snr, "seed": args.seed, "n_samples": int(len(smp["t"]))},
            "summary": {"dp_peak_truth_mmHg": float(dp.max()), "dp_peak_bernoulli_mmHg": float(dpb.max())}}
    save_dataset(Path(args.out), v, truth, mri, meta)
    print(f"[stenosis] wrote {args.out}")


if __name__ == "__main__":
    main()
