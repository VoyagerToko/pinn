"""The real-data comparison: peak trans-coarctation pressure drop from the catheter, against the physics-informed networks
(laminar and with an eddy viscosity, learned or Smagorinsky) and the two conventional estimates from the same 4D flow scan (the pressure
Poisson estimate and simplified Bernoulli), for every prepared case of the 4D-FlowP coarctation phantom.

    python scripts/coa_report.py                    # -> runs/coa_report/{report.json, report.md}
    python scripts/coa_report.py --baselines-only   # compute and cache the scan-based baselines (CPU, ~2 min per case)

Agreement is summarised as in the source publication: the slope of a line through the origin fitted to
(catheter, method) and its R^2, plus the per-case percentage error.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pinnflow.coa_phantom import BERNOULLI_VERSION, CASES, COA_DIR, PREP_VERSION, bernoulli_peak, ppe_pressure_drop  # noqa: E402

OUT = ROOT / "runs" / "coa_report"


def baselines(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "baselines.json"
    cache = json.loads(f.read_text()) if f.exists() and not force else {}
    for case in CASES:
        npz = COA_DIR / "prepared" / f"{case}.npz"
        if not npz.exists():
            continue
        c0 = cache.get(case, {})
        if c0.get("prep_version", 0) >= PREP_VERSION and c0.get("bernoulli_version", 0) >= BERNOULLI_VERSION:
            continue
        prep = np.load(npz)
        ppe = ppe_pressure_drop(case)
        ber = bernoulli_peak(case)
        cache[case] = {"catheter_peak_mmHg": float(prep["cath_dp_peak_mmHg"]), "ppe_peak_mmHg": ppe["dp_peak_mmHg"],
                       "bernoulli_peak_mmHg": ber["dp_peak_mmHg"], "v_peak_ms": ber["v_peak_ms"],
                       "aao_shift_mm": float(prep["aao_shift_mm"]) if "aao_shift_mm" in prep else 0.0, "prep_version": PREP_VERSION, "bernoulli_version": BERNOULLI_VERSION}
        f.write_text(json.dumps(cache, indent=2))
        print(case, {k: round(v, 2) for k, v in cache[case].items()}, flush=True)
    return cache


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    x, y = x[m], y[m]
    k = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - k * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return k, (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    base = baselines(args.force)
    if args.baselines_only:
        return
    rows = []
    for case in CASES:
        if case not in base:
            continue
        r = {"case": case, **base[case]}
        for tag, run in (("pinn_laminar", f"coa_{case}"), ("pinn_eddy", f"coa_{case}_eddy")):
            fe = ROOT / "runs" / run / "final_eval.json"
            ev = json.loads(fe.read_text()) if fe.exists() else {}
            current = int(ev.get("data_prep_version", 0)) >= PREP_VERSION
            r[f"{tag}_peak_mmHg"] = ev["dp_peak_pinn_mmHg"] if ev and current else None
        rows.append(r)
    methods = [("ppe", "conventional 4D flow (pressure Poisson)"), ("bernoulli", "simplified Bernoulli 4 v^2"),
               ("pinn_laminar", "network, laminar"), ("pinn_eddy", "network, eddy viscosity")]
    cath = [r["catheter_peak_mmHg"] for r in rows]
    summary = {}
    for key, label in methods:
        ys = [r.get(f"{key}_peak_mmHg") for r in rows]
        ys = [np.nan if y is None else y for y in ys]
        k, r2 = fit(cath, ys)
        errs = [abs(y - c) / c * 100 for y, c in zip(ys, cath) if np.isfinite(y)]
        summary[key] = {"label": label, "slope": k, "r2": r2, "mean_abs_err_pct": float(np.mean(errs)) if errs else None, "cases": len(errs)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    lines = ["# Real data: pressure drop across the coarctation phantom", "",
             "Peak trans-coarctation pressure drop (mmHg); catheter = ground truth (DAo location 3, beat-averaged).", "",
             "| case | catheter | pressure Poisson | Bernoulli | network, laminar | network, eddy |", "|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.1f}"
    for r in rows:
        lines.append(f"| {r['case']} | {r['catheter_peak_mmHg']:.1f} | {fmt(r['ppe_peak_mmHg'])} | {fmt(r['bernoulli_peak_mmHg'])} | "
                     f"{fmt(r['pinn_laminar_peak_mmHg'])} | {fmt(r['pinn_eddy_peak_mmHg'])} |")
    lines += ["", "| method | slope vs catheter | R^2 | mean abs error | cases |", "|---|---|---|---|---|"]
    for key, s in summary.items():
        lines.append(f"| {s['label']} | {fmt(s['slope'])} | {fmt(s['r2'])} | {fmt(s['mean_abs_err_pct'])}% | {s['cases']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
