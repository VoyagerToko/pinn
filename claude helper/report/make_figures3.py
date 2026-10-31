"""Third set of black-and-white figures: the 8-case phantom result (2026-10-27) and the reading at the catheter's own
marker locations (2026-10-30). Numbers from runs/coa_report/baselines.json, the runs' final_eval.json files and the
marker table in docs/STATUS.md."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent
FIG = D / "figs"
REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": "black", "axes.linewidth": 0.8,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})

base = json.loads((RUNS / "coa_report" / "baselines.json").read_text())
cases = ["P1_baseline", "P1_severe", "P1_baseline_lowflow", "P1_severe_lowflow",
         "P2_baseline", "P2_severe", "P2_baseline_lowflow", "P2_severe_lowflow"]
label = {c: c.replace("_lowflow", "\nlow flow").replace("_", " ") for c in cases}
cath = np.array([base[c]["catheter_peak_mmHg"] for c in cases])
bern = np.array([base[c]["bernoulli_peak_mmHg"] for c in cases])
ppe = np.array([base[c]["ppe_peak_mmHg"] for c in cases])
lam = np.array([json.loads((RUNS / f"coa_{c}_lam_60k/final_eval.json").read_text())["dp_peak_pinn_mmHg"] for c in cases])
acc = np.array([json.loads((RUNS / f"coa_{c}_accel_60k/final_eval.json").read_text())["dp_peak_pinn_mmHg"] for c in cases])
err = lambda e: 100 * np.mean(np.abs(e - cath) / cath)
print("mean errors: Bernoulli %.1f, PPE %.1f, laminar %.1f, accel %.1f" % (err(bern), err(ppe), err(lam), err(acc)))

# ---- F21: peak drop of every method on the 8 cases
series = [("catheter (truth)", cath, "black", ""), ("Bernoulli", bern, "white", "...."),
          ("pressure Poisson", ppe, "white", "\\\\\\"), ("PINN, laminar", lam, "white", "---"),
          ("PINN, Smagorinsky + acceleration", acc, "white", "///")]
x = np.arange(len(cases))
w = 0.16
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for k, (name, v, col, hatch) in enumerate(series):
    lab = name if k == 0 else f"{name} ({err(v):.1f}%)"
    ax.bar(x + (k - 2) * w, v, w, color=col, edgecolor="black", hatch=hatch, linewidth=0.6, label=lab)
ax.set_xticks(x, [label[c] for c in cases], fontsize=8.5)
ax.set_ylabel("peak pressure drop (mmHg)")
ax.set_ylim(0, 27)
ax.axvline(3.5, color="0.5", lw=0.6, ls=":")
ax.text(1.5, 25.6, "geometry P1", ha="center", fontsize=9)
ax.text(5.5, 25.6, "geometry P2 (P2 baseline = development case)", ha="center", fontsize=9)
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17),
          title="mean error over the 8 cases in brackets", title_fontsize=8)
fig.savefig(FIG / "fig21_phantom_cases.png")
plt.close(fig)

# ---- F22: error at the 2D-PC planes against the catheter's own marker locations (network + acceleration)
plane = np.abs(acc - cath) / cath * 100
marker_err = {"P1_baseline": 7, "P1_baseline_lowflow": 32, "P1_severe": 46, "P1_severe_lowflow": 29,
              "P2_baseline": 42, "P2_baseline_lowflow": 32, "P2_severe": 90, "P2_severe_lowflow": 70}
mk = np.array([marker_err[c] for c in cases])
bern_err = np.abs(bern - cath) / cath * 100
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ax.bar(x - 0.27, bern_err, 0.27, color="white", edgecolor="black", hatch="....", linewidth=0.6,
       label=f"Bernoulli (mean {bern_err.mean():.1f}%)")
ax.bar(x, plane, 0.27, color="white", edgecolor="black", hatch="///", linewidth=0.6,
       label=f"network read at the 2D-PC planes (mean {plane.mean():.1f}%)")
ax.bar(x + 0.27, mk, 0.27, color="0.35", edgecolor="black", linewidth=0.6,
       label=f"network read at the catheter markers (mean {mk.mean():.1f}%)")
ax.set_xticks(x, [label[c] for c in cases], fontsize=8.5)
ax.set_ylabel("peak-drop error against the catheter (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.savefig(FIG / "fig22_markers.png")
plt.close(fig)
print("figures: fig21_phantom_cases.png, fig22_markers.png")
