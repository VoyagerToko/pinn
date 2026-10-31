"""Black-and-white figures for the RM project report (data from docs/STATUS.md and the live server's curves)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps

D = Path(__file__).parent
FIG = D / "figs"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": "black", "axes.linewidth": 0.8,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, text, fill="white", bold=False, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fill, ec="black", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="black"))


# ---- Figure 1: end-to-end methodology
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 6.4)
ax.axis("off")
W, H = 2.8, 1.45
def b1(x, y, t, fill="white"):
    box(ax, x, y, W, H, t, fill, fs=7.2)
b1(0.1, 4.7, "INPUTS\nphysics only (benchmarks),\nor measured MRI velocity\n(+ acceleration)", "#eeeeee")
b1(3.2, 4.7, "DATA PREPARATION\nbackground correction, sign\nfix, segmentation, catheter\nplanes, clock alignment")
b1(6.3, 4.7, "NETWORKS\nvelocity net (Fourier features,\nModified MLP) + separate\nperiodic pressure net")
b1(9.4, 4.7, "AUTOMATIC\nDIFFERENTIATION\nforward-mode jvp:\ndu/dt, grad u, lap u, grad p", "#eeeeee")
b1(9.4, 2.55, "LOSS\ndata (voxel-averaged) +\nNavier-Stokes + continuity\n+ walls (+ acceleration)")
b1(6.3, 2.55, "TRAINING\nAdam, warm-up + decay;\ngrad-norm, RAD, causal\nweights; curriculum")
b1(3.2, 2.55, "EVALUATION\ngates vs exact / CFD / DNS;\npressure drop vs catheter,\nBernoulli, pressure Poisson")
b1(0.1, 2.55, "DIAGNOSTICS\nmomentum path integral,\nstand-alone pressure fit,\nablation rows A-H")
b1(6.3, 0.4, "LIVE VIEWER\nGPU inference server +\ninteractive viewer: fields,\nparticles, verdicts")
arrow(ax, 2.9, 5.42, 3.2, 5.42)
arrow(ax, 6.0, 5.42, 6.3, 5.42)
arrow(ax, 9.1, 5.42, 9.4, 5.42)
arrow(ax, 10.8, 4.7, 10.8, 4.0)
arrow(ax, 9.4, 3.27, 9.1, 3.27)
arrow(ax, 6.3, 3.27, 6.0, 3.27)
arrow(ax, 3.2, 3.27, 2.9, 3.27)
arrow(ax, 7.7, 2.55, 7.7, 1.85)
fig.savefig(FIG / "fig1_methodology.png")
plt.close(fig)

# ---- Figure 2: the separate pressure network next to the velocity network
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.4)
ax.axis("off")
box(ax, 0.1, 1.2, 1.4, 1.0, "input\n(t, x, y, z)", "#eeeeee")
box(ax, 2.0, 2.1, 2.6, 1.0, "periodic time embedding\n+ Fourier features (sigma 8)")
box(ax, 5.0, 2.1, 2.4, 1.0, "Modified MLP\n6 x 256, tanh")
box(ax, 2.0, 0.2, 2.6, 1.0, "12 heartbeat harmonics\n+ isotropic Fourier features\n(sigma 1, 4, 12)")
box(ax, 5.0, 0.2, 2.4, 1.0, "MLP 4 x 128, tanh")
box(ax, 7.9, 2.1, 2.0, 1.0, "velocity\n(u, v, w)", "#eeeeee")
box(ax, 7.9, 0.2, 2.0, 1.0, "pressure p", "#eeeeee")
for y in (2.6, 0.7):
    arrow(ax, 1.5, 1.7, 2.0, y)
    arrow(ax, 4.6, y, 5.0, y)
    arrow(ax, 7.4, y, 7.9, y)
fig.savefig(FIG / "fig2_networks.png")
plt.close(fig)

# ---- Figure 3: Gantt chart (Stage I done, Stage II planned)
tasks = [("Environment, JAX/WSL setup, tests", "2026-09-15", "2026-09-22", 1),
         ("Benchmarks A-B (TGV 2D, cavity)", "2026-09-20", "2026-09-27", 1),
         ("Benchmarks C-D, inverse wake, ablation", "2026-09-26", "2026-09-29", 1),
         ("PI-DeepONet cavity", "2026-10-01", "2026-10-04", 1),
         ("Synthetic stenosis from MRI", "2026-10-18", "2026-10-21", 1),
         ("Phantom data pipeline + baselines", "2026-10-20", "2026-10-23", 1),
         ("Phantom networks, diagnosis, fixes", "2026-10-22", "2026-10-24", 1),
         ("Live viewer", "2026-10-15", "2026-10-24", 1),
         ("Held-out test on 7 phantom cases", "2026-10-24", "2026-11-10", 0),
         ("Turbulence calibration, integral loss", "2026-11-10", "2026-12-20", 0),
         ("Code restructure + web viewer", "2026-11-15", "2027-01-31", 0),
         ("Paper writing and submission", "2027-01-15", "2027-03-31", 0),
         ("Final report, demo, viva", "2027-03-15", "2027-04-30", 0)]
import datetime as dt

fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, (name, a, b, done) in enumerate(tasks):
    a, b = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    ax.barh(i, (b - a).days or 1, left=a.toordinal(), color="black" if done else "white", edgecolor="black", hatch="" if done else "///", height=0.6)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in tasks], fontsize=8.5)
ax.invert_yaxis()
ticks = [dt.date(2026, m, 1) for m in (9, 10, 11, 12)] + [dt.date(2027, m, 1) for m in (1, 2, 3, 4, 5)]
ax.set_xticks([t.toordinal() for t in ticks])
ax.set_xticklabels([t.strftime("%b %Y") for t in ticks], fontsize=8.5)
ax.axvline(dt.date(2026, 10, 24).toordinal(), color="black", ls="--", lw=0.8)
ax.text(dt.date(2026, 10, 26).toordinal(), -0.8, "today (24 Oct 2026)", fontsize=8)
ax.grid(axis="x", color="0.85", lw=0.5)
from matplotlib.patches import Patch

ax.legend(handles=[Patch(fc="black", ec="black", label="Stage I, done"), Patch(fc="white", ec="black", hatch="///", label="Stage II, planned")],
          loc="upper right", fontsize=8, frameon=False)
fig.savefig(FIG / "fig3_gantt.png")
plt.close(fig)

# ---- Figure 4: pressure drop over one heartbeat, P2 baseline
c = json.load(open(D / "p2_curves.json"))
lam, acc = c["coa_P2_baseline"], c["coa_P2_baseline_eddy"]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(lam["cath_t_s"], lam["cath_dp_scan_mmHg"], color="black", lw=1.8, label="catheter (truth), peak 13.1 mmHg")
ax.plot(acc["t_s"], acc["dp_pinn_mmHg"], color="black", lw=1.4, ls="--", label="PINN + measured acceleration, peak 8.7 mmHg")
ax.plot(lam["t_s"], lam["dp_pinn_mmHg"], color="0.45", lw=1.4, ls=":", label="PINN, laminar, pressure network, peak 5.3 mmHg")
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlabel("time in the heartbeat (s)")
ax.set_ylabel("pressure drop (mmHg)")
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.set_ylim(top=19)
fig.savefig(FIG / "fig4_heartbeat.png")
plt.close(fig)

# ---- Figure 5: methods on the two finished phantom cases
cases = ["P2 baseline\n(development case)", "P2 severe, low flow\n(first held-out case)"]
vals = {"catheter (truth)": [13.10, 12.04], "Bernoulli 4v^2": [8.04, 7.45], "pressure Poisson (4D flow)": [7.87, 5.49],
        "PINN + measured acceleration": [8.74, 4.67]}
hatches = ["", "...", "\\\\\\", "///"]
fills = ["black", "white", "white", "white"]
import numpy as np

fig, ax = plt.subplots(figsize=(6.4, 3.0))
x = np.arange(len(cases))
w = 0.19
for k, (lab, v) in enumerate(vals.items()):
    bars = ax.bar(x + (k - 1.5) * w, v, w, color=fills[k], edgecolor="black", hatch=hatches[k], label=lab)
    for b_, vv in zip(bars, v):
        ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.25, f"{vv:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=9)
ax.set_ylabel("peak pressure drop (mmHg)")
ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
ax.set_ylim(0, 15.5)
fig.savefig(FIG / "fig5_methods.png")
plt.close(fig)

# ---- Figure 6: the P2 baseline journey
steps = ["first network\n(shared output)", "pressure fitted\nto frozen flow", "separate\npressure net", "+ Smagorinsky\neddy viscosity",
         "+ measured\nacceleration (60k)"]
v = [0.30, 7.26, 5.28, 6.05, 8.74]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
bars = ax.bar(range(len(v)), v, color="white", edgecolor="black", hatch="///")
bars[-1].set_hatch("xxx")
for b_, vv in zip(bars, v):
    ax.text(b_.get_x() + b_.get_width() / 2, vv + 0.25, f"{vv:.2f}", ha="center", fontsize=8.5)
ax.axhline(13.1, color="black", lw=1.2)
ax.text(4.45, 13.4, "catheter 13.1", ha="right", fontsize=8.5)
ax.axhline(8.04, color="black", lw=0.9, ls="--")
ax.text(-0.45, 8.35, "Bernoulli 8.0", fontsize=8.5)
ax.set_xticks(range(len(v)))
ax.set_xticklabels(steps, fontsize=8.3)
ax.set_ylabel("peak pressure drop (mmHg)")
ax.set_ylim(0, 15)
fig.savefig(FIG / "fig6_journey.png")
plt.close(fig)

# ---- Figure 7, 8: greyscale viewer screenshots
S = D.parent
for src, dst in (("p2_beat4.png", "fig7_viewer_phantom.png"), ("cyl2.png", "fig8_viewer_cylinder.png")):
    ImageOps.grayscale(Image.open(S / src)).save(FIG / dst)

# ---- Figure 9: catheter against Bernoulli and pressure Poisson, all 8 cases
cath = [16.23, 24.71, 13.10, 20.78, 8.95, 13.36, 7.17, 12.04]
bern = [10.34, 21.99, 8.04, 16.65, 5.12, 12.03, 3.85, 7.45]
ppe = [6.48, 10.21, 7.87, 14.55, 5.28, 6.75, 4.85, 5.49]
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot([0, 27], [0, 27], color="black", lw=0.8)
ax.scatter(cath, bern, marker="o", facecolor="black", edgecolor="black", s=26, label="Bernoulli 4v^2 (slope 0.77)")
ax.scatter(cath, ppe, marker="s", facecolor="white", edgecolor="black", s=26, label="pressure Poisson (slope 0.52)")
ax.set_xlabel("catheter peak drop (mmHg)")
ax.set_ylabel("estimated peak drop (mmHg)")
ax.set_xlim(0, 27)
ax.set_ylim(0, 27)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.savefig(FIG / "fig9_baselines.png")
plt.close(fig)
print("figures:", sorted(p.name for p in FIG.iterdir()))
