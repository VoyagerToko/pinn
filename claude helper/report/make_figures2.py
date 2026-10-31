"""Second set of black-and-white figures: images of the data, training curves, ablation, barriers (numbers from
docs/STATUS.md and the runs' metrics.csv files)."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).parent
FIG = D / "figs"
REPO = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": "black", "axes.linewidth": 0.8,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})


def evals(path, key):
    """(step, value) rows of metrics.csv where the evaluation column is filled."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get(key) not in (None, ""):
                out.append((float(r["step"]), float(r[key])))
    return np.array(out)


# ---- F10: the 8 phantom aortas (maximum speed projection at peak flow)
cases = ["P1_baseline", "P1_severe", "P1_baseline_lowflow", "P1_severe_lowflow",
         "P2_baseline", "P2_severe", "P2_baseline_lowflow", "P2_severe_lowflow"]
fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.6))
for ax, c in zip(axes.ravel(), cases):
    d = np.load(REPO / "data/stanford_coa/prepared" / f"{c}.npz")
    vel, pos = d["vel"], d["pos_mm"]
    sp = np.linalg.norm(vel, axis=-1)
    f = int(np.argmax(sp.mean(1)))
    s = sp[f]
    order = np.argsort(s)  # fastest drawn last, on top
    ax.scatter(pos[order, 0], pos[order, 1], c=s[order], cmap="gray_r", s=1.2, vmin=0, vmax=np.percentile(s, 99.5), linewidths=0)
    if c == "P2_baseline":
        for pl, mk in ((d["aao_mm"], "AAo"), (d["dao_mm"], "DAo")):
            ax.plot(pl[:, 0], pl[:, 1], ".", color="black", ms=1.5)
            ax.text(pl[:, 0].mean() + 6, pl[:, 1].mean(), mk, fontsize=7.5, va="center")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(c.replace("_lowflow", ", low flow").replace("_", " ") + f"\ncatheter {float(d['cath_dp_peak_mmHg']):.1f} mmHg", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig10_phantoms.png")
plt.close(fig)

# ---- F11: synthetic stenosis, CFD truth against the simulated MRI at peak flow
tr = np.load(REPO / "data/stenosis/truth.npz")
mr = np.load(REPO / "data/stenosis/mri.npz")
ft = int(np.argmax(np.abs(tr["u"]).mean((1, 2))))
fm = int(np.argmax(np.abs(mr["u"]).mean((1, 2))))
st = np.where(tr["fluid"], np.hypot(tr["u"][ft], tr["v"][ft]), np.nan)
sm = np.where(mr["lumen"], np.hypot(mr["u"][fm], mr["v"][fm]), np.nan)
vmax = np.nanpercentile(st, 99.5)
fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.8))
fig.subplots_adjust(hspace=0.55)
for ax, field, x, y, title in ((axes[0], st, tr["x"], tr["y"], "CFD truth (50 cells per diameter)"),
                               (axes[1], sm, mr["x"], mr["y"], "simulated 4D flow scan (0.6 mm voxels, SNR 20)")):
    im = ax.imshow(field.T, origin="lower", cmap="gray_r", vmin=0, vmax=vmax, aspect="auto",
                   extent=[x.min(), x.max(), y.min(), y.max()], interpolation="nearest")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_yticks([])
axes[1].set_xlabel("position along the artery (non-dimensional)")
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
cb.set_label("speed", fontsize=9)
fig.savefig(FIG / "fig11_stenosis_data.png")
plt.close(fig)

# ---- F12: catheter clock alignment on P2 baseline
d = np.load(REPO / "data/stanford_coa/prepared/P2_baseline.npz")
P = float(d["period_s"])
ph, q = d["cath_phase_s"], d["cath_flow_mLs"]
ts, qs = d["t_s"], d["aao_flux_frames_mLs"]
shift = float(d["cath_shift_s"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, sh, title in ((axes[0], 0.0, "before: the two clocks as recorded"), (axes[1], shift, f"after: catheter clock shifted by {shift * 1e3:.0f} ms")):
    tt = (ph - sh) % P
    o = np.argsort(tt)
    ax.plot(tt[o], q[o], color="black", lw=1.4, label="flow meter (catheter recording)")
    ax.plot(ts, qs, "o--", color="0.35", ms=3.5, mfc="white", lw=1.0, label="scan: flux through the upstream plane")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("time in the beat (s)")
axes[0].set_ylabel("flow (mL/s)")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig12_clock.png")
plt.close(fig)

# ---- F13: Taylor-Green 2D ablation rows A-G (3k steps)
rows = "ABCDEFG"
err = [8.7e-3, 2.25e-2, 3.2e-2, 2.18e-2, 1.80e-2, 1.71e-2, 1.17e-2]
fig, ax = plt.subplots(figsize=(6.0, 2.7))
xr = np.arange(len(rows))
bars = ax.bar(xr, err, width=0.6, color="white", edgecolor="black", hatch="///")
for xi_, v in zip(xr, err):
    ax.text(xi_, v + 0.0006, f"{v:.1e}", ha="center", fontsize=7.5)
ax.set_xticks(xr)
ax.set_xticklabels(list(rows))
ax.set_xlim(-0.6, len(rows) - 0.4)
ax.set_ylabel("velocity rel. L2 error (3k steps)")
ax.set_xlabel("ablation row (A: plain MLP ... G: all techniques)")
ax.set_ylim(0, 0.041)
ax.text(-0.45, 0.0375, "for comparison, row G after 50k steps: 4.05e-5", fontsize=7.5)
fig.savefig(FIG / "fig13_ablation_tgv.png")
plt.close(fig)

# ---- F14: PI-DeepONet error at the unseen Re = 400 across the training legs
legs = ["no curriculum\n(25k)", "curriculum,\nlr 3e-4 (200k)", "continuation\n(150k)", "slower decay\n(150k)",
        "constant lr\n(100k)", "4x batch\n(60k)"]
v = [81.0, 14.0, 11.1, 10.27, 10.11, 9.65]
fig, ax = plt.subplots(figsize=(6.6, 2.8))
ax.plot(range(len(v)), v, "o-", color="black", mfc="white")
for i, x in enumerate(v):
    ax.text(i, x * (1.12 if i < len(v) - 1 else 0.86), f"{x:g}%", ha="center", va="bottom" if i < len(v) - 1 else "top", fontsize=8)
ax.axhline(10, color="black", ls="--", lw=0.9)
ax.text(1.5, 9.3, "gate 10%", ha="center", va="top", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(7, 120)
ax.set_xticks(range(len(v)))
ax.set_xticklabels(legs, fontsize=7.8)
ax.set_ylabel("error at Re 400 (%)")
fig.savefig(FIG / "fig14_deeponet.png")
plt.close(fig)

# ---- F15: cylinder, share of the inflow reaching downstream stations
x = [0.6, 1.2, 2.2]
fig, ax = plt.subplots(figsize=(5.4, 2.7))
ax.plot([0.0] + x, [1.0, 0.12, 0.04, 0.02], "s--", color="0.35", mfc="white", label="velocity-pressure network (ramped inflow)")
ax.plot([0.0] + x, [1.0, 0.77, 0.77, 0.75], "o-", color="black", mfc="black", label="stream-function network (exact div u = 0)")
ax.axhline(1.0, color="black", lw=0.7)
ax.text(2.2, 1.03, "mass conserved", ha="right", fontsize=8)
ax.set_xlabel("distance along the channel (m)")
ax.set_ylabel("flow rate / inflow")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8, frameon=False, loc="center right")
fig.savefig(FIG / "fig15_cylinder_mass.png")
plt.close(fig)

# ---- F16: momentum balance along the vessel (P2 baseline, first network)
tt = ["125 ms", "185 ms", "245 ms\n(peak flow)", "490 ms"]
required = [4.53, 6.21, 5.52, -4.44]  # drop the network's own flow needs, mmHg (sign: AAo minus DAo)
network = [-0.05, 0.12, 0.34, -0.10]
xi = np.arange(len(tt))
fig, ax = plt.subplots(figsize=(6.0, 2.8))
ax.bar(xi - 0.18, required, 0.36, color="white", edgecolor="black", hatch="///", label="drop required by the network's own flow")
ax.bar(xi + 0.18, network, 0.36, color="black", label="drop in the network's pressure")
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(xi)
ax.set_xticklabels(tt, fontsize=8.5)
ax.set_ylabel("pressure drop AAo - DAo (mmHg)")
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.savefig(FIG / "fig16_momentum.png")
plt.close(fig)

# ---- F17: phantom training curves (peak pressure drop at each evaluation)
runs = [(D.parent / "rel/runs/coa_P2_baseline/metrics.csv", "shared output (first network)", "0.55", ":", "s"),
        (REPO / "runs/_test_pnet_cold/metrics.csv", "separate pressure network", "black", "--", "o"),
        (REPO / "runs/_test_smag_sub4/metrics.csv", "+ Smagorinsky eddy viscosity", "0.35", "-.", "^"),
        (REPO / "runs/coa_P2_baseline_accel_60k/metrics.csv", "+ measured acceleration", "black", "-", "D")]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for path, label, col, ls, mk in runs:
    e = evals(path, "eval/dp_peak_pinn_mmHg")
    e = e[e[:, 0] > 0]
    ax.plot(e[:, 0] / 1000, e[:, 1], ls=ls, marker=mk, color=col, mfc="white", ms=4, lw=1.3, label=label)
ax.axhline(13.1, color="black", lw=1.2)
ax.text(59, 13.4, "catheter 13.1", ha="right", fontsize=8)
ax.axhline(8.04, color="black", lw=0.8, ls="--")
ax.text(15, 8.35, "Bernoulli 8.0", ha="center", fontsize=8)
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("peak pressure drop (mmHg)")
ax.set_ylim(0, 15)
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.savefig(FIG / "fig17_phantom_training.png")
plt.close(fig)

# ---- F18: eddy viscosity during training (mean over the lumen, relative to the fluid's own)
fig, ax = plt.subplots(figsize=(6.0, 2.8))
for path, label, ls, mk in ((REPO / "runs/_test_pnet_cold_eddy/metrics.csv", "learned field, penalty 0.01", "--", "o"),
                            (REPO / "runs/_test_pnet_cold_eddy_reg0001/metrics.csv", "learned field, penalty 0.001", ":", "s"),
                            (REPO / "runs/_test_smag_sub4/metrics.csv", "Smagorinsky, from the flow (C_s 0.17)", "-", "^")):
    e = evals(path, "eval/nu_t_mean_over_nu")
    e = e[e[:, 0] > 0]
    ax.plot(e[:, 0] / 1000, e[:, 1], ls=ls, marker=mk, color="black", mfc="white", ms=4, lw=1.2, label=label)
ax.set_yscale("log")
ax.axhline(1.0, color="0.6", lw=0.7)
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("eddy viscosity / fluid viscosity")
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.savefig(FIG / "fig18_eddy.png")
plt.close(fig)

# ---- F19: synthetic stenosis, peak-drop error during training
fig, ax = plt.subplots(figsize=(6.0, 2.8))
for path, label, ls, mk in ((REPO / "runs/stenosis_mri_G/metrics.csv", "stenosis (Re 150), laptop GPU", "-", "o"),
                            (REPO / "runs/stenosis_mri_fast/metrics.csv", "fast jet (Re 300), desktop GPU", "--", "s")):
    e = evals(path, "eval/dp_peak_err_pct")
    e = e[e[:, 0] > 0]
    ax.plot(e[:, 0] / 1000, e[:, 1], ls=ls, marker=mk, color="black", mfc="white", ms=4, lw=1.2, label=label)
ax.axhline(10, color="black", lw=0.8, ls="--")
ax.text(79, 11, "gate 10%", ha="right", fontsize=8)
ax.axhline(31.5, color="0.5", lw=0.8)
ax.text(1, 32.5, "Bernoulli on the stenosis scan: 31.5%", fontsize=8)
ax.set_xlabel("training step (thousands)")
ax.set_ylabel("peak pressure-drop error (%)")
ax.set_ylim(0, 40)
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.savefig(FIG / "fig19_stenosis_training.png")
plt.close(fig)

# ---- F20: training wall-clock of the main runs
runs = [("tgv2d_G", 2.0), ("cavity_C_r2", 1.59), ("cavity_F_r2", 2.30), ("cavity_D_softlid", 3.27), ("cylinder_sf_2x6k", 1.04),
        ("tgv3d_H (SPINN)", 1.24), ("cylinder_inverse", 0.52), ("deeponet (5 legs)", 3.08), ("stenosis_mri_G", 1.6),
        ("stenosis_mri_fast", 1.6), ("coa P2 accel 60k", 3.9)]
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.barh([r[0] for r in runs], [r[1] for r in runs], color="white", edgecolor="black", hatch="///")
for i, (_, h) in enumerate(runs):
    ax.text(h + 0.05, i, f"{h:.1f} h", va="center", fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("training wall-clock (hours)")
ax.tick_params(axis="y", labelsize=8.5)
ax.set_xlim(0, 4.6)
fig.savefig(FIG / "fig20_cost.png")
plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("fig1[0-9]*.png")) + sorted(p.name for p in FIG.glob("fig20*.png")))
