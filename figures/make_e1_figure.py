#!/usr/bin/env python3
"""E1 main-text figure for main_v2: allocation -> reversal -> bridge.

Reads the frozen E1 outputs in the Overleaf project and writes
figures/e1_solved_class.pdf (vector) plus a PNG preview.
Palette validated with the dataviz six-checks script:
red #C24438, blue #3273B5, green #2E9662 on white; ink #2B2B2B.
"""
import csv, math, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/sampled_phase/outputs"
FIGDIR = ROOT / "figures"

RED, BLUE, GREEN = "#C24438", "#3273B5", "#2E9662"
INK, MUT, GRID = "#2B2B2B", "#6B6B6B", "#D9D9D9"
P_LO, P_HI = (3 - math.sqrt(5)) / 2, (math.sqrt(5) - 1) / 2

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

def rows(name):
    with open(OUT / name) as f:
        return [{k: (v if k in ("dimension", "p_index", "dataset_index")
                     else v) for k, v in r.items()} for r in csv.DictReader(f)]

fs = rows("figure_summary.csv")
p = [float(r["p"]) for r in fs]
thg = {k: [float(r[f"theory_{k}"]) for r in fs] for k in ("g12", "g13", "g23")}
mdg = {k: [float(r[f"{k}_median"]) for r in fs] for k in ("g12", "g13", "g23")}
thD = [float(r["theory_D"]) for r in fs]
mdD = [float(r["D_median"]) for r in fs]

hb = rows("heldout_branch_distortion.csv")
d2 = [r for r in hb if r["dimension"] == "2"]
d3 = [r for r in hb if r["dimension"] == "3"]
med = lambda xs: sorted(xs)[len(xs) // 2]
print("d2 rows", len(d2), "d3 rows", len(d3))
print("d2 median |bridge err|", med([float(r["absolute_bridge_error"]) for r in d2]))
if d3:
    print("d3 median |bridge err|", med([float(r["absolute_bridge_error"]) for r in d3]))

# p_delta for delta=0.5 (central phase root of D(p)=0.5)
D = lambda q: (1 - 2 * q) / 4 * ((q * (1 - q)) ** -2 - 1)
lo, hi = P_LO, P_HI
for _ in range(80):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if D(mid) > 0.5 else (lo, mid)
p_delta = (lo + hi) / 2
print("p_delta(0.5) =", round(p_delta, 4))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.05, 2.42))
fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.19, wspace=0.34)

def strip(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(color=MUT, labelcolor=INK, length=2.5)

# ---------------- Panel A: allocation ----------------
axA.axvspan(0, P_LO, color=RED, alpha=0.055, lw=0)
axA.axvspan(P_HI, 1, color=BLUE, alpha=0.055, lw=0)
for x in (P_LO, P_HI):
    axA.axvline(x, color=MUT, lw=0.7, ls=(0, (4, 3)))
for k, c in (("g12", RED), ("g13", GREEN), ("g23", BLUE)):
    axA.plot(p, thg[k], color=c, lw=1.4, solid_capstyle="round", zorder=3)
    axA.plot(p, mdg[k], "o", ms=3.4, mfc="white", mec=c, mew=0.9, zorder=4)
axA.text(0.075, 0.905, r"$g_{12}$", color=INK, fontsize=8.5)
axA.text(0.865, 0.905, r"$g_{23}$", color=INK, fontsize=8.5)
axA.text(0.5, 0.125, r"$g_{13}$", color=INK, fontsize=8.5, ha="center")
axA.text(P_LO / 2, 1.135, "collide $\\{1,2\\}$", color=MUT, ha="center", fontsize=7)
axA.text(0.5, 1.135, "central", color=MUT, ha="center", fontsize=7)
axA.text((P_HI + 1) / 2, 1.135, "collide $\\{2,3\\}$", color=MUT, ha="center", fontsize=7)
axA.set_xticks([0, P_LO, 0.5, P_HI, 1])
axA.set_xticklabels(["0", r"$p_-$", "0.5", r"$p_+$", "1"])
axA.set_xlim(0, 1); axA.set_ylim(-0.04, 1.12)
axA.set_xlabel(r"occupancy $p=\pi(E)$", labelpad=1.5)
axA.set_ylabel("squared Gram entry", labelpad=2)
strip(axA)

# ---------------- Panel B: reversal ----------------
axB.axvspan(0, p_delta, color=RED, alpha=0.055, lw=0)
axB.axhline(0.5, color=MUT, lw=0.8, ls=(0, (4, 3)))
axB.text(0.985, 0.545, r"$\delta=0.5$", color=MUT, fontsize=7, ha="right")
axB.plot(p, thD, color=INK, lw=1.4, solid_capstyle="round", zorder=3)
axB.plot(p, mdD, "o", ms=3.4, mfc="white", mec=INK, mew=0.9, zorder=4)
axB.plot([p_delta], [0.5], "o", ms=4.6, mfc=RED, mec="white", mew=0.8, zorder=5)
axB.annotate(r"$p_\delta$", (p_delta, 0.5), xytext=(p_delta + 0.06, 0.72),
             color=INK, fontsize=8.5,
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
axB.text(p_delta / 2, -0.82, "deployed preference\nreversed", color=MUT,
         ha="center", fontsize=7, linespacing=1.15)
axB.set_xticks([0, P_LO, 0.5, P_HI, 1])
axB.set_xticklabels(["0", r"$p_-$", "0.5", r"$p_+$", "1"])
axB.set_xlim(0, 1); axB.set_ylim(-1.12, 1.12)
axB.set_xlabel(r"occupancy $p=\pi(E)$", labelpad=1.5)
axB.set_ylabel(r"$\mathcal{D}(p)=g_{12}-g_{23}$", labelpad=0)
strip(axB)

# ---------------- Panel C: bridge ----------------
axC.plot([-0.8, 1.8], [-0.8, 1.8], color=MUT, lw=0.8, ls=(0, (4, 3)), zorder=2)
x2 = [float(r["gap_gram"]) for r in d2]
y2 = [float(r["gap_mc"]) for r in d2]
axC.plot(x2, y2, "o", ms=4.2, mfc=RED, mec="white", mew=0.55, alpha=0.85, zorder=4)
if d3:
    x3 = [float(r["gap_gram"]) for r in d3]
    y3 = [float(r["gap_mc"]) for r in d3]
    axC.plot(x3, y3, "^", ms=5.2, mfc=GREEN, mec="white", mew=0.55, zorder=6)
axC.annotate("$k{=}2$, $p{=}0.2$\nreversed", (-0.5, -0.5), xytext=(-0.32, -0.62),
             color=INK, fontsize=7, va="center",
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
axC.annotate("$k{=}2$, $p{=}0.8$", (1.47, 1.5), xytext=(0.56, 1.3),
             color=INK, fontsize=7, va="center",
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
axC.annotate("$k{=}3$ control (all $p$),\ngap $=\\delta$", (0.47, 0.52),
             xytext=(-0.62, 0.88), color=INK, fontsize=7, va="center",
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
axC.annotate("$k{=}2$, $p{=}0.5$\n(central phase)", (0.67, 0.67),
             xytext=(0.95, 0.05), color=INK, fontsize=7, va="center",
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.6))
axC.text(-0.62, 1.58, "median $|$error$|$ $3.9\\times10^{-3}$",
         color=MUT, fontsize=7)
axC.set_xlim(-0.8, 1.8); axC.set_ylim(-0.8, 1.8)
axC.set_xticks([-0.5, 0.5, 1.5]); axC.set_yticks([-0.5, 0.5, 1.5])
axC.set_xlabel("Gram-predicted gap", labelpad=1.5)
axC.set_ylabel("rollout-measured gap", labelpad=0)
strip(axC)

for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C")):
    ax.text(-0.14, 1.13, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color=INK)

fig.savefig(FIGDIR / "e1_solved_class.pdf")
fig.savefig(FIGDIR / "e1_solved_class.png", dpi=220)
print("written figures/e1_solved_class.pdf and .png")
