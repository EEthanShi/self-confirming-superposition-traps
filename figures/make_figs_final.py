"""Final paper figures from Block B + core finals. Colors: constr=deepblue,
unconstr=orange, intervention=deepgreen, trap-region=deepred only, gray=control."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, ".")
from e2 import model
from e2 import cohort_stats as cs

DB, DR, DG, OR, GRAY = "#245280", "#A63E37", "#33785B", "#C07A3E", "#8A8F98"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.18, "grid.linewidth": 0.5, "figure.dpi": 300,
    "axes.titleweight": "bold", "legend.frameon": False})

core = json.load(open("out/final_core.json"))
ms = json.load(open("out/multiseed_robustness.json"))
ph = json.load(open("out/posthoc_G6_basin_width.json"))
g6p = json.load(open("out/g6prime.json"))
blkB = json.load(open("out/cohort_blockB.json"))
tgB = json.load(open("cohort_targets_B.json"))
auth = json.load(open("out/du_authority.json"))
ext = json.load(open("out/du_authority_ext.json"))

# ---------------- fig 1 ----------------
fig = plt.figure(figsize=(7.4, 2.6), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
A = fig.add_subplot(gs[0]); Bx = fig.add_subplot(gs[1]); C = fig.add_subplot(gs[2])
recs = core["grids"]["warm_m20"]
ds = sorted({r["delta"] for r in recs}); p0s = sorted({r["p0"] for r in recs})
M = np.full((len(p0s), len(ds)), np.nan)
for r in recs:
    M[p0s.index(r["p0"]), ds.index(r["delta"])] = r["basin"]
A.pcolormesh(ds, p0s, M, cmap=ListedColormap([DR, DB]), alpha=0.33,
             shading="nearest", rasterized=True)
dd = np.linspace(min(ds), max(ds), 400)
A.plot(dd, [model.p_delta(x) for x in dd], color="black", lw=1.4)
s80 = sorted((r["delta"], r["sep"]) for r in core["separators"] if r["m"] == 80)
A.plot(*zip(*s80), "o", ms=3.4, mfc="white", mec="black", mew=0.8)
A.text(0.30, 0.09, "trap", color=DR, fontweight="bold")
A.text(0.30, 0.90, "escape", color=DB, fontweight="bold")
A.set_xlabel(r"$\delta$"); A.set_ylabel(r"$p_0$"); A.set_title("A  Two basins")

kappas = [1, 5, 20, 80]
mean_err, max_err = [], []
for m in kappas:
    e = [abs(r["sep"] - r["analytic"]) for r in core["separators"] if r["m"] == m]
    mean_err.append(np.mean(e)); max_err.append(np.max(e))
Bx.plot(kappas, mean_err, "o-", color=DB, lw=1.4, ms=4, label="mean over $\\delta$")
Bx.plot(kappas, max_err, "s--", color=DB, alpha=0.5, lw=1.0, ms=3.5, label="max")
for m in (20, 80):
    errs = [abs(r["sep"] - r["analytic"]) for r in ms["rows"] if r["m"] == m]
    Bx.plot([m, m], [min(errs), max(errs)], color=DB, lw=4, alpha=0.25,
            solid_capstyle="round")
kk = np.array([1.0, 100.0])
Bx.plot(kk, 0.15 / kk, ":", color="black", lw=1.0)
Bx.text(30, 0.011, r"$\propto 1/\kappa$", fontsize=7.5)
Bx.set_xscale("log"); Bx.set_yscale("log")
Bx.set_xlabel(r"time-scale ratio $\kappa$")
Bx.set_ylabel(r"$|\hat p_\delta - p_\delta|$")
Bx.set_title("B  Adiabatic limit"); Bx.legend(fontsize=6.5, loc="lower left")

lam, s = zip(*sorted(ph["sep_vs_lam"]))
C.plot(np.array(lam) / ph["lam_crit"], s, "-", lw=1.3, color=DG)
for d, mk, col in ((0.3, "o", DG), (0.6, "s", "#5E9E7E")):
    rows = sorted((r["lam"], r["s"]) for r in g6p["rows"] if r["delta"] == d)
    crit = g6p["lam_crit"][str(d)]
    C.plot([l / crit for l, _ in rows], [x for _, x in rows], mk, ms=3.6,
           mfc="white", mec=col, mew=1.1, label=rf"$\delta={d}$")
C.axvline(1.0, color="black", ls="--", lw=1.0)
C.text(0.99, 0.40, r"$\lambda_{\rm crit}$", ha="right", fontsize=8.5)
C.set_xlabel(r"penalty $\lambda\,/\,\lambda_{\rm crit}$")
C.set_ylabel("trap-basin width")
C.set_title("C  Removal at the threshold"); C.legend(fontsize=6.5)
fig.savefig("out/e2_core_final.png", bbox_inches="tight")
fig.savefig("out/e2_core_final.pdf", bbox_inches="tight")

# ---------------- fig 2 (Block B) ----------------
fig = plt.figure(figsize=(7.4, 2.7), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
A = fig.add_subplot(gs[0]); Bx = fig.add_subplot(gs[1])
recsB = blkB["records"]
seeds = sorted({r["seed"] for r in recsB})
idx = cs.boot_indices(len(seeds))
grids = {float(k): np.array(v) for k, v in tgB["grids"].items()}
T = {r["delta"]: r for r in tgB["rows"]}
tab = sorted({r["p"]: r["D_u"] for r in
              auth["continuation_table"] + ext["continuation_table"]}.items())
ps_ = np.array([x for x, _ in tab]); ds_ = np.array([y for _, y in tab])
def p_sep_u(delta):
    for i in range(len(ps_) - 1):
        if ds_[i] >= delta > ds_[i + 1]:
            t = (delta - ds_[i + 1]) / (ds_[i] - ds_[i + 1])
            return ps_[i + 1] - t * (ps_[i + 1] - ps_[i])
    return np.nan
dd = np.linspace(0.10, 0.66, 240)
A.plot(dd, [(model.p_delta(x) - 0.05) / 0.9 for x in dd], color=DB, lw=1.5)
A.plot(dd, [(p_sep_u(x) - 0.05) / 0.9 for x in dd], color=OR, lw=1.5)
A.text(0.13, 0.395, "constrained class", color=DB, fontsize=7.5)
A.text(0.13, 0.55, "unconstrained class", color=OR, fontsize=7.5)
for arm, col, mk in (("constr2", DB, "o"), ("unconstr2", OR, "s")):
    for d in (0.15, 0.30, 0.60):
        sel = [r for r in recsB if r["arm"] == arm and abs(r["delta"] - d) < 1e-9
               and abs(r["floor_eps"] - 0.05) < 1e-9]
        bs = cs.boot_boundaries(sel, grids[d], seeds, idx)
        ci, _ = cs.ci(bs)
        b = cs.boundary(cs.frac_curve(sel, grids[d]), grids[d])
        A.errorbar([d], [b], yerr=[[b - ci[0]], [ci[1] - b]], fmt=mk, ms=4.5,
                   mfc="white", mec=col, ecolor=col, elinewidth=1.2, mew=1.3,
                   capsize=2)
A.text(0.33, 0.06, "capacity control: 0 / 11 700 trapped", fontsize=7,
       color=GRAY)
A.axvspan(0.40, 0.55, color=GRAY, alpha=0.08)
A.text(0.475, 0.60, "exploratory\nband", fontsize=6.2, color=GRAY, ha="center")
A.set_xlabel(r"$\delta$"); A.set_ylabel("basin boundary")
A.set_title("A  Boundaries follow the representation class")

for arm, col, mk in (("constr2", DB, "o"), ("unconstr2", OR, "s")):
    sk = "shift_c" if arm == "constr2" else "shift_u"
    for d in (0.15, 0.30, 0.60):
        b05 = cs.boot_boundaries([r for r in recsB if r["arm"] == arm
                                  and abs(r["delta"] - d) < 1e-9
                                  and abs(r["floor_eps"] - 0.05) < 1e-9],
                                 grids[d], seeds, idx)
        b25 = cs.boot_boundaries([r for r in recsB if r["arm"] == arm
                                  and abs(r["delta"] - d) < 1e-9
                                  and abs(r["floor_eps"] - 0.25) < 1e-9],
                                 grids[d], seeds, idx)
        ci, _ = cs.ci(b05 - b25)
        mid = 0.5 * (ci[0] + ci[1])
        pred = T[d][sk]
        Bx.errorbar([pred], [mid], yerr=[[mid - ci[0]], [ci[1] - mid]], fmt=mk,
                    ms=4.5, mfc="white", mec=col, ecolor=col, elinewidth=1.2,
                    mew=1.3, capsize=2)
lims = [0, 0.16]
Bx.plot(lims, lims, "k--", lw=1.0)
Bx.set_xlim(*lims); Bx.set_ylim(*lims)
Bx.set_xlabel("predicted floor shift"); Bx.set_ylabel("measured shift")
Bx.set_title("B  Matched floor intervention")
Bx.plot([], [], "o", mfc="white", mec=DB, label="constrained")
Bx.plot([], [], "s", mfc="white", mec=OR, label="unconstrained")
Bx.legend(fontsize=6.5, loc="upper left")
fig.savefig("out/e2_bridge_final.png", bbox_inches="tight")
fig.savefig("out/e2_bridge_final.pdf", bbox_inches="tight")
print("final figs done")
