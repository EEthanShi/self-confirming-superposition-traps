"""Figures for E2. Palette matches the paper preamble exactly:
deepblue #245280, deepred #A63E37, deepgreen #33785B, softgray #F2F4F7.
High basin (p -> 1) = deepblue; low basin (the trap, p -> 0) = deepred.
Regions carry direct text labels, so identity is never color-alone.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DEEPBLUE, DEEPRED = "#245280", "#A63E37"
DEEPGREEN, SOFTGRAY = "#33785B", "#F2F4F7"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 300,
})


def fig_basin_map(grid_json, traj_json, out_png):
    G = json.load(open(grid_json)); Tj = json.load(open(traj_json))
    recs = G["records"]
    deltas = sorted({r["delta"] for r in recs})
    p0s = sorted({r["p0"] for r in recs})
    M = np.full((len(p0s), len(deltas)), np.nan)
    for r in recs:
        M[p0s.index(r["p0"]), deltas.index(r["delta"])] = r["basin"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                                  constrained_layout=True)
    cmap = ListedColormap([DEEPRED, DEEPBLUE])
    ax.pcolormesh(deltas, p0s, M, cmap=cmap, alpha=0.42, shading="nearest")
    ds = np.linspace(min(deltas), max(deltas), 400)
    ax.plot(ds, [__import__("e2.model", fromlist=["m"]).p_delta(d) for d in ds],
            "k--", lw=1.4, label=r"analytic separator $p_\delta$")
    ms = [(d, s) for d, s in G["separators"].items() if np.isfinite(s)]
    ax.plot([float(d) for d, _ in ms], [s for _, s in ms], "o", ms=4.5,
            mfc="white", mec="black", mew=0.9,
            label="measured separator (bisection)")
    ax.text(0.30, 0.12, "low basin (trap)", color=DEEPRED, fontsize=8.5,
            fontweight="bold")
    ax.text(0.30, 0.90, "high basin", color=DEEPBLUE, fontsize=8.5,
            fontweight="bold")
    ax.set_xlabel(r"task preference $\delta$")
    ax.set_ylabel(r"initial policy $p_0$")
    ax.set_title("Closed-loop basins (warm-start)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    pd = Tj["p_delta"]
    steps = None
    for tr in Tj["trajs"]:
        p = np.array(tr["p"]); x = np.arange(len(p))
        steps = x
        ax2.plot(x, p, lw=1.1,
                 color=DEEPBLUE if tr["basin"] == 1 else DEEPRED, alpha=0.85)
    ax2.axhline(pd, color="black", ls="--", lw=1.2)
    ax2.text(steps[-1] * 0.99, pd + 0.02, r"$p_\delta$", ha="right", fontsize=8.5)
    ax2.set_xlabel("outer step (recorded every 10)")
    ax2.set_ylabel(r"$p_t$")
    ax2.set_title(rf"Trajectories at $\delta={Tj['delta']}$", fontsize=9.5)
    ax2.set_ylim(-0.02, 1.02)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    return out_png
