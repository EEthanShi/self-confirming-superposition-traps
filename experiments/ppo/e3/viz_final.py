"""E3 final-cohort figure. Post-execution visualization only: reads the frozen
final report, computes nothing the gates do not already fix, and renders the
four-panel existence story. Palette identical to e2/viz.py and the paper
preamble: deepblue #245280 (escaped / high basin), deepred #A63E37 (trapped /
low basin), deepgreen #33785B (full-rank capacity control), softgray #F2F4F7.
Identity is never color-alone: every cluster carries a direct text label.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEEPBLUE, DEEPRED = "#245280", "#A63E37"
DEEPGREEN, SOFTGRAY = "#33785B", "#F2F4F7"
ORACLE = 0.16514          # analytic Bayes floor 2*sigma^2/(1+sigma^2)
RANK2_REF = 0.23511       # constructive rank-2 reference (occupied branch)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "figure.dpi": 300,
})


def _jitter(rng, n, w=0.10):
    return rng.uniform(-w, w, size=n)


def _mean_ci(v):
    v = np.asarray(v, dtype=float)
    m = v.mean()
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return m, h


def main(report_json, out_pdf):
    R = json.load(open(report_json))
    recs = R["records"]
    rng = np.random.default_rng(0)
    arms = {t: [r for r in recs if r["tag"] == t]
            for t in ("k2_low", "k2_high", "k64_low", "k64_high")}

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) endpoint occupancy by arm: bistability under rank 2, none at rank 64
    order = ["k2_low", "k2_high", "k64_low", "k64_high"]
    labels = ["rank 2\nlow start", "rank 2\nhigh start",
              "rank 64\nlow start", "rank 64\nhigh start"]
    for i, tag in enumerate(order):
        p = np.array([r["p_E_end"] for r in arms[tag]])
        cols = np.where(p < 0.5, DEEPRED, DEEPBLUE)
        ax_a.scatter(i + _jitter(rng, len(p)), p, s=9, c=cols, alpha=0.55,
                     linewidths=0)
        n_trap = int((p < 0.5).sum())
        ax_a.text(i, 1.06, f"{n_trap}/{len(p)} trapped", ha="center",
                  fontsize=7.5,
                  color=DEEPRED if n_trap else "0.45")
    ax_a.axhline(0.5, color="black", ls="--", lw=1.0)
    ax_a.text(3.42, 0.52, "separator", ha="right", fontsize=7.5)
    ax_a.axhline(0.05, color=DEEPRED, ls=":", lw=1.0)
    ax_a.text(3.42, 0.075, r"exploration floor $\epsilon$", ha="right",
              fontsize=7.5, color=DEEPRED)
    ax_a.set_xticks(range(4), labels)
    ax_a.set_ylim(-0.06, 1.13)
    ax_a.set_ylabel(r"final occupancy of $E$")
    ax_a.set_title("Endpoints across 100 seeds per arm", fontsize=9.5)

    # (b) endpoint-conditioned competence: occupied branch at the rank-2
    # reference, abandoned branch degraded; rank 64 near the oracle on both
    for tag, col in (("k64_low", DEEPGREEN), ("k64_high", DEEPGREEN)):
        s = np.array([r["mse_S"] for r in arms[tag]])
        e = np.array([r["mse_E"] for r in arms[tag]])
        ax_b.scatter(s, e, s=9, c=col, alpha=0.5, linewidths=0)
    for tag in ("k2_low", "k2_high"):
        for r in arms[tag]:
            col = DEEPRED if r["p_E_end"] < 0.5 else DEEPBLUE
            ax_b.scatter(r["mse_S"], r["mse_E"], s=9, c=col, alpha=0.5,
                         linewidths=0)
    for v, ls, lab in ((ORACLE, ":", "oracle"), (RANK2_REF, "--", "rank-2 ref")):
        ax_b.axvline(v, color="0.35", ls=ls, lw=0.9)
        ax_b.axhline(v, color="0.35", ls=ls, lw=0.9)
    ax_b.set_xlim(0.13, 0.82)
    ax_b.set_ylim(0.13, 0.82)
    ax_b.text(0.30, 0.68, "trapped:\nkeeps $S$, loses $E$", color=DEEPRED,
              fontsize=7.5)
    ax_b.text(0.52, 0.31, "escaped:\nkeeps $E$, loses $S$", color=DEEPBLUE,
              fontsize=7.5)
    ax_b.text(0.14, 0.245, "rank 64:\nkeeps both", color=DEEPGREEN,
              fontsize=7.5, va="bottom",
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                        pad=1.2))
    ax_b.text(ORACLE - 0.007, 0.79, "oracle", fontsize=7, color="0.35",
              rotation=90, va="top", ha="right")
    ax_b.text(RANK2_REF + 0.007, 0.40, "rank-2 ref", fontsize=7,
              color="0.35", rotation=90, va="bottom")
    ax_b.set_xlabel(r"reconstruction MSE on branch $S$")
    ax_b.set_ylabel(r"reconstruction MSE on branch $E$")
    ax_b.set_title("Competence is basin-conditional", fontsize=9.5)

    # (c) registered order gate: cross-talk on the E-exclusive pair (x12)
    # against the S-exclusive pair (x23); the collision sits on the pair
    # exclusive to whichever branch the run abandoned
    groups = [(DEEPRED, [r for t in ("k2_low", "k2_high") for r in arms[t]
                         if r["p_E_end"] < 0.5]),
              (DEEPBLUE, [r for t in ("k2_low", "k2_high") for r in arms[t]
                          if r["p_E_end"] >= 0.5]),
              (DEEPGREEN, [r for t in ("k64_low", "k64_high")
                           for r in arms[t]])]
    for col, rows in groups:
        x = np.array([r["xtalk"]["x12"] for r in rows])
        y = np.array([r["xtalk"]["x23"] for r in rows])
        ax_c.scatter(x, y, s=9, c=col, alpha=0.45, linewidths=0)
    lim = 0.82
    ax_c.plot([0, lim], [0, lim], color="0.35", ls="--", lw=0.9)
    ax_c.set_xlim(-0.02, lim)
    ax_c.set_ylim(-0.02, lim)
    ax_c.text(0.60, 0.10, "trapped:\n$E$ abandoned", color=DEEPRED,
              fontsize=7.5, ha="center")
    ax_c.text(0.16, 0.62, "escaped:\n$S$ abandoned", color=DEEPBLUE,
              fontsize=7.5, ha="center")
    ax_c.text(0.10, 0.115, "rank 64", color=DEEPGREEN, fontsize=7.5)
    ax_c.text(0.44, 0.475, "registered order boundary", color="0.35",
              fontsize=7, rotation=45, ha="center")
    ax_c.set_xlabel(r"cross-talk on the pair exclusive to $E$")
    ax_c.set_ylabel(r"cross-talk on the pair exclusive to $S$")
    ax_c.set_title("Cross-talk concentrates on the abandoned pair",
                   fontsize=9.5)

    # (d) deployed return: the rank-by-basin interaction (annotation values are
    # the frozen F7 bootstrap estimates; error bars here are normal-approx CIs)
    f7 = R["gates"]["F7"]
    xs = [0, 1]
    for kk, col, lab in ((2, "black", "rank 2"), (64, DEEPGREEN, "rank 64")):
        ms, hs = [], []
        for init in ("low", "high"):
            v = [r["ret"]["deterministic_return"]
                 for r in arms[f"k{kk}_{init}"]]
            m, h = _mean_ci(v)
            ms.append(m); hs.append(h)
        ax_d.errorbar(xs, ms, yerr=hs, color=col, marker="o", ms=4.5,
                      lw=1.4, capsize=3, label=lab)
    ax_d.annotate("", xy=(0.03, -0.248), xytext=(0.03, -0.023),
                  arrowprops=dict(arrowstyle="<->", color=DEEPRED, lw=1.1))
    ax_d.text(0.07, -0.14,
              "rank gap {:.3f}\nCI [{:.3f}, {:.3f}]".format(
                  f7["rank_gap_low"]["mean"], *f7["rank_gap_low"]["ci"]),
              fontsize=7.5, color=DEEPRED, va="center")
    ax_d.text(0.52, -0.235,
              "interaction (DiD) {:.3f}\nCI [{:.3f}, {:.3f}]".format(
                  f7["did"]["mean"], *f7["did"]["ci"]),
              fontsize=7.5, ha="center")
    ax_d.set_xticks(xs, ["low start", "high start"])
    ax_d.set_xlim(-0.25, 1.25)
    ax_d.set_ylabel("deployed return (deterministic)")
    ax_d.set_title("Return cost confined to rank 2, low start",
                   fontsize=9.5)
    ax_d.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    for ax, tag in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        t = ax.get_title()
        ax.set_title("")
        ax.set_title(f"({tag})  {t}", fontsize=9.5, loc="left")

    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"))
    print("wrote", out_pdf)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
