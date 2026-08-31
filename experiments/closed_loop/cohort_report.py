#!/usr/bin/env python3
"""One-command cohort gate table from raw block-A records + frozen targets +
du_authority(+ext). Also reports the outcome-informed exploratory band."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
from e2 import cohort_stats as cs

R = os.path.join(H, "..", "results_final")
blk = json.load(open(os.path.join(R, "cohort_blockA.json")))
tg = json.load(open(os.path.join(H, "cohort_targets.json")))
ext = json.load(open(os.path.join(R, "du_authority_ext.json")))
grids = {float(k): v for k, v in tg["grids"].items()}
targets = {r["delta"]: dict(r) for r in tg["rows"]}
pu6 = ext["p_sep_u_delta06"]
targets[0.6]["u05"] = round((pu6 - 0.05) / 0.9, 4)
targets[0.6]["u15"] = round((pu6 - 0.15) / 0.7, 4)
targets[0.6]["shift_u"] = round(targets[0.6]["u05"] - targets[0.6]["u15"], 4)

recs = blk["records"]
seeds = sorted({r["seed"] for r in recs})
idx = cs.boot_indices(len(seeds))
L = [f"failed runs: {sum(r['failed'] for r in recs)}/{len(recs)}"]
tr = [r for r in recs if r["arm"] == "fullrank3"]
f = np.mean([r["basin"] == 0 for r in tr])
L.append(f"C1 fullrank3 trapped {f:.4f} -> {'PASS' if f <= 0.01 else 'FAIL'}")
cache = {}
for d in (0.15, 0.30, 0.60):
    g = np.array(grids[d])
    for arm in ("constr2", "unconstr2"):
        for eps in (0.05, 0.15):
            sel = [r for r in recs if r["arm"] == arm
                   and abs(r["delta"] - d) < 1e-9
                   and abs(r["floor_eps"] - eps) < 1e-9]
            bs = cs.boot_boundaries(sel, g, seeds, idx)
            b = cs.boundary(cs.frac_curve(sel, g), g)
            cache[(d, arm, eps)] = bs
            pk = ("c" if arm == "constr2" else "u") + ("05" if eps == 0.05 else "15")
            pred = targets[d][pk]
            tol = 0.03 if arm == "constr2" else 0.04
            ci, badfrac = cs.ci(bs)
            gate = "C2" if arm == "constr2" else "C3"
            # post-execution targets (delta=0.6 unconstr, filled from
            # du_authority_ext AFTER block A ran) are development, mechanically
            dev = (arm == "unconstr2" and d == 0.60)
            if badfrac > 0.05 or not np.isfinite(b):
                L.append(f"{gate} {arm} d={d} eps={eps}: UNRESOLVED "
                         f"(grid-edge; {badfrac:.0%} non-finite bootstrap)")
                continue
            ok = (ci[0] - tol) <= pred <= (ci[1] + tol)
            lab = "DEV-CHECK" if dev else ("PASS" if ok else "FAIL")
            if dev:
                lab += " (consistent)" if ok else " (inconsistent)"
            L.append(f"{gate} {arm} d={d} eps={eps}: b={b:.4f}"
                     f" CI[{ci[0]:.4f},{ci[1]:.4f}] pred {pred} -> {lab}")
    for arm, sk in (("constr2", "shift_c"), ("unconstr2", "shift_u")):
        diff = cache[(d, arm, 0.05)] - cache[(d, arm, 0.15)]
        ci, badfrac = cs.ci(diff)
        pred = targets[d][sk]
        dev = (arm == "unconstr2" and d == 0.60)
        direction = "direction+" if ci[0] > 0 else "direction?"
        star = "" if ci[0] <= pred <= ci[1] else " [CI excludes point pred]"
        lab = "DEV-CHECK" if dev else "DIRECTION-ONLY GATE (frozen tol wider than signal)"
        L.append(f"C4 {arm} d={d}: shift CI[{ci[0]:.4f},{ci[1]:.4f}] pred {pred}"
                 f" -> {lab} {direction}{star}")
exp = json.load(open(os.path.join(R, "cohort_exploratory.json")))["records"]
gb = np.round(np.linspace(0.15, 0.55, 11), 4)
L.append("exploratory band (outcome-informed, no gates):")
for d in (0.40, 0.45, 0.50, 0.55):
    row = []
    for arm in ("constr2", "unconstr2", "fullrank3"):
        sel = [r for r in exp if r["arm"] == arm and abs(r["delta"] - d) < 1e-9]
        b = cs.boundary(cs.frac_curve(sel, gb), gb)
        row.append(f"{arm}={b if np.isfinite(b) else 'edge'}")
    L.append(f"  d={d}: " + "  ".join(
        x if isinstance(x, str) else f"{x:.3f}" for x in row))
out = "\n".join(str(x) for x in L)
open(os.path.join(R, "..", "cohort_gate_table.md"), "w").write(
    "# Cohort gate table (rebuilt from raw)\n\n" + out + "\n")
print(out)
