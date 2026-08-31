#!/usr/bin/env python3
"""Block B gate table, mechanical grading, rebuilt from raw only."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from e2 import cohort_stats as cs

R = os.path.join(H, "..", "results_final")
blk = json.load(open(os.path.join(R, "cohort_blockB.json")))
tg = json.load(open(os.path.join(H, "cohort_targets_B.json")))
grids = {float(k): np.array(v) for k, v in tg["grids"].items()}
T = {r["delta"]: r for r in tg["rows"]}
recs = blk["records"]
seeds = sorted({r["seed"] for r in recs})
idx = cs.boot_indices(len(seeds))
L = [f"records {len(recs)}, failed {sum(r['failed'] for r in recs)}"]
tr = [r for r in recs if r["arm"] == "fullrank3"]
f = np.mean([r["basin"] == 0 for r in tr])
L.append(f"B1 capacity control trapped {f:.4f} -> {'PASS' if f <= 0.005 else 'FAIL'}")
cache = {}
for d in (0.15, 0.30, 0.60):
    g = grids[d]
    for arm in ("constr2", "unconstr2"):
        m = 0.02 if arm == "constr2" else 0.03
        for eps in (0.05, 0.25):
            sel = [r for r in recs if r["arm"] == arm
                   and abs(r["delta"] - d) < 1e-9
                   and abs(r["floor_eps"] - eps) < 1e-9]
            bs = cs.boot_boundaries(sel, g, seeds, idx)
            cache[(d, arm, eps)] = bs
            b = cs.boundary(cs.frac_curve(sel, g), g)
            pk = ("c" if arm == "constr2" else "u") + ("05" if eps == 0.05 else "25")
            pred = T[d][pk]
            ci, bad = cs.ci(bs)
            gate = "B2" if arm == "constr2" else "B3"
            if bad > 0.05 or not np.isfinite(b):
                L.append(f"{gate} {arm} d={d} e={eps}: UNRESOLVED "
                         f"({bad:.0%} non-finite) -> FAIL")
                continue
            lo, hi = ci[0] - pred, ci[1] - pred
            ok = -m <= lo and hi <= m
            L.append(f"{gate} {arm} d={d} e={eps}: b={b:.4f} err CI"
                     f" [{lo:+.4f},{hi:+.4f}] margin ±{m} -> "
                     f"{'PASS' if ok else 'FAIL'}")
    for arm, sk in (("constr2", "shift_c"), ("unconstr2", "shift_u")):
        diff = cache[(d, arm, 0.05)] - cache[(d, arm, 0.25)]
        ci, bad = cs.ci(diff)
        pred = T[d][sk]
        if bad > 0.05:
            L.append(f"B4 {arm} d={d}: UNRESOLVED ({bad:.0%}) -> FAIL")
            continue
        ok = ci[0] > 0
        L.append(f"B4 {arm} d={d}: shift CI [{ci[0]:.4f},{ci[1]:.4f}] "
                 f"(pred {pred}, secondary) -> {'PASS' if ok else 'FAIL'}")
def chain(sel):
    trap = [r for r in sel if r["basin"] == 0]
    esc = [r for r in sel if r["basin"] == 1]
    return trap, esc


def seed_cluster_diff_ci(trap, esc, seeds, nboot=2000, seed0=779):
    """Cluster bootstrap over SEEDS (the experimental unit), vectorized."""
    rng = np.random.Generator(np.random.PCG64(seed0))
    n = len(seeds)
    smap = {s: i for i, s in enumerate(seeds)}
    def agg(rs):
        su = np.zeros(n); ct = np.zeros(n)
        for r in rs:
            i = smap[r["seed"]]; su[i] += r["ret_balanced"]; ct[i] += 1
        return su, ct
    ts, tc = agg(trap); es, ec = agg(esc)
    idxm = rng.integers(0, n, size=(nboot, n))
    tS = ts[idxm].sum(1); tC = tc[idxm].sum(1)
    eS = es[idxm].sum(1); eC = ec[idxm].sum(1)
    ok = (tC >= 10) & (eC >= 10)
    if ok.sum() < nboot * 0.5:
        return None
    d = tS[ok] / tC[ok] - eS[ok] / eC[ok]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
d2 = [r for r in recs if r["arm"] in ("constr2", "unconstr2") and not r["failed"]]
trap, esc = chain(d2)
c5t = np.mean([r["g12"] > r["g23"] for r in trap])
c5e = np.mean([r["g23"] > r["g12"] for r in esc])
L.append(f"B5 mediator: trapped g12>g23 {c5t:.3f}, escaped g23>g12 {c5e:.3f}"
         f" -> {'PASS' if c5t >= 0.95 and c5e >= 0.95 else 'FAIL'}")
c6t = np.mean([r["gap_forced"] < 0 for r in trap])
c6e = np.mean([r["gap_forced"] > 0 for r in esc])
L.append(f"B6 gap link: trapped gap<0 {c6t:.3f}, escaped gap>0 {c6e:.3f}"
         f" -> {'PASS' if c6t >= 0.95 and c6e >= 0.95 else 'FAIL'}")
rng = np.random.Generator(np.random.PCG64(777))
for d in (0.15, 0.30, 0.60):
    for arm in ("constr2", "unconstr2"):
        sel = [r for r in d2 if abs(r["delta"] - d) < 1e-9 and r["arm"] == arm]
        t_, e_ = chain(sel)
        if len(t_) < 10 or len(e_) < 10:
            L.append(f"B7 {arm} d={d}: N/A (groups {len(t_)}/{len(e_)})")
            continue
        rt = np.array([r["ret_balanced"] for r in t_])
        re_ = np.array([r["ret_balanced"] for r in e_])
        bs = [rt[rng.integers(0, len(rt), len(rt))].mean()
              - re_[rng.integers(0, len(re_), len(re_))].mean()
              for _ in range(2000)]
        hi = float(np.percentile(bs, 97.5))
        L.append(f"B7 {arm} d={d}: ret(trap)-ret(escape) CI hi {hi:+.4f}"
                 f" -> {'PASS' if hi < 0 else 'FAIL'}")
L.append("--- stratified re-report (delta x arm x eps, seed-clustered; "
         "supersedes pooled B5-B7 for interpretation, registered verdicts kept) ---")
for d in (0.15, 0.30, 0.60):
    for arm in ("constr2", "unconstr2"):
        for eps in (0.05, 0.25):
            sel = [r for r in d2 if abs(r["delta"] - d) < 1e-9
                   and r["arm"] == arm and abs(r["floor_eps"] - eps) < 1e-9]
            t_, e_ = chain(sel)
            b5t = np.mean([r["g12"] > r["g23"] for r in t_]) if t_ else float("nan")
            b6t = np.mean([r["gap_forced"] < 0 for r in t_]) if t_ else float("nan")
            b5e = np.mean([r["g23"] > r["g12"] for r in e_]) if e_ else float("nan")
            b6e = np.mean([r["gap_forced"] > 0 for r in e_]) if e_ else float("nan")
            ci = seed_cluster_diff_ci(t_, e_, seeds)
            b7 = ("n/a" if ci is None
                  else f"ret diff CI [{ci[0]:+.4f},{ci[1]:+.4f}]")
            L.append(f"  {arm} d={d} e={eps}: n_trap={len(t_)}/n_esc={len(e_)} "
                     f"B5 trap {b5t:.3f}/esc {b5e:.3f}  "
                     f"B6 trap {b6t:.3f}/esc {b6e:.3f}  B7 {b7}")
out = "\n".join(L)
open(os.path.join(R, "..", "cohort_B_gate_table.md"), "w").write(
    "# Block B gate table (rebuilt from raw)\n\n" + out + "\n")
print(out)
