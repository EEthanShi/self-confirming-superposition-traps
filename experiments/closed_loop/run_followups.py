"""Follow-ups frozen in E2_FROZEN_ADDENDUM.md (committed before this run)."""
import sys, json, time
import numpy as np
sys.path.insert(0, ".")
from multiprocessing import Pool
from e2 import model
from e2.dynamics import run_closed_loop, measure_separator

t0 = time.time()

# --- G6' ---
LAMS = {0.3: [1.5, 2.0, 2.3, 2.5, 2.65, 2.72, 2.77, 2.785],
        0.6: [0.8, 1.1, 1.3, 1.42, 1.48, 1.51]}
def g6p(args):
    d, lam = args
    fn = lambda dd, pp: run_closed_loop(dd, pp, m=80, ortho_lam=lam,
                                        eta_V=0.05, eta_a=0.05, T=8000)
    return dict(delta=d, lam=lam, s=measure_separator(d, fn, lo=0.002, hi=0.6))
jobs = [(d, l) for d, ls in LAMS.items() for l in ls]
with Pool(14) as pool:
    g6 = pool.map(g6p, jobs)
json.dump({"frozen": "E2_FROZEN_ADDENDUM.md", "rows": g6,
           "lam_crit": {d: model.lambda_crit(d) for d in LAMS}},
          open("out/g6prime.json", "w"))
print("G6' done", round(time.time()-t0), flush=True)

# --- repeated-solve control ---
def rs(args):
    d, p0 = args
    r = run_closed_loop(d, p0, mode="resolve", eta_V=0.05, eta_a=0.05, T=4000)
    return dict(delta=d, p0=p0, p_end=r["p_end"], basin=r["basin"])
jobs = [(d, float(p)) for d in (0.3, 0.6)
        for p in np.round(np.linspace(0.02, 0.98, 21), 4)]
with Pool(22) as pool:
    rows = pool.map(rs, jobs)
json.dump({"frozen": "E2_FROZEN_ADDENDUM.md", "rows": rows},
          open("out/repeated_solve.json", "w"))
print("repeated-solve done", round(time.time()-t0), flush=True)
