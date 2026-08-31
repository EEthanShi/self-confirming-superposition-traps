"""FINAL core runs under E2_FROZEN.md. Evidence-grade."""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from e2 import model
from e2.dynamics import run_closed_loop, measure_separator

DELTAS = np.round(np.linspace(0.05, 0.95, 19), 4)
P0S = np.round(np.linspace(0.02, 0.98, 41), 4)
BASE = dict(eta_V=0.05, eta_a=0.05, T=8000)
t0 = time.time()

def cell(args):
    d, p0, kw = args
    r = run_closed_loop(d, p0, **kw)
    return dict(delta=d, p0=p0, p_end=r["p_end"], basin=r["basin"], **{
        k: kw[k] for k in ("m", "mode", "dim") if k in kw})

grids = {}
for name, kw in [("warm_m20", dict(m=20, **BASE)),
                 ("cold_m20", dict(m=20, mode="cold", **BASE)),
                 ("dim3_m20", dict(m=20, dim=3, **BASE))]:
    jobs = [(float(d), float(p), kw) for d in DELTAS for p in P0S]
    with Pool(22) as pool:
        grids[name] = pool.map(cell, jobs)
    print(name, "done", round(time.time()-t0), "s", flush=True)

def sep(args):
    d, m = args
    fn = lambda dd, pp: run_closed_loop(dd, pp, m=m, **BASE)
    return dict(delta=float(d), m=int(m), sep=measure_separator(float(d), fn),
                analytic=model.p_delta(float(d)))

jobs = [(d, m) for m in (1, 5, 20, 80) for d in DELTAS]
with Pool(22) as pool:
    seps = pool.map(sep, jobs)
json.dump({"phase": "final", "frozen": "E2_FROZEN.md", "grids": grids,
           "separators": seps, "elapsed_s": time.time()-t0},
          open("out/final_core.json", "w"))
print("FINAL CORE DONE", round(time.time()-t0), "s", flush=True)
