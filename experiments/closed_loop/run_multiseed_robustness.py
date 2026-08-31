"""Development robustness check (code NOT pre-frozen; labeled as such):
does the main-E2 separator depend on the representation init seed?
10 V-init seeds x {m=20, m=80} x 5 deltas, frozen-core hyperparameters."""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
from e2 import model
from e2.dynamics import run_closed_loop, measure_separator

def sep(args):
    d, m, seed = args
    fn = lambda dd, pp: run_closed_loop(dd, pp, m=m, seed=seed,
                                        eta_V=0.05, eta_a=0.05, T=8000)
    return dict(delta=d, m=m, seed=seed, sep=measure_separator(d, fn),
                analytic=model.p_delta(d))

t0 = time.time()
jobs = [(d, m, s) for d in (0.15, 0.3, 0.45, 0.6, 0.8)
        for m in (20, 80) for s in range(1, 11)]
with Pool(22) as pool:
    rows = pool.map(sep, jobs)
json.dump({"tag": "development robustness check, code not pre-frozen",
           "rows": rows, "elapsed_s": time.time() - t0},
          open("out/multiseed_robustness.json", "w"))
print("MULTISEED DONE", round(time.time() - t0), flush=True)
