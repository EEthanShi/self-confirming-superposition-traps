"""FINAL intervention runs under E2_FROZEN.md. Evidence-grade."""
import sys, json, time
import numpy as np
sys.path.insert(0, ".")
from e2 import model
from e2.dynamics import run_closed_loop, measure_separator, is_bistable

BASE = dict(eta_V=0.05, eta_a=0.05, T=8000)
t0 = time.time()
out = {"phase": "final", "frozen": "E2_FROZEN.md"}

def bisect_knob(f, lo, hi, tol=5e-3):
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        if f(mid): lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

out["floor"] = []
for eps in (0.05, 0.1, 0.15, 0.2):
    fn = lambda dd, pp: run_closed_loop(dd, pp, m=20, floor_eps=eps, **BASE)
    out["floor"].append(dict(delta=0.3, eps=eps, m=20,
        sep=measure_separator(0.3, fn),
        pred=(model.p_delta(0.3)-eps)/(1-2*eps)))
print("floor done", round(time.time()-t0), flush=True)

out["entropy"] = []
for d in (0.2, 0.5, 0.8):
    def bist(tau):
        fn = lambda dd, pp: run_closed_loop(dd, pp, m=20, tau=tau, **BASE)
        return is_bistable(d, fn)[0]
    row = dict(delta=d, m=20, tau_star=bisect_knob(bist, 0.1, 3.0),
               bracket=[model.entropy_bistable_upper(d), 2.5])
    lo = run_closed_loop(d, 0.03, m=20, tau=0.8, **BASE)["p_end"]
    hi = run_closed_loop(d, 0.97, m=20, tau=0.8, **BASE)["p_end"]
    pL, pH = model.entropy_sinks(d, 0.8)
    row.update(sink_low=[lo, pL], sink_high=[hi, pH])
    phase_grid = []
    for tau in np.round(np.arange(0.2, 3.01, 0.2), 2):
        fn = lambda dd, pp: run_closed_loop(dd, pp, m=20, tau=float(tau), **BASE)
        phase_grid.append([float(tau), int(is_bistable(d, fn)[0])])
    row["phase_grid"] = phase_grid
    out["entropy"].append(row)
print("entropy done", round(time.time()-t0), flush=True)

out["ortho"] = []
for d in (0.3, 0.6):
    row = dict(delta=d, lam_crit=model.lambda_crit(d), lam_star={})
    for m in (5, 20, 80):
        def lowexists(lam):
            return run_closed_loop(d, 0.03, m=m, ortho_lam=lam, **BASE)["basin"] == 0
        row["lam_star"][m] = bisect_knob(lowexists, 0.2, 4.0)
    rl = run_closed_loop(d, 0.03, m=80, ortho_lam=1.5, **BASE)
    rh = run_closed_loop(d, 0.97, m=80, ortho_lam=1.5, **BASE)
    row["endpoint_gap"] = dict(low=[d - rl["gap"][-1], model.s_lambda(1.5)],
                               high=[d - rh["gap"][-1], -model.s_lambda(1.5)])
    out["ortho"].append(row)
print("ortho done", round(time.time()-t0), flush=True)

out["replay"] = []
for pbar in (0.2, 0.5):
    d = 0.3; pd = model.p_delta(d)
    pred = 1 - pd/pbar if pbar > pd else (pd - pbar)/(1 - pbar)
    row = dict(delta=d, pbar=pbar, lam_pred=pred, lam_star={})
    for m in (20, 80):
        def notbist(lam):
            fn = lambda dd, pp: run_closed_loop(dd, pp, m=m, replay=(lam, pbar), **BASE)
            return not is_bistable(d, fn)[0]
        row["lam_star"][m] = bisect_knob(notbist, 0.02, 0.95)
    out["replay"].append(row)
out["elapsed_s"] = time.time()-t0
json.dump(out, open("out/final_interventions.json", "w"))
print("FINAL INTERVENTIONS DONE", round(time.time()-t0), "s", flush=True)
