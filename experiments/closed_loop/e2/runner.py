"""Grid runner for the closed-loop experiment. All runs happen on the server.

Development runs are tagged phase="dev" in every record and are never evidence.
"""
import json, os, time
from multiprocessing import Pool
import numpy as np
from . import model
from .dynamics import run_closed_loop, measure_separator


def _one(args):
    delta, p0, kw = args
    r = run_closed_loop(delta, p0, **kw)
    return {"delta": delta, "p0": p0, "p_end": r["p_end"],
            "basin": r["basin"], **{k: v for k, v in r["config"].items()
                                    if k not in ("delta", "p0")}}


def basin_grid(deltas, p0s, out_path, phase="dev", workers=24, **kw):
    jobs = [(float(d), float(p), kw) for d in deltas for p in p0s]
    t0 = time.time()
    with Pool(workers) as pool:
        recs = pool.map(_one, jobs)
    seps = {}
    for d in deltas:
        run_fn = lambda dd, pp: run_closed_loop(dd, pp, **kw)
        seps[float(d)] = measure_separator(float(d), run_fn)
    out = {"phase": phase, "kw": {k: str(v) for k, v in kw.items()},
           "elapsed_s": time.time() - t0, "records": recs,
           "separators": seps,
           "analytic_separators": {float(d): model.p_delta(float(d))
                                   for d in deltas if -1 < d < 1}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out


def sample_trajectories(delta, p0s, out_path, phase="dev", **kw):
    trajs = []
    for p0 in p0s:
        r = run_closed_loop(delta, float(p0), **kw)
        trajs.append({"p0": float(p0), "p": r["p"].tolist(),
                      "basin": r["basin"]})
    out = {"phase": phase, "delta": delta, "trajs": trajs,
           "p_delta": model.p_delta(delta)}
    with open(out_path, "w") as f:
        json.dump(out, f)
    return out
