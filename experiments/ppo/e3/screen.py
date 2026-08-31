"""OUTCOME-INFORMED DEVELOPMENT SCREEN (ninth external review authorization:
OLD_PILOT_FAIL_RETAINED / NEW_DEVELOPMENT_ALLOWED / FINAL_HOLD).

Goal: raise Pr(trap | q0 low) while keeping Pr(escape | q0 high), k64 clean,
class-referenced competence, and mediator directions. Dev seeds 5100+ (never
used); full p_E(t) trajectories saved; no gates; results are development
evidence only. Final seeds 6000+ remain sealed.
"""
import json, os, time, socket
import numpy as np
from multiprocessing import Pool

SIGMA, EPS = 0.30, 0.05
CONFIGS = [dict(delta=d, rls=r, q0=q)
           for d in (0.15, 0.30) for r in (1.0, 0.3, 0.1)
           for q in (0.05, 0.10)]
LOW_SEEDS = list(range(5100, 5110))
HIGH_SEEDS = list(range(5110, 5115))
CTRL_SEEDS = list(range(5115, 5118))


def _pin():
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def cell(args):
    _pin()
    from .ppo import run_training
    from . import evaluators as ev
    from . import env as E
    cfg, seed, k, q0 = args
    net, logs, S = run_training(k_proj=k, delta=cfg["delta"], seed=seed,
                                p0=q0, updates=400, sigma_obs=SIGMA,
                                floor_eps=EPS, root_lr_scale=cfg["rls"],
                                log_every=1)
    comp = ev.competence(net, S["eval"], sigma_obs=SIGMA)
    ref = E.rank2_reference_mse(SIGMA)
    return dict(cfg=cfg, seed=seed, k=k, q0=q0,
                p_E_traj=[round(l["p_E"], 4) for l in logs],
                p_E_end=logs[-1]["p_E"],
                comp_S=comp["S"]["mse"], comp_E=comp["E"]["mse"],
                class_ratio=ref / max(min(comp["S"]["mse"],
                                          comp["E"]["mse"]), 1e-9),
                xtalk=ev.crosstalk(net, S["eval"], sigma_obs=SIGMA),
                gap=ev.forced_gap(net, S["eval"], sigma_obs=SIGMA,
                                  delta=cfg["delta"]))


if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for cfg in CONFIGS:
        for s in LOW_SEEDS:
            jobs.append((cfg, s, 2, cfg["q0"]))
        for s in HIGH_SEEDS:
            jobs.append((cfg, s, 2, 0.9))
        for s in CTRL_SEEDS:
            jobs.append((cfg, s, 64, cfg["q0"]))
    print("dev screen jobs:", len(jobs), flush=True)
    with Pool(20) as pool:
        recs = pool.map(cell, jobs)
    out = dict(tag="OUTCOME-INFORMED DEVELOPMENT SCREEN (no gates)",
               records=recs,
               receipt=dict(host=socket.gethostname(),
                            started_unix=t0, finished_unix=time.time(),
                            n_jobs=len(jobs)))
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/e3_dev_screen.json", "w"))
    for cfg in CONFIGS:
        low = [r for r in recs if r["cfg"] == cfg and r["k"] == 2
               and r["q0"] == cfg["q0"]]
        high = [r for r in recs if r["cfg"] == cfg and r["k"] == 2
                and r["q0"] == 0.9]
        ctrl = [r for r in recs if r["cfg"] == cfg and r["k"] == 64]
        tr = np.mean([r["p_E_end"] < 0.5 for r in low])
        esc = np.mean([r["p_E_end"] > 0.5 for r in high])
        ct = np.mean([r["p_E_end"] < 0.5 for r in ctrl])
        cr = np.mean([r["class_ratio"] for r in low])
        med = np.mean([r["xtalk"]["x12"] > r["xtalk"]["x23"]
                       for r in low if r["p_E_end"] < 0.5]) \
            if any(r["p_E_end"] < 0.5 for r in low) else float("nan")
        print(f"d={cfg['delta']} rls={cfg['rls']} q0={cfg['q0']}: "
              f"trap_low={tr:.2f} escape_high={esc:.2f} k64_trap={ct:.2f} "
              f"class_ratio={cr:.2f} med_dir={med:.2f}", flush=True)
