"""FINAL E2c neural runs under E2_FROZEN.md (GPU). Observed tier."""
import sys, json, time, torch
sys.path.insert(0, ".")
from e2.neural import run_batch
from e2 import model

t0 = time.time()
P0 = torch.linspace(0.05, 0.95, 21)
SEEDS = 30
res = {"phase": "final", "frozen": "E2_FROZEN.md", "cells": []}
for delta in (0.15, 0.3, 0.45, 0.6):
    p0s = P0.repeat_interleave(SEEDS)          # 630 runs per (dim, delta)
    for dim in (2, 3):                          # CRN: same seed per delta
        r = run_batch(delta, p0s, runs=len(p0s), dim=dim, T=3000, K=256,
                      eta_V=0.02, eta_a=0.02, ema=0.98, floor_eps=0.05,
                      burn_in=200, seed=1000 + int(delta*1000),
                      device="cuda")
        res["cells"].append(dict(delta=delta, dim=dim, p_end=r["p_end"],
                                 p0=p0s.tolist(),
                                 pred_sep=(model.p_delta(delta)-0.05)/0.9))
        print(f"delta={delta} dim={dim} done {time.time()-t0:.0f}s", flush=True)
res["elapsed_s"] = time.time()-t0
json.dump(res, open("out/final_neural.json", "w"))
print("FINAL NEURAL DONE", round(time.time()-t0), "s", flush=True)
