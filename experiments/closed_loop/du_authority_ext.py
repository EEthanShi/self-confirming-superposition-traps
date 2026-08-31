"""Append-only extension of du_authority to p in [0.20, 0.36] so the
delta=0.6 crossing gets an authority-grade target (gap in the original range,
disclosed; committed before block-A C3@0.6 analysis)."""
import json, time
import numpy as np, torch
import multiprocessing as mp
from du_authority import opt_from, multistart_point

torch.set_default_dtype(torch.float64)

def continuation(direction):
    ps = np.round(np.arange(0.20, 0.361, 0.005), 4)
    if direction == "down":
        ps = ps[::-1]
    torch.manual_seed(1)
    W = 0.5 * torch.randn(2, 3)
    rows = []
    for p in ps:
        L, d, W = opt_from(W, float(p), steps=3000)
        rows.append((float(p), L, d))
    return rows

if __name__ == "__main__":
    t0 = time.time()
    up = continuation("up"); down = continuation("down")
    dd = {p: d for p, _, d in down}
    hyst = max(abs(d - dd[p]) for p, _, d in up)
    with mp.get_context("spawn").Pool(3) as pool:
        ms = pool.map(multistart_point, [(0.32, 150), (0.34, 150), (0.36, 150)])
    tab = sorted((p, d) for p, _, d in up)
    ps = np.array([x for x, _ in tab]); ds = np.array([y for _, y in tab])
    p_sep = None
    for i in range(len(ps) - 1):
        if ds[i] >= 0.6 > ds[i + 1]:
            t = (0.6 - ds[i + 1]) / (ds[i] - ds[i + 1])
            p_sep = float(ps[i + 1] - t * (ps[i + 1] - ps[i]))
    out = dict(tag="du_authority extension p in [0.20,0.36] (append-only)",
               hysteresis_max=hyst, multistart=ms,
               continuation_table=[dict(p=float(p), D_u=float(d)) for p, d in tab],
               p_sep_u_delta06=p_sep, elapsed_s=time.time() - t0)
    json.dump(out, open("out/du_authority_ext.json", "w"))
    print(f"hyst={hyst:.2e}  p_sep_u(0.6)={p_sep:.4f}")
    for m in ms:
        print(f"p={m['p']}: frac_at_best={m['frac_at_best']:.2f} D={m['best_D']:+.4f}")
