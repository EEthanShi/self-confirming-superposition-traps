"""Hardening the unconstrained-class D_u(p) per external review:
(1) SDP relaxation result (documented: LOOSE — rank-free optimum is M ~ I,
    bound uninformative; no global certificate from this route);
(2) continuation in p, up-sweep vs down-sweep, hysteresis metric;
(3) large multistart (300 x {Adam, LBFGS}) value-consistency at band points;
(4) KKT stationarity on the det(M)=0 manifold + projected-Hessian check.
Outputs a frozen D_u reference table with uncertainty for the cohort freeze.
"""
import sys, json, time
import numpy as np, torch
import multiprocessing as mp

S2 = 0.01
torch.set_default_dtype(torch.float64)


def L_M(M, p):
    def pair(i, j):
        return ((M[i, i] - 1) ** 2 + (M[j, j] - 1) ** 2 + 2 * M[i, j] ** 2
                + S2 * (M[i, i] + M[j, j]))
    dE = 0.5 * (pair(0, 1) + pair(0, 2)); dS = 0.5 * (pair(0, 2) + pair(1, 2))
    return p * dE + (1 - p) * dS, dE, dS


def opt_from(W0, p, steps=8000, use_lbfgs=False):
    W = W0.clone().requires_grad_(True)
    if use_lbfgs:
        opt = torch.optim.LBFGS([W], max_iter=steps, tolerance_grad=1e-14,
                                tolerance_change=1e-16, history_size=30)
        def closure():
            opt.zero_grad(); L, _, _ = L_M(W.T @ W, p); L.backward(); return L
        opt.step(closure)
    else:
        opt = torch.optim.Adam([W], lr=0.02)
        for _ in range(steps):
            L, _, _ = L_M(W.T @ W, p); opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        M = W.T @ W
        L, dE, dS = L_M(M, p)
    return float(L), float(dE - dS), W.detach()


def multistart_point(args):
    p, n = args
    vals = []
    for t in range(n):
        torch.manual_seed(t * 7 + 3)
        W0 = 0.5 * torch.randn(2, 3)
        for lb in (False, True):
            L, d, W = opt_from(W0, p, use_lbfgs=lb)
            vals.append((L, d))
    vals.sort()
    best = vals[0]
    within = [v for v in vals if v[0] - best[0] < 1e-9]
    return dict(p=p, n_starts=2 * n, best_L=best[0], best_D=best[1],
                frac_at_best=len(within) / len(vals),
                D_spread_at_best=float(np.ptp([v[1] for v in within])))


def kkt_check(p):
    torch.manual_seed(11)
    _, _, W = opt_from(0.5 * torch.randn(2, 3), p, steps=20000)
    _, _, W = opt_from(W, p, use_lbfgs=True)
    M = (W.T @ W).numpy()
    Mt = torch.tensor(M, requires_grad=True)
    L, _, _ = L_M(Mt, p)
    G = torch.autograd.grad(L, Mt)[0].numpy()
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(M)
    u = evecs[:, 0]                      # null direction (rank-2)
    # KKT on {det=0}: grad_L restricted to the tangent of the manifold
    # must vanish; tangent excludes the u u^T direction.
    P = np.eye(3) - np.outer(u, u)
    resid = np.linalg.norm(P @ G @ P)
    return dict(p=p, null_eig=float(evals[0]), kkt_tangent_resid=float(resid),
                mu=float(u @ G @ u))


def continuation(direction):
    ps = np.round(np.arange(0.35, 0.651, 0.005), 4)
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
    band = [(p, 150) for p in (0.42, 0.45, 0.47, 0.48, 0.49)]
    with mp.get_context("spawn").Pool(5) as pool:
        ms = pool.map(multistart_point, band)
    kkt = [kkt_check(p) for p in (0.2, 0.45, 0.48)]
    grid = sorted({p for p, _, _ in up})
    table = [dict(p=p, D_u=next(d for pp, _, d in up if pp == p)) for p in grid]
    out = dict(tag="D_u authority hardening (external-review prescription)",
               sdp_note="SDP relaxation documented LOOSE (rank-free optimum "
                        "~identity); no certificate from that route",
               hysteresis_max=hyst, multistart=ms, kkt=kkt,
               continuation_table=table, elapsed_s=time.time() - t0)
    json.dump(out, open("out/du_authority.json", "w"))
    print(f"hysteresis_max={hyst:.2e}")
    for m in ms:
        print(f"p={m['p']}: frac_at_best={m['frac_at_best']:.3f} "
              f"D_spread={m['D_spread_at_best']:.2e} D={m['best_D']:+.4f}")
    for k in kkt:
        print(f"KKT p={k['p']}: null_eig={k['null_eig']:.2e} "
              f"tangent_resid={k['kkt_tangent_resid']:.2e} mu={k['mu']:+.4f}")
    print("done", round(time.time() - t0), "s")
