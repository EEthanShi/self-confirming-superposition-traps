"""Generates the unconstrained-class population predictions used in
results_final/posthoc_unconstrained_sep.json (previously run as an ad-hoc
remote snippet; committed here to close the reproducibility gap flagged by
external review). Exact quantities from M = W^T W; Adam multi-start."""
import torch, json, sys


def pop_loss(W, p, s2):
    M = W.T @ W; wn = (W ** 2).sum(0)
    def pair(i, j):
        return ((M[i, i] - 1) ** 2 + M[i, j] ** 2 + s2 * wn[i]
                + (M[j, j] - 1) ** 2 + M[j, i] ** 2 + s2 * wn[j])
    dE = 0.5 * (pair(0, 1) + pair(0, 2))
    dS = 0.5 * (pair(0, 2) + pair(1, 2))
    return p * dE + (1 - p) * dS, dE, dS


def D_u(p, s2=0.01, trials=12, steps=8000, lr=0.02):
    best = None
    for t in range(trials):
        torch.manual_seed(t * 13 + 1)
        W = (0.5 * torch.randn(2, 3)).requires_grad_(True)
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(steps):
            L, _, _ = pop_loss(W, p, s2)
            opt.zero_grad(); L.backward(); opt.step()
        L, dE, dS = pop_loss(W, p, s2)
        if best is None or L.item() < best[0]:
            best = (L.item(), (dE - dS).item())
    return best[1]


def separator(delta, iters=18):
    lo, hi = 0.05, 0.95
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if D_u(mid) > delta else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    out = []
    for d in [float(x) for x in (sys.argv[1:] or [0.15, 0.3, 0.45, 0.6])]:
        ps = separator(d)
        out.append(dict(delta=d, p_sep_unconstrained=round(ps, 4),
                        floored_boundary=round((ps - 0.05) / 0.9, 4)))
        print(out[-1])
    json.dump(out, open("out/unconstrained_sep_regen.json", "w"))
