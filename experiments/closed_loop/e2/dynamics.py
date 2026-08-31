"""Closed-loop two-timescale dynamics with intervention hooks.

Main arm (warm):   V_{t+1} = Update_m(V_t, occ(p_t));  u += eta_a * drive
where occ() is the training occupancy (identity / floor / replay) and
drive = gap - tau * u  (entropy regularization; tau=0 recovers NPG).
Interventions mirror the propositions exactly:
  floor:  occ(s) = eps + (1-2eps) s              (cor policy-floor)
  replay: occ(p) = lam * p + (1-lam) * pbar      (prop replay)
  entropy: drive = gap - tau * u                 (prop entropy-trap)
  ortho:  inner objective L_p + lam_o * sum g_ij (prop gram-threshold)
Cold-restart control re-draws V each outer step. Exact ODE = calibration only.
"""
import numpy as np
from . import model

U_CLIP = 60.0


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -U_CLIP, U_CLIP)))


def logit(p):
    return np.log(p / (1.0 - p))


def make_occ(floor_eps=0.0, replay=None):
    if replay is not None:
        lam, pbar = replay
        return lambda p: lam * p + (1.0 - lam) * pbar
    if floor_eps > 0.0:
        return lambda p: floor_eps + (1.0 - 2.0 * floor_eps) * p
    return lambda p: p


def run_closed_loop(delta, p0, *, dim=2, mode="warm", m=5, eta_V=0.05,
                    eta_a=0.05, T=4000, seed=0, record_every=10,
                    tau=0.0, floor_eps=0.0, replay=None, ortho_lam=0.0):
    rng = np.random.default_rng(seed)
    occ = make_occ(floor_eps, replay)
    V = model.random_V(dim, rng)
    u = logit(np.clip(p0, 1e-12, 1 - 1e-12))
    ps, gaps = [], []
    for t in range(T):
        p = sigmoid(u)
        if mode == "cold":
            V = model.random_V(dim, rng)
        p_train = occ(p)
        if mode == "resolve":
            V = model.random_V(dim, rng)
            for k in range(5000):
                V = model.riemannian_step(V, p_train, eta_V, ortho_lam)
                if k % 10 == 0:
                    g = model.euclidean_grad(V, p_train, ortho_lam)
                    g = g - V * (g * V).sum(axis=0, keepdims=True)
                    if (g ** 2).sum() ** 0.5 <= 1e-8:
                        break
        else:
            for _ in range(m):
                V = model.riemannian_step(V, p_train, eta_V, ortho_lam)
        gap = model.deployed_gap(V, delta)
        u = np.clip(u + eta_a * (gap - tau * u), -U_CLIP, U_CLIP)
        if t % record_every == 0 or t == T - 1:
            ps.append(sigmoid(u)); gaps.append(gap)
    p_end = sigmoid(u)
    g = model.gram_sq(V)
    return {
        "p": np.array(ps), "gap": np.array(gaps),
        "p_end": float(p_end), "basin": int(p_end > 0.5),
        "g_end": g.tolist(),
        "config": dict(delta=delta, p0=p0, dim=dim, mode=mode, m=m,
                       eta_V=eta_V, eta_a=eta_a, T=T, seed=seed, tau=tau,
                       floor_eps=floor_eps, replay=replay,
                       ortho_lam=ortho_lam),
    }


def run_exact_ode(delta, p0, *, dt=0.01, T=20000, tau=0.0, floor_eps=0.0,
                  replay=None):
    """Calibration: p' = p(1-p)[delta - D(occ(p)) + tau*log((1-p)/p)]."""
    occ = make_occ(floor_eps, replay)
    p = float(np.clip(p0, 1e-12, 1 - 1e-12))
    ps = [p]
    for _ in range(T):
        ent = tau * (np.log(1.0 - p) - np.log(p)) if tau > 0 else 0.0
        p = p + dt * p * (1.0 - p) * (delta - model.D_analytic(occ(p)) + ent)
        p = min(max(p, 1e-15), 1 - 1e-15)
        ps.append(p)
    return np.array(ps)


def measure_separator(delta, run_fn, *, tol=1e-3, lo=0.02, hi=0.98):
    if run_fn(delta, lo)["basin"] == 1 or run_fn(delta, hi)["basin"] == 0:
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if run_fn(delta, mid)["basin"] == 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def is_bistable(delta, run_fn, split=0.5):
    a = run_fn(delta, 0.03)["p_end"]; b = run_fn(delta, 0.97)["p_end"]
    return abs(a - b) > 0.2, a, b
