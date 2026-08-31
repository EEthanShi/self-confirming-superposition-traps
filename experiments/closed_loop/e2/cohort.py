"""Medium cohort: three representation arms, five bound RNG streams, CRN.

Only the representation constraint differs across arms; per (delta_idx,
p0_idx, seed) every arm consumes identical draws from identical streams:
  init   W0 raw normals (3x3 drawn once; 2D arms use the first two rows)
  data   pair pick + feature values Z
  noise  encoder noise, always drawn K x 3, sliced to the arm's dim
  action behavior-branch uniforms (thresholded by the arm's own p: CRN)
  eval   fresh evaluation episodes at the final policy
Deterministic given (arm, delta, p0, indices, seed): pure numpy PCG64.
"""
import numpy as np

PAIRS = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
ARMS = ("constr2", "unconstr2", "fullrank3")


def streams(delta_idx, p0_idx, seed):
    base = (int(seed) * 1_000_003 + int(delta_idx) * 1009 + int(p0_idx)) * 8
    return [np.random.Generator(np.random.PCG64(base + k)) for k in range(5)]


def init_W(arm, init_rng):
    R = 0.5 * init_rng.standard_normal((3, 3))
    if arm == "fullrank3":
        return R.copy()
    W = R[:2, :].copy()
    if arm == "constr2":
        W /= np.linalg.norm(W, axis=0, keepdims=True)
    return W


def draw_batch(pb, K, action_rng, data_rng, noise_rng):
    U = action_rng.random(K)
    B = U < pb
    pick = data_rng.random(K) < 0.5
    sel = np.stack([B & pick, B & ~pick, ~B & pick, ~B & ~pick], axis=1)
    mask = sel.astype(float) @ PAIRS
    Z = data_rng.standard_normal((K, 3))
    N3 = noise_rng.standard_normal((K, 3))
    return B, mask, Z, N3


def dist_batch(W, mask, Z, N3, sigma):
    h = (Z * mask) @ W.T + sigma * N3[:, :W.shape[0]]
    a = h @ W
    return ((a - Z) ** 2 * mask).sum(axis=1)


def grad_W(W, mask, Z, N3, sigma):
    """dL/dW for L = mean_k sum_j mask_kj (a_kj - Z_kj)^2, a = W^T(Wx+sn)."""
    K = mask.shape[0]
    x = Z * mask
    h = x @ W.T + sigma * N3[:, :W.shape[0]]
    r = (h @ W - Z) * mask
    return (2.0 / K) * (h.T @ r + (r @ W.T).T @ x)


def step_W(arm, W, g, eta):
    if arm == "constr2":
        g = g - W * (g * W).sum(axis=0, keepdims=True)
        W = W - eta * g
        return W / np.linalg.norm(W, axis=0, keepdims=True)
    return W - eta * g


def run_cell(arm, delta, p0, delta_idx, p0_idx, seed, *, floor_eps=0.05,
             T=3000, burn_in=200, m=2, K=256, eta_V=0.02, eta_a=0.02,
             sigma=0.1, ema=0.98, eval_episodes=1000):
    init_rng, data_rng, noise_rng, action_rng, eval_rng = streams(
        delta_idx, p0_idx, seed)
    W = init_W(arm, init_rng)
    u = float(np.log(p0 / (1 - p0)))
    dE = dS = 0.0
    failed = False
    for t in range(-burn_in, T):
        p = 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60)))
        pb = floor_eps + (1 - 2 * floor_eps) * p
        for _ in range(m):
            B, mask, Z, N3 = draw_batch(pb, K, action_rng, data_rng, noise_rng)
            W = step_W(arm, W, grad_W(W, mask, Z, N3, sigma), eta_V)
        B, mask, Z, N3 = draw_batch(pb, K, action_rng, data_rng, noise_rng)
        d = dist_batch(W, mask, Z, N3, sigma)
        if B.any():
            dE = ema * dE + (1 - ema) * float(d[B].mean())
        if (~B).any():
            dS = ema * dS + (1 - ema) * float(d[~B].mean())
        if t >= 0:
            u = float(np.clip(u + eta_a * (delta - (dE - dS)), -60, 60))
        if not np.isfinite(u):
            failed = True
            break
    p_end = 1.0 / (1.0 + np.exp(-u))
    pb_end = floor_eps + (1 - 2 * floor_eps) * p_end
    Bv, maskv, Zv, N3v = draw_batch(pb_end, eval_episodes, eval_rng, eval_rng,
                                    eval_rng)
    ret = float((np.where(Bv, delta, 0.0)
                 - dist_batch(W, maskv, Zv, N3v, sigma)).mean())
    # balanced forced evaluation (block B chain measurement): equal numbers of
    # forced-E and forced-S episodes from the same eval stream
    _, mE, ZE, NE = draw_batch(1.0, eval_episodes, eval_rng, eval_rng, eval_rng)
    _, mS, ZS, NS = draw_batch(0.0, eval_episodes, eval_rng, eval_rng, eval_rng)
    dE_f = float(dist_batch(W, mE, ZE, NE, sigma).mean())
    dS_f = float(dist_batch(W, mS, ZS, NS, sigma).mean())
    ret_bal = float(0.5 * (delta - dE_f) + 0.5 * (0.0 - dS_f))
    Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    G = Wn.T @ Wn
    return dict(arm=arm, delta=float(delta), p0=float(p0), seed=int(seed),
                floor_eps=float(floor_eps), p_end=float(p_end),
                basin=int(p_end > 0.5), ret=ret,
                g12=float(G[0, 1] ** 2), g23=float(G[1, 2] ** 2),
                dE_forced=dE_f, dS_forced=dS_f,
                gap_forced=float(delta - (dE_f - dS_f)), ret_balanced=ret_bal,
                failed=bool(failed))
