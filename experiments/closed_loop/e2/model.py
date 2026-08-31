"""Solved three-feature model: analytic ground truth and optimization objects.

Everything here mirrors the paper's Section 3 exactly:
  pair weights   q12 = p/2, q13 = 1/2, q23 = (1-p)/2
  objective      L_p(V) = p*g12 + g13 + (1-p)*g23,  g_ij = <v_i, v_j>^2
  branch gap     D_E - D_S = g12 - g23   (deployed gap = delta - (g12 - g23))
Phase boundaries p_- = (3-sqrt5)/2, p_+ = (sqrt5-1)/2 (golden ratio).
"""
import numpy as np

P_MINUS = (3.0 - np.sqrt(5.0)) / 2.0
P_PLUS = (np.sqrt(5.0) - 1.0) / 2.0


def pair_weights(p):
    """Coefficients of (g12, g13, g23) in the nonconstant objective L_p."""
    return np.array([p, 1.0, 1.0 - p])


def gram_sq(V):
    """(g12, g13, g23) with g_ij = <v_i, v_j>^2 for unit columns of V."""
    G = V.T @ V
    return np.array([G[0, 1] ** 2, G[0, 2] ** 2, G[1, 2] ** 2])


def loss(V, p):
    g = gram_sq(V)
    return float(pair_weights(p) @ g)


def euclidean_grad(V, p, ortho_lam=0.0):
    """dL/dV for L = sum_{i<j} w_ij <v_i,v_j>^2 (+ ortho penalty lam*sum g_ij)."""
    w = {(0, 1): p + ortho_lam, (0, 2): 1.0 + ortho_lam,
         (1, 2): 1.0 - p + ortho_lam}
    G = V.T @ V
    grad = np.zeros_like(V)
    for (i, j), wij in w.items():
        grad[:, i] += 2.0 * wij * G[i, j] * V[:, j]
        grad[:, j] += 2.0 * wij * G[i, j] * V[:, i]
    return grad


def riemannian_step(V, p, eta, ortho_lam=0.0):
    """One projected-gradient step on the product of unit spheres (columns)."""
    g = euclidean_grad(V, p, ortho_lam)
    g = g - V * np.sum(g * V, axis=0, keepdims=True)   # tangent projection
    V = V - eta * g
    return V / np.linalg.norm(V, axis=0, keepdims=True)  # retraction


def random_V(dim, rng):
    V = rng.standard_normal((dim, 3))
    return V / np.linalg.norm(V, axis=0, keepdims=True)


def deployed_gap(V, delta):
    """delta - (D_E - D_S) = delta - (g12 - g23) at the CURRENT code V."""
    g = gram_sq(V)
    return delta - (g[0] - g[2])


# ---------- analytic ground truth (calibration only, never evidence) ----------

def L_star(p):
    """Optimal value of L_p over unit vectors in R^2 (Theorem frame)."""
    p = float(p)
    if p <= P_MINUS:
        return p
    if p >= P_PLUS:
        return 1.0 - p
    chi = p * (1.0 - p)
    return 1.5 - 0.25 * (chi + 1.0 / chi)


def D_analytic(p):
    """D(p) = g12(p) - g23(p) = dL*/dp, piecewise (order parameter)."""
    p = float(p)
    if p < P_MINUS:
        return 1.0
    if p > P_PLUS:
        return -1.0
    chi = p * (1.0 - p)
    return (1.0 - 2.0 * p) / 4.0 * (chi ** -2 - 1.0)


def p_delta(delta, tol=1e-12):
    """Unique separator in (p_-, p_+) with D(p_delta) = delta, |delta| < 1."""
    assert -1.0 < delta < 1.0
    lo, hi = P_MINUS, P_PLUS
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if D_analytic(mid) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def s_lambda(lam):
    """Endpoint gap magnitude under ortho penalty (prop gram-threshold)."""
    return 1.0 if lam <= 1.0 else (3.0 * lam + 1.0) / (4.0 * lam ** 2)


def lambda_crit(delta):
    return (3.0 + np.sqrt(9.0 + 16.0 * delta)) / (8.0 * delta)


def entropy_sinks(delta, tau):
    """Closed-form sinks p_L, p_H of the entropy-regularized flow."""
    return (1.0 / (1.0 + np.exp((1.0 - delta) / tau)),
            1.0 / (1.0 + np.exp(-(1.0 + delta) / tau)))


def entropy_bistable_upper(delta):
    """delta-dependent guaranteed-bistable bound min((1-delta)/log phi, 15/8)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    return min((1.0 - delta) / np.log(phi), 15.0 / 8.0)
