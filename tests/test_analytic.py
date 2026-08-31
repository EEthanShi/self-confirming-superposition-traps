import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from e2 import model
from e2.dynamics import run_closed_loop, run_exact_ode, measure_separator


def test_lstar_continuity_and_values():
    assert abs(model.L_star(0.5) - 7.0 / 16.0) < 1e-12
    for pb in (model.P_MINUS, model.P_PLUS):
        lo = model.L_star(pb - 1e-9); hi = model.L_star(pb + 1e-9)
        assert abs(lo - hi) < 1e-6
    assert abs(model.L_star(model.P_MINUS) - model.P_MINUS) < 1e-9


def test_D_analytic():
    assert model.D_analytic(0.1) == 1.0
    assert model.D_analytic(0.9) == -1.0
    assert abs(model.D_analytic(0.5)) < 1e-12
    grid = np.linspace(model.P_MINUS + 1e-6, model.P_PLUS - 1e-6, 200)
    vals = [model.D_analytic(p) for p in grid]
    assert all(a > b for a, b in zip(vals, vals[1:]))          # strict decrease
    eps = 1e-7                                                  # D = dL*/dp
    for p in (0.42, 0.5, 0.58):
        fd = (model.L_star(p + eps) - model.L_star(p - eps)) / (2 * eps)
        assert abs(fd - model.D_analytic(p)) < 1e-5


def test_p_delta():
    assert abs(model.p_delta(0.0) - 0.5) < 1e-9
    for d in (-0.8, -0.3, 0.3, 0.8):
        assert abs(model.D_analytic(model.p_delta(d)) - d) < 1e-9


def test_gradient_finite_difference():
    rng = np.random.default_rng(0)
    V = model.random_V(2, rng); p = 0.37; eps = 1e-6
    g = model.euclidean_grad(V, p)
    for i in range(V.shape[0]):
        for j in range(3):
            Vp, Vm = V.copy(), V.copy()
            Vp[i, j] += eps; Vm[i, j] -= eps
            fd = (model.loss(Vp, p) - model.loss(Vm, p)) / (2 * eps)
            assert abs(fd - g[i, j]) < 1e-5


def test_inner_optimizer_reaches_lstar():
    rng = np.random.default_rng(1)
    for p in (0.2, 0.5, 0.8):
        best = np.inf; best_g = None
        for _ in range(8):
            V = model.random_V(2, rng)
            for _ in range(4000):
                V = model.riemannian_step(V, p, 0.05)
            if model.loss(V, p) < best:
                best = model.loss(V, p); best_g = model.gram_sq(V)
        assert abs(best - model.L_star(p)) < 1e-6
        assert abs((best_g[0] - best_g[2]) - model.D_analytic(p)) < 1e-3


def test_exact_ode_basins():
    for d in (0.3, 0.6):
        pd = model.p_delta(d)
        assert run_exact_ode(d, pd - 0.05)[-1] < 1e-3
        assert run_exact_ode(d, pd + 0.05)[-1] > 1 - 1e-3


def test_closed_loop_warm_smoke_and_orthogonal_control():
    r = run_closed_loop(0.3, 0.2, T=1500, seed=0)
    assert np.isfinite(r["p"]).all() and 0.0 < r["p_end"] < 1.0
    r3 = run_closed_loop(0.3, 0.2, dim=3, T=3000, seed=0)
    assert r3["basin"] == 1        # matched capacity: no trap, goes high


def test_entropy_ode_sinks_and_single_phase():
    from e2.dynamics import run_exact_ode
    d, tau = 0.3, 1.0            # bistable side: tau < (1-d)/log(phi) = 1.455
    pL, pH = model.entropy_sinks(d, tau)
    lo = run_exact_ode(d, 0.05, tau=tau, T=60000)[-1]
    hi = run_exact_ode(d, 0.95, tau=tau, T=60000)[-1]
    assert abs(lo - pL) < 1e-4 and abs(hi - pH) < 1e-4
    lo2 = run_exact_ode(d, 0.05, tau=2.6, T=60000)[-1]   # tau >= 5/2: single
    hi2 = run_exact_ode(d, 0.95, tau=2.6, T=60000)[-1]
    assert abs(lo2 - hi2) < 1e-4


def test_floor_ode_separator_mapping():
    from e2.dynamics import run_exact_ode
    d, eps = 0.3, 0.1
    s_pred = (model.p_delta(d) - eps) / (1.0 - 2.0 * eps)
    assert run_exact_ode(d, s_pred - 0.02, floor_eps=eps, T=60000)[-1] < 1e-3
    assert run_exact_ode(d, s_pred + 0.02, floor_eps=eps, T=60000)[-1] > 1 - 1e-3


def test_replay_ode_condition():
    from e2.dynamics import run_exact_ode
    d, pbar = 0.3, 0.5
    pd = model.p_delta(d)                       # 0.4596; pbar > pd here
    lam_hi = 1.0 - 1e-6                          # condition holds -> bistable
    assert run_exact_ode(d, 0.03, replay=(0.9, pbar), T=60000)[-1] < 0.2
    assert run_exact_ode(d, 0.97, replay=(0.9, pbar), T=60000)[-1] > 0.8
    # lam small: (1-lam)*pbar > pd -> low endpoint destabilized, all go high
    lam = 0.05
    assert (1 - lam) * pbar > pd
    assert run_exact_ode(d, 0.03, replay=(lam, pbar), T=60000)[-1] > 0.8


def test_ortho_gradient_and_threshold_formula():
    rng = np.random.default_rng(3)
    V = model.random_V(2, rng); p, lam, eps = 0.4, 0.7, 1e-6
    g = model.euclidean_grad(V, p, ortho_lam=lam)
    def loss_o(V):
        gs = model.gram_sq(V)
        return model.loss(V, p) + lam * gs.sum()
    for i in range(2):
        for j in range(3):
            Vp, Vm = V.copy(), V.copy(); Vp[i, j] += eps; Vm[i, j] -= eps
            fd = (loss_o(Vp) - loss_o(Vm)) / (2 * eps)
            assert abs(fd - g[i, j]) < 1e-5
    lc = model.lambda_crit(0.3)
    assert abs(model.s_lambda(lc) - 0.3) < 1e-12   # threshold: s_lambda = delta
