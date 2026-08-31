"""E3 evaluators v3 (bound eval stream; balanced forced episodes).

PRIMARY mediator: deployed-interface cross-talk (audit v2 #1) — the Jacobian
of the deployed reconstruction with respect to the sensory FEATURE inputs,
xtalk_ij = mean(|J_ij| + |J_ji|)/2 for i != j. Behavioral, identifiable,
invariant to hidden reparameterization. probe-Gram kept as DIAGNOSTIC only.
Deployment returns come in two flavors and are named honestly:
deterministic (Gaussian mean) and sampled (terminal policy sampled).
"""
import numpy as np
import torch
from . import env as E


def _forced(net, branch, n, eval_rng, sigma_obs):
    b = np.full(n, int(branch))
    mask, Z, sens = E.draw_batch(n, b, eval_rng, eval_rng,
                                 sigma_obs=sigma_obs)
    bo = torch.zeros(n, 2); bo[:, branch] = 1.0
    with torch.no_grad():
        _, mu, _ = net(torch.tensor(sens), bo)
    return mask, Z, sens, bo, mu.numpy()


def competence(net, eval_rng, *, n=20000, sigma_obs=0.30):
    oracle = E.oracle_recon_mse(sigma_obs)
    out = {}
    for br, name in ((0, "S"), (1, "E")):
        mask, Z, _, _, mu = _forced(net, br, n, eval_rng, sigma_obs)
        mse = float((((mu - Z) ** 2) * mask).sum(1).mean())
        out[name] = dict(mse=mse, oracle_mse=oracle,
                         ratio_to_oracle=oracle / max(mse, 1e-9))
    return out


def crosstalk(net, eval_rng, *, n=4096, sigma_obs=0.30):
    """Deployed-interface Jacobian cross-talk on feature coords (PRIMARY)."""
    J_acc = np.zeros((3, 3))
    for br in (0, 1):
        mask, Z, sens, bo, _ = _forced(net, br, n // 2, eval_rng, sigma_obs)
        st = torch.tensor(sens, requires_grad=True)
        _, mu, _ = net(st, bo)
        for i in range(3):
            g = torch.autograd.grad(mu[:, i].sum(), st, retain_graph=True)[0]
            J_acc[i] += np.abs(g[:, :3].numpy()).mean(axis=0) / 2
    x = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        x[f"x{i+1}{j+1}"] = float((J_acc[i, j] + J_acc[j, i]) / 2)
    return x


def probe_gram(net, eval_rng, *, n=20000, sigma_obs=0.30):
    """DIAGNOSTIC only (not identifiable under hidden reparameterization)."""
    feats, Zs, Mks = [], [], []
    for br in (0, 1):
        mask, Z, sens, bo, _ = _forced(net, br, n // 2, eval_rng, sigma_obs)
        with torch.no_grad():
            h = net.features(torch.tensor(sens), bo).numpy()
        feats.append(h); Zs.append(Z); Mks.append(mask)
    H = np.concatenate(feats); Z = np.concatenate(Zs)
    Mk = np.concatenate(Mks)
    W = []
    for i in range(3):
        sel = Mk[:, i] == 1
        X = H[sel]; y = Z[sel][:, i]
        w, *_ = np.linalg.lstsq(X.T @ X + 1e-3 * np.eye(X.shape[1]),
                                X.T @ y, rcond=None)
        W.append(w / (np.linalg.norm(w) + 1e-12))
    G = np.array(W) @ np.array(W).T
    return dict(g12=float(G[0, 1] ** 2), g13=float(G[0, 2] ** 2),
                g23=float(G[1, 2] ** 2))


def forced_gap(net, eval_rng, *, n=20000, sigma_obs=0.30, delta=0.3):
    vals = {}
    for br, name in ((0, "S"), (1, "E")):
        mask, Z, _, _, mu = _forced(net, br, n, eval_rng, sigma_obs)
        vals[name] = float((-((mu - Z) ** 2) * mask).sum(1).mean())
    return dict(gap_no_delta=vals["E"] - vals["S"],
                gap_with_delta=vals["E"] - vals["S"] + delta)


def deployment_return(net, eval_rng, *, n=20000, sigma_obs=0.30, delta=0.3,
                      floor_eps=0.05, sampled=False):
    from .ppo import floor_pi_E
    with torch.no_grad():
        rl, _, _ = net(torch.zeros(1, E.SENS_DIM), torch.zeros(1, 2))
        pE = float(floor_pi_E(rl, floor_eps)[0])
    b = (eval_rng.random(n) < pE).astype(int)
    mask, Z, sens = E.draw_batch(n, b, eval_rng, eval_rng,
                                 sigma_obs=sigma_obs)
    bo = torch.zeros(n, 2); bo[np.arange(n), b] = 1.0
    with torch.no_grad():
        _, mu, _ = net(torch.tensor(sens), bo)
        a = mu.numpy()
        if sampled:
            a = a + np.exp(net.log_std.numpy()) \
                * eval_rng.standard_normal((n, 3))
    r = E.reward(a, Z, mask, b, delta)
    key = "sampled_return" if sampled else "deterministic_return"
    return {("p_E"): pE, key: float(r.mean())}
