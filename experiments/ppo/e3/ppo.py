"""Minimal line-auditable PPO+GAE (v3).

Architecture: sensory encoder -> FIXED rank-k projector -> concat branch
context (routed AROUND the projector; only branch identity can bypass, the
sensory dims cannot) -> trunk -> heads. Root policy uses the THEORY floor
pi_E = eps + (1-2eps) * softmax_E. Forced-branch training freezes the root
actor loss (audit v2 #4): forced actions are not policy samples, so no root
surrogate or root entropy is applied; terminal head, value, and
representation still train. Six bound RNG streams; no global-RNG use;
CPU execution declared. No checkpoint/resume (runs are minutes; claim
removed per audit v2 #5).
"""
import numpy as np
import torch
import torch.nn as nn
from . import env as E


class TrunkPolicy(nn.Module):
    def __init__(self, k_proj, hidden=64, init_rng=None):
        super().__init__()
        g = torch.Generator().manual_seed(int(init_rng.integers(2 ** 31)))
        self.enc = nn.Linear(E.SENS_DIM, hidden)
        R = torch.randn(hidden, hidden, generator=g)
        Q, _ = torch.linalg.qr(R)
        self.register_buffer("proj", Q[:, :k_proj] @ Q[:, :k_proj].T)
        self.trunk = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.root_head = nn.Linear(hidden, 2)
        self.mu_head = nn.Linear(hidden, 3)
        self.log_std = nn.Parameter(torch.full((3,), -0.5))
        self.v_head = nn.Linear(hidden, 1)
        for m in (self.enc, self.trunk, self.root_head, self.mu_head,
                  self.v_head):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p, gain=1.0)
                else:
                    nn.init.zeros_(p)

    def features(self, sens, branch_onehot):
        bott = torch.tanh(self.enc(sens)) @ self.proj
        return self.trunk(torch.cat([bott, branch_onehot], dim=-1))

    def forward(self, sens, branch_onehot):
        h = self.features(sens, branch_onehot)
        return self.root_head(h), self.mu_head(h), self.v_head(h).squeeze(-1)


def floor_pi_E(root_logits, eps):
    """Theory floor: pi_E = eps + (1-2eps) * softmax_E (cor:policy-floor)."""
    return eps + (1.0 - 2.0 * eps) * torch.softmax(root_logits, -1)[..., 1]


def gaussian_logp(a, mu, log_std):
    var = torch.exp(2 * log_std)
    return (-0.5 * ((a - mu) ** 2 / var + 2 * log_std
                    + np.log(2 * np.pi))).sum(-1)


def make_streams(seed):
    return {k: np.random.Generator(np.random.PCG64(seed * 16 + i))
            for i, k in enumerate(
                ("init", "data", "noise", "action", "opt", "eval"))}


def state_hash(net):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def run_training(*, k_proj=2, delta=0.3, p0=0.5, floor_eps=0.05, seed=0,
                 updates=300, batch=2048, ppo_epochs=4, minibatch=512,
                 lr=3e-4, clip=0.2, ent_coef=0.01, gae_lam=0.95,
                 sigma_obs=0.30, log_every=10, forced_branch=None,
                 root_lr_scale=1.0):
    S = make_streams(seed)
    torch.manual_seed(int(S["init"].integers(2 ** 31)))
    net = TrunkPolicy(k_proj, init_rng=S["init"])
    with torch.no_grad():
        net.root_head.bias.copy_(torch.tensor(
            [0.0, float(np.log(p0 / (1 - p0)))]))
    root_params = list(net.root_head.parameters())
    other = [q for q in net.parameters()
             if not any(q is rp for rp in root_params)]
    opt = torch.optim.Adam([{"params": other, "lr": lr},
                            {"params": root_params,
                             "lr": lr * root_lr_scale}])
    train_root = forced_branch is None
    sens0 = torch.zeros(batch, E.SENS_DIM)
    b0 = torch.zeros(batch, 2)                      # no branch yet at t=0
    perm_gen = torch.Generator().manual_seed(int(S["opt"].integers(2 ** 31)))
    logs = []
    for u in range(updates):
        with torch.no_grad():
            rl, _, v0 = net(sens0, b0)
            pE = floor_pi_E(rl, floor_eps)
            if train_root:
                aE = (S["action"].random(batch) < pE.numpy()).astype(int)
            else:
                aE = np.full(batch, int(forced_branch))
            mask, Z, sens = E.draw_batch(batch, aE, S["data"], S["noise"],
                                         sigma_obs=sigma_obs)
            s1 = torch.tensor(sens)
            bo = torch.zeros(batch, 2); bo[np.arange(batch), aE] = 1.0
            _, mu, v1 = net(s1, bo)
            a = mu + torch.exp(net.log_std) * torch.tensor(
                S["action"].standard_normal((batch, 3)), dtype=torch.float32)
            r = torch.tensor(E.reward(a.numpy(), Z, mask, aE, delta),
                             dtype=torch.float32)
            aEt = torch.tensor(aE)
            lpr = torch.where(aEt == 1, torch.log(pE + 1e-9),
                              torch.log(1 - pE + 1e-9))
            lpt = gaussian_logp(a, mu, net.log_std)
            adv1 = r - v1
            adv0 = (v1 - v0) + gae_lam * adv1
            ret = r.clone()
        for _ in range(ppo_epochs):
            perm = torch.randperm(batch, generator=perm_gen)
            for st in range(0, batch, minibatch):
                mb = perm[st:st + minibatch]
                rln, _, v0n = net(sens0[mb], b0[mb])
                _, mut, v1n = net(s1[mb], bo[mb])
                loss = 0.5 * ((v0n - ret[mb]) ** 2).mean() \
                    + 0.5 * ((v1n - ret[mb]) ** 2).mean()
                lptn = gaussian_logp(a[mb], mut, net.log_std)
                ratio = torch.exp(lptn - lpt[mb])
                an = (adv1[mb] - adv1[mb].mean()) / (adv1[mb].std() + 1e-8)
                loss = loss - torch.min(
                    ratio * an,
                    torch.clamp(ratio, 1 - clip, 1 + clip) * an).mean()
                loss = loss - ent_coef * net.log_std.sum()
                if train_root:
                    pEn = floor_pi_E(rln, floor_eps)
                    lprn = torch.where(aEt[mb] == 1, torch.log(pEn + 1e-9),
                                       torch.log(1 - pEn + 1e-9))
                    ratio0 = torch.exp(lprn - lpr[mb])
                    a0n = (adv0[mb] - adv0[mb].mean()) / (adv0[mb].std() + 1e-8)
                    loss = loss - torch.min(
                        ratio0 * a0n,
                        torch.clamp(ratio0, 1 - clip, 1 + clip) * a0n).mean()
                    loss = loss - ent_coef * (
                        -(pEn * torch.log(pEn + 1e-9)
                          + (1 - pEn) * torch.log(1 - pEn + 1e-9)).mean())
                opt.zero_grad(); loss.backward(); opt.step()
        if u % log_every == 0 or u == updates - 1:
            with torch.no_grad():
                rl, _, _ = net(sens0[:1], b0[:1])
                logs.append(dict(update=u,
                                 p_E=float(floor_pi_E(rl, floor_eps)[0]),
                                 mean_reward=float(r.mean())))
    return net, logs, S
