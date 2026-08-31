"""E2c neural bridge: minimal learned system, batched on GPU.

Per run r (all runs advance in lockstep as one batched tensor computation):
  encoder  h = W_r x + sigma*noise   (learnable, 2D or 3D bottleneck)
  readout  a = W_r^T h               (tied; untied variant adds a decoder)
  reward   b_B - sum_{i in G} (a_i - Z_i)^2   (distortion IS the reward)
  actor    logit u_r updated ONLY through the learned on-policy gap estimate
           (EMA of per-branch distortion difference from the agent's own
           episodes; a branch it never takes keeps a stale estimate).
Competence, Gram mediator, gap, policy, and return are all logged; nothing
is presumed. Control arm: dim=3 bottleneck, same seeds (CRN pairing).
"""
import torch


def run_batch(delta, p0, *, runs=256, dim=2, T=2000, K=64, m=2,
              eta_V=0.02, eta_a=0.05, sigma=0.1, ema=0.9, seed=0,
              device="cuda", record_every=20, burn_in=200, floor_eps=0.0):
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(runs, dim, 3, generator=g, device=device) * 0.5
    W.requires_grad_(True)
    opt = torch.optim.SGD([W], lr=eta_V)
    if torch.is_tensor(p0):
        u = torch.logit(p0.to(device).clamp(1e-6, 1 - 1e-6))
    else:
        u = torch.full((runs,), float(torch.logit(torch.tensor(p0))),
                       device=device)
    dS = torch.zeros(runs, device=device)   # EMA distortion estimates
    dE = torch.zeros(runs, device=device)
    log = {"p": [], "gap_hat": [], "g12": [], "g23": [], "ret": [],
           "comp": []}

    def sample(u):
        p = torch.sigmoid(u)
        if floor_eps > 0.0:
            p = floor_eps + (1.0 - 2.0 * floor_eps) * p
        B = (torch.rand(runs, K, generator=g, device=device) < p[:, None])
        pick = torch.rand(runs, K, generator=g, device=device) < 0.5
        # S: {1,3} or {2,3};  E: {1,2} or {1,3}   (features 0-indexed)
        mask = torch.zeros(runs, K, 3, device=device)
        mask[..., 2] = (~B).float()                      # S always has f3
        mask[..., 0] = (B | (~B & pick)).float()         # wait: S pick {1,3}
        # explicit per-branch pair construction:
        mask.zero_()
        sel = torch.stack([B & pick, B & ~pick, ~B & pick, ~B & ~pick], -1)
        pairs = torch.tensor([[1, 1, 0], [1, 0, 1],      # E: {1,2}, {1,3}
                              [1, 0, 1], [0, 1, 1]],     # S: {1,3}, {2,3}
                             dtype=torch.float32, device=device)
        mask = (sel.float() @ pairs)
        Z = torch.randn(runs, K, 3, generator=g, device=device)
        return B.float(), mask, Z

    for t in range(-burn_in, T):
        frozen = t < 0            # burn-in: representation and EMAs adapt to
        for _ in range(m):        # the INITIAL occupancy, actor frozen
            B, mask, Z = sample(u)
            x = (Z * mask)
            h = torch.einsum("rij,rkj->rki", W, x)
            h = h + sigma * torch.randn_like(h)
            a = torch.einsum("rij,rki->rkj", W, h)
            dist = (mask * (a - Z) ** 2).sum(-1)         # (runs, K)
            opt.zero_grad(); dist.mean(dim=1).sum().backward(); opt.step()
        with torch.no_grad():
            B, mask, Z = sample(u)
            x = Z * mask
            h = torch.einsum("rij,rkj->rki", W, x)
            h = h + sigma * torch.randn_like(h)
            a = torch.einsum("rij,rki->rkj", W, h)
            dist = (mask * (a - Z) ** 2).sum(-1)
            nE = B.sum(1).clamp(min=1); nS = (1 - B).sum(1).clamp(min=1)
            dE_b = (dist * B).sum(1) / nE
            dS_b = (dist * (1 - B)).sum(1) / nS
            hasE = B.sum(1) > 0; hasS = (1 - B).sum(1) > 0
            dE = torch.where(hasE, ema * dE + (1 - ema) * dE_b, dE)
            dS = torch.where(hasS, ema * dS + (1 - ema) * dS_b, dS)
            gap_hat = delta - (dE - dS)
            if not frozen:
                u = (u + eta_a * gap_hat).clamp(-60, 60)
            if t >= 0 and (t % record_every == 0 or t == T - 1):
                Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-9)
                G = torch.einsum("rij,rik->rjk", Wn, Wn)
                ret = (torch.sigmoid(u) * delta - dist.mean(1))
                comp = 1.0 - (dist.mean(1) / (2.0 + 1e-9))  # crude recovery
                log["p"].append(torch.sigmoid(u).cpu())
                log["gap_hat"].append(gap_hat.cpu())
                log["g12"].append((G[:, 0, 1] ** 2).cpu())
                log["g23"].append((G[:, 1, 2] ** 2).cpu())
                log["ret"].append(ret.cpu())
                log["comp"].append(comp.cpu())
    out = {k: torch.stack(v).numpy().tolist() for k, v in log.items()}
    out["p_end"] = torch.sigmoid(u).cpu().numpy().tolist()
    out["config"] = dict(delta=delta, runs=runs, dim=dim, T=T, K=K, m=m,
                         eta_V=eta_V, eta_a=eta_a, sigma=sigma, ema=ema,
                         seed=seed)
    return out


def pop_quantities(W, sigma2):
    """Exact population branch distortions from M = W^T W (batched)."""
    M = torch.einsum("rij,rik->rjk", W, W)
    wn = (W ** 2).sum(1)
    def pair(i, j):
        return ((M[:, i, i] - 1) ** 2 + M[:, i, j] ** 2 + sigma2 * wn[:, i]
                + (M[:, j, j] - 1) ** 2 + M[:, j, i] ** 2 + sigma2 * wn[:, j])
    dE = 0.5 * (pair(0, 1) + pair(0, 2))
    dS = 0.5 * (pair(0, 2) + pair(1, 2))
    return dE, dS


def four_arm_batch(delta, p0, *, encoder="sgd", gapmode="ema", runs=None,
                   T=3000, K=256, m=2, eta_V=0.02, eta_a=0.02, sigma=0.1,
                   ema=0.98, seed=0, device="cuda", burn_in=200,
                   floor_eps=0.05):
    """Decomposition arms: {sgd,pop} encoder x {ema,exact} gap. dim=2 tied."""
    runs = runs or len(p0)
    s2 = sigma * sigma
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(runs, 2, 3, generator=g, device=device) * 0.5
    W.requires_grad_(True)
    opt = torch.optim.SGD([W], lr=eta_V)
    u = torch.logit(p0.to(device).clamp(1e-6, 1 - 1e-6))
    dS_e = torch.zeros(runs, device=device); dE_e = torch.zeros(runs, device=device)

    def sample(pb):
        B = (torch.rand(runs, K, generator=g, device=device) < pb[:, None])
        pick = torch.rand(runs, K, generator=g, device=device) < 0.5
        sel = torch.stack([B & pick, B & ~pick, ~B & pick, ~B & ~pick], -1)
        pairs = torch.tensor([[1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]],
                             dtype=torch.float32, device=device)
        mask = sel.float() @ pairs
        Z = torch.randn(runs, K, 3, generator=g, device=device)
        return B.float(), mask, Z

    for t in range(-burn_in, T):
        frozen = t < 0
        p = torch.sigmoid(u)
        pb = floor_eps + (1 - 2 * floor_eps) * p if floor_eps > 0 else p
        for _ in range(m):
            if encoder == "pop":
                dE, dS = pop_quantities(W, s2)
                loss = (pb.detach() * dE + (1 - pb.detach()) * dS).sum()
            else:
                B, mask, Z = sample(pb)
                x = Z * mask
                h = torch.einsum("rij,rkj->rki", W, x)
                h = h + sigma * torch.randn_like(h)
                a = torch.einsum("rij,rki->rkj", W, h)
                loss = (mask * (a - Z) ** 2).sum(-1).mean(dim=1).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if gapmode == "exact":
                dE, dS = pop_quantities(W, s2)
                gap = delta - (dE - dS)
            else:
                B, mask, Z = sample(pb)
                x = Z * mask
                h = torch.einsum("rij,rkj->rki", W, x)
                h = h + sigma * torch.randn_like(h)
                a = torch.einsum("rij,rki->rkj", W, h)
                dist = (mask * (a - Z) ** 2).sum(-1)
                nE = B.sum(1).clamp(min=1); nS = (1 - B).sum(1).clamp(min=1)
                dEb = (dist * B).sum(1) / nE; dSb = (dist * (1 - B)).sum(1) / nS
                hasE = B.sum(1) > 0; hasS = (1 - B).sum(1) > 0
                dE_e = torch.where(hasE, ema * dE_e + (1 - ema) * dEb, dE_e)
                dS_e = torch.where(hasS, ema * dS_e + (1 - ema) * dSb, dS_e)
                gap = delta - (dE_e - dS_e)
            if not frozen:
                u = (u + eta_a * gap).clamp(-60, 60)
    return {"p_end": torch.sigmoid(u).cpu().numpy().tolist(),
            "config": dict(delta=delta, encoder=encoder, gapmode=gapmode,
                           K=K, T=T, m=m, eta_V=eta_V, eta_a=eta_a,
                           sigma=sigma, ema=ema, seed=seed,
                           floor_eps=floor_eps, burn_in=burn_in)}
