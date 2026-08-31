"""E3 environment v3. Continuous masked-reconstruction reward; sensory
observation and branch context returned SEPARATELY (branch context is routed
around the sensory bottleneck in the policy; audit v2 #2).

  t=0  fixed start; discrete action {S, E} selects the branch.
  t=1  two of three features co-activate (S: {1,3}/{2,3}; E: {1,2}/{1,3});
       sensory obs = (Z*mask + sigma_obs*noise, nuisance) in R^8.
       action a in R^3; reward = -sum_{i in G}(a_i - Z_i)^2 + delta*1{E}.
Analytic Bayes oracle (audit v2 #3): scored coordinates are active by
definition, so the optimal response is x_i/(1+sigma^2) and
MSE_oracle = 2 sigma^2/(1+sigma^2). Reward-only training.
"""
import numpy as np

PAIR_SETS = {0: [(0, 2), (1, 2)], 1: [(0, 1), (0, 2)]}
N_NUIS = 5
SENS_DIM = 3 + N_NUIS


def draw_batch(n, branch, data_rng, noise_rng, *, sigma_obs=0.30):
    pick = (data_rng.random(n) < 0.5).astype(int)
    pairs = np.array([PAIR_SETS[b][k] for b, k in zip(branch, pick)])
    mask = np.zeros((n, 3))
    mask[np.arange(n), pairs[:, 0]] = 1.0
    mask[np.arange(n), pairs[:, 1]] = 1.0
    Z = data_rng.standard_normal((n, 3))
    feat = Z * mask + sigma_obs * noise_rng.standard_normal((n, 3))
    nuis = noise_rng.standard_normal((n, N_NUIS))
    sens = np.concatenate([feat, nuis], axis=1).astype(np.float32)
    return mask, Z, sens


def reward(a, Z, mask, branch, delta):
    return -((a - Z) ** 2 * mask).sum(axis=1) + delta * branch


def oracle_recon_mse(sigma_obs=0.30):
    """Analytic full-information Bayes MSE on active coordinates."""
    s2 = sigma_obs ** 2
    return 2.0 * s2 / (1.0 + s2)


def rank2_reference_mse(sigma_obs=0.30):
    """Explicit ACHIEVABLE rank-2 linear reference for forced-branch data
    (external review, ninth round): one dimension stores the always-active
    feature, one stores the sum of the two mutually-exclusive features
    (masking scores only the active coordinate, so emitting the shrunk sum on
    both channels is valid). L = s2/(1+s2) + 2 s2/(1+2 s2). This is an
    achievable REFERENCE, not a proven class ceiling."""
    s2 = sigma_obs ** 2
    return s2 / (1 + s2) + 2 * s2 / (1 + 2 * s2)
