#!/usr/bin/env python3
"""Executable algebra checks and phase-data generator.

The checks are numerical evidence only. They are not proof certificates.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


SEED = 20260808
ROOT = Path(__file__).resolve().parents[1]
P_MINUS = (3.0 - math.sqrt(5.0)) / 2.0
P_PLUS = (math.sqrt(5.0) - 1.0) / 2.0


def tied_solution(p: float) -> tuple[float, np.ndarray, float]:
    if p <= P_MINUS:
        return p, np.array([1.0, 0.0, 0.0]), 1.0
    if p >= P_PLUS:
        return 1.0 - p, np.array([0.0, 0.0, 1.0]), -1.0
    q = p * (1.0 - p)
    value = 1.5 - 0.25 * (q + 1.0 / q)
    g12 = 0.5 - 0.25 * (
        1.0 / (1.0 - p) + (1.0 - p) - (1.0 - p) / p**2
    )
    g13 = 1.0 + q / 4.0 - 1.0 / (4.0 * q)
    g23 = 0.5 - 0.25 * (p + 1.0 / p - p / (1.0 - p) ** 2)
    d = (1.0 - 2.0 * p) * (q**-2 - 1.0) / 4.0
    return value, np.array([g12, g13, g23]), d


def weighted_solution(a: float, b: float, c: float) -> tuple[float, np.ndarray]:
    if a <= b * c / (b + c):
        return a, np.array([1.0, 0.0, 0.0])
    if b <= a * c / (a + c):
        return b, np.array([0.0, 1.0, 0.0])
    if c <= a * b / (a + b):
        return c, np.array([0.0, 0.0, 1.0])
    value = (a + b + c) / 2.0 - 0.25 * (
        a * b / c + a * c / b + b * c / a
    )
    g12 = 0.5 - 0.25 * (b / c + c / b - b * c / a**2)
    g13 = 0.5 - 0.25 * (a / c + c / a - a * c / b**2)
    g23 = 0.5 - 0.25 * (a / b + b / a - a * b / c**2)
    return value, np.array([g12, g13, g23])


def write_phase_data() -> None:
    path = ROOT / "figures" / "phase_data.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["p", "g12", "g13", "g23", "D"])
        for p in np.linspace(0.0, 1.0, 201):
            _, g, d = tied_solution(float(p))
            writer.writerow([f"{p:.8f}", *(f"{x:.10f}" for x in g), f"{d:.10f}"])


def pair_loss_check(rng: np.random.Generator) -> float:
    theta = 0.731
    sigma = 0.37
    count = 800_000
    z = rng.normal(size=(count, 2))
    eps = rng.normal(scale=sigma, size=(count, 2))
    v1 = np.array([1.0, 0.0])
    v2 = np.array([math.cos(theta), math.sin(theta)])
    h = z[:, [0]] * v1 + z[:, [1]] * v2 + eps
    empirical = np.mean((h @ v1 - z[:, 0]) ** 2 + (h @ v2 - z[:, 1]) ** 2)
    exact = 2.0 * (math.cos(theta) ** 2 + sigma**2)
    error = abs(float(empirical) - exact)
    assert error < 0.005
    return error


def partial_observation_reduction_check(
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Check the vector-block reduction and quotient-space realizability."""

    block_dims = [2, 1, 2]
    code_dim = 3
    nuisance_dim = 2
    observation_dim = nuisance_dim + sum(block_dims)
    vectors = []
    for block_dim in block_dims:
        q, _ = np.linalg.qr(rng.normal(size=(code_dim, block_dim)))
        vectors.append(q[:, :block_dim])

    encoder = np.zeros((code_dim, observation_dim))
    observation_maps = []
    offset = nuisance_dim
    for block_dim, vector in zip(block_dims, vectors):
        observation_map = np.zeros((observation_dim, block_dim))
        observation_map[:nuisance_dim] = rng.normal(
            size=(nuisance_dim, block_dim)
        )
        observation_map[offset : offset + block_dim] = np.eye(block_dim)
        encoder[:, offset : offset + block_dim] = vector
        observation_maps.append(observation_map)
        offset += block_dim

    realization_error = max(
        float(np.max(np.abs(encoder @ observation_map - vector)))
        for observation_map, vector in zip(observation_maps, vectors)
    )

    active_law = {
        (0,): 0.10,
        (1,): 0.15,
        (2,): 0.05,
        (0, 1): 0.20,
        (0, 2): 0.25,
        (1, 2): 0.15,
        (0, 1, 2): 0.10,
    }
    assert abs(sum(active_law.values()) - 1.0) < 1e-15
    sigma = 0.37
    direct = 0.0
    activation = np.zeros(len(block_dims))
    coactivation = np.zeros((len(block_dims), len(block_dims)))
    for active, probability in active_law.items():
        for i in active:
            activation[i] += probability
            direct += probability * sigma**2 * block_dims[i]
            for j in active:
                if i == j:
                    continue
                cross = vectors[i].T @ vectors[j]
                direct += probability * float(np.sum(cross**2))
        for i in active:
            for j in active:
                if i < j:
                    coactivation[i, j] += probability

    reduced = sigma**2 * float(np.dot(block_dims, activation))
    for i in range(len(block_dims)):
        for j in range(i + 1, len(block_dims)):
            reduced += 2.0 * coactivation[i, j] * float(
                np.sum((vectors[i].T @ vectors[j]) ** 2)
            )
    reduction_error = abs(direct - reduced)
    assert realization_error < 1e-12
    assert reduction_error < 1e-12
    return reduction_error, realization_error


def weighted_grid_check(rng: np.random.Generator) -> tuple[float, float]:
    size = 1201
    angles = np.linspace(0.0, math.pi, size, endpoint=False)
    c2 = np.cos(angles) ** 2
    max_value_error = 0.0
    max_gram_error = 0.0
    phases = set()
    for _ in range(24):
        a, b, c = np.exp(rng.uniform(-2.0, 2.0, size=3))
        value, gram = weighted_solution(float(a), float(b), float(c))
        objective = (
            a * c2[:, None]
            + b * c2[None, :]
            + c * np.cos(angles[:, None] - angles[None, :]) ** 2
        )
        i, j = np.unravel_index(np.argmin(objective), objective.shape)
        brute_value = float(objective[i, j])
        brute_gram = np.array(
            [c2[i], c2[j], math.cos(angles[i] - angles[j]) ** 2]
        )
        max_value_error = max(max_value_error, abs(brute_value - value))
        max_gram_error = max(max_gram_error, float(np.max(np.abs(brute_gram - gram))))
        phases.add(int(np.argmax(gram)) if np.max(gram) > 0.999 else 3)
    assert max_value_error < 5e-5
    assert max_gram_error < 5e-3
    assert phases == {0, 1, 2, 3}
    return max_value_error, max_gram_error


def envelope_check() -> float:
    maximum = 0.0
    for p in [0.1, 0.4, 0.5, 0.6, 0.9]:
        step = 1e-6
        left = tied_solution(p - step)[0]
        right = tied_solution(p + step)[0]
        finite_difference = (right - left) / (2.0 * step)
        _, gram, d = tied_solution(p)
        maximum = max(maximum, abs(finite_difference - d), abs(d - gram[0] + gram[2]))
    assert maximum < 1e-8
    return maximum


def angle_representative(p: float) -> tuple[float, float]:
    _, gram, _ = tied_solution(p)
    alpha0 = math.acos(math.sqrt(max(0.0, min(1.0, float(gram[1])))))
    beta0 = math.acos(math.sqrt(max(0.0, min(1.0, float(gram[2])))))
    candidates = []
    for alpha in [alpha0, math.pi - alpha0]:
        for beta in [beta0, math.pi - beta0]:
            residual = abs(math.cos(alpha - beta) ** 2 - float(gram[0]))
            candidates.append((residual, alpha, beta))
    residual, alpha, beta = min(candidates)
    assert residual < 1e-9
    return alpha, beta


def codimension_lift_check(
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Check the exact block construction and its spectral lower bound."""

    maximum_block_error = 0.0
    minimum_objective_margin = math.inf
    minimum_tail_bound_slack = math.inf
    occupancies = [0.0, 0.2, P_MINUS, 0.5, P_PLUS, 0.8, 1.0]

    for dimension in range(2, 9):
        for p in occupancies:
            alpha, beta = angle_representative(p)
            vectors = np.zeros((dimension, dimension + 1))
            vectors[:2, 0] = [math.cos(alpha), math.sin(alpha)]
            vectors[:2, 1] = [math.cos(beta), math.sin(beta)]
            vectors[0, 2] = 1.0
            for column in range(3, dimension + 1):
                vectors[column - 1, column] = 1.0
            squared_gram = (vectors.T @ vectors) ** 2
            total = float(np.sum(np.triu(squared_gram, k=1)))
            objective = (
                total
                + (p - 1.0) * squared_gram[0, 1]
                - p * squared_gram[1, 2]
            )
            maximum_block_error = max(
                maximum_block_error, abs(objective - tied_solution(p)[0])
            )

        if dimension == 2:
            continue
        tail_count = dimension - 2
        kappa = (tail_count - 1.0) / (2.0 * tail_count)
        for _ in range(80):
            p = float(rng.uniform(0.0, 1.0))
            vectors = rng.normal(size=(dimension, dimension + 1))
            vectors /= np.linalg.norm(vectors, axis=0, keepdims=True)
            squared_gram = (vectors.T @ vectors) ** 2
            total = float(np.sum(np.triu(squared_gram, k=1)))
            objective = (
                total
                + (p - 1.0) * squared_gram[0, 1]
                - p * squared_gram[1, 2]
            )
            minimum_objective_margin = min(
                minimum_objective_margin, objective - tied_solution(p)[0]
            )

            core = vectors[:, :3]
            tail = vectors[:, 3:]
            core_gram = core.T @ core
            lam = max(0.0, float(np.min(np.linalg.eigvalsh(core_gram))))
            a = core @ core.T
            w = tail @ tail.T
            tail_energy = float(
                np.trace(a @ w)
                + 0.5 * (np.trace(w @ w) - tail_count)
            )
            lower_bound = lam - kappa * lam**2
            minimum_tail_bound_slack = min(
                minimum_tail_bound_slack, tail_energy - lower_bound
            )

    assert maximum_block_error < 1e-12
    assert minimum_objective_margin > -1e-10
    assert minimum_tail_bound_slack > -1e-10
    return (
        maximum_block_error,
        minimum_objective_margin,
        minimum_tail_bound_slack,
    )


def finite_rate_check() -> tuple[float, float]:
    maximum_stationarity = 0.0
    minimum_absolute_eigenvalue = math.inf
    for p in [0.4, 0.5, 0.6]:
        alpha, beta = angle_representative(p)
        difference = alpha - beta
        gradient = np.array(
            [
                -p * math.sin(2.0 * difference) - math.sin(2.0 * alpha),
                p * math.sin(2.0 * difference)
                - (1.0 - p) * math.sin(2.0 * beta),
            ]
        )
        maximum_stationarity = max(
            maximum_stationarity, float(np.max(np.abs(gradient)))
        )
        hessian = np.array(
            [
                [
                    -2.0 * p * math.cos(2.0 * difference)
                    - 2.0 * math.cos(2.0 * alpha),
                    2.0 * p * math.cos(2.0 * difference),
                ],
                [
                    2.0 * p * math.cos(2.0 * difference),
                    -2.0 * p * math.cos(2.0 * difference)
                    - 2.0 * (1.0 - p) * math.cos(2.0 * beta),
                ],
            ]
        )
        assert float(np.min(np.linalg.eigvalsh(hessian))) > 1e-8
        g = np.array(
            [
                -math.sin(2.0 * difference),
                math.sin(2.0 * difference) + math.sin(2.0 * beta),
            ]
        )
        _, _, delta = tied_solution(p)
        for epsilon in [0.0, 0.1, 0.3]:
            scale = 1.0 - 2.0 * epsilon
            actor_state = (p - epsilon) / scale
            assert 0.0 < actor_state < 1.0
            for preconditioner in [
                np.eye(2),
                np.array([[2.0, 1.0], [1.0, 2.0]]),
                np.array([[1.0, 0.2], [0.2, 3.0]]),
            ]:
                assert float(np.min(np.linalg.eigvalsh(preconditioner))) > 0.0
                for kappa in [0.01, 1.0, 100.0]:
                    actor_mobility = actor_state * (1.0 - actor_state)
                    jacobian = np.block(
                        [
                            [
                                -kappa * preconditioner @ hessian,
                                -kappa
                                * scale
                                * (preconditioner @ g)[:, None],
                            ],
                            [
                                -scale * actor_mobility * g[None, :],
                                np.zeros((1, 1)),
                            ],
                        ]
                    )
                    eigenvalues = np.linalg.eigvals(jacobian)
                    assert float(np.max(np.abs(eigenvalues.imag))) < 1e-8
                    real = np.sort(eigenvalues.real)
                    assert real[0] < 0.0 and real[1] < 0.0 and real[2] > 0.0
                    minimum_absolute_eigenvalue = min(
                        minimum_absolute_eigenvalue, float(np.min(np.abs(real)))
                    )
        assert abs(delta - (math.cos(difference) ** 2 - math.cos(beta) ** 2)) < 1e-9

    for kappa in [0.01, 1.0, 100.0]:
        for delta in [-0.8, 0.0, 0.8]:
            low = np.array([-2.0 * kappa, -2.0 * kappa, delta - 1.0])
            high = np.array(
                [
                    -kappa * (3.0 + math.sqrt(5.0)),
                    -kappa * (3.0 - math.sqrt(5.0)),
                    -(delta + 1.0),
                ]
            )
            assert float(np.max(low)) < 0.0
            assert float(np.max(high)) < 0.0
            for epsilon in [0.1, 0.3]:
                low_hessian = 2.0 * np.array(
                    [
                        [1.0 - epsilon, epsilon],
                        [epsilon, 1.0 - 2.0 * epsilon],
                    ]
                )
                high_hessian = 2.0 * np.array(
                    [
                        [2.0 - epsilon, -(1.0 - epsilon)],
                        [-(1.0 - epsilon), 1.0 - 2.0 * epsilon],
                    ]
                )
                assert float(np.min(np.linalg.eigvalsh(low_hessian))) > 0.0
                assert float(np.min(np.linalg.eigvalsh(high_hessian))) > 0.0
    assert maximum_stationarity < 1e-8
    return maximum_stationarity, minimum_absolute_eigenvalue


def policy_floor_check() -> float:
    maximum_residual = 0.0
    delta = 0.3
    grid = np.linspace(P_MINUS, P_PLUS, 100_001)
    slopes = np.array([tied_solution(float(p))[2] for p in grid])
    p_delta = float(grid[np.argmin(np.abs(slopes - delta))])
    for epsilon in [0.05, 0.2, 0.35]:
        s_delta = (p_delta - epsilon) / (1.0 - 2.0 * epsilon)
        assert 0.0 < s_delta < 1.0
        low_value = tied_solution(epsilon)[0]
        high_value = tied_solution(1.0 - epsilon)[0]
        maximum_residual = max(maximum_residual, abs(low_value - high_value))
        assert tied_solution(epsilon)[1][0] == 1.0
        assert tied_solution(1.0 - epsilon)[1][2] == 1.0
    assert maximum_residual < 1e-12
    return maximum_residual


def entropy_regularization_check() -> tuple[float, float]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    q_boundary = P_MINUS * P_PLUS
    q_grid = np.linspace(q_boundary, 0.25, 20_001)
    curvature = 0.5 * (q_grid**-2 - 3.0 * q_grid**-1 - q_grid)
    curvature_error = max(
        abs(float(np.min(curvature)) - 15.0 / 8.0),
        abs(float(np.max(curvature)) - 5.0 / 2.0),
    )
    assert curvature_error < 1e-12

    maximum_root_residual = 0.0
    for delta in [0.2, 0.5, 0.8]:
        cap = min((1.0 - delta) / math.log(phi), 15.0 / 8.0)
        for fraction in [0.3, 0.7, 0.99]:
            tau = fraction * cap
            p_low = 1.0 / (1.0 + math.exp((1.0 - delta) / tau))
            p_high = 1.0 / (1.0 + math.exp(-(1.0 + delta) / tau))
            assert 0.0 < p_low < P_MINUS < P_PLUS < p_high < 1.0
            low_residual = delta - 1.0 + tau * math.log((1.0 - p_low) / p_low)
            high_residual = delta + 1.0 + tau * math.log(
                (1.0 - p_high) / p_high
            )
            maximum_root_residual = max(
                maximum_root_residual, abs(low_residual), abs(high_residual)
            )

            central = np.linspace(P_MINUS, 0.5, 20_001)
            values = np.array(
                [
                    delta
                    - tied_solution(float(p))[2]
                    + tau * math.log((1.0 - p) / p)
                    for p in central
                ]
            )
            crossings = np.where(values[:-1] * values[1:] < 0.0)[0]
            assert len(crossings) == 1
            assert values[0] < 0.0 and values[-1] > 0.0

            environmental_low = delta * p_low - p_low
            environmental_high = delta * p_high - (1.0 - p_high)
            assert environmental_high > environmental_low

    assert maximum_root_residual < 1e-10
    return curvature_error, maximum_root_residual


def gram_regularization_check() -> tuple[float, float]:
    maximum_slope_error = 0.0
    maximum_threshold_error = 0.0
    for regularizer in [0.0, 0.2, 1.0, 1.0001, 2.0, 10.0, 100.0]:
        if regularizer == 0.0:
            _, gram, slope = tied_solution(0.0)
        else:
            _, gram = weighted_solution(
                regularizer, 1.0 + regularizer, 1.0 + regularizer
            )
            slope = float(gram[0] - gram[2])
        exact = (
            1.0
            if regularizer <= 1.0
            else (3.0 * regularizer + 1.0) / (4.0 * regularizer**2)
        )
        maximum_slope_error = max(maximum_slope_error, abs(slope - exact))
        if regularizer <= 1.0:
            assert np.max(np.abs(gram - np.array([1.0, 0.0, 0.0]))) < 1e-12

    for delta in [0.05, 0.2, 0.5, 0.9, 0.999]:
        critical = (3.0 + math.sqrt(9.0 + 16.0 * delta)) / (8.0 * delta)
        assert critical > 1.0
        threshold_residual = abs(
            (3.0 * critical + 1.0) / (4.0 * critical**2) - delta
        )
        maximum_threshold_error = max(maximum_threshold_error, threshold_residual)

        slopes = []
        for p in np.linspace(0.0, 1.0, 1001):
            _, gram = weighted_solution(
                p + critical, 1.0 + critical, 1.0 - p + critical
            )
            slopes.append(float(gram[0] - gram[2]))
        assert abs(slopes[0] - delta) < 1e-11
        assert all(slopes[i] > slopes[i + 1] for i in range(len(slopes) - 1))
        assert max(slopes[1:]) < delta

        above = 1.001 * critical
        above_slopes = []
        for p in np.linspace(0.0, 1.0, 101):
            _, gram = weighted_solution(p + above, 1.0 + above, 1.0 - p + above)
            above_slopes.append(float(gram[0] - gram[2]))
        assert max(above_slopes) < delta

    assert maximum_slope_error < 1e-11
    assert maximum_threshold_error < 1e-12
    return maximum_slope_error, maximum_threshold_error


def untied_loss_grid(p: float, eta: float, angles: np.ndarray) -> np.ndarray:
    g12 = np.cos(angles[:, None]) ** 2
    g13 = np.cos(angles[None, :]) ** 2
    g23 = np.cos(angles[:, None] - angles[None, :]) ** 2
    q12, q13, q23 = p / 2.0, 0.5, (1.0 - p) / 2.0
    r = [q12 + q13, q12 + q23, q13 + q23]
    rows = [
        (r[0], q12 / r[0], q13 / r[0], g23, g12, g13),
        (r[1], q12 / r[1], q23 / r[1], g13, g12, g23),
        (r[2], q13 / r[2], q23 / r[2], g12, g13, g23),
    ]
    total = np.zeros_like(g12 + g13)
    for ri, aj, ak, gjk, gij, gik in rows:
        cross = aj * ak * (1.0 - gjk)
        numerator = eta * (eta + 1.0) + cross
        denominator = (
            eta * (eta + 2.0)
            + cross
            + aj * (1.0 - gij)
            + ak * (1.0 - gik)
        )
        total += ri * numerator / denominator
    return total


def untied_rational_check(rng: np.random.Generator) -> float:
    maximum = 0.0
    for _ in range(300):
        p = float(rng.uniform(0.0, 1.0))
        eta = float(np.exp(rng.uniform(math.log(0.02), math.log(200.0))))
        theta = rng.uniform(0.0, math.pi, size=3)
        vectors = [np.array([math.cos(t), math.sin(t)]) for t in theta]
        q = {(0, 1): p / 2.0, (0, 2): 0.5, (1, 2): (1.0 - p) / 2.0}
        r = [q[(0, 1)] + q[(0, 2)], q[(0, 1)] + q[(1, 2)], q[(0, 2)] + q[(1, 2)]]
        direct = 0.0
        rational = 0.0
        for i in range(3):
            others = [j for j in range(3) if j != i]
            j, k = others
            alpha_j = q[tuple(sorted((i, j)))] / r[i]
            alpha_k = q[tuple(sorted((i, k)))] / r[i]
            matrix = eta * np.eye(2) + np.outer(vectors[i], vectors[i])
            matrix += alpha_j * np.outer(vectors[j], vectors[j])
            matrix += alpha_k * np.outer(vectors[k], vectors[k])
            direct += r[i] * (
                1.0 - vectors[i] @ np.linalg.solve(matrix, vectors[i])
            )

            gij = float(vectors[i] @ vectors[j]) ** 2
            gik = float(vectors[i] @ vectors[k]) ** 2
            gjk = float(vectors[j] @ vectors[k]) ** 2
            cross = alpha_j * alpha_k * (1.0 - gjk)
            numerator = eta * (eta + 1.0) + cross
            denominator = (
                eta * (eta + 2.0)
                + cross
                + alpha_j * (1.0 - gij)
                + alpha_k * (1.0 - gik)
            )
            rational += r[i] * numerator / denominator
        maximum = max(maximum, abs(float(direct) - rational))
    assert maximum < 1e-11
    return maximum


def untied_checks() -> tuple[float, float]:
    angles = np.linspace(0.0, math.pi, 1001, endpoint=False)
    maximum_endpoint_error = 0.0
    minimum_signed_slope = math.inf
    for eta in [0.05, 0.2, 1.0, 5.0, 20.0, 100.0]:
        objective = untied_loss_grid(0.0, eta, angles)
        i, j = np.unravel_index(np.argmin(objective), objective.shape)
        v = [
            np.array([1.0, 0.0]),
            np.array([math.cos(angles[i]), math.sin(angles[i])]),
            np.array([math.cos(angles[j]), math.sin(angles[j])]),
        ]
        q = {(0, 1): 0.0, (0, 2): 0.5, (1, 2): 0.5}
        r = [0.5, 0.5, 1.0]
        w = []
        for feature in range(3):
            matrix = eta * np.eye(2) + np.outer(v[feature], v[feature])
            for other in range(3):
                if feature == other:
                    continue
                weight = q[tuple(sorted((feature, other)))] / r[feature]
                matrix += weight * np.outer(v[other], v[other])
            w.append(np.linalg.solve(matrix, v[feature]))

        def pair_loss(first: int, second: int) -> float:
            result = 0.0
            for target, other in [(first, second), (second, first)]:
                result += (
                    (w[target] @ v[target] - 1.0) ** 2
                    + (w[target] @ v[other]) ** 2
                    + eta * (w[target] @ w[target])
                )
            return float(result)

        d_s = 0.5 * (pair_loss(0, 2) + pair_loss(1, 2))
        d_e = 0.5 * (pair_loss(0, 1) + pair_loss(0, 2))
        error = abs((d_e - d_s) - 1.0 / (1.0 + eta) ** 2)
        maximum_endpoint_error = max(maximum_endpoint_error, error)

        slopes = []
        for p in [0.4, 0.5, 0.6]:
            step = 2e-4
            low = float(np.min(untied_loss_grid(p - step, eta, angles)))
            high = float(np.min(untied_loss_grid(p + step, eta, angles)))
            slopes.append((high - low) / (2.0 * step))
        assert slopes[0] > 1e-6 and abs(slopes[1]) < 5e-5 and slopes[2] < -1e-6
        minimum_signed_slope = min(minimum_signed_slope, slopes[0], -slopes[2])
    assert maximum_endpoint_error < 1e-6
    return maximum_endpoint_error, minimum_signed_slope


def main() -> None:
    rng = np.random.default_rng(SEED)
    write_phase_data()
    pair_error = pair_loss_check(rng)
    partial_reduction_error, partial_realization_error = (
        partial_observation_reduction_check(rng)
    )
    value_error, gram_error = weighted_grid_check(rng)
    envelope_error = envelope_check()
    lift_error, lift_margin, lift_tail_slack = codimension_lift_check(rng)
    finite_stationarity, finite_eigenvalue = finite_rate_check()
    floor_residual = policy_floor_check()
    entropy_curvature_error, entropy_root_error = entropy_regularization_check()
    gram_slope_error, gram_threshold_error = gram_regularization_check()
    rational_error = untied_rational_check(rng)
    untied_error, signed_slope = untied_checks()
    print(f"seed={SEED}")
    print(f"pair_loss_abs_error={pair_error:.10g}")
    print(f"partial_observation_reduction_error={partial_reduction_error:.10g}")
    print(f"partial_observation_realization_error={partial_realization_error:.10g}")
    print(f"weighted_max_value_error={value_error:.10g}")
    print(f"weighted_max_gram_error={gram_error:.10g}")
    print(f"envelope_max_error={envelope_error:.10g}")
    print(f"codimension_lift_block_error={lift_error:.10g}")
    print(f"codimension_lift_min_objective_margin={lift_margin:.10g}")
    print(f"codimension_lift_min_tail_bound_slack={lift_tail_slack:.10g}")
    print(f"finite_rate_stationarity_error={finite_stationarity:.10g}")
    print(f"finite_rate_min_abs_eigenvalue={finite_eigenvalue:.10g}")
    print(f"policy_floor_symmetry_error={floor_residual:.10g}")
    print(f"entropy_curvature_error={entropy_curvature_error:.10g}")
    print(f"entropy_root_error={entropy_root_error:.10g}")
    print(f"gram_endpoint_slope_error={gram_slope_error:.10g}")
    print(f"gram_threshold_error={gram_threshold_error:.10g}")
    print(f"untied_rational_max_error={rational_error:.10g}")
    print(f"untied_endpoint_max_error={untied_error:.10g}")
    print(f"untied_min_signed_slope={signed_slope:.10g}")
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()
