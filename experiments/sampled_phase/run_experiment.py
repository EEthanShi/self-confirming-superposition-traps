#!/usr/bin/env python3
"""Finite-sample optimization check for the tied phase diagram.

The primary protocol is frozen in PROTOCOL.md.  The experiment samples the
paper's horizon-two data generator, trains the normalized tied code from random
initializations, certifies optimization error against the exact empirical
weighted-frame optimum, and writes auditable CSV/JSON outputs for the paper.

This is explanatory evidence inside the solved model, not prevalence evidence
for deep RL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PAPER_DIR = EXPERIMENT_DIR.parents[1]
SCRIPTS_DIR = PAPER_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_theory import P_MINUS, P_PLUS, tied_solution, weighted_solution  # noqa: E402


BASE_SEED = 20260809
PRIMARY_N = 4096
PRIMARY_DATASETS = 50
PRIMARY_INITIALIZATIONS = 8
PRIMARY_STEP_SIZE = 0.05
PRIMARY_MAX_STEPS = 20_000
PRIMARY_GRAD_TOL = 1e-10
DELTA = 0.5
HELDOUT_N = 65_536

PAIR_12 = 0
PAIR_13 = 1
PAIR_23 = 2


@dataclass(frozen=True)
class Dataset:
    p: float
    x: np.ndarray
    mask: np.ndarray
    pair_ids: np.ndarray
    coefficients: np.ndarray


def seed_for(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *parts]))


def primary_p_grid() -> np.ndarray:
    values = list(np.linspace(0.0, 1.0, 21)) + [P_MINUS, P_PLUS]
    return np.asarray(sorted(set(float(value) for value in values)), dtype=float)


def sample_dataset(p: float, count: int, rng: np.random.Generator) -> Dataset:
    """Sample exactly from the branch/pair generator in Section 3."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0,1]")
    branch_e = rng.random(count) < p
    pair_coin = rng.integers(0, 2, size=count)
    pair_ids = np.empty(count, dtype=np.int8)
    pair_ids[branch_e & (pair_coin == 0)] = PAIR_12
    pair_ids[branch_e & (pair_coin == 1)] = PAIR_13
    pair_ids[(~branch_e) & (pair_coin == 0)] = PAIR_13
    pair_ids[(~branch_e) & (pair_coin == 1)] = PAIR_23

    z = rng.standard_normal((count, 3))
    mask = np.zeros((count, 3), dtype=float)
    mask[pair_ids == PAIR_12, 0:2] = 1.0
    mask[pair_ids == PAIR_13, 0] = 1.0
    mask[pair_ids == PAIR_13, 2] = 1.0
    mask[pair_ids == PAIR_23, 1:3] = 1.0
    x = z * mask

    coefficients = np.asarray(
        [
            float(np.sum(x[pair_ids == pair_id] ** 2)) / count
            for pair_id in (PAIR_12, PAIR_13, PAIR_23)
        ],
        dtype=float,
    )
    return Dataset(p=p, x=x, mask=mask, pair_ids=pair_ids, coefficients=coefficients)


def vectors_from_angles(alpha: float | np.ndarray, beta: float | np.ndarray) -> np.ndarray:
    alpha_array = np.asarray(alpha, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    shape = np.broadcast_shapes(alpha_array.shape, beta_array.shape)
    alpha_array = np.broadcast_to(alpha_array, shape)
    beta_array = np.broadcast_to(beta_array, shape)
    vectors = np.empty(shape + (2, 3), dtype=float)
    vectors[..., 0, 0] = np.cos(alpha_array)
    vectors[..., 1, 0] = np.sin(alpha_array)
    vectors[..., 0, 1] = np.cos(beta_array)
    vectors[..., 1, 1] = np.sin(beta_array)
    vectors[..., 0, 2] = 1.0
    vectors[..., 1, 2] = 0.0
    return vectors


def gram_from_vectors(vectors: np.ndarray) -> np.ndarray:
    dots = np.swapaxes(vectors, -2, -1) @ vectors
    return np.stack(
        [dots[..., 0, 1] ** 2, dots[..., 0, 2] ** 2, dots[..., 1, 2] ** 2],
        axis=-1,
    )


def gram_from_angles(alpha: float | np.ndarray, beta: float | np.ndarray) -> np.ndarray:
    return gram_from_vectors(vectors_from_angles(alpha, beta))


def raw_loss_and_angle_gradient(dataset: Dataset, alpha: float, beta: float) -> tuple[float, np.ndarray]:
    """Unreduced tied forward pass and exact gradient on a fixed raw dataset."""
    vectors = vectors_from_angles(alpha, beta)
    hidden = dataset.x @ vectors.T
    predictions = hidden @ vectors
    errors = (predictions - dataset.x) * dataset.mask
    count = len(dataset.x)
    loss = float(np.mean(np.sum(errors**2, axis=1)))

    prediction_gradient = 2.0 * errors / count
    decoder_gradient = hidden.T @ prediction_gradient
    hidden_gradient = prediction_gradient @ vectors.T
    encoder_gradient = hidden_gradient.T @ dataset.x
    vector_gradient = decoder_gradient + encoder_gradient

    tangent_alpha = np.asarray([-math.sin(alpha), math.cos(alpha)])
    tangent_beta = np.asarray([-math.sin(beta), math.cos(beta)])
    angle_gradient = np.asarray(
        [
            float(vector_gradient[:, 0] @ tangent_alpha),
            float(vector_gradient[:, 1] @ tangent_beta),
        ]
    )
    return loss, angle_gradient


def reduced_loss_and_angle_gradient(
    coefficients: np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact sufficient-statistic form of the fixed raw reconstruction loss."""
    coefficients = np.asarray(coefficients, dtype=float)
    alpha_array = np.asarray(alpha, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    a, b, c = np.moveaxis(coefficients, -1, 0)
    difference = alpha_array - beta_array
    loss = (
        a * np.cos(difference) ** 2
        + b * np.cos(alpha_array) ** 2
        + c * np.cos(beta_array) ** 2
    )
    gradient = np.stack(
        [
            -a * np.sin(2.0 * difference) - b * np.sin(2.0 * alpha_array),
            a * np.sin(2.0 * difference) - c * np.sin(2.0 * beta_array),
        ],
        axis=-1,
    )
    return loss, gradient


def raw_reduction_check(trials: int = 32) -> tuple[float, float]:
    maximum_loss_error = 0.0
    maximum_gradient_error = 0.0
    rng = seed_for(10)
    for trial in range(trials):
        p = float(rng.uniform())
        dataset = sample_dataset(p, 257 + trial, rng)
        alpha, beta = rng.uniform(0.0, math.pi, size=2)
        raw_loss, raw_gradient = raw_loss_and_angle_gradient(dataset, float(alpha), float(beta))
        reduced_loss, reduced_gradient = reduced_loss_and_angle_gradient(
            dataset.coefficients, alpha, beta
        )
        maximum_loss_error = max(maximum_loss_error, abs(raw_loss - float(reduced_loss)))
        maximum_gradient_error = max(
            maximum_gradient_error,
            float(np.max(np.abs(raw_gradient - reduced_gradient))),
        )
    return maximum_loss_error, maximum_gradient_error


def optimize_angles(
    coefficients: np.ndarray,
    initial_angles: np.ndarray,
    step_size: float,
    max_steps: int,
    gradient_tolerance: float,
) -> dict[str, np.ndarray | int]:
    """Vectorized full-batch gradient descent on fixed empirical risks."""
    coefficients = np.asarray(coefficients, dtype=float)
    angles = np.asarray(initial_angles, dtype=float).copy()
    if angles.shape[-1] != 2:
        raise ValueError("initial_angles must end in (alpha,beta)")
    expanded_coefficients = np.broadcast_to(
        coefficients[..., None, :], angles.shape[:-1] + (3,)
    )
    converged_step = np.full(angles.shape[:-1], max_steps, dtype=int)
    active = np.ones(angles.shape[:-1], dtype=bool)
    steps_used = max_steps
    for step in range(max_steps):
        _, gradient = reduced_loss_and_angle_gradient(
            expanded_coefficients, angles[..., 0], angles[..., 1]
        )
        gradient_norm = np.max(np.abs(gradient), axis=-1)
        newly_converged = active & (gradient_norm < gradient_tolerance)
        converged_step[newly_converged] = step
        active &= ~newly_converged
        if not np.any(active):
            steps_used = step
            break
        angles[active] -= step_size * gradient[active]
        angles[active] %= math.pi
    losses, gradients = reduced_loss_and_angle_gradient(
        expanded_coefficients, angles[..., 0], angles[..., 1]
    )
    return {
        "angles": angles,
        "losses": losses,
        "gradient_norms": np.max(np.abs(gradients), axis=-1),
        "converged_steps": converged_step,
        "steps_used": steps_used,
    }


def d3_loss_and_gradient(coefficients: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = np.moveaxis(np.asarray(coefficients, dtype=float), -1, 0)
    v1, v2, v3 = [vectors[..., :, index] for index in range(3)]
    c12 = np.sum(v1 * v2, axis=-1)
    c13 = np.sum(v1 * v3, axis=-1)
    c23 = np.sum(v2 * v3, axis=-1)
    loss = a * c12**2 + b * c13**2 + c * c23**2
    gradient = np.empty_like(vectors)
    gradient[..., :, 0] = 2.0 * a[..., None] * c12[..., None] * v2 + 2.0 * b[..., None] * c13[..., None] * v3
    gradient[..., :, 1] = 2.0 * a[..., None] * c12[..., None] * v1 + 2.0 * c[..., None] * c23[..., None] * v3
    gradient[..., :, 2] = 2.0 * b[..., None] * c13[..., None] * v1 + 2.0 * c[..., None] * c23[..., None] * v2
    radial = np.sum(gradient * vectors, axis=-2, keepdims=True)
    tangent_gradient = gradient - vectors * radial
    return loss, tangent_gradient


def optimize_d3(
    coefficients: np.ndarray,
    initial_vectors: np.ndarray,
    step_size: float,
    max_steps: int,
    gradient_tolerance: float,
) -> dict[str, np.ndarray | int]:
    vectors = np.asarray(initial_vectors, dtype=float).copy()
    vectors /= np.linalg.norm(vectors, axis=-2, keepdims=True)
    expanded_coefficients = np.broadcast_to(
        coefficients[..., None, :], vectors.shape[:-2] + (3,)
    )
    converged_step = np.full(vectors.shape[:-2], max_steps, dtype=int)
    active = np.ones(vectors.shape[:-2], dtype=bool)
    steps_used = max_steps
    for step in range(max_steps):
        _, gradient = d3_loss_and_gradient(expanded_coefficients, vectors)
        gradient_norm = np.max(np.abs(gradient), axis=(-2, -1))
        newly_converged = active & (gradient_norm < gradient_tolerance)
        converged_step[newly_converged] = step
        active &= ~newly_converged
        if not np.any(active):
            steps_used = step
            break
        vectors[active] -= step_size * gradient[active]
        vectors[active] /= np.linalg.norm(vectors[active], axis=-2, keepdims=True)
    losses, gradients = d3_loss_and_gradient(expanded_coefficients, vectors)
    return {
        "vectors": vectors,
        "losses": losses,
        "gradient_norms": np.max(np.abs(gradients), axis=(-2, -1)),
        "converged_steps": converged_step,
        "steps_used": steps_used,
    }


def forced_branch_loss(vectors: np.ndarray, branch: str, count: int, rng: np.random.Generator) -> float:
    coin = rng.integers(0, 2, size=count)
    pair_ids = np.empty(count, dtype=np.int8)
    if branch == "S":
        pair_ids[coin == 0] = PAIR_13
        pair_ids[coin == 1] = PAIR_23
    elif branch == "E":
        pair_ids[coin == 0] = PAIR_12
        pair_ids[coin == 1] = PAIR_13
    else:
        raise ValueError("branch must be S or E")
    z = rng.standard_normal((count, 3))
    mask = np.zeros((count, 3), dtype=float)
    mask[pair_ids == PAIR_12, 0:2] = 1.0
    mask[pair_ids == PAIR_13, 0] = 1.0
    mask[pair_ids == PAIR_13, 2] = 1.0
    mask[pair_ids == PAIR_23, 1:3] = 1.0
    x = z * mask
    hidden = x @ vectors.T
    predictions = hidden @ vectors
    errors = (predictions - x) * mask
    return float(np.mean(np.sum(errors**2, axis=1)))


def quantiles(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    return tuple(float(value) for value in np.quantile(array, [0.1, 0.5, 0.9]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_primary(dataset_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    p_values = sorted({float(row["p"]) for row in dataset_rows})
    for p in p_values:
        subset = [row for row in dataset_rows if float(row["p"]) == p]
        theory = tied_solution(p)
        row: dict[str, object] = {
            "p": f"{p:.10f}",
            "theory_g12": f"{theory[1][0]:.10f}",
            "theory_g13": f"{theory[1][1]:.10f}",
            "theory_g23": f"{theory[1][2]:.10f}",
            "theory_D": f"{theory[2]:.10f}",
        }
        for metric in ("g12", "g13", "g23", "D"):
            low, median, high = quantiles(float(item[metric]) for item in subset)
            row[f"{metric}_low"] = f"{low:.10f}"
            row[f"{metric}_median"] = f"{median:.10f}"
            row[f"{metric}_high"] = f"{high:.10f}"
            row[f"{metric}_errminus"] = f"{median - low:.10f}"
            row[f"{metric}_errplus"] = f"{high - median:.10f}"
        output.append(row)
    return output


def squared_gram_medoid_index(grams: np.ndarray) -> int:
    """Return the true medoid under squared Euclidean distance in Gram space."""
    grams = np.asarray(grams, dtype=float)
    if grams.ndim != 2 or grams.shape[1] != 3 or len(grams) == 0:
        raise ValueError("grams must have shape (n,3) with n>0")
    pairwise_differences = grams[:, None, :] - grams[None, :, :]
    total_distances = np.sum(pairwise_differences**2, axis=(1, 2))
    return int(np.argmin(total_distances))


def representative_macros(dataset_rows: list[dict[str, object]]) -> str:
    labels = {0.2: "Low", 0.5: "Mid", 0.8: "High"}
    feature_names = ["One", "Two", "Three"]
    lines = ["% Generated by experiments/sampled_phase/run_experiment.py"]
    for p, label in labels.items():
        subset = [row for row in dataset_rows if abs(float(row["p"]) - p) < 1e-12]
        grams = np.asarray([[row["g12"], row["g13"], row["g23"]] for row in subset], dtype=float)
        medoid_index = squared_gram_medoid_index(grams)
        chosen = subset[medoid_index]
        vectors = vectors_from_angles(float(chosen["alpha"]), float(chosen["beta"]))
        lines.append(f"\\def\\{label}DatasetSeed{{{int(chosen['dataset_index'])}}}")
        for feature_index, feature_name in enumerate(feature_names):
            x, y = vectors[:, feature_index]
            lines.append(f"\\def\\{label}V{feature_name}X{{{x:.10f}}}")
            lines.append(f"\\def\\{label}V{feature_name}Y{{{y:.10f}}}")
    return "\n".join(lines) + "\n"


def run_primary(
    dataset_count: int,
    initialization_count: int,
    sample_count: int,
    max_steps: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[float, int], Dataset]]:
    p_grid = primary_p_grid()
    datasets: dict[tuple[float, int], Dataset] = {}
    coefficient_rows = []
    for p_index, p in enumerate(p_grid):
        for dataset_index in range(dataset_count):
            dataset = sample_dataset(
                float(p), sample_count, seed_for(100, p_index, dataset_index, sample_count)
            )
            datasets[(float(p), dataset_index)] = dataset
            coefficient_rows.append(dataset.coefficients)
    coefficients = np.asarray(coefficient_rows, dtype=float)
    optimizer_rng = seed_for(200, dataset_count, initialization_count, sample_count)
    initial_angles = optimizer_rng.uniform(
        0.0, math.pi, size=(len(coefficients), initialization_count, 2)
    )
    optimized = optimize_angles(
        coefficients,
        initial_angles,
        PRIMARY_STEP_SIZE,
        max_steps,
        PRIMARY_GRAD_TOL,
    )
    angles = np.asarray(optimized["angles"])
    losses = np.asarray(optimized["losses"])
    gradient_norms = np.asarray(optimized["gradient_norms"])
    grams = gram_from_angles(angles[..., 0], angles[..., 1])

    run_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    flat_index = 0
    for p_index, p in enumerate(p_grid):
        theory_value, theory_gram, theory_d = tied_solution(float(p))
        for dataset_index in range(dataset_count):
            dataset = datasets[(float(p), dataset_index)]
            empirical_value, empirical_gram = weighted_solution(*dataset.coefficients)
            run_slice = losses[flat_index]
            best_initialization = int(np.argmin(run_slice))
            for initialization in range(initialization_count):
                gram = grams[flat_index, initialization]
                run_rows.append(
                    {
                        "p": float(p),
                        "p_index": p_index,
                        "dataset_index": dataset_index,
                        "initialization": initialization,
                        "A": float(dataset.coefficients[0]),
                        "B": float(dataset.coefficients[1]),
                        "C": float(dataset.coefficients[2]),
                        "alpha": float(angles[flat_index, initialization, 0]),
                        "beta": float(angles[flat_index, initialization, 1]),
                        "g12": float(gram[0]),
                        "g13": float(gram[1]),
                        "g23": float(gram[2]),
                        "D": float(gram[0] - gram[2]),
                        "empirical_loss": float(run_slice[initialization]),
                        "empirical_optimum": float(empirical_value),
                        "optimization_error": float(run_slice[initialization] - empirical_value),
                        "gradient_norm": float(gradient_norms[flat_index, initialization]),
                        "is_best": int(initialization == best_initialization),
                    }
                )
            best_gram = grams[flat_index, best_initialization]
            learned_d = float(best_gram[0] - best_gram[2])
            dataset_rows.append(
                {
                    "p": float(p),
                    "p_index": p_index,
                    "dataset_index": dataset_index,
                    "A": float(dataset.coefficients[0]),
                    "B": float(dataset.coefficients[1]),
                    "C": float(dataset.coefficients[2]),
                    "empirical_optimum": float(empirical_value),
                    "empirical_g12": float(empirical_gram[0]),
                    "empirical_g13": float(empirical_gram[1]),
                    "empirical_g23": float(empirical_gram[2]),
                    "theory_optimum": float(theory_value),
                    "theory_g12": float(theory_gram[0]),
                    "theory_g13": float(theory_gram[1]),
                    "theory_g23": float(theory_gram[2]),
                    "theory_D": float(theory_d),
                    "best_initialization": best_initialization,
                    "alpha": float(angles[flat_index, best_initialization, 0]),
                    "beta": float(angles[flat_index, best_initialization, 1]),
                    "g12": float(best_gram[0]),
                    "g13": float(best_gram[1]),
                    "g23": float(best_gram[2]),
                    "D": learned_d,
                    "deployed_gap": float(DELTA - learned_d),
                    "optimization_error": float(run_slice[best_initialization] - empirical_value),
                    "gram_error_to_empirical": float(np.max(np.abs(best_gram - empirical_gram))),
                    "gram_error_to_population": float(np.max(np.abs(best_gram - theory_gram))),
                    "gradient_norm": float(gradient_norms[flat_index, best_initialization]),
                }
            )
            flat_index += 1
    return run_rows, dataset_rows, datasets


def run_d3_control(
    datasets: dict[tuple[float, int], Dataset],
    dataset_count: int,
    initialization_count: int,
    max_steps: int,
) -> list[dict[str, object]]:
    p_values = [0.2, 0.5, 0.8]
    selected = [datasets[(p, index)] for p in p_values for index in range(dataset_count)]
    coefficients = np.asarray([dataset.coefficients for dataset in selected], dtype=float)
    rng = seed_for(300, dataset_count, initialization_count)
    initial_vectors = rng.standard_normal(
        (len(selected), initialization_count, 3, 3)
    )
    optimized = optimize_d3(
        coefficients,
        initial_vectors,
        PRIMARY_STEP_SIZE,
        max_steps,
        PRIMARY_GRAD_TOL,
    )
    vectors = np.asarray(optimized["vectors"])
    losses = np.asarray(optimized["losses"])
    gradient_norms = np.asarray(optimized["gradient_norms"])
    grams = gram_from_vectors(vectors)
    rows: list[dict[str, object]] = []
    flat_index = 0
    for p in p_values:
        for dataset_index in range(dataset_count):
            best_initialization = int(np.argmin(losses[flat_index]))
            gram = grams[flat_index, best_initialization]
            learned_d = float(gram[0] - gram[2])
            row: dict[str, object] = {
                "p": p,
                "dataset_index": dataset_index,
                "best_initialization": best_initialization,
                "g12": float(gram[0]),
                "g13": float(gram[1]),
                "g23": float(gram[2]),
                "D": learned_d,
                "deployed_gap": float(DELTA - learned_d),
                "empirical_loss": float(losses[flat_index, best_initialization]),
                "gradient_norm": float(gradient_norms[flat_index, best_initialization]),
            }
            for vector_index in range(3):
                for coordinate in range(3):
                    row[f"v{vector_index + 1}_{coordinate + 1}"] = float(
                        vectors[flat_index, best_initialization, coordinate, vector_index]
                    )
            rows.append(row)
            flat_index += 1
    return rows


def run_heldout(
    dataset_rows: list[dict[str, object]],
    d3_rows: list[dict[str, object]],
    heldout_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dimension in (2, 3):
        source = dataset_rows if dimension == 2 else d3_rows
        source = [row for row in source if float(row["p"]) in (0.2, 0.5, 0.8)]
        for row in source:
            p = float(row["p"])
            dataset_index = int(row["dataset_index"])
            if dimension == 2:
                vectors = vectors_from_angles(float(row["alpha"]), float(row["beta"]))
            else:
                vectors = np.asarray(
                    [
                        [row[f"v{feature}_{coordinate}"] for feature in range(1, 4)]
                        for coordinate in range(1, 4)
                    ],
                    dtype=float,
                )
            d_s = forced_branch_loss(
                vectors, "S", heldout_count, seed_for(400, dimension, int(round(p * 1000)), dataset_index, 0)
            )
            d_e = forced_branch_loss(
                vectors, "E", heldout_count, seed_for(400, dimension, int(round(p * 1000)), dataset_index, 1)
            )
            gram_d = float(row["D"])
            rows.append(
                {
                    "dimension": dimension,
                    "p": p,
                    "dataset_index": dataset_index,
                    "D_S_mc": d_s,
                    "D_E_mc": d_e,
                    "D_mc": d_e - d_s,
                    "D_gram": gram_d,
                    "absolute_bridge_error": abs((d_e - d_s) - gram_d),
                    "gap_mc": DELTA - (d_e - d_s),
                    "gap_gram": DELTA - gram_d,
                }
            )
    return rows


def run_sample_size_robustness() -> list[dict[str, object]]:
    sample_sizes = [256, 1024, 4096, 16_384]
    p_values = [0.2, 0.35, P_MINUS, 0.5, P_PLUS, 0.65, 0.8]
    rows: list[dict[str, object]] = []
    for sample_index, sample_count in enumerate(sample_sizes):
        for p_index, p in enumerate(p_values):
            _, population_gram, _ = tied_solution(float(p))
            for dataset_index in range(100):
                dataset = sample_dataset(
                    float(p),
                    sample_count,
                    seed_for(500, sample_index, p_index, dataset_index),
                )
                _, empirical_gram = weighted_solution(*dataset.coefficients)
                rows.append(
                    {
                        "N": sample_count,
                        "p": float(p),
                        "dataset_index": dataset_index,
                        "empirical_g12": float(empirical_gram[0]),
                        "empirical_g13": float(empirical_gram[1]),
                        "empirical_g23": float(empirical_gram[2]),
                        "population_max_gram_error": float(
                            np.max(np.abs(empirical_gram - population_gram))
                        ),
                    }
                )
    return rows


def sample_size_table_rows(sample_size_rows: list[dict[str, object]]) -> str:
    """Render the appendix table directly from the retained robustness runs."""
    targets = [P_MINUS, 0.5, P_PLUS]
    lines = [
        "% Generated by experiments/sampled_phase/run_experiment.py",
        r"\def\SampleSizeTableRows{%",
    ]
    for sample_count in sorted({int(row["N"]) for row in sample_size_rows}):
        cells = []
        for p in targets:
            subset = [
                float(row["population_max_gram_error"])
                for row in sample_size_rows
                if int(row["N"]) == sample_count
                and abs(float(row["p"]) - p) < 1e-12
            ]
            if not subset:
                raise ValueError(f"missing sample-size rows for N={sample_count}, p={p}")
            median, high = np.quantile(np.asarray(subset, dtype=float), [0.5, 0.9])
            cells.append(f"{median:.3f} / {high:.3f}")
        lines.append(f"{sample_count} & " + " & ".join(cells) + r" \\%")
    lines.append("}")
    return "\n".join(lines) + "\n"


def summarize_results(
    raw_loss_error: float,
    raw_gradient_error: float,
    run_rows: list[dict[str, object]],
    dataset_rows: list[dict[str, object]],
    d3_rows: list[dict[str, object]],
    heldout_rows: list[dict[str, object]],
    sample_size_rows: list[dict[str, object]],
) -> dict[str, object]:
    best_success = np.mean([float(row["optimization_error"]) < 1e-8 for row in dataset_rows])
    single_success = np.mean([float(row["optimization_error"]) < 1e-8 for row in run_rows])
    empirical_gram_errors = np.asarray(
        [float(row["gram_error_to_empirical"]) for row in dataset_rows], dtype=float
    )
    median_empirical_gram_error = float(np.median(empirical_gram_errors))
    low = [row for row in dataset_rows if abs(float(row["p"]) - 0.2) < 1e-12]
    high = [row for row in dataset_rows if abs(float(row["p"]) - 0.8) < 1e-12]
    low_collision = np.mean(
        [
            float(row["g12"]) > 0.99
            and float(row["g13"]) < 0.01
            and float(row["g23"]) < 0.01
            for row in low
        ]
    )
    high_collision = np.mean(
        [
            float(row["g23"]) > 0.99
            and float(row["g12"]) < 0.01
            and float(row["g13"]) < 0.01
            for row in high
        ]
    )
    low_reversal = np.mean([float(row["deployed_gap"]) < 0.0 for row in low])
    d3_capacity = np.mean(
        [
            max(float(row["g12"]), float(row["g13"]), float(row["g23"])) < 0.01
            and float(row["deployed_gap"]) > 0.0
            for row in d3_rows
        ]
    )
    bridge_errors_by_dimension = {
        dimension: np.asarray(
            [
                float(row["absolute_bridge_error"])
                for row in heldout_rows
                if int(row["dimension"]) == dimension
            ],
            dtype=float,
        )
        for dimension in (2, 3)
    }
    robustness_summary = []
    for sample_count in sorted({int(row["N"]) for row in sample_size_rows}):
        subset = [
            float(row["population_max_gram_error"])
            for row in sample_size_rows
            if int(row["N"]) == sample_count
        ]
        low_q, median, high_q = quantiles(subset)
        robustness_summary.append(
            {"N": sample_count, "error_q10": low_q, "error_median": median, "error_q90": high_q}
        )
    criteria = {
        "raw_reduction_identity": bool(
            raw_loss_error < 1e-10 and raw_gradient_error < 1e-10
        ),
        "best_of_eight_optimization": bool(best_success >= 0.99),
        "median_empirical_gram_error": bool(median_empirical_gram_error < 0.02),
        "low_collision": bool(low_collision >= 0.95),
        "high_collision": bool(high_collision >= 0.95),
        "low_gap_reversal": bool(low_reversal >= 0.95),
        "d3_capacity_control": bool(d3_capacity >= 0.95),
    }
    return {
        "protocol": {
            "base_seed": BASE_SEED,
            "N": PRIMARY_N,
            "dataset_count": PRIMARY_DATASETS,
            "initialization_count": PRIMARY_INITIALIZATIONS,
            "step_size": PRIMARY_STEP_SIZE,
            "max_steps": PRIMARY_MAX_STEPS,
            "gradient_tolerance": PRIMARY_GRAD_TOL,
            "delta": DELTA,
            "heldout_N_per_branch": HELDOUT_N,
        },
        "metrics": {
            "raw_loss_reduction_max_error": raw_loss_error,
            "raw_gradient_reduction_max_error": raw_gradient_error,
            "best_of_eight_optimization_success_rate": float(best_success),
            "single_start_optimization_success_rate": float(single_success),
            "median_gram_error_to_empirical_optimum": median_empirical_gram_error,
            "q90_gram_error_to_empirical_optimum": float(
                np.quantile(empirical_gram_errors, 0.9)
            ),
            "q99_gram_error_to_empirical_optimum": float(
                np.quantile(empirical_gram_errors, 0.99)
            ),
            "max_gram_error_to_empirical_optimum": float(np.max(empirical_gram_errors)),
            "low_collision_rate": float(low_collision),
            "high_collision_rate": float(high_collision),
            "low_gap_reversal_rate": float(low_reversal),
            "d3_capacity_and_positive_gap_rate": float(d3_capacity),
            "heldout_bridge_error_d2_median": float(
                np.median(bridge_errors_by_dimension[2])
            ),
            "heldout_bridge_error_d2_q90": float(
                np.quantile(bridge_errors_by_dimension[2], 0.9)
            ),
            "heldout_bridge_error_d3_median": float(
                np.median(bridge_errors_by_dimension[3])
            ),
            "heldout_bridge_error_d3_q90": float(
                np.quantile(bridge_errors_by_dimension[3], 0.9)
            ),
            "sample_size_robustness": robustness_summary,
        },
        "criteria": criteria,
        "all_primary_criteria_passed": all(criteria.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="small smoke run; never use for paper outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (defaults to outputs, or outputs_quick with --quick)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_output_dir = EXPERIMENT_DIR / "outputs"
    output_dir = args.output_dir or (
        EXPERIMENT_DIR / "outputs_quick" if args.quick else official_output_dir
    )
    if args.quick and output_dir.resolve() == official_output_dir.resolve():
        raise SystemExit("--quick refuses to overwrite the official outputs directory")

    dataset_count = 4 if args.quick else PRIMARY_DATASETS
    initialization_count = 3 if args.quick else PRIMARY_INITIALIZATIONS
    sample_count = 512 if args.quick else PRIMARY_N
    max_steps = 2000 if args.quick else PRIMARY_MAX_STEPS
    heldout_count = 4096 if args.quick else HELDOUT_N

    raw_loss_error, raw_gradient_error = raw_reduction_check(8 if args.quick else 32)
    run_rows, dataset_rows, datasets = run_primary(
        dataset_count, initialization_count, sample_count, max_steps
    )
    d3_rows = run_d3_control(datasets, dataset_count, initialization_count, max_steps)
    heldout_rows = run_heldout(dataset_rows, d3_rows, heldout_count)
    sample_size_rows = [] if args.quick else run_sample_size_robustness()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_optimizer_runs.csv", run_rows)
    write_csv(output_dir / "dataset_best_runs.csv", dataset_rows)
    write_csv(output_dir / "d3_capacity_control.csv", d3_rows)
    write_csv(output_dir / "heldout_branch_distortion.csv", heldout_rows)
    if sample_size_rows:
        write_csv(output_dir / "sample_size_robustness.csv", sample_size_rows)
        (output_dir / "sample_size_table_rows.tex").write_text(
            sample_size_table_rows(sample_size_rows), encoding="utf-8"
        )

    figure_summary = aggregate_primary(dataset_rows)
    write_csv(output_dir / "figure_summary.csv", figure_summary)
    (output_dir / "representative_lines.tex").write_text(
        representative_macros(dataset_rows), encoding="utf-8"
    )

    if args.quick:
        results = {
            "quick_smoke_run": True,
            "raw_loss_reduction_max_error": raw_loss_error,
            "raw_gradient_reduction_max_error": raw_gradient_error,
        }
    else:
        results = summarize_results(
            raw_loss_error,
            raw_gradient_error,
            run_rows,
            dataset_rows,
            d3_rows,
            heldout_rows,
            sample_size_rows,
        )
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    if not args.quick and not results["all_primary_criteria_passed"]:
        raise SystemExit("one or more frozen primary criteria failed; see results.json")


if __name__ == "__main__":
    main()
