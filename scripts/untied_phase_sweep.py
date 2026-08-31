#!/usr/bin/env python3
"""Deterministic moderate-noise sweep for the optimized untied decoder.

This script supplies numerical evidence only. It eliminates the linear decoder
exactly and performs a global coarse grid followed by local angle refinements.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ETAS = [0.2, 1.0, 5.0, 20.0]
P_GRID = np.linspace(0.0, 1.0, 51)


def objective_grid(
    p: float, eta: float, theta2: np.ndarray, theta3: np.ndarray
) -> np.ndarray:
    """Reduced untied objective with v1 fixed and v2,v3 on angle grids."""
    g12 = np.cos(theta2[:, None]) ** 2
    g13 = np.cos(theta3[None, :]) ** 2
    g23 = np.cos(theta2[:, None] - theta3[None, :]) ** 2
    q12, q13, q23 = p / 2.0, 0.5, (1.0 - p) / 2.0
    r = [q12 + q13, q12 + q23, q13 + q23]
    rows = [
        (r[0], q12 / r[0], q13 / r[0], g23, g12, g13),
        (r[1], q12 / r[1], q23 / r[1], g13, g12, g23),
        (r[2], q13 / r[2], q23 / r[2], g12, g13, g23),
    ]
    total = np.zeros_like(g12 + g13)
    for ri, alpha_j, alpha_k, gjk, gij, gik in rows:
        cross = alpha_j * alpha_k * (1.0 - gjk)
        numerator = eta * (eta + 1.0) + cross
        denominator = (
            eta * (eta + 2.0)
            + cross
            + alpha_j * (1.0 - gij)
            + alpha_k * (1.0 - gik)
        )
        total += ri * numerator / denominator
    return total


def optimize_angles(p: float, eta: float) -> tuple[float, float, float]:
    coarse_size = 361
    coarse = np.linspace(0.0, math.pi, coarse_size, endpoint=False)
    objective = objective_grid(p, eta, coarse, coarse)
    row, column = np.unravel_index(np.argmin(objective), objective.shape)
    theta2, theta3 = float(coarse[row]), float(coarse[column])
    value = float(objective[row, column])

    width = 2.0 * math.pi / coarse_size
    for _ in range(3):
        offsets = np.linspace(-width, width, 101)
        grid2 = np.mod(theta2 + offsets, math.pi)
        grid3 = np.mod(theta3 + offsets, math.pi)
        objective = objective_grid(p, eta, grid2, grid3)
        row, column = np.unravel_index(np.argmin(objective), objective.shape)
        theta2, theta3 = float(grid2[row]), float(grid3[column])
        value = float(objective[row, column])
        width *= 0.04
    return value, theta2, theta3


def optimized_slope(
    p: float, eta: float, theta2: float, theta3: float
) -> float:
    vectors = [
        np.array([1.0, 0.0]),
        np.array([math.cos(theta2), math.sin(theta2)]),
        np.array([math.cos(theta3), math.sin(theta3)]),
    ]
    q = {(0, 1): p / 2.0, (0, 2): 0.5, (1, 2): (1.0 - p) / 2.0}
    r = [
        q[(0, 1)] + q[(0, 2)],
        q[(0, 1)] + q[(1, 2)],
        q[(0, 2)] + q[(1, 2)],
    ]
    decoder = []
    for feature in range(3):
        matrix = eta * np.eye(2) + np.outer(vectors[feature], vectors[feature])
        for other in range(3):
            if feature == other:
                continue
            weight = q[tuple(sorted((feature, other)))] / r[feature]
            matrix += weight * np.outer(vectors[other], vectors[other])
        decoder.append(np.linalg.solve(matrix, vectors[feature]))

    def pair_loss(first: int, second: int) -> float:
        loss = 0.0
        for target, other in [(first, second), (second, first)]:
            weight = decoder[target]
            loss += (
                (weight @ vectors[target] - 1.0) ** 2
                + (weight @ vectors[other]) ** 2
                + eta * (weight @ weight)
            )
        return float(loss)

    return 0.5 * (pair_loss(0, 1) - pair_loss(1, 2))


def key(eta: float) -> str:
    return f"eta_{str(eta).replace('.', 'p')}"


def main() -> None:
    rows = [{"p": float(p)} for p in P_GRID]
    summaries = []
    for eta in ETAS:
        values = []
        slopes = []
        scale = 1.0 / (1.0 + eta) ** 2
        for index, p in enumerate(P_GRID):
            value, theta2, theta3 = optimize_angles(float(p), eta)
            slope = optimized_slope(float(p), eta, theta2, theta3)
            if index == 0:
                slope = scale
            elif index == len(P_GRID) - 1:
                slope = -scale
            values.append(value)
            slopes.append(slope)
            rows[index][key(eta)] = slope / scale

        values_array = np.asarray(values)
        slopes_array = np.asarray(slopes)
        symmetry_error = float(np.max(np.abs(values_array - values_array[::-1])))
        slope_symmetry_error = float(
            np.max(np.abs(slopes_array + slopes_array[::-1]))
        )
        monotonic_violation = float(np.max(np.diff(slopes_array)))
        assert symmetry_error < 2e-8
        assert slope_symmetry_error < 2e-4
        assert monotonic_violation < 2e-4
        summaries.append(
            (
                eta,
                symmetry_error,
                slope_symmetry_error,
                monotonic_violation,
                slopes[len(P_GRID) * 2 // 5],
            )
        )

    output = ROOT / "figures" / "untied_phase_data.csv"
    fieldnames = ["p", *(key(eta) for eta in ETAS)]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for eta, value_error, slope_error, violation, slope_at_point_four in summaries:
        print(
            f"eta={eta:g} value_symmetry={value_error:.3g} "
            f"slope_symmetry={slope_error:.3g} "
            f"monotonic_violation={violation:.3g} "
            f"slope_p0.4={slope_at_point_four:.8g}"
        )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
