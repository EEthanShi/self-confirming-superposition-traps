#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import unittest

import numpy as np

import run_experiment as experiment


class SampledPhaseTests(unittest.TestCase):
    def test_raw_and_reduced_objective_match(self) -> None:
        loss_error, gradient_error = experiment.raw_reduction_check(12)
        self.assertLess(loss_error, 1e-10)
        self.assertLess(gradient_error, 1e-10)

    def test_sampled_pair_weights_are_unbiased(self) -> None:
        p = 0.27
        dataset = experiment.sample_dataset(p, 300_000, experiment.seed_for(800))
        np.testing.assert_allclose(dataset.coefficients, [p, 1.0, 1.0 - p], atol=0.012)

    def test_angle_gradient_finite_difference(self) -> None:
        coefficients = np.asarray([0.37, 1.04, 0.61])
        alpha, beta = 0.73, 2.11
        _, gradient = experiment.reduced_loss_and_angle_gradient(coefficients, alpha, beta)
        step = 1e-6
        finite = []
        for index in range(2):
            plus = [alpha, beta]
            minus = [alpha, beta]
            plus[index] += step
            minus[index] -= step
            high = experiment.reduced_loss_and_angle_gradient(coefficients, *plus)[0]
            low = experiment.reduced_loss_and_angle_gradient(coefficients, *minus)[0]
            finite.append((float(high) - float(low)) / (2.0 * step))
        np.testing.assert_allclose(gradient, finite, atol=1e-8)

    def test_two_dimensional_optimizer_recovers_endpoint_phases(self) -> None:
        coefficients = np.asarray([[0.2, 1.0, 0.8], [0.8, 1.0, 0.2]])
        rng = experiment.seed_for(801)
        initial = rng.uniform(0.0, math.pi, size=(2, 8, 2))
        optimized = experiment.optimize_angles(coefficients, initial, 0.05, 20_000, 1e-10)
        angles = np.asarray(optimized["angles"])
        losses = np.asarray(optimized["losses"])
        grams = experiment.gram_from_angles(angles[..., 0], angles[..., 1])
        low = grams[0, int(np.argmin(losses[0]))]
        high = grams[1, int(np.argmin(losses[1]))]
        np.testing.assert_allclose(low, [1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(high, [0.0, 0.0, 1.0], atol=1e-7)

    def test_three_dimensional_control_orthogonalizes(self) -> None:
        coefficients = np.asarray([[0.2, 1.0, 0.8], [0.5, 1.0, 0.5], [0.8, 1.0, 0.2]])
        initial = experiment.seed_for(802).standard_normal((3, 6, 3, 3))
        optimized = experiment.optimize_d3(coefficients, initial, 0.05, 20_000, 1e-10)
        vectors = np.asarray(optimized["vectors"])
        losses = np.asarray(optimized["losses"])
        grams = experiment.gram_from_vectors(vectors)
        for index in range(3):
            best = grams[index, int(np.argmin(losses[index]))]
            self.assertLess(float(np.max(best)), 1e-7)

    def test_heldout_branch_bridge(self) -> None:
        vectors = experiment.vectors_from_angles(math.pi / 2.0, math.pi / 2.0)
        d_s = experiment.forced_branch_loss(vectors, "S", 250_000, experiment.seed_for(803, 0))
        d_e = experiment.forced_branch_loss(vectors, "E", 250_000, experiment.seed_for(803, 1))
        self.assertAlmostEqual(d_e - d_s, 1.0, delta=0.015)

    def test_squared_gram_medoid_is_a_data_point(self) -> None:
        grams = np.asarray(
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.0, 0.0, 0.0], [1.0, 0.2, 0.0]]
        )
        index = experiment.squared_gram_medoid_index(grams)
        self.assertEqual(index, 1)

    def test_saved_outputs_are_self_consistent(self) -> None:
        output_dir = experiment.EXPERIMENT_DIR / "outputs"
        with (output_dir / "results.json").open(encoding="utf-8") as stream:
            results = json.load(stream)
        with (output_dir / "dataset_best_runs.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            dataset_rows = list(csv.DictReader(stream))
        with (output_dir / "sample_size_robustness.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            sample_size_rows = list(csv.DictReader(stream))

        self.assertTrue(results["all_primary_criteria_passed"])
        self.assertEqual(len(dataset_rows), 1150)
        self.assertEqual(len(sample_size_rows), 2800)
        best_success = np.mean(
            [float(row["optimization_error"]) < 1e-8 for row in dataset_rows]
        )
        self.assertAlmostEqual(
            best_success,
            results["metrics"]["best_of_eight_optimization_success_rate"],
        )
        self.assertEqual(
            experiment.sample_size_table_rows(sample_size_rows),
            (output_dir / "sample_size_table_rows.tex").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
