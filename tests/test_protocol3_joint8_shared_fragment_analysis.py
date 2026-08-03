import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from digit_writing.protocol3_joint8_shared_fragment_analysis import (
    ANALYSIS_DIGITS,
    FragmentOccurrence,
    _candidate_maximal_matches,
    _eligible_interval_mask,
    _expected_families,
    _fragment_equal,
    _occurrence_activity,
    _rotation_category,
    cca_metrics,
    cross_projection_metrics,
    delay_embed_trials,
    fit_dsa_linear_system,
    pavf_distance,
    principal_angle_metrics,
    procrustes_metrics,
    round_half_up_nonnegative,
    solve_fixed_points,
    validate_analysis_config,
    _fixed_point_stability,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configurations"
    / "digit_writing_original_protocol3_joint8_shared_fragment_analysis_v1.json"
)


class GeometryInvariantTests(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.7, 0.7], [1.2, 1.8], [0.4, 2.1]],
            dtype=np.float64,
        )

    def equal(self, first, second):
        return _fragment_equal(first, second, 1e-10, 1e-9, 1e-10)

    def test_translation_and_rotation_are_allowed(self):
        angle = 0.63
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        transformed = self.points @ rotation.T + np.asarray([3.0, -4.0])
        self.assertTrue(self.equal(self.points, transformed))

    def test_scaling_reflection_and_reversal_are_rejected(self):
        self.assertFalse(self.equal(self.points, 1.1 * self.points))
        reflected = self.points.copy()
        reflected[:, 0] *= -1.0
        self.assertFalse(self.equal(self.points, reflected))
        self.assertFalse(self.equal(self.points, self.points[::-1]))

    def test_pointwise_lengths_and_signed_turns_are_both_required(self):
        different_steps = np.asarray(
            [[0.0, 0.0], [0.8, 0.0], [1.7, 0.7], [1.2, 1.8], [0.4, 2.1]]
        )
        self.assertFalse(self.equal(self.points, different_steps))
        same_lengths_different_turn = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.7, -0.7], [1.2, -1.8], [0.4, -2.1]]
        )
        self.assertFalse(self.equal(self.points, same_lengths_different_turn))

    def test_corner_exclusion_uses_actual_fifteen_interval_half_window(self):
        trajectory = SimpleNamespace(
            movement_intervals=100,
            corner_ease_boundaries=(
                {"qualifies": True, "corner_sample_index": 50},
                {"qualifies": False, "corner_sample_index": 80},
            ),
        )
        mask = _eligible_interval_mask(trajectory, 15)
        self.assertTrue(np.all(mask[:35]))
        self.assertTrue(np.all(~mask[35:65]))
        self.assertTrue(np.all(mask[65:]))

    def test_rotation_category_rejects_mixed_turns(self):
        mixed = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
        self.assertEqual(_rotation_category(mixed, 1e-9), "mixed_rotation")


class FrozenConfigAndOracleTests(unittest.TestCase):
    def test_exact_key_validation_rejects_extra_setting(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["cca"]["unapproved"] = 1
        with self.assertRaises(ValueError):
            validate_analysis_config(ROOT, changed)

    def test_frozen_geometry_reproduces_three_families_and_four_pairs(self):
        from digit_writing.geometry import build_digit_trajectory, load_geometry_config

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        training = json.loads(
            (ROOT / config["training_identity"]["training_config"]).read_text(
                encoding="utf-8"
            )
        )
        geometry = load_geometry_config(ROOT / training["geometry_config"])
        trajectories = {}
        masks = {}
        for digit in ANALYSIS_DIGITS:
            trajectory = build_digit_trajectory(
                digit,
                geometry,
                100,
                spatial_angle_rad=0.0,
                anchor=(0.0, 0.0),
            )
            trajectories[digit] = np.asarray(trajectory.points)
            masks[digit] = _eligible_interval_mask(trajectory, 15)
        matching = config["fragment_matching"]
        discovered = set()
        for index, first_digit in enumerate(ANALYSIS_DIGITS):
            for second_digit in ANALYSIS_DIGITS[index + 1 :]:
                for pair in _candidate_maximal_matches(
                    first_digit,
                    second_digit,
                    trajectories,
                    masks,
                    2,
                    1e-10,
                    1e-9,
                    1e-10,
                ):
                    if pair[0].intervals >= 30:
                        discovered.add(pair)
        expected = set()
        families = _expected_families(config)
        for family in families:
            for first_index, first in enumerate(family.occurrences):
                for second in family.occurrences[first_index + 1 :]:
                    if first.digit != second.digit:
                        expected.add((first, second))
        self.assertEqual(len(families), matching["expected_family_count"])
        self.assertEqual(discovered, expected)
        self.assertEqual(len(discovered), matching["expected_primary_pair_count"])


class ActivityBoundaryTests(unittest.TestCase):
    def test_half_open_fragment_uses_t_plus_one_activity_samples(self):
        activity = {
            "deterministic_h": np.arange(30, dtype=np.float64).reshape(10, 3),
            "stochastic_h": np.arange(120, dtype=np.float64).reshape(4, 10, 3),
        }
        occurrence = FragmentOccurrence(2, 2, 7)
        deterministic = _occurrence_activity(
            activity, occurrence, "deterministic", state="h"
        )
        stochastic = _occurrence_activity(
            activity, occurrence, "stochastic_trials", state="h"
        )
        self.assertEqual(deterministic.shape, (6, 3))
        self.assertEqual(stochastic.shape, (4, 6, 3))
        np.testing.assert_array_equal(deterministic, activity["deterministic_h"][2:8])

    def test_phase_index_uses_round_half_up(self):
        self.assertEqual(round_half_up_nonnegative(2.5), 3)
        self.assertEqual(round_half_up_nonnegative(2.49), 2)


class PopulationMetricTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(4)
        latent = generator.normal(size=(80, 3))
        basis, _ = np.linalg.qr(generator.normal(size=(12, 3)))
        score_rotation, _ = np.linalg.qr(generator.normal(size=(3, 3)))
        self.first = latent @ basis.T
        self.same_subspace = (latent @ score_rotation) @ basis.T
        neural_rotation, _ = np.linalg.qr(generator.normal(size=(12, 12)))
        self.orthogonal_copy = self.first @ neural_rotation

    def test_principal_angles_and_cross_projection_identical_subspace(self):
        angles = principal_angle_metrics(self.first, self.same_subspace, 3)
        projection = cross_projection_metrics(self.first, self.same_subspace, 3)
        self.assertLess(max(angles["angles_deg"]), 1e-5)
        self.assertAlmostEqual(
            projection["symmetric_normalized_cross_projection"], 1.0, places=10
        )

    def test_cca_recovers_shared_latent_scores(self):
        result = cca_metrics(self.first, self.orthogonal_copy, 3, 3)
        self.assertGreater(min(result["canonical_correlations"]), 0.999999)

    def test_procrustes_removes_translation_scale_and_orthogonal_transform(self):
        transformed = 3.2 * self.orthogonal_copy + 7.0
        result = procrustes_metrics(self.first, transformed)
        self.assertLess(result["disparity"], 1e-20)


class FixedPointTests(unittest.TestCase):
    def test_known_stable_linear_fixed_point(self):
        import torch

        class LinearMRNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "matrix", torch.diag(torch.tensor([0.4, 0.7], dtype=torch.float64))
                )
                self.register_buffer(
                    "bias", torch.tensor([0.3, -0.2], dtype=torch.float64)
                )
                self.activation = lambda value: value

            def forward(self, x, observation, noise=False, h0=None):
                next_x = x @ self.matrix.T + self.bias
                return next_x[:, None, :], next_x[:, None, :]

        policy = SimpleNamespace(mrnn=LinearMRNN().double())
        settings = {
            "learning_rate": 0.03,
            "maximum_steps": 3000,
            "residual_rms_max": 1e-7,
            "dedup_activation_rms_max": 1e-5,
        }
        initial = np.random.default_rng(2).normal(size=(16, 2))
        result = solve_fixed_points(policy, np.zeros(1), initial, settings)
        self.assertEqual(result["converged_count"], 16)
        self.assertEqual(len(result["clusters"]), 1)
        expected = np.asarray([0.3 / 0.6, -0.2 / 0.3])
        np.testing.assert_allclose(result["clusters"][0]["x"], expected, atol=2e-5)
        stability = _fixed_point_stability(
            policy, np.zeros(1), result["clusters"][0]["x"]
        )
        self.assertTrue(stability["stable"])
        self.assertAlmostEqual(stability["spectral_radius"], 0.7, places=8)


class DSATests(unittest.TestCase):
    @staticmethod
    def linear_trials(matrix, seed=3):
        generator = np.random.default_rng(seed)
        trials = np.empty((24, 45, matrix.shape[0]), dtype=np.float64)
        trials[:, 0] = generator.normal(size=(24, matrix.shape[0]))
        for index in range(1, trials.shape[1]):
            trials[:, index] = trials[:, index - 1] @ matrix.T
        return trials

    def test_delay_embedding_shape(self):
        value = np.zeros((7, 20, 4))
        self.assertEqual(delay_embed_trials(value, 5, 1).shape, (7, 16, 20))

    def test_trial_boundaries_do_not_create_regression_transitions(self):
        value = np.zeros((2, 8, 1), dtype=np.float64)
        value[1] = 1e6
        result = fit_dsa_linear_system(
            value,
            n_delays=1,
            delay_interval=1,
            steps_ahead=1,
            rank=1,
            ridge=1e-8,
        )
        self.assertEqual(result["regression_samples"], 2 * 7)
        paired_changes = result["plus"] - result["minus"]
        self.assertLess(float(np.max(np.abs(paired_changes))), 1e-10)

    def test_trial_permutation_preserves_fitted_dynamics(self):
        matrix = np.diag([0.93, 0.78, 0.61])
        trials = self.linear_trials(matrix)
        first = fit_dsa_linear_system(
            trials, n_delays=3, delay_interval=1, steps_ahead=1, rank=3, ridge=1e-8
        )
        second = fit_dsa_linear_system(
            trials[::-1], n_delays=3, delay_interval=1, steps_ahead=1, rank=3, ridge=1e-8
        )
        result = pavf_distance(
            first["system"], second["system"], learning_rate=0.01, iterations=200
        )
        self.assertLess(result["distance"], 1e-6)

    def test_pavf_handles_orthogonal_basis_and_separates_different_spectrum(self):
        angle = 0.7
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        first = np.diag([0.95, 0.75, 0.55])
        equivalent = rotation.T @ first @ rotation
        different = np.diag([0.95, 0.75, 0.15])
        same_result = pavf_distance(
            first, equivalent, learning_rate=0.02, iterations=500
        )
        different_result = pavf_distance(
            first, different, learning_rate=0.02, iterations=500
        )
        self.assertLess(same_result["distance"], 1e-3)
        self.assertGreater(different_result["distance"], same_result["distance"] + 0.1)


if __name__ == "__main__":
    unittest.main()
