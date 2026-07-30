import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from digit_writing.geometry import build_digit_trajectory, load_geometry_config
from digit_writing.protocol3_closed_loop_diagnostic import (
    CHECKPOINT_LABELS,
    AUDIT_IDENTITY,
    _padded_derivatives,
    _worker_directory,
    closure_metrics,
    constrained_monotonic_alignment,
    digit0_region_metrics,
    digit8_segment_metrics,
    gradient_decomposition,
    load_diagnostic_config,
    regional_kinematic_metrics,
    validate_digit8_landmarks,
)


class Protocol3ClosedLoopDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.root
            / "configurations"
            / "digit_writing_original_protocol3_digit0_digit8_closed_loop_diagnostic.json"
        )
        cls.config = load_diagnostic_config(cls.config_path)

    def test_frozen_seven_checkpoint_zero_step_design(self):
        self.assertEqual(
            tuple(row["label"] for row in self.config["checkpoints"]),
            CHECKPOINT_LABELS,
        )
        self.assertEqual(
            tuple(row["digit"] for row in self.config["checkpoints"]),
            (0, 0, 0, 0, 8, 8, 9),
        )
        self.assertEqual(self.config["audit"], AUDIT_IDENTITY)
        self.assertEqual(self.config["audit"]["optimizer_steps"], 0)
        self.assertFalse(self.config["decision"]["automatic_stage_b_start"])
        self.assertFalse(self.config["decision"]["automatic_training"])
        self.assertFalse(self.config["decision"]["formal_full10_start"])

    def test_checkpoint_and_summary_hashes_are_frozen(self):
        self.assertEqual(
            self.config["source"]["repository_head"],
            "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925",
        )
        self.assertEqual(
            self.config["continuation"]["repository_head"],
            "76349d51222eed62bdb26d78d251b7d57760f2e8",
        )
        self.assertEqual(
            self.config["mrnntorch_head"],
            "ac0c4f589eae37bbde63968912925de99232e306",
        )
        for row in self.config["checkpoints"]:
            self.assertEqual(len(row["checkpoint_sha256"]), 64)
            self.assertEqual(len(row["run_summary_sha256"]), 64)
            self.assertIn(row["origin"], {"source", "continuation"})

    def test_design_mutations_are_rejected(self):
        mutations = []
        changed = copy.deepcopy(self.config)
        changed["checkpoints"][0]["checkpoint_sha256"] = "0" * 64
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["audit"]["phase_alignment_band_steps"] = 30
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["audit"]["optimizer_steps"] = 1
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["decision"]["automatic_stage_b_start"] = True
        mutations.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in mutations:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_diagnostic_config(path)

    def test_exact_alignment_has_zero_error_and_identity_mapping(self):
        time = np.linspace(0.0, 1.0, 61)
        target = np.column_stack((time, time**2))
        result = constrained_monotonic_alignment(target, target, band_steps=25)
        np.testing.assert_array_equal(result["mapping"], np.arange(len(target)))
        self.assertEqual(result["phase_aligned_normalized_mean_error"], 0.0)
        self.assertEqual(result["phase_lag_steps_final"], 0)

    def test_alignment_recovers_fixed_ten_step_lag(self):
        time = np.arange(61, dtype=np.float64)
        target = np.column_stack((time, time**2 / 100.0))
        actual = target[np.maximum(np.arange(len(target)) - 10, 0)]
        result = constrained_monotonic_alignment(actual, target, band_steps=25)
        self.assertEqual(result["phase_aligned_normalized_mean_error"], 0.0)
        self.assertEqual(result["phase_lag_steps_final"], 10)
        self.assertEqual(result["mapping"][-1], len(target) - 11)

    def test_alignment_does_not_hide_spatial_compression(self):
        angle = np.linspace(0.0, 2.0 * math.pi, 101)
        target = np.column_stack((np.cos(angle), 1.4 * np.sin(angle)))
        center = (target.min(axis=0) + target.max(axis=0)) / 2.0
        actual = center + 0.8 * (target - center)
        result = constrained_monotonic_alignment(actual, target, band_steps=25)
        self.assertGreater(result["phase_aligned_normalized_mean_error"], 0.0)
        self.assertLessEqual(
            result["phase_aligned_normalized_mean_error"],
            result["time_aligned_normalized_mean_error"],
        )

    def test_closure_search_excludes_early_movement_start(self):
        movement = np.ones((20, 2), dtype=np.float64)
        movement[:5] = 0.0
        hold = np.ones((5, 2), dtype=np.float64)
        result = closure_metrics(
            movement,
            hold,
            np.zeros(2),
            target_diagonal=1.0,
            movement_start_episode_index=75,
        )
        self.assertTrue(result["closure_never_reached"])
        self.assertFalse(result["closure_reached_during_movement"])
        self.assertIsNone(result["first_closure_entry_episode_index"])
        self.assertIn("persistent_nonclosure", result["diagnostic_labels"])

    def test_closure_categories_are_exclusive_and_hold_late_is_detected(self):
        movement = np.ones((20, 2), dtype=np.float64)
        hold = np.ones((5, 2), dtype=np.float64)
        hold[2:] = 0.0
        result = closure_metrics(
            movement,
            hold,
            np.zeros(2),
            target_diagonal=1.0,
            movement_start_episode_index=75,
        )
        categories = (
            result["closure_reached_during_movement"],
            result["closure_reached_only_during_hold"],
            result["closure_never_reached"],
        )
        self.assertEqual(sum(categories), 1)
        self.assertTrue(result["closure_reached_only_during_hold"])
        self.assertEqual(result["first_closure_entry_phase"], "hold")
        self.assertIn("late_closure", result["diagnostic_labels"])

    def test_digit0_regions_cover_every_sample_and_interval_once(self):
        angle = np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 171)
        target = np.column_stack((0.36 * np.cos(angle), 0.48 * np.sin(angle)))
        result = digit0_region_metrics(target, target, np.arange(len(target)))
        self.assertEqual(
            sum(row["sample_count"] for row in result["regions"].values()), 171
        )
        self.assertEqual(
            sum(row["interval_count"] for row in result["regions"].values()), 170
        )
        self.assertAlmostEqual(result["x_bbox_ratio"], 1.0)
        self.assertAlmostEqual(result["y_bbox_ratio"], 1.0)

    def test_authoritative_digit8_landmarks_and_segments_are_exact(self):
        geometry = load_geometry_config(
            self.root
            / "configurations"
            / "digit_writing_original_protocol3_geometry_scale2p50_ref100_corner_ease_v3.json"
        )
        target = build_digit_trajectory(8, geometry, 100).points
        validate_digit8_landmarks(target)
        result = digit8_segment_metrics(target, target, np.arange(201))
        for row in result["segments"].values():
            self.assertAlmostEqual(row["actual_to_target_path_ratio"], 1.0)
            self.assertEqual(row["time_aligned_error_m"], 0.0)
            self.assertEqual(row["phase_aligned_error_m"], 0.0)
        self.assertEqual(
            result["crossings"]["crossing_1"]["actual_crossing_time"], 50
        )
        self.assertEqual(
            result["crossings"]["crossing_2"]["actual_crossing_time"], 150
        )
        self.assertFalse(
            result["crossings"]["crossing_1"]["wrong_branch_departure"]
        )
        self.assertFalse(
            result["crossings"]["crossing_2"]["wrong_branch_departure"]
        )
        kinematics = regional_kinematic_metrics(
            target, target, digit=8, dt_seconds=geometry.dt_seconds
        )
        self.assertEqual(tuple(kinematics["segments"]), (
            "segment_1", "segment_2", "segment_3", "segment_4"
        ))

    def test_digit0_kinematics_are_reported_for_all_four_regions(self):
        angle = np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 171)
        target = np.column_stack((0.36 * np.cos(angle), 0.48 * np.sin(angle)))
        result = regional_kinematic_metrics(
            target, target, digit=0, dt_seconds=0.01
        )
        self.assertEqual(
            tuple(result["regions"]), ("upper", "right", "lower", "left")
        )
        for region in result["regions"].values():
            self.assertEqual(region["actual"], region["target"])

    def test_gradient_decomposition_uses_autograd_without_parameter_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
        stable = 0.1 * torch.sum(parameter**2)
        delay = 0.1 * torch.sum(torch.abs(parameter))
        movement = 0.6 * torch.sum((parameter - 1.0) ** 2)
        hold = 0.2 * torch.sum((parameter + 0.5) ** 2)
        total_position = stable + delay + movement + hold
        rate = 0.001 * torch.sum(parameter**2)
        weight = 0.001 * torch.sum(torch.abs(parameter))
        muscle = 0.01 * torch.sum((parameter - 0.2) ** 2)
        dynamics = 0.001 * torch.linalg.vector_norm(parameter)
        total_regularization = rate + weight + muscle + dynamics
        result = gradient_decomposition(
            {
                "stable_position": stable,
                "delay_position": delay,
                "movement_position": movement,
                "hold_position": hold,
                "total_position": total_position,
                "weighted_l1_rate": rate,
                "weighted_l1_weight": weight,
                "weighted_l1_muscle_act": muscle,
                "weighted_simple_dynamics": dynamics,
                "total_regularization": total_regularization,
                "total_objective": total_position + total_regularization,
            },
            (parameter,),
        )
        self.assertLess(result["component_sum_residual_norm"], 1e-6)
        self.assertTrue(result["component_sum_consistent"])
        self.assertGreater(result["component_sum_tolerance"], 0.0)
        self.assertIsNone(parameter.grad)
        self.assertEqual(result["optimizer_steps_during_audit"], 0)

    def test_derivative_arrays_keep_time_axis_and_explicit_zero_padding(self):
        points = np.column_stack((np.arange(6, dtype=np.float64), np.zeros(6)))
        result = _padded_derivatives(points, 0.5)
        self.assertEqual(result["speed"].shape, (6,))
        self.assertEqual(result["acceleration"].shape, (6,))
        self.assertEqual(result["jerk"].shape, (6,))
        self.assertEqual(result["speed"][0], 0.0)
        np.testing.assert_allclose(result["speed"][1:], 2.0)

    def test_worker_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory).resolve()
            output = run_root / "per_checkpoint" / CHECKPOINT_LABELS[0]
            output.mkdir(parents=True)
            (output / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _worker_directory(run_root, CHECKPOINT_LABELS[0])

    def test_runner_is_seven_way_read_only_and_requires_authorization(self):
        script = (
            self.root
            / "server"
            / "run_digit_writing_original_protocol3_digit0_digit8_closed_loop_diagnostic_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("PROTOCOL3_CLOSED_LOOP_DIAGNOSTIC_AUTHORIZED", script)
        self.assertIn("AVAILABLE_CPUS < 7", script)
        self.assertIn("CPU_QUOTA / CPU_PERIOD < 7", script)
        self.assertIn("COMPLETED_CHECKPOINTS=7", script)
        self.assertIn("OPTIMIZER_STEPS=0", script)
        self.assertIn("AUTOMATIC_STAGE_B_STARTED=0", script)
        self.assertNotIn("train_subsets_base_model", script)
        self.assertNotIn("protocol3_digit8_v3_lr_continuation", script)

    def test_module_contains_no_optimizer_or_training_entry(self):
        source = (
            self.root
            / "digit_writing"
            / "protocol3_closed_loop_diagnostic.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("train_subsets_base_model", source)
        self.assertNotIn("protocol3_digit8_v3_lr_continuation", source)


if __name__ == "__main__":
    unittest.main()
