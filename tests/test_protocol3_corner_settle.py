from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import motornet as mn

from digit_writing.corner_settle import (
    CORNER_SETTLE_INTERVALS,
    CORNER_SETTLE_PHYSICAL_MS,
    build_corner_settle_trajectory,
    save_corner_audit_artifacts,
)
from digit_writing.geometry import build_digit_trajectory, load_geometry_config
from digit_writing.protocol3_corner_settle import (
    _validate_config,
    validate_next_review_target,
)
from digit_writing.protocol3_checkpoint import state_dict_sha256
from envs import DlyFullCircleCClk, DlySinusoidInv


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configurations"
    / "digit_writing_original_protocol3_corner_settle_scale2p50_ref50.json"
)


class Protocol3CornerSettleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.geometry = load_geometry_config(ROOT / cls.config["geometry_config"])

    def test_config_is_a_strict_fast_6000_update_manual_gate(self):
        source = _validate_config(ROOT, self.config)
        self.assertEqual(source["selected_reference_steps"], 50)
        self.assertEqual(self.config["corner_settle"]["settle_intervals"], 10)
        self.assertEqual(self.config["corner_settle"]["physical_pause_ms"], 100)
        self.assertEqual(self.config["training"]["max_updates"], 6000)
        self.assertFalse(self.config["corner_settle"]["automatic_continuation"])
        self.assertFalse(self.config["corner_settle"]["automatic_medium_fallback"])

    def test_zero_intervals_is_bitwise_identical_to_formal_baseline(self):
        for digit in (4, 7):
            baseline = build_digit_trajectory(digit, self.geometry, 50)
            candidate = build_corner_settle_trajectory(
                digit, self.geometry, 50, settle_intervals=0
            )
            np.testing.assert_array_equal(candidate.points, baseline.points)
            self.assertEqual(candidate.movement_intervals, baseline.movement_intervals)
            self.assertEqual(candidate.base.boundaries, baseline.boundaries)

    def test_settle_only_adds_ten_repeated_connection_points_per_corner(self):
        for digit, expected_corners in ((4, 2), (7, 1)):
            candidate = build_corner_settle_trajectory(
                digit,
                self.geometry,
                50,
                settle_intervals=CORNER_SETTLE_INTERVALS,
            )
            self.assertEqual(len(candidate.corners), expected_corners)
            self.assertTrue(all(corner.turn_qualifies for corner in candidate.corners))
            self.assertTrue(all(corner.settle_applied for corner in candidate.corners))
            self.assertTrue(
                all(corner.turn_angle_deg >= 60.0 for corner in candidate.corners)
            )
            keep = np.ones(len(candidate.points), dtype=bool)
            for corner in candidate.corners:
                keep[corner.settle_start_index : corner.settle_end_index] = False
                repeated = candidate.points[
                    corner.settle_start_index : corner.settle_end_index
                ]
                expected = np.repeat(
                    candidate.points[corner.connection_index][None, :],
                    CORNER_SETTLE_INTERVALS,
                    axis=0,
                )
                np.testing.assert_array_equal(repeated, expected)
            np.testing.assert_array_equal(candidate.points[keep], candidate.base.points)
            self.assertEqual(
                candidate.movement_intervals,
                candidate.base.movement_intervals
                + expected_corners * CORNER_SETTLE_INTERVALS,
            )
            self.assertEqual(len(candidate.points), candidate.movement_intervals + 1)
            self.assertEqual(CORNER_SETTLE_PHYSICAL_MS, 100)

    def test_original_primitive_hashes_do_not_change(self):
        for digit in (4, 7):
            baseline = build_digit_trajectory(digit, self.geometry, 50)
            candidate = build_corner_settle_trajectory(
                digit, self.geometry, 50, settle_intervals=10
            )
            self.assertEqual(
                [boundary.canonical_template_sha256 for boundary in candidate.base.boundaries],
                [boundary.canonical_template_sha256 for boundary in baseline.boundaries],
            )
            self.assertEqual(
                [boundary.ordered_instance_sha256 for boundary in candidate.base.boundaries],
                [boundary.ordered_instance_sha256 for boundary in baseline.boundaries],
            )

    def test_environment_keeps_28_features_and_deterministic_targets(self):
        for environment_class in (DlySinusoidInv, DlyFullCircleCClk):
            effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
            environment = environment_class(
                effector=effector,
                action_frame_stacking=0,
                geometry_config_path=ROOT / self.config["geometry_config"],
                corner_settle_intervals=10,
            )
            options = {
                "batch_size": 1,
                "reach_conds": 0,
                "speed_cond": 0,
                "delay_cond": 1,
                "deterministic": True,
            }
            observation, _ = environment.reset(seed=1042, options=options)
            first_target = environment.traj.detach().cpu().numpy().copy()
            self.assertEqual(tuple(observation.shape), (1, 28))
            self.assertIsNotNone(environment.corner_records)
            self.assertTrue(bool(torch.isfinite(environment.traj).all()))
            environment.reset(seed=1042, options=options)
            np.testing.assert_array_equal(
                environment.traj.detach().cpu().numpy(), first_target
            )

    def test_corner_metric_definition_is_exact_on_perfect_tracking(self):
        candidate = build_corner_settle_trajectory(
            4, self.geometry, 50, settle_intervals=10
        )
        target = torch.as_tensor(candidate.points, dtype=torch.float32).unsqueeze(0)
        with tempfile.TemporaryDirectory() as directory:
            audit = save_corner_audit_artifacts(
                directory,
                target,
                target,
                candidate.corners,
                self.geometry.dt_seconds,
            )
            self.assertTrue((Path(directory) / "corner_local_overlay.png").is_file())
            self.assertTrue((Path(directory) / "fingertip_speed.png").is_file())
        for corner in audit["corners"]:
            self.assertEqual(corner["corner_local_mean_error_m"], 0.0)
            self.assertEqual(corner["corner_local_max_error_m"], 0.0)
            self.assertEqual(corner["corner_miss_distance_m"], 0.0)
            self.assertAlmostEqual(corner["post_corner_progress_ratio"], 1.0)

    def test_continuation_adds_exactly_one_manual_review_segment(self):
        self.assertEqual(validate_next_review_target(6000, 12000), 12000)
        self.assertEqual(validate_next_review_target(12000, 18000), 18000)
        for source, target in ((6000, 18000), (5000, 11000), (6000, 11900)):
            with self.subTest(source=source, target=target):
                with self.assertRaises(ValueError):
                    validate_next_review_target(source, target)

    def test_parallel_runner_has_four_single_thread_cpu_cases_and_no_medium(self):
        runner = (
            ROOT
            / "server"
            / "run_digit_writing_original_protocol3_corner_settle_fast_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CASES=(digit4_baseline digit4_settle100ms digit7_baseline digit7_settle100ms)", runner)
        self.assertIn("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1", runner)
        self.assertIn("AUTOMATIC_CONTINUATION_STARTED=0", runner)
        self.assertIn("AUTOMATIC_MEDIUM_FALLBACK_STARTED=0", runner)
        self.assertNotIn("ref100", runner)

        continuation = (
            ROOT
            / "server"
            / "run_digit_writing_original_protocol3_corner_settle_continue_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("MANUAL_RESUME_AUTHORIZED:-", continuation)
        self.assertIn("SOURCE_UPDATES + 6000", continuation)
        self.assertIn("--manual-resume-authorized", continuation)
        self.assertIn("AUTOMATIC_CONTINUATION_STARTED=0", continuation)
        self.assertIn("AUTOMATIC_MEDIUM_FALLBACK_STARTED=0", continuation)

    def test_corner_settle_resume_matches_uninterrupted_updates(self):
        from train import (
            PROTOCOL3_DELAYS,
            _base_hp_from_config,
            _digit_env_dict,
            train_subsets_base_model,
        )

        def hp(max_updates):
            value = _base_hp_from_config(self.config)
            value.update(
                {
                    "variant": "digit4_settle100ms",
                    "condition_schedule": "protocol3_corner_settle_single_condition",
                    "gate2_direction_index": 0,
                    "gate2_delay_index": PROTOCOL3_DELAYS.index(50),
                    "batch_size": 1,
                    "hid_size": 16,
                    "epochs": max_updates,
                    "save_iter": 1,
                    "stop_after_updates": max_updates,
                    "initial_model_file": None,
                    "final_model_file": "final.pt",
                    "env_kwargs": {
                        "geometry_config_path": self.config["geometry_config"],
                        "corner_settle_intervals": 10,
                    },
                }
            )
            return value

        def assert_nested_equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_nested_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for left_value, right_value in zip(left, right):
                    assert_nested_equal(left_value, right_value)
            else:
                self.assertEqual(left, right)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_subsets_base_model(
                str(directory / "uninterrupted"),
                "best.pt",
                hp=hp(4),
                env_dict=_digit_env_dict((4,)),
            )
            train_subsets_base_model(
                str(directory / "prefix"),
                "best.pt",
                hp=hp(2),
                env_dict=_digit_env_dict((4,)),
            )
            source_checkpoint = directory / "prefix" / "final.pt"
            source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
            source_head = "b" * 40
            source["git_identity"]["repository_head"] = source_head
            source["hp"]["git_identity"]["repository_head"] = source_head
            torch.save(source, source_checkpoint)
            train_subsets_base_model(
                str(directory / "resumed"),
                "best.pt",
                hp=hp(2),
                env_dict=_digit_env_dict((4,)),
                resume_checkpoint=str(source_checkpoint),
                target_completed_updates=4,
                manual_resume_authorized=True,
                expected_resume_repository_head=source_head,
            )
            uninterrupted = torch.load(
                directory / "uninterrupted" / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed = torch.load(
                directory / "resumed" / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(
            state_dict_sha256(uninterrupted["agent_state_dict"]),
            state_dict_sha256(resumed["agent_state_dict"]),
        )
        assert_nested_equal(
            uninterrupted["optimizer_state_dict"], resumed["optimizer_state_dict"]
        )
        assert_nested_equal(
            uninterrupted["training_state"], resumed["training_state"]
        )
        assert_nested_equal(uninterrupted["rng_state"], resumed["rng_state"])


if __name__ == "__main__":
    unittest.main()
