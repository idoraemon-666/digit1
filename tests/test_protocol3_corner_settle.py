from __future__ import annotations

import json
import math
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
MEDIUM_CONFIG_PATH = (
    ROOT
    / "configurations"
    / "digit_writing_original_protocol3_corner_settle_scale2p50_ref100.json"
)


class Protocol3CornerSettleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.geometry = load_geometry_config(ROOT / cls.config["geometry_config"])
        cls.medium_config = json.loads(
            MEDIUM_CONFIG_PATH.read_text(encoding="utf-8")
        )
        cls.medium_geometry = load_geometry_config(
            ROOT / cls.medium_config["geometry_config"]
        )

    def test_config_is_a_strict_fast_6000_update_manual_gate(self):
        source = _validate_config(ROOT, self.config)
        self.assertEqual(source["selected_reference_steps"], 50)
        self.assertEqual(self.config["corner_settle"]["settle_intervals"], 10)
        self.assertEqual(self.config["corner_settle"]["physical_pause_ms"], 100)
        self.assertEqual(self.config["training"]["max_updates"], 6000)
        self.assertFalse(self.config["corner_settle"]["automatic_continuation"])
        self.assertFalse(self.config["corner_settle"]["automatic_medium_fallback"])

    def test_medium_config_is_a_strict_five_digit_6000_update_manual_gate(self):
        source = _validate_config(ROOT, self.medium_config)
        self.assertEqual(source["selected_reference_steps"], 100)
        self.assertEqual(self.medium_config["selected_reference_steps"], 100)
        self.assertEqual(
            self.medium_config["geometry_config"],
            "configurations/"
            "digit_writing_original_protocol3_geometry_scale2p50_ref100.json",
        )
        self.assertEqual(
            [case["label"] for case in self.medium_config["cases"]],
            [
                "digit4_medium_baseline",
                "digit4_medium_settle100ms",
                "digit7_medium_baseline",
                "digit7_medium_settle100ms",
                "digit2_medium_baseline",
                "digit2_medium_settle100ms",
                "digit5_medium_baseline",
                "digit5_medium_settle100ms",
                "digit3_medium_baseline",
                "digit3_medium_settle100ms",
            ],
        )
        self.assertEqual(self.medium_config["corner_settle"]["settle_intervals"], 10)
        self.assertEqual(self.medium_config["corner_settle"]["physical_pause_ms"], 100)
        self.assertEqual(self.medium_config["training"]["max_updates"], 6000)
        self.assertFalse(
            self.medium_config["corner_settle"]["automatic_continuation"]
        )
        self.assertFalse(
            self.medium_config["corner_settle"]["automatic_medium_fallback"]
        )

    def test_medium_five_digit_directions_sizes_corners_and_steps_are_exact(self):
        angle = math.radians(55.0)
        expected = {
            2: {
                "names": ("curve_A_2", "diagonal_55_2", "horizontal_2"),
                "intervals": (80, 60, 60),
                "deltas": (
                    (0.48 * math.sin(angle), -0.48 * math.cos(angle)),
                    (-0.64 * math.cos(angle), -0.64 * math.sin(angle)),
                    (0.64, 0.0),
                ),
                "qualifies": (False, True),
                "angle_ranges": ((0.0, 5.0), (124.999, 125.001)),
                "baseline_intervals": 200,
                "settle_intervals": 210,
                "baseline_episode_steps": 301,
                "settle_episode_steps": 311,
            },
            3: {
                "names": ("curve_A_3", "curve_B_3"),
                "intervals": (80, 90),
                "deltas": ((0.0, -0.48), (0.0, -0.48)),
                "qualifies": (True,),
                "angle_ranges": ((170.0, 180.001),),
                "baseline_intervals": 170,
                "settle_intervals": 180,
                "baseline_episode_steps": 271,
                "settle_episode_steps": 281,
            },
            4: {
                "names": ("horizontal_4", "diagonal_55_4", "vertical_4"),
                "intervals": (60, 60, 60),
                "deltas": ((-0.72, 0.0), (0.78 / math.tan(angle), 0.78), (0.0, -1.0)),
                "qualifies": (True, True),
                "angle_ranges": ((124.999, 125.001), (144.999, 145.001)),
                "baseline_intervals": 180,
                "settle_intervals": 200,
                "baseline_episode_steps": 281,
                "settle_episode_steps": 301,
            },
            5: {
                "names": ("horizontal_5", "vertical_5", "curve_B_5"),
                "intervals": (60, 60, 90),
                "deltas": ((-0.42, 0.0), (0.0, -0.48), (0.0, -0.48)),
                "qualifies": (True, True),
                "angle_ranges": ((89.999, 90.001), (80.0, 90.001)),
                "baseline_intervals": 210,
                "settle_intervals": 230,
                "baseline_episode_steps": 311,
                "settle_episode_steps": 331,
            },
            7: {
                "names": ("horizontal_7", "diagonal_63_435_7"),
                "intervals": (60, 60),
                "deltas": ((0.64, 0.0), (-0.48, -0.96)),
                "qualifies": (True,),
                "angle_ranges": ((116.564, 116.566),),
                "baseline_intervals": 120,
                "settle_intervals": 130,
                "baseline_episode_steps": 221,
                "settle_episode_steps": 231,
            },
        }
        scale = self.medium_geometry.global_scale_m_per_unit
        self.assertEqual(self.medium_geometry.selected_reference_steps, 100)
        self.assertEqual(self.medium_geometry.dt_seconds, 0.01)
        self.assertAlmostEqual(scale, 0.16025641025641027, places=15)
        for digit, frozen in expected.items():
            with self.subTest(digit=digit):
                baseline = build_digit_trajectory(
                    digit, self.medium_geometry, 100
                )
                no_settle = build_corner_settle_trajectory(
                    digit, self.medium_geometry, 100, settle_intervals=0
                )
                settle = build_corner_settle_trajectory(
                    digit,
                    self.medium_geometry,
                    100,
                    settle_intervals=CORNER_SETTLE_INTERVALS,
                )
                np.testing.assert_array_equal(no_settle.points, baseline.points)
                self.assertEqual(
                    tuple(boundary.name for boundary in baseline.boundaries),
                    frozen["names"],
                )
                self.assertEqual(
                    tuple(boundary.intervals for boundary in baseline.boundaries),
                    frozen["intervals"],
                )
                self.assertEqual(
                    baseline.movement_intervals, frozen["baseline_intervals"]
                )
                self.assertEqual(
                    settle.movement_intervals, frozen["settle_intervals"]
                )
                self.assertAlmostEqual(
                    baseline.movement_duration_s,
                    frozen["baseline_intervals"] * 0.01,
                    places=15,
                )
                self.assertAlmostEqual(
                    settle.movement_duration_s,
                    frozen["settle_intervals"] * 0.01,
                    places=15,
                )
                baseline_episode_steps = (
                    self.medium_geometry.stable_steps
                    + 50
                    + baseline.movement_intervals
                    + 1
                    + self.medium_geometry.hold_steps
                )
                settle_episode_steps = (
                    self.medium_geometry.stable_steps
                    + 50
                    + settle.movement_intervals
                    + 1
                    + self.medium_geometry.hold_steps
                )
                self.assertEqual(
                    baseline_episode_steps, frozen["baseline_episode_steps"]
                )
                self.assertEqual(
                    settle_episode_steps, frozen["settle_episode_steps"]
                )
                for boundary, expected_delta in zip(
                    baseline.boundaries, frozen["deltas"]
                ):
                    actual_delta = (
                        baseline.points[boundary.end_index]
                        - baseline.points[boundary.start_index]
                    )
                    np.testing.assert_allclose(
                        actual_delta,
                        np.asarray(expected_delta) * scale,
                        rtol=0.0,
                        atol=1e-14,
                    )
                self.assertEqual(
                    tuple(corner.turn_qualifies for corner in settle.corners),
                    frozen["qualifies"],
                )
                self.assertEqual(
                    tuple(corner.settle_applied for corner in settle.corners),
                    frozen["qualifies"],
                )
                for corner, (lower, upper) in zip(
                    settle.corners, frozen["angle_ranges"]
                ):
                    self.assertGreaterEqual(corner.turn_angle_deg, lower)
                    self.assertLessEqual(corner.turn_angle_deg, upper)
                keep = np.ones(len(settle.points), dtype=bool)
                for corner in settle.corners:
                    if not corner.settle_applied:
                        self.assertEqual(corner.settle_intervals, 0)
                        continue
                    keep[
                        corner.settle_start_index : corner.settle_end_index
                    ] = False
                    repeated = settle.points[
                        corner.settle_start_index : corner.settle_end_index
                    ]
                    expected_repeated = np.repeat(
                        settle.points[corner.connection_index][None, :],
                        CORNER_SETTLE_INTERVALS,
                        axis=0,
                    )
                    np.testing.assert_array_equal(repeated, expected_repeated)
                np.testing.assert_array_equal(settle.points[keep], baseline.points)
                self.assertEqual(
                    [boundary.ordered_instance_sha256 for boundary in settle.base.boundaries],
                    [boundary.ordered_instance_sha256 for boundary in baseline.boundaries],
                )

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

    def test_medium_runner_has_ten_isolated_single_thread_cpu_cases(self):
        runner = (
            ROOT
            / "server"
            / "run_digit_writing_original_protocol3_corner_settle_medium_five_digit_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "CASES=(digit4_medium_baseline digit4_medium_settle100ms "
            "digit7_medium_baseline digit7_medium_settle100ms "
            "digit2_medium_baseline digit2_medium_settle100ms "
            "digit5_medium_baseline digit5_medium_settle100ms "
            "digit3_medium_baseline digit3_medium_settle100ms)",
            runner,
        )
        self.assertIn("selected_reference_steps\"] == 100", runner)
        self.assertIn("AVAILABLE_CPUS < 10", runner)
        self.assertIn("CPU_QUOTA / CPU_PERIOD < 10", runner)
        self.assertIn(
            "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
            runner,
        )
        self.assertIn("MEDIUM_FIVE_DIGIT_CASES=10", runner)
        self.assertIn("AUTOMATIC_CONTINUATION_STARTED=0", runner)
        self.assertIn("AUTOMATIC_REFERENCE_FALLBACK_STARTED=0", runner)
        self.assertIn("FORMAL_FULL10_STARTED=0", runner)

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
