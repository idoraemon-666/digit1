import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from digit_writing.geometry import build_digit_trajectory, load_geometry_config
from digit_writing.protocol3_movement_interval_lr_scratch import (
    DIGITS,
    EXPECTED_ARM_COUNT,
    LEARNING_RATE_ARMS,
    MAX_UPDATES,
    MOVEMENT_INTERVALS_BY_DIGIT,
    RUN_KIND,
    SOURCE_INTERVALS,
    VARIANT,
    load_experiment_config,
)


class Protocol3MovementIntervalLRScratchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.root
            / "configurations"
            / "digit_writing_original_protocol3_digit0_digit8_movement_interval_lr_scratch6000.json"
        )
        cls.config = load_experiment_config(cls.root, cls.config_path)

    def test_twelve_arm_matrix_is_exact_and_from_scratch(self):
        self.assertEqual(self.config["run_kind"], RUN_KIND)
        self.assertEqual(self.config["variant"], VARIANT)
        self.assertEqual(tuple(self.config["digits"]), DIGITS)
        self.assertEqual(
            self.config["source_movement_intervals"],
            {"0": SOURCE_INTERVALS[0], "8": SOURCE_INTERVALS[8]},
        )
        self.assertEqual(
            self.config["movement_interval_arms"],
            {
                str(digit): list(MOVEMENT_INTERVALS_BY_DIGIT[digit])
                for digit in DIGITS
            },
        )
        self.assertEqual(
            tuple(
                (row["label"], row["learning_rate"])
                for row in self.config["learning_rate_arms"]
            ),
            LEARNING_RATE_ARMS,
        )
        self.assertEqual(len(self.config["arms"]), EXPECTED_ARM_COUNT)
        self.assertEqual(self.config["training"]["max_updates"], MAX_UPDATES)
        self.assertTrue(self.config["training"]["from_scratch"])
        self.assertFalse(self.config["training"]["early_stopping"])
        self.assertTrue(
            self.config["optimizer"]["fixed_learning_rate_no_schedule"]
        )
        self.assertTrue(
            self.config["timing_intervention"][
                "digit0_time170_is_original_control"
            ]
        )
        self.assertTrue(
            self.config["timing_intervention"][
                "digit8_time200_is_original_control"
            ]
        )

    def test_only_interval_count_changes_spatial_identity(self):
        geometry = load_geometry_config(
            self.root / self.config["geometry_config"]
        )
        for digit in DIGITS:
            source = build_digit_trajectory(digit, geometry, 100)
            self.assertEqual(
                source.movement_intervals,
                SOURCE_INTERVALS[digit],
            )
            self.assertEqual(len(source.boundaries), 1)
            for movement_intervals in MOVEMENT_INTERVALS_BY_DIGIT[digit]:
                candidate = build_digit_trajectory(
                    digit,
                    geometry,
                    100,
                    movement_intervals_override=movement_intervals,
                )
                self.assertEqual(
                    candidate.movement_intervals,
                    movement_intervals,
                )
                self.assertEqual(
                    len(candidate.points),
                    movement_intervals + 1,
                )
                self.assertTrue(
                    np.array_equal(
                        source.points[[0, -1]],
                        candidate.points[[0, -1]],
                    )
                )
                self.assertEqual(
                    source.boundaries[0].source_geometry_sha256,
                    candidate.boundaries[0].source_geometry_sha256,
                )
                self.assertEqual(
                    source.boundaries[0].derived_path_geometry_sha256,
                    candidate.boundaries[0].derived_path_geometry_sha256,
                )
                self.assertAlmostEqual(
                    source.arc_length_m,
                    candidate.arc_length_m,
                    places=14,
                )
                self.assertEqual(
                    (
                        source.boundaries[0].temporal_sampling_sha256
                        != candidate.boundaries[0].temporal_sampling_sha256
                    ),
                    movement_intervals != SOURCE_INTERVALS[digit],
                )

    def test_interval_override_scope_is_closed(self):
        geometry = load_geometry_config(
            self.root / self.config["geometry_config"]
        )
        with self.assertRaises(ValueError):
            build_digit_trajectory(
                1,
                geometry,
                100,
                movement_intervals_override=220,
            )
        with self.assertRaises(ValueError):
            build_digit_trajectory(
                0,
                geometry,
                100,
                movement_intervals_override=210,
            )
        with self.assertRaises(ValueError):
            build_digit_trajectory(
                8,
                geometry,
                100,
                movement_intervals_override=170,
            )

    def test_config_mutations_are_rejected(self):
        mutations = []
        changed = copy.deepcopy(self.config)
        changed["movement_interval_arms"]["0"] = [170, 190, 210]
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["learning_rate_arms"][1]["learning_rate"] = 0.0001
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["arms"][0]["digit"] = 8
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["training"]["from_scratch"] = False
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["training"]["max_updates"] = 8000
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["optimizer"]["fixed_learning_rate_no_schedule"] = False
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["decision"]["automatic_interval_selection"] = True
        mutations.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in mutations:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_experiment_config(self.root, path)

    def test_runner_is_twelve_way_fixed_lr_and_scratch_only(self):
        script = (
            self.root
            / "server"
            / "run_digit_writing_original_protocol3_digit0_digit8_movement_interval_lr_scratch_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PROTOCOL3_DIGIT0_DIGIT8_INTERVAL_LR_SCRATCH_AUTHORIZED",
            script,
        )
        self.assertIn("AVAILABLE_CPUS < 12", script)
        self.assertIn("CPU_QUOTA / CPU_PERIOD < 12", script)
        self.assertIn("COMPLETED_ARMS=12", script)
        self.assertIn("COMPLETED_UPDATES_PER_ARM=6000", script)
        self.assertIn("FROM_SCRATCH=1", script)
        self.assertIn("FIXED_LR_NO_SCHEDULE=1", script)
        self.assertIn("DIGIT0_SOURCE_INTERVALS=170", script)
        self.assertIn("DIGIT8_SOURCE_INTERVALS=200", script)
        self.assertIn(
            "DIGIT0_MOVEMENT_INTERVAL_ARMS=170,200,220",
            script,
        )
        self.assertIn(
            "DIGIT8_MOVEMENT_INTERVAL_ARMS=200,220,240",
            script,
        )
        self.assertIn(
            "===== FAILED ARM ${ARM_LABELS[$INDEX]} =====",
            script,
        )
        self.assertNotIn("resume_checkpoint", script)
        self.assertNotIn("target_completed_updates", script)


@unittest.skipUnless(
    importlib.util.find_spec("motornet")
    and importlib.util.find_spec("configargparse"),
    "MotorNet integration dependencies are server-only",
)
class Protocol3MovementIntervalEnvironmentTests(unittest.TestCase):
    def test_environment_uses_exact_interval_override(self):
        import motornet as mn

        from train import DIGIT_ENV_CLASSES

        root = Path(__file__).resolve().parents[1]
        geometry = (
            root
            / "configurations"
            / "digit_writing_original_protocol3_geometry_scale2p50_ref100_corner_ease_v3.json"
        )
        for digit in DIGITS:
            for movement_intervals in MOVEMENT_INTERVALS_BY_DIGIT[digit]:
                effector = mn.effector.RigidTendonArm26(
                    mn.muscle.MujocoHillMuscle()
                )
                environment = DIGIT_ENV_CLASSES[digit](
                    effector=effector,
                    geometry_config_path=geometry,
                    movement_intervals_override=movement_intervals,
                )
                environment.reset(
                    testing=False,
                    seed=42,
                    options={
                        "batch_size": 1,
                        "reach_conds": 0,
                        "speed_cond": 0,
                        "custom_delay": 50,
                        "deterministic": True,
                    },
                )
                self.assertEqual(
                    environment.movement_intervals,
                    movement_intervals,
                )
                self.assertEqual(
                    environment.traj.shape[1],
                    movement_intervals + 1,
                )


if __name__ == "__main__":
    unittest.main()
