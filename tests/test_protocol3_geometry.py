from pathlib import Path
import unittest

import numpy as np

from digit_writing import digit_geometry_final as final
from digit_writing.geometry import build_digit_trajectory, load_geometry_config
from digit_writing.protocol3_gate1 import (
    _shared_assertions,
    primitive_manifest_rows,
    target_kinematics_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configurations"
EXPECTED_FAST_INTERVALS = (85, 30, 100, 85, 90, 105, 100, 60, 100, 100)
EXPECTED_MEDIUM_INTERVALS = tuple(value * 2 for value in EXPECTED_FAST_INTERVALS)


class Protocol3GeometryTests(unittest.TestCase):
    @staticmethod
    def load(scale, reference):
        return load_geometry_config(
            CONFIG_ROOT
            / (
                "digit_writing_original_protocol3_geometry_"
                f"scale{scale}_ref{reference}.json"
            )
        )

    def test_protocol3_configuration_is_explicit_and_isolated(self):
        for scale, multiplier in (("2p50", 2.5), ("2p25", 2.25)):
            for reference in (50, 100):
                with self.subTest(scale=scale, reference=reference):
                    config = self.load(scale, reference)
                    self.assertEqual(
                        config.protocol, "digit_writing_original_protocol3"
                    )
                    self.assertEqual(config.timing_mode, "fixed_segment_timing")
                    self.assertEqual(config.scale_multiplier, multiplier)
                    self.assertEqual(config.selected_reference_steps, reference)
                    self.assertEqual(config.training_reference_steps, (reference,))
                    self.assertEqual(config.validation_reference_steps, (reference,))

    def test_fast_and_medium_digit_interval_tables_are_exact(self):
        for reference, expected in (
            (50, EXPECTED_FAST_INTERVALS),
            (100, EXPECTED_MEDIUM_INTERVALS),
        ):
            config = self.load("2p50", reference)
            actual = tuple(
                build_digit_trajectory(digit, config, reference).movement_intervals
                for digit in range(10)
            )
            self.assertEqual(actual, expected)

    def test_all_protocol3_paths_use_one_isotropic_scale(self):
        segments = final.build_digit_segments_units()
        for scale, multiplier in (("2p50", 2.5), ("2p25", 2.25)):
            config = self.load(scale, 50)
            self.assertAlmostEqual(
                config.global_scale_m_per_unit,
                final.GLOBAL_SCALE_M_PER_UNIT * multiplier,
                places=15,
            )
            for digit in range(10):
                sampled = build_digit_trajectory(digit, config, 50)
                self.assertEqual(len(sampled.points), sampled.movement_intervals + 1)
                self.assertTrue(np.isfinite(sampled.points).all())
                expected_length = sum(
                    final.arc_length(segment.points_units)
                    * config.global_scale_m_per_unit
                    for segment in segments[digit]
                )
                self.assertAlmostEqual(sampled.arc_length_m, expected_length, places=13)

    def test_only_curve_a_and_curve_b_require_shared_template_hashes(self):
        config = self.load("2p50", 50)
        trajectories = {
            digit: build_digit_trajectory(digit, config, 50)
            for digit in (2, 3, 5, 6, 9)
        }
        self.assertEqual(
            trajectories[2].boundaries[0].canonical_template_sha256,
            trajectories[3].boundaries[0].canonical_template_sha256,
        )
        self.assertEqual(
            trajectories[3].boundaries[1].canonical_template_sha256,
            trajectories[5].boundaries[2].canonical_template_sha256,
        )
        ellipse6 = trajectories[6].boundaries[1]
        ellipse9 = trajectories[9].boundaries[0]
        self.assertEqual(ellipse6.timing_key, "ellipse_5_4")
        self.assertEqual(ellipse9.timing_key, "ellipse_5_4")
        self.assertEqual(ellipse6.intervals, ellipse9.intervals)
        self.assertNotEqual(
            ellipse6.ordered_instance_sha256,
            ellipse9.ordered_instance_sha256,
        )

    def test_protocol2_geometry_behavior_remains_unchanged(self):
        config = load_geometry_config(
            CONFIG_ROOT / "digit_writing_original_protocol2_geometry.json"
        )
        self.assertEqual(config.timing_mode, "physical_speed_arclength")
        self.assertEqual(config.scale_multiplier, 1.0)
        for digit in range(10):
            for reference in (*config.training_reference_steps, *config.validation_reference_steps):
                with self.subTest(digit=digit, reference=reference):
                    trajectory = build_digit_trajectory(digit, config, reference)
                    authoritative = final.sample_digit(
                        digit,
                        config.reach_distance_m / (reference * config.dt_seconds),
                    )
                    np.testing.assert_array_equal(
                        trajectory.points,
                        authoritative["path_m"],
                    )
                    self.assertEqual(
                        trajectory.movement_intervals,
                        authoritative["movement_intervals"],
                    )

    def test_gate1_manifest_and_target_kinematics_are_self_consistent(self):
        rows = []
        for reference in (50, 100):
            config = self.load("2p50", reference)
            rows.extend(primitive_manifest_rows(config))
            kinematics = target_kinematics_rows(config)
            self.assertEqual(len(kinematics), 10)
            self.assertEqual(
                tuple(row["movement_intervals"] for row in kinematics),
                (
                    EXPECTED_FAST_INTERVALS
                    if reference == 50
                    else EXPECTED_MEDIUM_INTERVALS
                ),
            )
            self.assertTrue(
                all(row["peak_speed_m_s"] > 0.0 for row in kinematics)
            )
        assertions = _shared_assertions(rows)
        self.assertTrue(assertions["passed"])
        self.assertTrue(assertions["ellipse_5_4_is_not_strict_shared"]["passed"])


if __name__ == "__main__":
    unittest.main()
