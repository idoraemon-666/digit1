import inspect
import math
from pathlib import Path
import unittest

import numpy as np

from digit_writing import geometry


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "digit_original_protocol_geometry.json"
)


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _signed_area(points):
    return 0.5 * np.sum(
        points[:-1, 0] * points[1:, 1]
        - points[1:, 0] * points[:-1, 1]
    )


class DigitGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = geometry.load_geometry_config(CONFIG_PATH)
        cls.library = geometry.primitive_library(cls.config)

    def occurrence_points(self, digit, index):
        occurrence = geometry.prescribed_digit_primitives(digit, self.config)[index]
        return geometry._rotate(
            self.library[occurrence.template_key], occurrence.rotation_rad
        )

    def test_frozen_geometry_and_time_values(self):
        config = self.config
        self.assertEqual(config.a, 0.035)
        self.assertEqual(config.b, 0.02625)
        self.assertEqual(config.H, 0.035)
        self.assertEqual(config.V, 0.06125)
        self.assertEqual(config.D_dx, 0.02625)
        self.assertEqual(config.D_dy, 0.0525)
        self.assertEqual(config.global_scale, 0.8791208791208792)
        self.assertEqual(config.dt_seconds, 0.01)
        self.assertEqual(config.stable_steps, 25)
        self.assertEqual(config.delay_steps, (25, 50, 75))
        self.assertEqual(config.hold_steps, 25)
        self.assertEqual(config.training_reference_steps, (50, 100, 150))
        self.assertEqual(config.validation_reference_steps, tuple(range(50, 150, 10)))
        self.assertNotIn("lobe_rx", CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertNotIn("lobe_ry", CONFIG_PATH.read_text(encoding="utf-8"))

    def test_canonical_lines_are_shared_and_strictly_reversed(self):
        for forward, reverse in (
            ("H_right", "H_left"),
            ("V_down", "V_up"),
            ("D_up", "D_down"),
        ):
            expected = self.library[forward][::-1] - self.library[forward][-1]
            np.testing.assert_array_equal(self.library[reverse], expected)

        expected_lengths = {
            "H_right": self.config.H,
            "H_left": self.config.H,
            "V_down": self.config.V,
            "V_up": self.config.V,
            "D_up": math.hypot(self.config.D_dx, self.config.D_dy),
            "D_down": math.hypot(self.config.D_dx, self.config.D_dy),
        }
        for key, unscaled_length in expected_lengths.items():
            actual = geometry._arc_table(self.library[key])[1]
            self.assertAlmostEqual(
                actual, unscaled_length * self.config.global_scale, places=14
            )

    def test_diagonal_angle_is_frozen(self):
        diagonal = self.library["D_up"][-1]
        angle_deg = math.degrees(math.atan2(diagonal[1], diagonal[0]))
        self.assertAlmostEqual(angle_deg, 63.43494882292201, places=12)

    def test_every_digit_starts_at_origin_and_is_finite(self):
        for digit in range(10):
            trajectory = geometry.build_digit_trajectory(digit, self.config, 100)
            np.testing.assert_array_equal(trajectory.points[0], np.zeros(2))
            self.assertTrue(np.isfinite(trajectory.points).all())

    def test_joins_are_c0_and_retained_once(self):
        for digit in range(10):
            trajectory = geometry.build_digit_trajectory(digit, self.config, 100)
            self.assertEqual(
                len(trajectory.points),
                sum(boundary.intervals for boundary in trajectory.boundaries) + 1,
            )
            for left, right in zip(trajectory.boundaries, trajectory.boundaries[1:]):
                self.assertEqual(left.end_index, right.start_index)
                np.testing.assert_array_equal(
                    trajectory.points[left.end_index],
                    trajectory.points[right.start_index],
                )

    def test_all_ellipse_templates_use_the_same_axes(self):
        horizontal = geometry.standard_ellipse(
            self.config, "horizontal", 0.0, 2 * math.pi, "increasing"
        )
        vertical = geometry.standard_ellipse(
            self.config, "vertical", 0.0, 2 * math.pi, "increasing"
        )
        self.assertAlmostEqual(np.max(np.abs(horizontal[:, 0])), self.config.a)
        self.assertAlmostEqual(np.max(np.abs(horizontal[:, 1])), self.config.b)
        self.assertAlmostEqual(np.max(np.abs(vertical[:, 0])), self.config.b)
        self.assertAlmostEqual(np.max(np.abs(vertical[:, 1])), self.config.a)

    def test_digit_zero_is_top_starting_counterclockwise_loop(self):
        points = self.occurrence_points(0, 0)
        np.testing.assert_array_equal(points[0], points[-1])
        self.assertAlmostEqual(float(points[:, 1].max()), 0.0, places=14)
        self.assertLess(points[1, 0], 0.0)
        self.assertGreater(_signed_area(points), 0.0)

    def test_digit_two_control_points_and_tangent(self):
        points = self.occurrence_points(2, 0)
        u = _unit(np.array((self.config.D_dx, self.config.D_dy)))
        v = np.array((u[1], -u[0]))
        scale = self.config.global_scale
        expected_midpoint = scale * (self.config.a * u + self.config.b * v)
        expected_endpoint = scale * (2 * self.config.b * v)
        np.testing.assert_allclose(
            points[len(points) // 2], expected_midpoint, atol=1e-14, rtol=0
        )
        np.testing.assert_allclose(points[-1], expected_endpoint, atol=1e-14, rtol=0)
        tangent = _unit(points[-1] - points[-2])
        d_down = _unit(self.library["D_down"][-1])
        self.assertGreater(float(np.dot(tangent, d_down)), 0.999999999)

    def test_digit_three_uses_two_identical_right_halves(self):
        occurrences = geometry.prescribed_digit_primitives(3, self.config)
        self.assertEqual(
            [item.template_key for item in occurrences],
            [
                "half_ellipse_right_top_to_bottom",
                "half_ellipse_right_top_to_bottom",
            ],
        )
        np.testing.assert_array_equal(
            self.occurrence_points(3, 0), self.occurrence_points(3, 1)
        )

    def test_digit_five_uses_standard_h_v_and_right_half(self):
        occurrences = geometry.prescribed_digit_primitives(5, self.config)
        self.assertEqual(
            [item.template_key for item in occurrences],
            ["H_left", "V_down", "half_ellipse_right_top_to_bottom"],
        )

    def test_digit_six_descends_to_upper_left_tangent_then_ccw_loop(self):
        occurrences = geometry.prescribed_digit_primitives(6, self.config)
        self.assertEqual(
            [item.template_key for item in occurrences],
            ["D_down", "ellipse_vertical_ccw_tangent6"],
        )
        diagonal = self.occurrence_points(6, 0)
        self.assertLess(diagonal[-1, 0], 0.0)
        self.assertLess(diagonal[-1, 1], 0.0)
        loop = self.occurrence_points(6, 1)
        np.testing.assert_array_equal(loop[0], loop[-1])
        self.assertGreater(_signed_area(loop), 0.0)
        tangent = _unit(loop[1] - loop[0])
        d_down = _unit(self.library["D_down"][-1])
        self.assertGreater(float(np.dot(tangent, d_down)), 0.999999999)

    def test_digit_eight_has_the_required_three_part_order(self):
        occurrences = geometry.prescribed_digit_primitives(8, self.config)
        self.assertEqual(
            [item.template_key for item in occurrences],
            [
                "half_ellipse_left_top_to_bottom",
                "ellipse_horizontal_cw_top",
                "half_ellipse_right_bottom_to_top",
            ],
        )
        self.assertGreater(_signed_area(self.occurrence_points(8, 0)), 0.0)
        self.assertLess(_signed_area(self.occurrence_points(8, 1)), 0.0)
        self.assertGreater(_signed_area(self.occurrence_points(8, 2)), 0.0)

    def test_digit_nine_is_right_starting_cw_loop_then_vertical(self):
        occurrences = geometry.prescribed_digit_primitives(9, self.config)
        self.assertEqual(
            [item.template_key for item in occurrences],
            ["ellipse_vertical_cw_right", "V_down"],
        )
        loop = self.occurrence_points(9, 0)
        np.testing.assert_array_equal(loop[0], loop[-1])
        self.assertLess(loop[1, 1], 0.0)
        self.assertLess(_signed_area(loop), 0.0)

    def test_forbidden_prefix_relations_do_not_hold(self):
        digit3 = self.occurrence_points(3, 0)
        digit8 = self.occurrence_points(8, 0)
        self.assertGreater(digit3[1, 0], 0.0)
        self.assertLess(digit8[1, 0], 0.0)

        digit0 = self.occurrence_points(0, 0)
        digit9 = self.occurrence_points(9, 0)
        self.assertFalse(np.allclose(digit0[1], digit9[1], atol=1e-14, rtol=0))

    def test_training_and_validation_rotations_preserve_arc_length_and_anchor(self):
        anchor = np.array((0.17, 0.23))
        for angles in (geometry.training_angles(), geometry.validation_angles()):
            for digit in range(10):
                reference = geometry.build_digit_trajectory(
                    digit, self.config, 100, anchor=anchor
                )
                reference_chord_length = np.linalg.norm(
                    np.diff(reference.points, axis=0), axis=1
                ).sum()
                for angle in angles:
                    rotated = geometry.build_digit_trajectory(
                        digit,
                        self.config,
                        100,
                        spatial_angle_rad=float(angle),
                        anchor=anchor,
                    )
                    np.testing.assert_array_equal(rotated.points[0], anchor)
                    rotated_chord_length = np.linalg.norm(
                        np.diff(rotated.points, axis=0), axis=1
                    ).sum()
                    self.assertAlmostEqual(
                        float(rotated_chord_length),
                        float(reference_chord_length),
                        places=13,
                    )


class DigitTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = geometry.load_geometry_config(CONFIG_PATH)

    def test_training_speed_mapping_is_global(self):
        expected = {
            50: (0.5, 2.0 / 3.0),
            100: (0.25, 1.0 / 3.0),
            150: (1.0 / 6.0, 0.0),
        }
        for reference_steps, (speed, scalar) in expected.items():
            for digit in range(10):
                trajectory = geometry.build_digit_trajectory(
                    digit, self.config, reference_steps
                )
                self.assertAlmostEqual(trajectory.physical_speed_m_s, speed)
                self.assertAlmostEqual(trajectory.speed_scalar, scalar)

    def test_validation_speed_mapping_is_global(self):
        for reference_steps in self.config.validation_reference_steps:
            expected_speed = 0.25 / (reference_steps * 0.01)
            expected_scalar = 1.0 - reference_steps / 150.0
            values = []
            for digit in range(10):
                trajectory = geometry.build_digit_trajectory(
                    digit, self.config, reference_steps
                )
                values.append(
                    (trajectory.physical_speed_m_s, trajectory.speed_scalar)
                )
            self.assertTrue(
                all(
                    math.isclose(speed, expected_speed)
                    and math.isclose(scalar, expected_scalar)
                    for speed, scalar in values
                )
            )

    def test_shared_templates_have_identical_counts_and_hashes(self):
        for reference_steps in self.config.training_reference_steps:
            groups = {}
            for digit in range(10):
                trajectory = geometry.build_digit_trajectory(
                    digit, self.config, reference_steps
                )
                for boundary in trajectory.boundaries:
                    groups.setdefault(boundary.template_key, set()).add(
                        (boundary.intervals, boundary.canonical_sample_sha256)
                    )
            for values in groups.values():
                self.assertEqual(len(values), 1)

    def test_duration_obeys_per_primitive_arc_length_ceiling(self):
        for digit in range(10):
            trajectory = geometry.build_digit_trajectory(digit, self.config, 100)
            ideal_duration = trajectory.arc_length_m / trajectory.physical_speed_m_s
            ceiling_allowance = len(trajectory.boundaries) * self.config.dt_seconds
            self.assertGreaterEqual(trajectory.movement_duration_s, ideal_duration)
            self.assertLess(
                trajectory.movement_duration_s, ideal_duration + ceiling_allowance
            )

    def test_fast_medium_slow_duration_order(self):
        for digit in range(10):
            durations = [
                geometry.build_digit_trajectory(digit, self.config, steps).movement_duration_s
                for steps in (50, 100, 150)
            ]
            self.assertLess(durations[0], durations[1])
            self.assertLess(durations[1], durations[2])

    def test_no_primitive_has_fewer_than_two_intervals(self):
        audit = geometry.build_time_audit(self.config)
        self.assertEqual(audit["primitives_with_fewer_than_two_intervals"], [])
        self.assertEqual(len(audit["training_timing"]), 30)
        self.assertEqual(len(audit["validation_timing"]), 100)

    def test_linear_arc_sampling_is_nearly_uniform(self):
        for digit in range(10):
            trajectory = geometry.build_digit_trajectory(digit, self.config, 100)
            for boundary in trajectory.boundaries:
                segment = trajectory.points[
                    boundary.start_index : boundary.end_index + 1
                ]
                step_lengths = np.linalg.norm(np.diff(segment, axis=0), axis=1)
                relative_spread = np.ptp(step_lengths) / np.mean(step_lengths)
                self.assertLess(relative_spread, 1e-3)

    def test_hard_corner_is_not_smoothed(self):
        trajectory = geometry.build_digit_trajectory(4, self.config, 100)
        join = trajectory.boundaries[0].end_index
        incoming = _unit(trajectory.points[join] - trajectory.points[join - 1])
        outgoing = _unit(trajectory.points[join + 1] - trajectory.points[join])
        self.assertLess(float(np.dot(incoming, outgoing)), 0.0)

    def test_digits_two_and_six_have_discrete_tangent_continuity(self):
        for digit in (2, 6):
            trajectory = geometry.build_digit_trajectory(digit, self.config, 100)
            join = trajectory.boundaries[0].end_index
            incoming = _unit(trajectory.points[join] - trajectory.points[join - 1])
            outgoing = _unit(trajectory.points[join + 1] - trajectory.points[join])
            self.assertGreater(float(np.dot(incoming, outgoing)), 0.999)

    def test_minimum_jerk_is_absent(self):
        source = inspect.getsource(geometry)
        forbidden_name = "minimum" + "_jerk"
        self.assertNotIn(forbidden_name, source)


if __name__ == "__main__":
    unittest.main()
