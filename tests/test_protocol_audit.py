from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from digit_writing.geometry import load_geometry_config
from digit_writing.protocol_audit import (
    build_digit_component_audit,
    build_shared_constraint_audit,
    trajectory_metrics,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "digit_original_protocol_geometry.json"
)


class ProtocolMetricTests(unittest.TestCase):
    def test_straight_line_has_exact_length_and_no_smooth_curvature(self):
        points = np.column_stack((np.linspace(0.0, 1.0, 101), np.zeros(101)))
        metrics = trajectory_metrics(points, dt_seconds=0.01, movement_steps=100)
        self.assertAlmostEqual(metrics["discrete_arc_length_m"], 1.0, places=14)
        self.assertEqual(metrics["kinematic_interval_count"], 100)
        self.assertEqual(metrics["movement_steps"], 100)
        self.assertEqual(metrics["maximum_smooth_curvature_1_m"], 0.0)
        self.assertIsNone(metrics["minimum_smooth_curvature_radius_m"])
        self.assertEqual(metrics["maximum_discrete_acceleration_m_s2"], 0.0)
        self.assertEqual(metrics["near_duplicate_segment_count"], 0)

    def test_circle_curvature_matches_known_radius(self):
        radius = 0.125
        phase = np.linspace(0.0, 2.0 * math.pi, 2001)
        points = radius * np.column_stack((np.cos(phase), np.sin(phase)))
        metrics = trajectory_metrics(points, dt_seconds=0.01, movement_steps=2000)
        self.assertAlmostEqual(
            metrics["maximum_smooth_curvature_1_m"], 1.0 / radius, places=7
        )
        self.assertAlmostEqual(
            metrics["minimum_smooth_curvature_radius_m"], radius, places=7
        )

    def test_join_direction_ignores_original_duplicate_boundary_point(self):
        points = np.array(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
            dtype=np.float64,
        )
        metrics = trajectory_metrics(
            points,
            dt_seconds=0.01,
            movement_steps=4,
            joins=(
                {
                    "label": "out_to_return",
                    "classification": "original_concatenation_boundary",
                    "left_index": 1,
                    "right_index": 2,
                },
            ),
        )
        self.assertTrue(metrics["position_continuous"])
        self.assertEqual(metrics["near_duplicate_segment_count"], 1)
        self.assertAlmostEqual(metrics["joins"][0]["direction_change_deg"], 180.0)


class ProtocolConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_geometry_config(CONFIG_PATH)

    def test_all_frozen_shared_constraints_and_analytic_tangencies_pass(self):
        audit = build_shared_constraint_audit(self.config)
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["tangency"]["digit_2"]["analytic_angle_error_deg"], 0.0)
        self.assertEqual(audit["tangency"]["digit_6"]["analytic_angle_error_deg"], 0.0)
        self.assertLess(
            audit["tangency"]["digit_2"]["high_resolution_finite_difference_angle_error_deg"],
            0.01,
        )
        self.assertLess(
            audit["tangency"]["digit_6"]["high_resolution_finite_difference_angle_error_deg"],
            0.01,
        )

    def test_component_audit_uses_all_digits_and_exactly_continuous_joins(self):
        rows = build_digit_component_audit(self.config)
        self.assertEqual([row["digit"] for row in rows], list(range(10)))
        self.assertTrue(all(row["position_continuous"] for row in rows))
        self.assertEqual(
            [primitive["template_key"] for primitive in rows[6]["primitives"]],
            ["D_down", "ellipse_vertical_ccw_tangent6"],
        )


if __name__ == "__main__":
    unittest.main()
