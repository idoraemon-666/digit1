import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

from digit_writing.geometry import load_geometry_config
from digit_writing.geometry_audit import (
    FK_TOLERANCE_M,
    _inverse_kinematics,
    build_workspace_audit,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "digit_writing_original_protocol2_geometry.json"
)
MOTORNET_AVAILABLE = importlib.util.find_spec("motornet") is not None


class GeometryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_geometry_config(CONFIG_PATH)

    def test_analytic_inverse_matches_two_link_forward_equation(self):
        link1 = 0.31
        link2 = 0.33
        expected = np.array(
            (
                (0.4, 0.7),
                (1.0, 1.3),
                (1.7, 2.1),
            ),
            dtype=np.float64,
        )
        shoulder = expected[:, 0]
        elbow = expected[:, 1]
        points = np.column_stack(
            (
                link1 * np.cos(shoulder)
                + link2 * np.cos(shoulder + elbow),
                link1 * np.sin(shoulder)
                + link2 * np.sin(shoulder + elbow),
            )
        )
        actual, reachable = _inverse_kinematics(points, link1, link2)
        self.assertTrue(reachable.all())
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)

    @unittest.skipUnless(MOTORNET_AVAILABLE, "MotorNet is required")
    def test_full_motornet_workspace_and_dynamic_smoke(self):
        audit = build_workspace_audit(self.config)
        self.assertEqual(audit["motornet_version"], "0.2.0")
        self.assertEqual(audit["device"], "cpu")
        self.assertEqual(audit["condition_count"], 320)
        self.assertEqual(len(audit["conditions"]), 320)
        self.assertEqual(len(audit["dynamic_smoke"]), 64)
        np.testing.assert_allclose(
            audit["anchor_m"],
            (-0.24382227659225464, 0.29947587847709656),
            atol=1e-7,
            rtol=0,
        )
        self.assertFalse(audit["touches_limit"])
        self.assertGreater(
            audit["radial_reach_margin_m"]["minimum_inner"], 0.0
        )
        self.assertGreater(
            audit["radial_reach_margin_m"]["minimum_outer"], 0.0
        )
        self.assertGreater(
            audit["joint_angle_margin_rad"]["overall_minimum"], 0.0
        )
        self.assertLessEqual(
            audit["inverse_forward_consistency"]["maximum_error_m"],
            FK_TOLERANCE_M,
        )
        self.assertTrue(audit["dynamic_smoke_passed"])
        self.assertTrue(audit["passed"])
        self.assertIn(audit["most_dangerous_condition"]["digit"], range(10))
        self.assertIn(
            audit["most_dangerous_condition"]["direction_index"], range(32)
        )


if __name__ == "__main__":
    unittest.main()
