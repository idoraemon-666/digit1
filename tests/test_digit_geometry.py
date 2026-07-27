import inspect
from pathlib import Path
import unittest

import numpy as np

from digit_writing import digit_geometry_final as final
from digit_writing import geometry


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "digit_writing_original_protocol2_geometry.json"
)


EXPECTED_ARC_LENGTHS_M = {
    0: 0.170026832,
    1: 0.061538462,
    2: 0.145800867,
    3: 0.132654179,
    4: 0.171295140,
    5: 0.126596902,
    6: 0.185072096,
    7: 0.109827733,
    8: 0.206257428,
    9: 0.194046455,
}


class FinalDigitGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = geometry.load_geometry_config(CONFIG_PATH)

    def test_final_authority_self_test(self):
        final.self_test()

    def test_configuration_only_references_the_final_authority(self):
        self.assertEqual(
            self.config.geometry_source,
            "digit_writing/digit_geometry_final.py",
        )
        self.assertEqual(
            self.config.global_scale_m_per_unit,
            final.GLOBAL_SCALE_M_PER_UNIT,
        )
        self.assertEqual(self.config.dt_seconds, 0.01)
        self.assertEqual(self.config.stable_steps, 25)
        self.assertEqual(self.config.delay_steps, (25, 50, 75))
        self.assertEqual(self.config.hold_steps, 25)
        config_text = CONFIG_PATH.read_text(encoding="utf-8")
        for obsolete in ("curve_C", "minimum_jerk", "movement_steps"):
            self.assertNotIn(obsolete, config_text)

    def test_arc_lengths_match_the_frozen_reference(self):
        audit = final.build_audit()
        for digit, expected in EXPECTED_ARC_LENGTHS_M.items():
            actual = audit["geometry"][str(digit)]["total_arc_length_m"]
            self.assertAlmostEqual(actual, expected, delta=1e-9)

    def test_all_digits_start_at_origin_and_join_once(self):
        for digit in range(10):
            result = final.sample_digit(digit, final.TRAIN_SPEEDS_MPS["medium"])
            path = result["path_m"]
            np.testing.assert_array_equal(path[0], np.zeros(2))
            self.assertTrue(np.isfinite(path).all())
            self.assertEqual(len(path), result["movement_intervals"] + 1)
            self.assertEqual(result["segment_boundaries"][0], 0)
            self.assertEqual(result["segment_boundaries"][-1], len(path) - 1)

    def test_shared_curves_keep_shape_length_and_sampling(self):
        for speed in final.TRAIN_SPEEDS_MPS.values():
            sampled = {digit: final.sample_digit(digit, speed) for digit in (2, 3, 5)}
            a2 = sampled[2]["segments"][0]
            a3 = sampled[3]["segments"][0]
            b3 = sampled[3]["segments"][1]
            b5 = sampled[5]["segments"][2]
            self.assertEqual(a2["intervals"], a3["intervals"])
            self.assertEqual(b3["intervals"], b5["intervals"])
            self.assertAlmostEqual(a2["arc_length_m"], a3["arc_length_m"], places=14)
            self.assertAlmostEqual(b3["arc_length_m"], b5["arc_length_m"], places=14)
        adapter = {
            digit: geometry.build_digit_trajectory(digit, self.config, 100)
            for digit in (2, 3, 5)
        }
        self.assertEqual(
            adapter[2].boundaries[0].canonical_sample_sha256,
            adapter[3].boundaries[0].canonical_sample_sha256,
        )
        self.assertEqual(
            adapter[3].boundaries[1].canonical_sample_sha256,
            adapter[5].boundaries[2].canonical_sample_sha256,
        )

    def test_runtime_adapter_is_identical_to_the_authoritative_sampler(self):
        for digit in range(10):
            trajectory = geometry.build_digit_trajectory(
                digit, self.config, 100
            )
            authoritative = final.sample_digit(
                digit, geometry.physical_speed(self.config, 100)
            )
            np.testing.assert_array_equal(
                trajectory.points, authoritative["path_m"]
            )
            self.assertEqual(
                trajectory.movement_intervals,
                authoritative["movement_intervals"],
            )

    def test_all_spatial_rotations_preserve_arc_length(self):
        for digit in range(10):
            expected = geometry.build_digit_trajectory(
                digit, self.config, 100
            ).arc_length_m
            for angle in geometry.validation_angles():
                actual = geometry.build_digit_trajectory(
                    digit,
                    self.config,
                    100,
                    spatial_angle_rad=float(angle),
                ).arc_length_m
                self.assertAlmostEqual(actual, expected, places=14)

    def test_no_obsolete_time_parameterization_is_reintroduced(self):
        source = inspect.getsource(final)
        self.assertNotIn("minimum_jerk", source)
        self.assertIn("resample_linear_arclength", source)


if __name__ == "__main__":
    unittest.main()
