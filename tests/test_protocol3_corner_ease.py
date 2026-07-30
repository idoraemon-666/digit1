from pathlib import Path
import unittest

import numpy as np

from digit_writing import digit_geometry_final as final
from digit_writing.corner_time_reparameterization import (
    BASE_WINDOW_INTERVALS,
    CORNER_EASE_TIMING,
    EXTRA_INTERVALS_PER_SIDE,
    RESAMPLED_WINDOW_INTERVALS,
    V3_STEP_WEIGHT_DENOMINATOR,
    V3_STEP_WEIGHT_NUMERATORS,
    _corner_ease_fractions,
    corner_ease_step_weights_v3,
)
from digit_writing.geometry import (
    _identified_json_hash,
    _identified_sample_hash,
    build_digit_trajectory,
    load_geometry_config,
)
from digit_writing.protocol3_corner_ease import (
    EXPECTED_INTERVALS,
    EXPECTED_QUALIFYING_BOUNDARIES,
    _kinematic_metrics,
    _local_corner_points,
    _read_json,
    _validate_config,
    select_validation_history,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configurations"
BASE_GEOMETRY = (
    CONFIG_ROOT / "digit_writing_original_protocol3_geometry_scale2p50_ref100.json"
)
EASE_GEOMETRY = (
    CONFIG_ROOT
    / "digit_writing_original_protocol3_geometry_scale2p50_ref100_corner_ease_v3.json"
)
EXPERIMENT_CONFIG = (
    CONFIG_ROOT / "digit_writing_original_protocol3_ten_digit_corner_ease_overfit_v3.json"
)


class Protocol3CornerEaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_config = load_geometry_config(BASE_GEOMETRY)
        cls.ease_config = load_geometry_config(EASE_GEOMETRY)

    def test_v3_step_table_is_frozen_positive_and_exact(self):
        weights = corner_ease_step_weights_v3()
        self.assertEqual(V3_STEP_WEIGHT_DENOMINATOR, 20)
        self.assertEqual(
            V3_STEP_WEIGHT_NUMERATORS,
            (20,) * 5 + (19, 17, 15, 13, 11, 9, 7, 5, 3, 1),
        )
        self.assertEqual(len(weights), RESAMPLED_WINDOW_INTERVALS)
        self.assertTrue(np.all(weights > 0.0))
        self.assertTrue(np.all(np.diff(weights) <= 0.0))
        self.assertAlmostEqual(float(weights.sum()), BASE_WINDOW_INTERVALS)
        self.assertAlmostEqual(float(weights[-1]), 0.05)
        changes = np.abs(np.diff(np.concatenate(([1.0], weights))))
        self.assertAlmostEqual(float(changes.max()), 0.10)

    def test_v3_maps_are_strictly_monotone_and_time_reversed(self):
        to_stop = _corner_ease_fractions(
            60, ease_start=False, ease_end=True
        )
        from_stop = _corner_ease_fractions(
            60, ease_start=True, ease_end=False
        )
        self.assertTrue(np.all(np.diff(to_stop) > 0.0))
        self.assertTrue(np.all(np.diff(from_stop) > 0.0))
        np.testing.assert_allclose(
            np.diff(from_stop)[:RESAMPLED_WINDOW_INTERVALS],
            np.diff(to_stop)[-RESAMPLED_WINDOW_INTERVALS:][::-1],
            rtol=0.0,
            atol=1e-15,
        )

    def test_corner_ease_configuration_is_frozen_and_isolated(self):
        config = _read_json(EXPERIMENT_CONFIG)
        source, baseline = _validate_config(ROOT, config)
        self.assertEqual(self.ease_config.timing_mode, CORNER_EASE_TIMING)
        self.assertEqual(self.ease_config.selected_reference_steps, 100)
        self.assertEqual(source["selected_reference_steps"], 100)
        self.assertEqual(source["training"]["batch_size"], 8)
        self.assertEqual(len(baseline["cases"]), 5)
        self.assertFalse(
            config["geometry_identity"]["legacy_canonical_template_sha256_redefined"]
        )

    def test_expected_corner_manifest_and_interval_table(self):
        for digit in range(10):
            trajectory = build_digit_trajectory(digit, self.ease_config, 100)
            qualifying = tuple(
                int(row["boundary_index"])
                for row in trajectory.corner_ease_boundaries
                if row["qualifies"]
            )
            self.assertEqual(qualifying, EXPECTED_QUALIFYING_BOUNDARIES[digit])
            self.assertEqual(trajectory.movement_intervals, EXPECTED_INTERVALS[digit])
            self.assertEqual(len(trajectory.points), EXPECTED_INTERVALS[digit] + 1)

    def test_non_corner_targets_are_bitwise_unchanged(self):
        for digit in (0, 1, 6, 8, 9):
            baseline = build_digit_trajectory(digit, self.baseline_config, 100)
            candidate = build_digit_trajectory(digit, self.ease_config, 100)
            np.testing.assert_array_equal(candidate.points, baseline.points)

    def test_corner_targets_preserve_geometry_without_pause_or_reversal(self):
        segments = final.build_digit_segments_units()
        for digit in (2, 3, 4, 5, 7):
            baseline = build_digit_trajectory(digit, self.baseline_config, 100)
            candidate = build_digit_trajectory(digit, self.ease_config, 100)
            np.testing.assert_array_equal(
                candidate.points[[0, -1]], baseline.points[[0, -1]]
            )
            steps = np.linalg.norm(np.diff(candidate.points, axis=0), axis=1)
            self.assertTrue(np.all(steps > 0.0))
            for old, new, source_segment in zip(
                baseline.boundaries, candidate.boundaries, segments[digit]
            ):
                np.testing.assert_array_equal(
                    baseline.points[old.end_index], candidate.points[new.end_index]
                )
                self.assertEqual(
                    old.source_geometry_sha256, new.source_geometry_sha256
                )
                self.assertEqual(
                    old.derived_path_geometry_sha256,
                    new.derived_path_geometry_sha256,
                )
                self.assertTrue(np.isfinite(source_segment.points_units).all())

    def test_three_hash_layers_are_independent(self):
        source = {"primitive": "polyline", "anchors_units": [[0.0, 0.0], [1.0, 0.0]]}
        source_hash = _identified_json_hash(source)
        changed_source = {
            "primitive": "polyline",
            "anchors_units": [[0.0, 0.0], [1.0, 0.1]],
        }
        rounded = np.asarray(((0.0, 0.0), (0.5, 0.1), (1.0, 0.0)))
        straight = np.asarray(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)))
        self.assertEqual(source_hash, _identified_json_hash(source))
        self.assertNotEqual(source_hash, _identified_json_hash(changed_source))
        self.assertNotEqual(
            _identified_sample_hash(straight),
            _identified_sample_hash(rounded),
        )
        self.assertNotEqual(
            _identified_sample_hash(straight),
            _identified_sample_hash(straight[[0, 2]]),
        )

    def test_temporal_hash_changes_only_for_retimed_segments(self):
        for digit in range(10):
            baseline = build_digit_trajectory(digit, self.baseline_config, 100)
            candidate = build_digit_trajectory(digit, self.ease_config, 100)
            equality = tuple(
                old.temporal_sampling_sha256 == new.temporal_sampling_sha256
                for old, new in zip(baseline.boundaries, candidate.boundaries)
            )
            retimed_segments = {
                segment_index
                for boundary_index in EXPECTED_QUALIFYING_BOUNDARIES[digit]
                for segment_index in (boundary_index, boundary_index + 1)
            }
            self.assertEqual(
                equality,
                tuple(
                    index not in retimed_segments
                    for index in range(len(equality))
                ),
            )

    def test_legacy_canonical_template_hash_keeps_its_sampled_array_semantics(self):
        baseline = build_digit_trajectory(3, self.baseline_config, 100)
        expected = _identified_sample_hash(
            final.resample_linear_arclength(
                final.canonical_curve_a(), baseline.boundaries[0].intervals
            )
        )
        self.assertEqual(
            baseline.boundaries[0].canonical_template_sha256,
            expected,
        )
        candidate = build_digit_trajectory(3, self.ease_config, 100)
        self.assertNotEqual(
            candidate.boundaries[0].canonical_template_sha256,
            baseline.boundaries[0].canonical_template_sha256,
        )

    def test_each_qualifying_corner_adds_ten_intervals(self):
        for digit in range(10):
            baseline = build_digit_trajectory(digit, self.baseline_config, 100)
            candidate = build_digit_trajectory(digit, self.ease_config, 100)
            expected_added = 2 * EXTRA_INTERVALS_PER_SIDE * len(
                EXPECTED_QUALIFYING_BOUNDARIES[digit]
            )
            self.assertEqual(
                candidate.movement_intervals - baseline.movement_intervals,
                expected_added,
            )
        self.assertEqual(BASE_WINDOW_INTERVALS, 10)
        self.assertEqual(RESAMPLED_WINDOW_INTERVALS, 15)

    def test_qualifying_corners_are_near_zero_but_never_stopped(self):
        for digit in (2, 3, 4, 5, 7):
            baseline = build_digit_trajectory(digit, self.baseline_config, 100)
            candidate = build_digit_trajectory(digit, self.ease_config, 100)
            for boundary_index in EXPECTED_QUALIFYING_BOUNDARIES[digit]:
                old_left = baseline.boundaries[boundary_index]
                old_right = baseline.boundaries[boundary_index + 1]
                new_left = candidate.boundaries[boundary_index]
                connection = new_left.end_index
                regular = np.median(
                    np.concatenate(
                        (
                            np.linalg.norm(
                                np.diff(
                                    baseline.points[
                                        old_left.end_index - BASE_WINDOW_INTERVALS :
                                        old_left.end_index + 1
                                    ],
                                    axis=0,
                                ),
                                axis=1,
                            ),
                            np.linalg.norm(
                                np.diff(
                                    baseline.points[
                                        old_right.start_index :
                                        old_right.start_index + BASE_WINDOW_INTERVALS + 1
                                    ],
                                    axis=0,
                                ),
                                axis=1,
                            ),
                        )
                    )
                )
                incoming = np.linalg.norm(
                    candidate.points[connection] - candidate.points[connection - 1]
                )
                outgoing = np.linalg.norm(
                    candidate.points[connection + 1] - candidate.points[connection]
                )
                self.assertGreater(incoming, 0.0)
                self.assertGreater(outgoing, 0.0)
                self.assertLessEqual(incoming / regular, 0.10)
                self.assertLessEqual(outgoing / regular, 0.10)

    def test_v3_local_kinematic_gate(self):
        for digit in (2, 3, 4, 5, 7):
            baseline = build_digit_trajectory(digit, self.baseline_config, 100)
            candidate = build_digit_trajectory(digit, self.ease_config, 100)
            for boundary_index in EXPECTED_QUALIFYING_BOUNDARIES[digit]:
                baseline_metrics = _kinematic_metrics(
                    _local_corner_points(
                        baseline, boundary_index, BASE_WINDOW_INTERVALS
                    ),
                    self.baseline_config.dt_seconds,
                )
                candidate_metrics = _kinematic_metrics(
                    _local_corner_points(
                        candidate, boundary_index, RESAMPLED_WINDOW_INTERVALS
                    ),
                    self.ease_config.dt_seconds,
                )
                with self.subTest(digit=digit, boundary=boundary_index):
                    self.assertLessEqual(
                        candidate_metrics["p95_acceleration_m_s2"],
                        baseline_metrics["p95_acceleration_m_s2"],
                    )
                    self.assertLessEqual(
                        candidate_metrics["p95_jerk_m_s3"],
                        baseline_metrics["p95_jerk_m_s3"],
                    )

    def test_frozen_checkpoint_selection_rule(self):
        def row(update, mean, endpoint=0.02, path=1.0):
            return {
                "completed_updates": update,
                "normalized_mean_error": mean,
                "normalized_endpoint_error": endpoint,
                "path_length_ratio": path,
            }

        stable = select_validation_history(
            [row(0, 1.0), row(100, 0.04), row(200, 0.03), row(300, 0.02)]
        )
        self.assertEqual(stable["status"], "STABLE_PASS")
        self.assertEqual(stable["best_update"], 300)
        unstable = select_validation_history(
            [row(0, 1.0), row(100, 0.03), row(200, 0.2), row(300, 0.03)]
        )
        self.assertEqual(unstable["status"], "PASS_UNSTABLE")
        self.assertEqual(unstable["best_update"], 100)
        failed = select_validation_history([row(0, 1.0), row(100, 0.2)])
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["best_update"], 100)

    def test_server_runner_enforces_manual_scope_and_cpu_parallelism(self):
        script = (
            ROOT
            / "server"
            / "run_digit_writing_original_protocol3_ten_digit_corner_ease_v3_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("nproc >= 10", script)
        self.assertIn(
            'AVAILABLE_CPUS="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"',
            script,
        )
        self.assertIn("OMP_NUM_THREADS=1", script)
        self.assertIn("COMPLETED_CASES=10", script)
        self.assertIn("AUTOMATIC_EXTENSION_STARTED=0", script)
        self.assertIn("AUTOMATIC_SECOND_SEED_STARTED=0", script)
        self.assertIn("FORMAL_FULL10_STARTED=0", script)
        self.assertNotIn("10000", script)


if __name__ == "__main__":
    unittest.main()
