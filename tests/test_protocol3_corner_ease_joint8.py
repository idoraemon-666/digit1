import json
import unittest
from pathlib import Path

from digit_writing.protocol3_corner_ease_joint8 import (
    JOINT8_DIGITS,
    JOINT8_FINAL_LR,
    JOINT8_INITIAL_LR,
    JOINT8_LR_SWITCH_UPDATE,
    JOINT8_TOTAL_UPDATES,
    digit_metrics_pass,
    digit_relative_threshold_violation,
    joint8_block_order,
    joint8_digit_for_update,
    joint8_learning_rate,
    joint8_schedule_sha256,
    select_joint8_validation_history,
    validate_config,
    validate_joint8_condition_counts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configurations"
    / "digit_writing_original_protocol3_corner_ease_joint8_excluding_digit0_digit8.json"
)


def digit_metrics(mean=0.04, endpoint=0.02, path=1.0):
    return {
        "normalized_mean_error": mean,
        "normalized_endpoint_error": endpoint,
        "path_length_ratio": path,
    }


def validation_row(update, failing=(), mean=0.04):
    per_digit = {
        str(digit): digit_metrics(
            mean=0.09 if digit in failing else mean,
        )
        for digit in JOINT8_DIGITS
    }
    return {
        "completed_updates": update,
        "normalized_mean_error": sum(
            row["normalized_mean_error"] for row in per_digit.values()
        )
        / len(per_digit),
        "per_digit": per_digit,
    }


class Joint8ScheduleTests(unittest.TestCase):
    def test_each_block_is_an_exact_permutation(self):
        for block in (0, 1, 2, 5_999, 7_999):
            order = joint8_block_order(block, 42)
            self.assertEqual(len(order), 8)
            self.assertEqual(set(order), set(JOINT8_DIGITS))
            self.assertEqual(
                order,
                tuple(
                    joint8_digit_for_update(block * 8 + offset, 42)
                    for offset in range(8)
                ),
            )

    def test_full_schedule_has_exact_counts_and_is_seeded(self):
        counts = {digit: 0 for digit in JOINT8_DIGITS}
        for update in range(JOINT8_TOTAL_UPDATES):
            counts[joint8_digit_for_update(update, 42)] += 1
        self.assertEqual(counts, {digit: 8_000 for digit in JOINT8_DIGITS})
        self.assertEqual(joint8_schedule_sha256(42), joint8_schedule_sha256(42))
        self.assertNotEqual(joint8_schedule_sha256(42), joint8_schedule_sha256(43))

    def test_learning_rate_boundary_is_unambiguous(self):
        self.assertEqual(joint8_learning_rate(0), JOINT8_INITIAL_LR)
        self.assertEqual(
            joint8_learning_rate(JOINT8_LR_SWITCH_UPDATE - 1),
            JOINT8_INITIAL_LR,
        )
        self.assertEqual(
            joint8_learning_rate(JOINT8_LR_SWITCH_UPDATE),
            JOINT8_FINAL_LR,
        )
        self.assertEqual(
            joint8_learning_rate(JOINT8_TOTAL_UPDATES - 1),
            JOINT8_FINAL_LR,
        )
        with self.assertRaises(ValueError):
            joint8_learning_rate(JOINT8_TOTAL_UPDATES)

    def test_condition_count_validator_excludes_zero_and_eight(self):
        completed = 800
        digit_counts = [0, 100, 100, 100, 100, 100, 100, 100, 0, 100]
        counts = {
            "digit_update_counts": digit_counts,
            "delay_update_counts": {"25": 0, "50": completed, "75": 0},
            "digit_delay_update_counts": [
                [0, value, 0] for value in digit_counts
            ],
        }
        validate_joint8_condition_counts(counts, completed)
        counts["digit_update_counts"][0] = 1
        with self.assertRaises(ValueError):
            validate_joint8_condition_counts(counts, completed)


class Joint8SelectionTests(unittest.TestCase):
    def test_digit_thresholds_and_relative_violation(self):
        self.assertTrue(digit_metrics_pass(digit_metrics(0.08, 0.05, 0.85)))
        self.assertFalse(digit_metrics_pass(digit_metrics(mean=0.09)))
        self.assertAlmostEqual(
            digit_relative_threshold_violation(digit_metrics(mean=0.12)),
            0.5,
        )

    def test_stable_selection_uses_only_stable_interval(self):
        history = [
            validation_row(0, failing=(1,), mean=0.01),
            validation_row(800, mean=0.04),
            validation_row(1600, mean=0.03),
            validation_row(2400, mean=0.05),
        ]
        selected = select_joint8_validation_history(history)
        self.assertEqual(selected["status"], "STABLE_PASS")
        self.assertEqual(selected["best_update"], 1600)
        self.assertEqual(
            selected["stable_intervals"],
            [{"start_update": 800, "end_update": 2400}],
        )

    def test_unstable_and_fail_fallbacks(self):
        unstable = select_joint8_validation_history(
            [validation_row(0, failing=(1,)), validation_row(800)]
        )
        self.assertEqual(unstable["status"], "PASS_UNSTABLE")
        self.assertEqual(unstable["best_update"], 800)

        failed = select_joint8_validation_history(
            [
                validation_row(0, failing=(1, 2)),
                validation_row(800, failing=(1,)),
            ]
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["best_update"], 800)

    def test_exact_tie_prefers_earlier_update(self):
        selected = select_joint8_validation_history(
            [validation_row(0, failing=(1,)), validation_row(800, failing=(1,))]
        )
        self.assertEqual(selected["best_update"], 0)


class Joint8ConfigurationTests(unittest.TestCase):
    def test_checked_in_config_is_frozen_and_geometry_matches(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_config(ROOT, config)
        self.assertEqual(config["train_digits"], list(JOINT8_DIGITS))
        self.assertEqual(config["excluded_digits"], [0, 8])
        self.assertEqual(config["training"]["max_updates"], 64_000)

    def test_train_core_applies_schedule_before_optimizer_step(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        loop_start = source.index("for batch in range(completed_updates, stop_after_updates):")
        loop_end = source.index("def train_subsets_held_out_base_model", loop_start)
        loop_source = source[loop_start:loop_end]
        lr_assignment = loop_source.index(
            "expected_learning_rate = joint8_learning_rate(completed_updates)"
        )
        optimizer_step = loop_source.index("optimizer.step()")
        self.assertLess(lr_assignment, optimizer_step)
        self.assertIn(
            "_protocol3_training_condition(\n                env_list, hp, completed_updates=completed_updates",
            loop_source,
        )
        self.assertNotIn("optimizer = torch.optim.Adam", loop_source)

    def test_server_runner_is_authorized_single_process_and_test_gated(self):
        runner = (
            ROOT
            / "server"
            / "run_digit_writing_original_protocol3_corner_ease_joint8.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("PROTOCOL3_CORNER_EASE_JOINT8_AUTHORIZED", runner)
        self.assertIn("tests.test_protocol3_corner_ease_joint8", runner)
        self.assertEqual(
            runner.count("-m digit_writing.protocol3_corner_ease_joint8 run"),
            1,
        )
        self.assertIn("OMP_NUM_THREADS=1", runner)


if __name__ == "__main__":
    unittest.main()
