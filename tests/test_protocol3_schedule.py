import random
import unittest

import numpy as np

from digit_writing.protocol3_schedule import (
    balanced_direction_indices,
    new_condition_counts,
    record_condition,
    sample_independent_digit_delay,
    sample_protocol3_update,
    validate_condition_counts,
)


class Protocol3ScheduleTests(unittest.TestCase):
    def test_batch_has_four_copies_of_each_training_direction(self):
        directions = balanced_direction_indices(32)
        self.assertEqual(tuple(directions.shape), (32,))
        np.testing.assert_array_equal(
            np.bincount(directions, minlength=8),
            np.full(8, 4),
        )

    def test_balanced_directions_require_a_multiple_of_eight(self):
        with self.assertRaises(ValueError):
            balanced_direction_indices(31)

    def test_digit_and_delay_sequence_is_independent_and_reproducible(self):
        first = random.Random(42)
        second = random.Random(42)
        sequence_a = [sample_independent_digit_delay(first) for _ in range(200)]
        sequence_b = [sample_independent_digit_delay(second) for _ in range(200)]
        self.assertEqual(sequence_a, sequence_b)
        self.assertTrue(all(0 <= digit <= 9 for digit, _ in sequence_a))
        self.assertTrue(all(0 <= delay <= 2 for _, delay in sequence_a))

    def test_update_sampling_includes_a_reproducible_environment_seed(self):
        first = random.Random(42)
        second = random.Random(42)
        sequence_a = [sample_protocol3_update(first) for _ in range(20)]
        sequence_b = [sample_protocol3_update(second) for _ in range(20)]
        self.assertEqual(sequence_a, sequence_b)
        self.assertTrue(all(0 <= seed < 2**32 for _, _, seed in sequence_a))

    def test_audit_counts_do_not_change_the_rng_sequence(self):
        counted_rng = random.Random(7)
        control_rng = random.Random(7)
        counts = new_condition_counts()
        counted = []
        for _ in range(100):
            condition = sample_independent_digit_delay(counted_rng)
            counted.append(condition)
            record_condition(counts, *condition)
        control = [sample_independent_digit_delay(control_rng) for _ in range(100)]
        self.assertEqual(counted, control)
        self.assertEqual(sum(counts["digit_update_counts"]), 100)
        self.assertEqual(sum(counts["delay_update_counts"].values()), 100)
        self.assertEqual(
            sum(sum(row) for row in counts["digit_delay_update_counts"]),
            100,
        )

    def test_condition_count_totals_are_audited(self):
        counts = new_condition_counts()
        for digit, delay_index in ((0, 0), (9, 2), (0, 1)):
            record_condition(counts, digit, delay_index)
        validate_condition_counts(counts, 3)
        counts["digit_update_counts"][0] += 1
        with self.assertRaisesRegex(ValueError, "totals"):
            validate_condition_counts(counts, 3)


if __name__ == "__main__":
    unittest.main()
