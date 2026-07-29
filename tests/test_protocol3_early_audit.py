import unittest

from digit_writing.protocol3_early_audit import learning_progress_summary


class Protocol3EarlyAuditTests(unittest.TestCase):
    def test_confirmed_learning_progress_rule_passes(self):
        before = [1.0] * 10
        after = [0.8] * 8 + [1.0, 1.1]
        summary = learning_progress_summary(before, after)
        self.assertTrue(summary["conditions_met"])
        self.assertEqual(summary["improved_digit_count"], 8)
        self.assertAlmostEqual(summary["worst_digit_relative_change"], 0.1)

    def test_rule_reports_but_does_not_define_automatic_failure(self):
        before = [1.0] * 10
        after = [0.8] * 7 + [1.0, 1.0, 1.16]
        summary = learning_progress_summary(before, after)
        self.assertFalse(summary["conditions_met"])
        self.assertEqual(summary["improved_digit_count"], 7)
        self.assertGreater(summary["worst_digit_relative_change"], 0.15)


if __name__ == "__main__":
    unittest.main()
