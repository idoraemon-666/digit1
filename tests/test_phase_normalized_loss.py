import unittest

import torch

from digit_writing.phase_normalized_loss import (
    PHASE_WEIGHTS,
    phase_normalized_l1,
    position_l1_metrics,
)


def _trial(stable, delay, movement, hold):
    values = [1.0] * stable + [2.0] * delay + [3.0] * movement + [4.0] * hold
    prediction = torch.zeros((1, len(values), 2), dtype=torch.float32)
    prediction[0, :, 0] = torch.tensor(values)
    target = torch.zeros_like(prediction)
    bounds = {
        "stable": (0, stable),
        "delay": (stable, stable + delay),
        "movement": (stable + delay, stable + delay + movement),
        "hold": (stable + delay + movement, len(values)),
    }
    return prediction, target, bounds


class PhaseNormalizedLossTests(unittest.TestCase):
    def test_fixed_phase_weights_and_internal_means(self):
        prediction, target, bounds = _trial(2, 3, 4, 2)
        metrics = position_l1_metrics(prediction, target, bounds)
        self.assertEqual(
            PHASE_WEIGHTS,
            {"stable": 0.1, "delay": 0.1, "movement": 0.6, "hold": 0.2},
        )
        self.assertAlmostEqual(float(metrics["stable_mean_l1"]), 1.0)
        self.assertAlmostEqual(float(metrics["delay_mean_l1"]), 2.0)
        self.assertAlmostEqual(float(metrics["movement_mean_l1"]), 3.0)
        self.assertAlmostEqual(float(metrics["hold_mean_l1"]), 4.0)
        self.assertAlmostEqual(
            float(metrics["phase_normalized_position_l1"]), 2.9, places=6
        )
        self.assertAlmostEqual(float(metrics["full_trial_position_l1"]), 28.0 / 11.0)
        self.assertAlmostEqual(float(metrics["endpoint_error"]), 3.0)
        self.assertAlmostEqual(float(metrics["hold_drift"]), 1.0)

    def test_variable_movement_length_does_not_change_its_total_weight(self):
        short = _trial(2, 3, 2, 2)
        long = _trial(2, 3, 20, 2)
        short_loss = phase_normalized_l1(*short)
        long_loss = phase_normalized_l1(*long)
        self.assertAlmostEqual(float(short_loss), 2.9, places=6)
        self.assertAlmostEqual(float(long_loss), 2.9, places=6)

    def test_phase_bounds_must_be_mutually_exclusive_and_exhaustive(self):
        prediction, target, bounds = _trial(2, 3, 4, 2)
        bounds["movement"] = (4, 9)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            phase_normalized_l1(prediction, target, bounds)

        prediction, target, bounds = _trial(2, 3, 4, 2)
        bounds["hold"] = (9, 10)
        with self.assertRaisesRegex(ValueError, "cover"):
            phase_normalized_l1(prediction, target, bounds)

    def test_movement_first_and_last_samples_are_in_the_movement_slice(self):
        prediction, target, bounds = _trial(2, 3, 4, 2)
        movement_start, movement_end = bounds["movement"]
        prediction[0, movement_start, 0] = 30.0
        prediction[0, movement_end - 1, 0] = 30.0
        metrics = position_l1_metrics(prediction, target, bounds)
        self.assertGreater(float(metrics["movement_mean_l1"]), 3.0)
        self.assertEqual(float(metrics["endpoint_error"]), 30.0)


if __name__ == "__main__":
    unittest.main()
