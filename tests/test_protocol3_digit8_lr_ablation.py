import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from digit_writing.protocol3_digit8_lr_ablation import (
    ARM_LABELS,
    ARM_LEARNING_RATES,
    load_ablation_config,
    movement_point_error_summary,
)
from digit_writing.phase_normalized_loss import position_l1_metrics


class Protocol3Digit8LearningRateAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.root
            / "configurations"
            / "digit_writing_original_protocol3_digit8_equal_point_lr_ablation.json"
        )
        cls.config = load_ablation_config(cls.config_path)

    def test_frozen_three_arm_equal_point_design(self):
        self.assertEqual(
            tuple(arm["label"] for arm in self.config["arms"]), ARM_LABELS
        )
        self.assertEqual(
            tuple(arm["learning_rate"] for arm in self.config["arms"]),
            ARM_LEARNING_RATES,
        )
        objective = self.config["objective"]
        self.assertTrue(objective["movement_points_equal_weight"])
        self.assertEqual(objective["start_point_extra_weight"], 0.0)
        self.assertEqual(objective["endpoint_extra_weight"], 0.0)
        self.assertEqual(objective["closure_extra_weight"], 0.0)
        self.assertEqual(self.config["source"]["completed_updates"], 6000)
        self.assertEqual(
            self.config["training"]["target_completed_updates"], 7000
        )
        self.assertEqual(self.config["training"]["additional_updates"], 1000)
        self.assertTrue(self.config["training"]["fixed_target_no_early_stop"])
        self.assertFalse(
            self.config["decision"]["automatic_further_continuation"]
        )
        self.assertFalse(
            self.config["decision"]["automatic_reference_fallback"]
        )
        self.assertFalse(self.config["decision"]["formal_full10_start"])

    def test_source_gate2_loss_and_case_remain_frozen(self):
        source_path = self.root / self.config["source"]["protocol_config"]
        with source_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        self.assertEqual(source["optimizer"]["learning_rate"], 0.001)
        self.assertEqual(
            source["position_loss"],
            {
                "type": "phase_normalized_l1",
                "stable_weight": 0.1,
                "delay_weight": 0.1,
                "movement_weight": 0.6,
                "hold_weight": 0.2,
            },
        )
        self.assertEqual(
            [case for case in source["cases"] if case["label"] == "o3_digit8"],
            [
                {
                    "label": "o3_digit8",
                    "digit": 8,
                    "direction_index": 0,
                    "delay_steps": 50,
                }
            ],
        )

    def test_design_mutations_are_rejected(self):
        mutations = []
        changed = copy.deepcopy(self.config)
        changed["objective"]["endpoint_extra_weight"] = 0.05
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["arms"][1]["learning_rate"] = 0.0004
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["frozen_case"]["movement_samples"] = 200
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["training"]["target_completed_updates"] = 9000
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["decision"]["automatic_further_continuation"] = True
        mutations.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in mutations:
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(value, handle)
                with self.assertRaises(ValueError):
                    load_ablation_config(path)

    def test_movement_point_summary_keeps_all_201_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "movement_overlay.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "movement_index",
                        "actual_x_m",
                        "actual_y_m",
                        "target_x_m",
                        "target_y_m",
                    )
                )
                for index in range(201):
                    writer.writerow((index, index / 1000.0, 0.0, 0.0, 0.0))
            summary = movement_point_error_summary(path)
        self.assertEqual(summary["movement_samples"], 201)
        self.assertTrue(summary["all_movement_points_equal_weight_in_training"])
        self.assertAlmostEqual(summary["mean_euclidean_error_m"], 0.1)
        self.assertAlmostEqual(summary["median_euclidean_error_m"], 0.1)
        self.assertAlmostEqual(summary["p95_euclidean_error_m"], 0.19)
        self.assertAlmostEqual(summary["maximum_euclidean_error_m"], 0.2)
        self.assertEqual(summary["maximum_error_index"], 200)
        self.assertAlmostEqual(summary["first_10_mean_error_m"], 0.0045)
        self.assertAlmostEqual(summary["middle_181_mean_error_m"], 0.1)
        self.assertAlmostEqual(summary["last_10_mean_error_m"], 0.1955)

    def test_all_201_movement_points_have_identical_loss_gradient_weight(self):
        prediction = torch.zeros((1, 301, 2), requires_grad=True)
        target = torch.ones_like(prediction)
        bounds = {
            "stable": (0, 25),
            "delay": (25, 75),
            "movement": (75, 276),
            "hold": (276, 301),
        }
        loss = position_l1_metrics(prediction, target, bounds)[
            "phase_normalized_position_l1"
        ]
        loss.backward()
        movement_gradient = prediction.grad[0, 75:276, 0].abs()
        self.assertEqual(movement_gradient.numel(), 201)
        self.assertTrue(
            torch.equal(
                movement_gradient,
                movement_gradient[0].expand_as(movement_gradient),
            )
        )
        self.assertAlmostEqual(
            float(movement_gradient[0]),
            0.6 / 201,
            places=9,
        )


@unittest.skipUnless(
    importlib.util.find_spec("motornet")
    and importlib.util.find_spec("configargparse"),
    "MotorNet integration dependencies are server-only",
)
class Protocol3Digit8LearningRateResumeIntegrationTests(unittest.TestCase):
    def test_adam_restore_then_override_changes_only_learning_rate(self):
        from train import _apply_digit8_lr_ablation_optimizer_override

        source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        source_optimizer = torch.optim.Adam([source_parameter], lr=0.001)
        source_parameter.grad = torch.tensor([0.5])
        source_optimizer.step()
        target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        target_optimizer = torch.optim.Adam([target_parameter], lr=0.0003)
        target_optimizer.load_state_dict(source_optimizer.state_dict())
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.001)
        before = copy.deepcopy(target_optimizer.state_dict())
        _apply_digit8_lr_ablation_optimizer_override(
            target_optimizer,
            {"hp": {"lr": 0.001}},
            {
                "experimental_resume_learning_rate": 0.0003,
                "experimental_source_learning_rate": 0.001,
            },
        )
        after = target_optimizer.state_dict()
        self.assertEqual(after["param_groups"][0]["lr"], 0.0003)
        before_group = dict(before["param_groups"][0])
        after_group = dict(after["param_groups"][0])
        before_group.pop("lr")
        after_group.pop("lr")
        self.assertEqual(before_group, after_group)
        self.assertEqual(before["state"].keys(), after["state"].keys())
        for key in before["state"]:
            self.assertEqual(
                before["state"][key].keys(), after["state"][key].keys()
            )
            for name in before["state"][key]:
                self.assertTrue(
                    torch.equal(
                        before["state"][key][name],
                        after["state"][key][name],
                    )
                )


if __name__ == "__main__":
    unittest.main()
