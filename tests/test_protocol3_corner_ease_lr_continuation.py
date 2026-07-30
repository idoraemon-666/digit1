import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from digit_writing.protocol3_corner_ease import (
    reconstruct_corner_ease_selection_state,
)
from digit_writing.protocol3_corner_ease_lr_continuation import (
    ARM_LABELS,
    ARM_LEARNING_RATES,
    CASE_DIGITS,
    load_continuation_config,
)


class Protocol3CornerEaseLearningRateContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.root
            / "configurations"
            / "digit_writing_original_protocol3_corner_ease_lr_continuation_6000_to8000.json"
        )
        cls.config = load_continuation_config(cls.config_path)

    def test_frozen_six_digit_two_arm_2000_update_design(self):
        self.assertEqual(
            tuple(case["digit"] for case in self.config["source"]["cases"]),
            CASE_DIGITS,
        )
        self.assertNotIn(8, CASE_DIGITS)
        self.assertEqual(
            tuple(arm["label"] for arm in self.config["arms"]), ARM_LABELS
        )
        self.assertEqual(
            tuple(arm["learning_rate"] for arm in self.config["arms"]),
            ARM_LEARNING_RATES,
        )
        self.assertEqual(self.config["source"]["completed_updates"], 6000)
        self.assertEqual(
            self.config["training"]["target_completed_updates"], 8000
        )
        self.assertEqual(self.config["training"]["additional_updates"], 2000)
        self.assertTrue(self.config["training"]["fixed_target_no_early_stop"])
        self.assertFalse(self.config["decision"]["automatic_winner_selection"])
        self.assertFalse(
            self.config["decision"]["automatic_further_continuation"]
        )
        self.assertFalse(self.config["decision"]["automatic_second_seed"])
        self.assertFalse(self.config["decision"]["digit8_included"])
        self.assertFalse(self.config["decision"]["formal_full10_start"])

    def test_source_v3_identity_and_artifact_hashes_are_frozen(self):
        self.assertEqual(
            self.config["source"]["repository_head"],
            "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925",
        )
        source_config = json.loads(
            (self.root / self.config["source"]["protocol_config"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(source_config["variant"], "ten_digit_corner_ease_v3_overfit6000")
        self.assertEqual(source_config["optimizer"]["learning_rate"], 0.001)
        for case in self.config["source"]["cases"]:
            self.assertEqual(case["source_status"], "PASS_UNSTABLE")
            self.assertEqual(len(case["best_checkpoint_sha256"]), 64)
            self.assertEqual(len(case["final_checkpoint_sha256"]), 64)
            self.assertEqual(len(case["run_summary_sha256"]), 64)

    def test_design_mutations_are_rejected(self):
        mutations = []
        changed = copy.deepcopy(self.config)
        changed["source"]["cases"].append(copy.deepcopy(changed["source"]["cases"][0]))
        changed["source"]["cases"][-1]["digit"] = 8
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["arms"][1]["learning_rate"] = 0.0004
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["training"]["target_completed_updates"] = 9000
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["decision"]["automatic_winner_selection"] = True
        mutations.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in mutations:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_continuation_config(path)

    def test_online_selector_state_is_reconstructed_across_resume_boundary(self):
        def row(update, mean, endpoint=0.02, path=1.0):
            return {
                "completed_updates": update,
                "normalized_mean_error": mean,
                "normalized_endpoint_error": endpoint,
                "path_length_ratio": path,
            }

        state = reconstruct_corner_ease_selection_state(
            [
                row(5700, 0.2),
                row(5800, 0.04),
                row(5900, 0.03),
                row(6000, 0.02),
            ]
        )
        self.assertEqual(state["pass_streak"], 3)
        self.assertEqual(state["current_interval_start"], 5800)
        self.assertEqual(state["current_streak_best"]["completed_updates"], 6000)
        self.assertEqual(state["stable_best"]["completed_updates"], 6000)
        self.assertEqual(
            state["stable_intervals"],
            [{"start_update": 5800, "end_update": 6000}],
        )

    def test_server_runner_requires_authorization_and_twelve_cpus(self):
        script = (
            self.root
            / "server"
            / "run_digit_writing_original_protocol3_corner_ease_lr_continuation_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("PROTOCOL3_CORNER_EASE_LR_CONTINUATION_AUTHORIZED", script)
        self.assertIn("AVAILABLE_CPUS < 12", script)
        self.assertIn("CPU_QUOTA / CPU_PERIOD < 12", script)
        self.assertIn("COMPLETED_ARMS=12", script)
        self.assertIn("COMPLETED_UPDATES=8000", script)
        self.assertIn("AUTOMATIC_WINNER_SELECTION_STARTED=0", script)
        self.assertIn("DIGIT8_STARTED=0", script)
        self.assertIn("FORMAL_FULL10_STARTED=0", script)


@unittest.skipUnless(
    importlib.util.find_spec("motornet")
    and importlib.util.find_spec("configargparse"),
    "MotorNet integration dependencies are server-only",
)
class Protocol3CornerEaseLearningRateOverrideIntegrationTests(unittest.TestCase):
    def test_adam_restore_then_override_changes_only_learning_rate(self):
        from train import _apply_protocol3_resume_learning_rate_override

        source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        source_optimizer = torch.optim.Adam([source_parameter], lr=0.001)
        source_parameter.grad = torch.tensor([0.5])
        source_optimizer.step()
        target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        target_optimizer = torch.optim.Adam([target_parameter], lr=0.0003)
        target_optimizer.load_state_dict(source_optimizer.state_dict())
        before = copy.deepcopy(target_optimizer.state_dict())
        _apply_protocol3_resume_learning_rate_override(
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
            for name in before["state"][key]:
                self.assertTrue(
                    torch.equal(
                        before["state"][key][name],
                        after["state"][key][name],
                    )
                )


if __name__ == "__main__":
    unittest.main()
