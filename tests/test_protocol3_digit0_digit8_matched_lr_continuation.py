import copy
import json
import tempfile
import unittest
from pathlib import Path

from digit_writing.protocol3_corner_ease_lr_continuation import (
    ARM_LABELS,
    ARM_LEARNING_RATES,
    CASE_DIGITS,
    JOINT_CASE_DIGITS,
    JOINT_RUN_KIND,
    JOINT_VARIANT,
    load_continuation_config,
)


class Protocol3Digit0Digit8MatchedContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = (
            cls.root
            / "configurations"
            / "digit_writing_original_protocol3_digit0_digit8_matched_lr_continuation_6000_to8000.json"
        )
        cls.config = load_continuation_config(cls.config_path)

    def test_four_independent_arms_are_frozen_and_synchronized(self):
        self.assertEqual(self.config["run_kind"], JOINT_RUN_KIND)
        self.assertEqual(self.config["variant"], JOINT_VARIANT)
        self.assertEqual(
            tuple(case["digit"] for case in self.config["source"]["cases"]),
            JOINT_CASE_DIGITS,
        )
        self.assertEqual(JOINT_CASE_DIGITS, (0, 8))
        self.assertEqual(CASE_DIGITS, (0, 3, 4, 5, 6, 7))
        self.assertEqual(
            tuple(arm["label"] for arm in self.config["arms"]), ARM_LABELS
        )
        self.assertEqual(
            tuple(arm["learning_rate"] for arm in self.config["arms"]),
            ARM_LEARNING_RATES,
        )
        self.assertEqual(
            self.config["training"]["synchronized_parallel_arms"], 4
        )
        self.assertTrue(
            self.config["decision"]["synchronized_parallel_training"]
        )
        self.assertTrue(self.config["decision"]["digit8_included"])
        self.assertEqual(
            {
                case["digit"]: case["source_status"]
                for case in self.config["source"]["cases"]
            },
            {0: "PASS_UNSTABLE", 8: "FAIL"},
        )

    def test_source_checkpoints_and_summaries_are_exactly_frozen(self):
        self.assertEqual(
            self.config["source"]["repository_head"],
            "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925",
        )
        expected = {
            0: (
                "629e353d161281265e40d221cf5bcd30d2906db05b459f2a7be2b62efe81cff7",
                "fd4c42c7a5ee546ccdc65615df2648379fdc1b36f36dbe0d918de5ed476ae908",
                "0df65f909782fab0fcd8a1fbcda90511720a451ce9112a18aebbd63070efc65a",
            ),
            8: (
                "f70e0ba8e8585414f499f8ae2cc54e9061923983ea444ca62117e5faa5822a39",
                "2812c1c336c14d8cb58bb02638592499d0bc018fb6628287d73e10a5dfa445ce",
                "15df42d78db01d52a802f7dcf10c6a8661c342d04c6ade5e30e32fe61817a5b1",
            ),
        }
        for case in self.config["source"]["cases"]:
            self.assertEqual(
                (
                    case["best_checkpoint_sha256"],
                    case["final_checkpoint_sha256"],
                    case["run_summary_sha256"],
                ),
                expected[case["digit"]],
            )

    def test_training_and_decision_boundaries_are_exact(self):
        self.assertEqual(
            self.config["training"],
            {
                "target_completed_updates": 8000,
                "additional_updates": 2000,
                "validation_interval": 100,
                "batch_size": 8,
                "fixed_target_no_early_stop": True,
                "synchronized_parallel_arms": 4,
            },
        )
        self.assertEqual(
            self.config["decision"],
            {
                "automatic_winner_selection": False,
                "automatic_further_continuation": False,
                "automatic_second_seed": False,
                "digit8_included": True,
                "formal_full10_start": False,
                "qualitative_overlay_review_required": True,
                "automatic_geometry_change": False,
                "automatic_loss_change": False,
                "synchronized_parallel_training": True,
            },
        )

    def test_joint_design_mutations_are_rejected(self):
        mutations = []
        changed = copy.deepcopy(self.config)
        changed["source"]["repository_head"] = "0" * 40
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["source"]["cases"].reverse()
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["source"]["cases"][1]["source_status"] = "PASS_UNSTABLE"
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["arms"][1]["learning_rate"] = 0.0001
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["training"]["synchronized_parallel_arms"] = 2
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["decision"]["automatic_winner_selection"] = True
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["decision"]["automatic_loss_change"] = True
        mutations.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in mutations:
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_continuation_config(path)

    def test_runner_is_four_way_and_requires_separate_authorization(self):
        script = (
            self.root
            / "server"
            / "run_digit_writing_original_protocol3_digit0_digit8_matched_lr_continuation_parallel.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PROTOCOL3_DIGIT0_DIGIT8_MATCHED_LR_CONTINUATION_AUTHORIZED",
            script,
        )
        self.assertIn("AVAILABLE_CPUS < 4", script)
        self.assertIn("CPU_QUOTA / CPU_PERIOD < 4", script)
        self.assertIn("CASE_LABELS=(digit0_corner_ease digit8_corner_ease)", script)
        self.assertIn("COMPLETED_ARMS=4", script)
        self.assertIn("SYNCHRONIZED_PARALLEL_TRAINING=1", script)
        self.assertIn("DIGIT0_STARTED=1", script)
        self.assertIn("DIGIT8_STARTED=1", script)
        self.assertIn("matched continuation prepare failed", script)
        self.assertIn("matched continuation summarize failed", script)
        self.assertNotIn("digit3_corner_ease", script)
        self.assertNotIn("lr1e4", script)
        self.assertNotIn("protocol3_digit8_lr_ablation run", script)


if __name__ == "__main__":
    unittest.main()
