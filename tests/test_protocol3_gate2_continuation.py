import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from digit_writing.protocol3_checkpoint import (
    capture_rng_state,
    protocol_config_sha256,
    state_dict_sha256,
)
from digit_writing.protocol3_gate2_continuation import (
    review_gate2_continuation,
    validate_gate2_continuation_target,
)
from digit_writing.protocol3_schedule import new_condition_counts, record_condition


class Protocol3Gate2ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "protocol": "digit_writing_original_protocol3",
            "run_kind": "protocol3_gate2",
            "training": {
                "batch_size": 8,
                "max_updates": 3000,
                "validation_interval": 100,
            },
            "cases": [
                {
                    "label": "o1_digit1",
                    "digit": 1,
                    "direction_index": 0,
                    "delay_steps": 50,
                }
            ],
        }
        self.source_identity = {
            "repository_head": "a" * 40,
            "branch": "codex/digit-writing-original-protocol3",
            "mrnntorch_recorded_head": "b" * 40,
            "mrnntorch_worktree_head": "b" * 40,
        }
        self.current_identity = {
            **self.source_identity,
            "repository_head": "c" * 40,
        }
        counts = new_condition_counts()
        for _ in range(3000):
            record_condition(counts, 1, 1)
        history = []
        for update in range(0, 3001, 100):
            fraction = update / 3000
            history.append(
                {
                    "completed_updates": update,
                    "phase_normalized_position_l1": 1.0 - 0.5 * fraction,
                    "normalized_mean_error": 1.0 - 0.4 * fraction,
                    "normalized_endpoint_error": 1.0 - 0.3 * fraction,
                    "path_length_ratio": 0.2 + 0.4 * fraction,
                }
            )
        self.checkpoint = {
            "agent_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {"state": {0: {}}, "param_groups": [{}]},
            "hp": {
                "condition_schedule": "protocol3_gate2_single_condition",
                "gate2_direction_index": 0,
                "gate2_delay_index": 1,
                "batch_size": 8,
                "save_iter": 100,
                "git_identity": self.source_identity,
            },
            "protocol_config": self.config,
            "protocol_config_sha256": protocol_config_sha256(self.config),
            "variant": "o1_digit1",
            "git_identity": self.source_identity,
            "rng_state": capture_rng_state(),
            "training_state": {
                "completed_updates": 3000,
                "next_update": 3000,
                "condition_counts": counts,
                "validation_history": history,
                "best_validation_loss": 0.5,
                "best_checkpoint_update": 3000,
                "last_validation_loss": 0.5,
                "gate2_consecutive_passes": 0,
            },
        }
        self.summary = {
            "protocol": "digit_writing_original_protocol3",
            "run_kind": "protocol3_gate2",
            "git_identity": self.source_identity,
            "cases": [
                {
                    **self.config["cases"][0],
                    "updates": 3000,
                    "engineering_passed": True,
                    "behavior_passed": False,
                    "classification": "behavior_failure",
                }
            ],
        }

    def review(self, *, summary=None, checkpoint=None):
        return review_gate2_continuation(
            self.config,
            self.summary if summary is None else summary,
            self.checkpoint if checkpoint is None else checkpoint,
            checkpoint_sha256="d" * 64,
            case_label="o1_digit1",
            expected_source_repository_head="a" * 40,
            current_identity=self.current_identity,
        )

    def test_behavior_failure_is_eligible_for_manual_continuation(self):
        review = self.review()
        self.assertTrue(review["eligible_for_manual_continuation"])
        self.assertFalse(review["automatic_continuation_allowed"])
        self.assertEqual(review["source_completed_updates"], 3000)
        self.assertEqual(review["validation_points"], 31)
        self.assertLess(
            review["metrics"]["normalized_mean_error"]["last_10_mean"],
            review["metrics"]["normalized_mean_error"]["previous_10_mean"],
        )
        self.assertEqual(validate_gate2_continuation_target(review, 4000), 4000)

    def test_target_must_advance_to_a_validation_boundary(self):
        review = self.review()
        with self.assertRaisesRegex(ValueError, "exceed"):
            validate_gate2_continuation_target(review, 3000)
        with self.assertRaisesRegex(ValueError, "validation boundary"):
            validate_gate2_continuation_target(review, 4050)

    def test_previous_continuation_can_be_the_next_source(self):
        continuation = {
            "protocol": "digit_writing_original_protocol3",
            "run_kind": "protocol3_gate2_continuation",
            "git_identity": self.source_identity,
            "case": self.config["cases"][0],
            "completed_updates": 3000,
            "engineering_passed": True,
            "behavior_passed": False,
            "classification": "behavior_failure",
        }
        review = self.review(summary=continuation)
        self.assertTrue(review["eligible_for_manual_continuation"])

    def test_engineering_failure_and_passed_case_are_rejected(self):
        engineering = copy.deepcopy(self.summary)
        engineering["cases"][0]["engineering_passed"] = False
        engineering["cases"][0]["classification"] = (
            "engineering_or_safety_failure"
        )
        with self.assertRaisesRegex(ValueError, "behavior-failure"):
            self.review(summary=engineering)

        passed = copy.deepcopy(self.summary)
        passed["cases"][0]["behavior_passed"] = True
        passed["cases"][0]["classification"] = (
            "provisional_pass_pending_qualitative_review"
        )
        with self.assertRaisesRegex(ValueError, "behavior-failure"):
            self.review(summary=passed)

    def test_source_identity_and_validation_history_are_strict(self):
        with self.assertRaisesRegex(ValueError, "repository HEAD"):
            review_gate2_continuation(
                self.config,
                self.summary,
                self.checkpoint,
                checkpoint_sha256="d" * 64,
                case_label="o1_digit1",
                expected_source_repository_head="e" * 40,
                current_identity=self.current_identity,
            )
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["training_state"]["validation_history"].pop()
        with self.assertRaisesRegex(ValueError, "history is incomplete"):
            self.review(checkpoint=checkpoint)


@unittest.skipUnless(
    importlib.util.find_spec("motornet")
    and importlib.util.find_spec("configargparse"),
    "MotorNet integration dependencies are server-only",
)
class Protocol3Gate2ContinuationIntegrationTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_value, right_value in zip(left, right):
                self.assert_nested_equal(left_value, right_value)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_updates(self):
        from train import (
            PROTOCOL3_DELAYS,
            _base_hp_from_config,
            _digit_env_dict,
            train_subsets_base_model,
        )

        root = Path(__file__).resolve().parents[1]
        with (root / "configurations" / "digit_writing_original_protocol3_gate2_scale2p50_ref50.json").open(
            "r", encoding="utf-8"
        ) as handle:
            config = json.load(handle)

        def hp(max_updates):
            value = _base_hp_from_config(config)
            value.update(
                {
                    "variant": "o1_digit1",
                    "condition_schedule": "protocol3_gate2_single_condition",
                    "gate2_direction_index": 0,
                    "gate2_delay_index": PROTOCOL3_DELAYS.index(50),
                    "batch_size": 1,
                    "hid_size": 16,
                    "epochs": max_updates,
                    "save_iter": 1,
                    "stop_after_updates": max_updates,
                    "initial_model_file": None,
                    "final_model_file": "final.pt",
                }
            )
            return value

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            uninterrupted_hp = hp(4)
            train_subsets_base_model(
                str(directory / "uninterrupted"),
                "best.pt",
                hp=uninterrupted_hp,
                env_dict=_digit_env_dict((1,)),
            )
            prefix_hp = hp(2)
            train_subsets_base_model(
                str(directory / "prefix"),
                "best.pt",
                hp=prefix_hp,
                env_dict=_digit_env_dict((1,)),
            )
            source_checkpoint = directory / "prefix" / "final.pt"
            source = torch.load(
                source_checkpoint, map_location="cpu", weights_only=False
            )
            source_head = "a" * 40
            source["git_identity"]["repository_head"] = source_head
            source["hp"]["git_identity"]["repository_head"] = source_head
            torch.save(source, source_checkpoint)
            resume_hp = hp(2)
            train_subsets_base_model(
                str(directory / "resumed"),
                "best.pt",
                hp=resume_hp,
                env_dict=_digit_env_dict((1,)),
                resume_checkpoint=str(source_checkpoint),
                target_completed_updates=4,
                manual_resume_authorized=True,
                expected_resume_repository_head=source_head,
            )
            uninterrupted = torch.load(
                directory / "uninterrupted" / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed = torch.load(
                directory / "resumed" / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(
            state_dict_sha256(uninterrupted["agent_state_dict"]),
            state_dict_sha256(resumed["agent_state_dict"]),
        )
        self.assert_nested_equal(
            uninterrupted["optimizer_state_dict"],
            resumed["optimizer_state_dict"],
        )
        self.assertEqual(
            uninterrupted["training_state"], resumed["training_state"]
        )
        self.assertTrue(
            torch.equal(
                uninterrupted["rng_state"]["torch"],
                resumed["rng_state"]["torch"],
            )
        )
        self.assertEqual(
            uninterrupted["rng_state"]["python"],
            resumed["rng_state"]["python"],
        )
        uninterrupted_numpy = uninterrupted["rng_state"]["numpy"]
        resumed_numpy = resumed["rng_state"]["numpy"]
        self.assertEqual(uninterrupted_numpy[0], resumed_numpy[0])
        np.testing.assert_array_equal(uninterrupted_numpy[1], resumed_numpy[1])
        self.assertEqual(uninterrupted_numpy[2:], resumed_numpy[2:])


if __name__ == "__main__":
    unittest.main()
