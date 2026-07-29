import random
import unittest

import numpy as np
import torch

from digit_writing.protocol3_checkpoint import (
    capture_rng_state,
    protocol_config_sha256,
    restore_rng_state,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)
from digit_writing.protocol3_schedule import (
    new_condition_counts,
    record_condition,
    sample_protocol3_update,
)


class Protocol3CheckpointTests(unittest.TestCase):
    def test_rng_restore_reproduces_uninterrupted_condition_sequence(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        prefix = [sample_protocol3_update(random) for _ in range(17)]
        saved = capture_rng_state()
        expected_suffix = [
            (
                sample_protocol3_update(random),
                float(np.random.random()),
                float(torch.rand(())),
            )
            for _ in range(23)
        ]

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restore_rng_state(saved)
        resumed_suffix = [
            (
                sample_protocol3_update(random),
                float(np.random.random()),
                float(torch.rand(())),
            )
            for _ in range(23)
        ]

        self.assertEqual(len(prefix), 17)
        self.assertEqual(resumed_suffix, expected_suffix)

    def test_state_hash_is_stable_and_detects_changes(self):
        state = {"b": torch.tensor([3.0]), "a": torch.tensor([[1.0, 2.0]])}
        reordered = {"a": state["a"].clone(), "b": state["b"].clone()}
        self.assertEqual(state_dict_sha256(state), state_dict_sha256(reordered))
        changed = {"a": state["a"].clone(), "b": torch.tensor([4.0])}
        self.assertNotEqual(state_dict_sha256(state), state_dict_sha256(changed))

    def test_resume_validator_requires_complete_identity_and_state(self):
        config = {"protocol": "digit_writing_original_protocol3", "seed": 42}
        counts = new_condition_counts()
        for index in range(5000):
            record_condition(counts, index % 10, index % 3)
        checkpoint = {
            "agent_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {"state": {0: {}}, "param_groups": [{}]},
            "protocol_config": config,
            "protocol_config_sha256": protocol_config_sha256(config),
            "git_identity": {
                "repository_head": "abc",
                "branch": "codex/digit-writing-original-protocol3",
                "mrnntorch_recorded_head": "def",
                "mrnntorch_worktree_head": "def",
            },
            "rng_state": capture_rng_state(),
            "training_state": {
                "completed_updates": 5000,
                "next_update": 5000,
                "condition_counts": counts,
                "validation_history": [],
                "best_validation_loss": 1.0,
                "best_checkpoint_update": 5000,
                "last_validation_loss": 1.0,
            },
        }
        state = validate_protocol3_resume_checkpoint(checkpoint, config)
        self.assertEqual(state["next_update"], 5000)
        del checkpoint["rng_state"]["numpy"]
        with self.assertRaisesRegex(ValueError, "RNG state"):
            validate_protocol3_resume_checkpoint(checkpoint, config)


if __name__ == "__main__":
    unittest.main()
