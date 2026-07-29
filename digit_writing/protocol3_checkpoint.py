"""Checkpoint identity and RNG helpers for digit-writing Protocol3."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from digit_writing.protocol3_schedule import validate_condition_counts


REQUIRED_RNG_KEYS = ("python", "numpy", "torch")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_RNG_KEYS if key not in state]
    if missing:
        raise ValueError(f"checkpoint RNG state is missing: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def protocol_config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def current_git_identity(repository_root: str | Path) -> dict[str, str]:
    root = Path(repository_root).resolve()

    def git(*arguments: str, cwd: Path = root) -> str:
        result = subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    recorded_submodule = git("rev-parse", "HEAD:mRNNTorch")
    worktree_submodule = git("rev-parse", "HEAD", cwd=root / "mRNNTorch")
    if recorded_submodule != worktree_submodule:
        raise RuntimeError("mRNNTorch worktree HEAD differs from the recorded gitlink")
    return {
        "repository_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "mrnntorch_recorded_head": recorded_submodule,
        "mrnntorch_worktree_head": worktree_submodule,
    }


def validate_protocol3_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    expected_protocol_config: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "agent_state_dict",
        "optimizer_state_dict",
        "protocol_config",
        "protocol_config_sha256",
        "git_identity",
        "rng_state",
        "training_state",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"continuation checkpoint is missing: {missing}")
    if checkpoint["protocol_config"] != expected_protocol_config:
        raise ValueError("continuation checkpoint protocol config does not match")
    expected_hash = protocol_config_sha256(expected_protocol_config)
    if checkpoint["protocol_config_sha256"] != expected_hash:
        raise ValueError("continuation checkpoint protocol config hash does not match")
    restore_state = checkpoint["training_state"]
    if not checkpoint["agent_state_dict"]:
        raise ValueError("continuation checkpoint model state is empty")
    optimizer_state = checkpoint["optimizer_state_dict"]
    if not isinstance(optimizer_state, Mapping) or not {
        "state",
        "param_groups",
    }.issubset(optimizer_state):
        raise ValueError("continuation checkpoint optimizer state is incomplete")
    git_identity = checkpoint["git_identity"]
    if not isinstance(git_identity, Mapping) or not {
        "repository_head",
        "branch",
        "mrnntorch_recorded_head",
        "mrnntorch_worktree_head",
    }.issubset(git_identity):
        raise ValueError("continuation checkpoint Git identity is incomplete")
    for key in (
        "completed_updates",
        "next_update",
        "condition_counts",
        "validation_history",
        "best_validation_loss",
        "best_checkpoint_update",
        "last_validation_loss",
    ):
        if key not in restore_state:
            raise ValueError(f"continuation training state is missing: {key}")
    completed = int(restore_state["completed_updates"])
    if int(restore_state["next_update"]) != completed:
        raise ValueError("continuation next_update must equal completed_updates")
    missing_rng = [
        key for key in REQUIRED_RNG_KEYS if key not in checkpoint["rng_state"]
    ]
    if missing_rng:
        raise ValueError(f"continuation checkpoint RNG state is missing: {missing_rng}")
    validate_condition_counts(restore_state["condition_counts"], completed)
    return dict(restore_state)
