"""Read-only update-5000 audit for Protocol3 full10 continuation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from digit_writing.final_protocol_audit import _all_finite, _json_safe
from digit_writing.geometry import load_geometry_config
from digit_writing.geometry_audit import build_workspace_audit
from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)


DELAYS = (25, 50, 75)


def learning_progress_summary(
    update0_by_digit: Sequence[float],
    update5000_by_digit: Sequence[float],
) -> dict[str, Any]:
    if len(update0_by_digit) != 10 or len(update5000_by_digit) != 10:
        raise ValueError("learning-progress audit requires exactly 10 digits")
    if any(value <= 0.0 for value in update0_by_digit):
        raise ValueError("update-0 normalized errors must be positive")
    per_digit = []
    for digit, (before, after) in enumerate(
        zip(update0_by_digit, update5000_by_digit)
    ):
        per_digit.append(
            {
                "digit": digit,
                "update0_normalized_mean_error": float(before),
                "update5000_normalized_mean_error": float(after),
                "improved": bool(after < before),
                "relative_change": float((after - before) / before),
            }
        )
    overall_before = float(np.mean(update0_by_digit))
    overall_after = float(np.mean(update5000_by_digit))
    improved_digits = sum(row["improved"] for row in per_digit)
    worst_relative_change = max(row["relative_change"] for row in per_digit)
    return {
        "overall_update0_normalized_mean_error": overall_before,
        "overall_update5000_normalized_mean_error": overall_after,
        "improved_digit_count": improved_digits,
        "worst_digit_relative_change": worst_relative_change,
        "per_digit": per_digit,
        "conditions_met": bool(
            overall_after < overall_before
            and improved_digits >= 8
            and worst_relative_change <= 0.15
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_policy(checkpoint, build_policy):
    policy = build_policy(checkpoint["hp"], 6, torch.device("cpu"))
    policy.load_state_dict(checkpoint["agent_state_dict"])
    policy.eval()
    return policy


def _rollout(policy, hp, environment_class, digit, delay_index):
    import motornet as mn

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    environment = environment_class(
        effector=effector,
        **hp.get("env_kwargs", {}),
    )
    observation, info = environment.reset(
        testing=True,
        seed=int(hp["validation_seed"]) + digit * len(DELAYS) + delay_index,
        options={
            "batch_size": 32,
            "reach_conds": np.arange(32, dtype=np.int64),
            "speed_cond": 0,
            "delay_cond": delay_index,
            "deterministic": True,
        },
    )
    x = torch.zeros((32, hp["hid_size"]))
    h = torch.zeros_like(x)
    positions = []
    targets = []
    joint_positions = []
    finite = _all_finite(observation) and _all_finite(info)
    terminated = False
    timestep = 0
    while not terminated:
        with torch.no_grad():
            x, h, action = policy(observation, x, h, noise=False)
            observation, _, terminated, info = environment.step(timestep, action)
        positions.append(info["states"]["fingertip"][:, None, :])
        targets.append(info["goal"][:, None, :])
        joint_positions.append(
            info["states"]["joint"][:, : environment.effector.dof].unsqueeze(1)
        )
        finite &= _all_finite(observation) and _all_finite(info)
        timestep += 1
    positions = torch.cat(positions, dim=1)
    targets = torch.cat(targets, dim=1)
    joint_positions = torch.cat(joint_positions, dim=1)
    movement_start, movement_end = environment.epoch_bounds["movement"]
    actual = positions[:, movement_start:movement_end]
    target = targets[:, movement_start:movement_end]
    error = torch.linalg.vector_norm(actual - target, dim=-1)
    diagonal = torch.linalg.vector_norm(
        target.amax(dim=1) - target.amin(dim=1), dim=-1
    )
    actual_length = torch.linalg.vector_norm(
        torch.diff(actual, dim=1), dim=-1
    ).sum(dim=1)
    target_length = torch.linalg.vector_norm(
        torch.diff(target, dim=1), dim=-1
    ).sum(dim=1)
    if bool(torch.any(diagonal <= 0.0)) or bool(torch.any(target_length <= 0.0)):
        raise RuntimeError("early-audit normalization denominator is not positive")
    radius = torch.linalg.vector_norm(positions, dim=-1)
    inner_radius = abs(float(effector.skeleton.L1) - float(effector.skeleton.L2))
    outer_radius = float(effector.skeleton.L1) + float(effector.skeleton.L2)
    lower = torch.as_tensor(effector.pos_lower_bound, dtype=joint_positions.dtype)
    upper = torch.as_tensor(effector.pos_upper_bound, dtype=joint_positions.dtype)
    joint_margin = torch.minimum(joint_positions - lower, upper - joint_positions)
    metrics = {
        "normalized_mean_error": float((error.mean(dim=1) / diagonal).mean()),
        "normalized_endpoint_error": float((error[:, -1] / diagonal).mean()),
        "path_length_ratio": float((actual_length / target_length).mean()),
        "min_inner_radius_margin_m": float((radius - inner_radius).min()),
        "min_outer_radius_margin_m": float((outer_radius - radius).min()),
        "min_joint_margin_rad": float(joint_margin.min()),
        "all_values_finite": finite,
    }
    return metrics, actual.detach().cpu().numpy(), target.detach().cpu().numpy()


def _evaluate(policy, hp, env_classes):
    rows = []
    overlays = {}
    for digit, environment_class in enumerate(env_classes):
        for delay_index, delay in enumerate(DELAYS):
            metrics, actual, target = _rollout(
                policy, hp, environment_class, digit, delay_index
            )
            rows.append({"digit": digit, "delay_steps": delay, **metrics})
            if delay == 50:
                overlays[digit] = (actual, target)
    return rows, overlays


def _save_overlays(output, baseline_overlays, final_overlays):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for digit in range(10):
        baseline, target = baseline_overlays[digit]
        final, final_target = final_overlays[digit]
        if not np.array_equal(target, final_target):
            raise RuntimeError("update 0 and 5000 target grids differ")
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for direction in range(32):
            axes[0].plot(target[direction, :, 0], target[direction, :, 1], "k--", alpha=0.15)
            axes[0].plot(baseline[direction, :, 0], baseline[direction, :, 1], alpha=0.35)
            axes[1].plot(target[direction, :, 0], target[direction, :, 1], "k--", alpha=0.15)
            axes[1].plot(final[direction, :, 0], final[direction, :, 1], alpha=0.35)
        axes[0].set_title(f"digit {digit}: update 0")
        axes[1].set_title(f"digit {digit}: update 5000")
        for axis in axes:
            axis.set_aspect("equal", adjustable="box")
        figure.tight_layout()
        figure.savefig(output / f"digit{digit}_movement_overlay.png", dpi=180)
        plt.close(figure)


def run_early_audit(
    initial_checkpoint_path: str | Path,
    continuation_checkpoint_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    from train import DIGIT_ENV_CLASSES, _build_policy

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"early-audit output already exists: {output}")
    output.mkdir(parents=True)
    initial = torch.load(
        initial_checkpoint_path, map_location="cpu", weights_only=False
    )
    continuation = torch.load(
        continuation_checkpoint_path, map_location="cpu", weights_only=False
    )
    config = continuation.get("protocol_config")
    if config is None or initial.get("protocol_config") != config:
        raise ValueError("update 0 and 5000 protocol identities differ")
    initial_state = validate_protocol3_resume_checkpoint(initial, config)
    continuation_state = validate_protocol3_resume_checkpoint(continuation, config)
    if int(initial_state["completed_updates"]) != 0:
        raise ValueError("initial checkpoint is not update 0")
    if int(continuation_state["completed_updates"]) != 5000:
        raise ValueError("continuation checkpoint is not update 5000")
    repository_root = Path(__file__).resolve().parents[1]
    git_identity = current_git_identity(repository_root)
    if initial["git_identity"] != git_identity or continuation["git_identity"] != git_identity:
        raise ValueError("checkpoint and audit Git identities differ")

    initial_policy = _load_policy(initial, _build_policy)
    continuation_policy = _load_policy(continuation, _build_policy)
    initial_hash_before = state_dict_sha256(initial_policy.state_dict())
    continuation_hash_before = state_dict_sha256(continuation_policy.state_dict())
    initial_rows, initial_overlays = _evaluate(
        initial_policy, initial["hp"], DIGIT_ENV_CLASSES
    )
    continuation_rows, continuation_overlays = _evaluate(
        continuation_policy, continuation["hp"], DIGIT_ENV_CLASSES
    )
    initial_hash_after = state_dict_sha256(initial_policy.state_dict())
    continuation_hash_after = state_dict_sha256(continuation_policy.state_dict())
    _save_overlays(output, initial_overlays, continuation_overlays)

    comparison_rows = []
    update0_by_digit = []
    update5000_by_digit = []
    for before, after in zip(initial_rows, continuation_rows):
        if (before["digit"], before["delay_steps"]) != (
            after["digit"],
            after["delay_steps"],
        ):
            raise RuntimeError("evaluation grids differ")
        comparison_rows.append(
            {
                "digit": before["digit"],
                "delay_steps": before["delay_steps"],
                **{
                    f"update0_{key}": value
                    for key, value in before.items()
                    if key not in {"digit", "delay_steps"}
                },
                **{
                    f"update5000_{key}": value
                    for key, value in after.items()
                    if key not in {"digit", "delay_steps"}
                },
            }
        )
    for digit in range(10):
        before_values = [
            row["normalized_mean_error"]
            for row in initial_rows
            if row["digit"] == digit
        ]
        after_values = [
            row["normalized_mean_error"]
            for row in continuation_rows
            if row["digit"] == digit
        ]
        before_mean = float(np.mean(before_values))
        after_mean = float(np.mean(after_values))
        update0_by_digit.append(before_mean)
        update5000_by_digit.append(after_mean)
    progress = learning_progress_summary(
        update0_by_digit,
        update5000_by_digit,
    )
    all_finite = all(
        row["all_values_finite"]
        and all(
            math.isfinite(value)
            for key, value in row.items()
            if key not in {"digit", "delay_steps", "all_values_finite"}
        )
        for row in initial_rows + continuation_rows
    )
    workspace = build_workspace_audit(
        load_geometry_config(config["geometry_config"])
    )
    dynamic_safety_passed = all(
        row["min_inner_radius_margin_m"] > 0.0
        and row["min_outer_radius_margin_m"] > 0.0
        and row["min_joint_margin_rad"] > 0.0
        for row in initial_rows + continuation_rows
    )
    engineering_passed = bool(
        all_finite
        and workspace["passed"]
        and dynamic_safety_passed
        and initial_hash_before == initial_hash_after
        and continuation_hash_before == continuation_hash_after
    )
    with (output / "condition_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    with (output / "per_digit_progress.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(progress["per_digit"][0]))
        writer.writeheader()
        writer.writerows(progress["per_digit"])

    result = {
        "protocol": "digit_writing_original_protocol3",
        "audit": "update5000_read_only",
        "optimizer_steps_during_audit": 0,
        "initial_state_sha256_before": initial_hash_before,
        "initial_state_sha256_after": initial_hash_after,
        "continuation_state_sha256_before": continuation_hash_before,
        "continuation_state_sha256_after": continuation_hash_after,
        "all_values_finite": all_finite,
        "workspace_passed": workspace["passed"],
        "dynamic_safety_passed": dynamic_safety_passed,
        "engineering_passed": engineering_passed,
        "overall_update0_normalized_mean_error": progress[
            "overall_update0_normalized_mean_error"
        ],
        "overall_update5000_normalized_mean_error": progress[
            "overall_update5000_normalized_mean_error"
        ],
        "improved_digit_count": progress["improved_digit_count"],
        "worst_digit_relative_change": progress["worst_digit_relative_change"],
        "learning_progress_conditions_met": progress["conditions_met"],
        "per_digit": progress["per_digit"],
        "automatic_training_failure": False,
        "manual_resume_approval_required": True,
        "resume_allowed_without_user_approval": False,
    }
    _write_json(output / "early_audit_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--continuation-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_early_audit(
        args.initial_checkpoint,
        args.continuation_checkpoint,
        args.output,
    )
    print(f"EARLY_AUDIT_ENGINEERING_PASS={int(result['engineering_passed'])}")
    print(
        "EARLY_AUDIT_LEARNING_PROGRESS_CONDITIONS_MET="
        f"{int(result['learning_progress_conditions_met'])}"
    )
    print("MANUAL_RESUME_APPROVAL_REQUIRED=1")
    return 0 if result["engineering_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
