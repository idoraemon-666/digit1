"""Read-only final-candidate audit for Protocol3 Gate 2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from digit_writing.final_protocol_audit import _all_finite, _json_safe
from digit_writing.protocol3_checkpoint import state_dict_sha256
from losses import (
    detached_position_metrics,
    l1_muscle_act,
    l1_rate,
    l1_weight,
    position_l1_metrics,
    simple_dynamics,
)


GRADIENT_EPSILON = 1e-12


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _gradient_vector(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def _movement_metrics(actual, target, epoch_bounds):
    movement_start, movement_end = epoch_bounds["movement"]
    actual_movement = actual[:, movement_start:movement_end]
    target_movement = target[:, movement_start:movement_end]
    error = torch.linalg.vector_norm(actual_movement - target_movement, dim=-1)
    target_span = target_movement.amax(dim=1) - target_movement.amin(dim=1)
    diagonal = torch.linalg.vector_norm(target_span, dim=-1)
    target_length = torch.linalg.vector_norm(
        torch.diff(target_movement, dim=1), dim=-1
    ).sum(dim=1)
    actual_length = torch.linalg.vector_norm(
        torch.diff(actual_movement, dim=1), dim=-1
    ).sum(dim=1)
    if bool(torch.any(diagonal <= 0.0)) or bool(torch.any(target_length <= 0.0)):
        raise RuntimeError("Gate 2 target normalization denominator is not positive")
    return actual_movement, target_movement, {
        "mean_euclidean_error_m": float(error.mean()),
        "normalized_mean_error": float((error.mean(dim=1) / diagonal).mean()),
        "endpoint_error_m": float(error[:, -1].mean()),
        "normalized_endpoint_error": float((error[:, -1] / diagonal).mean()),
        "actual_path_length_m": float(actual_length.mean()),
        "target_path_length_m": float(target_length.mean()),
        "path_length_ratio": float((actual_length / target_length).mean()),
    }


def _dynamic_safety_metrics(environment, positions, joint_positions):
    skeleton = environment.effector.skeleton
    inner_radius = abs(float(skeleton.L1) - float(skeleton.L2))
    outer_radius = float(skeleton.L1) + float(skeleton.L2)
    radius = torch.linalg.vector_norm(positions, dim=-1)
    lower = torch.as_tensor(
        environment.effector.pos_lower_bound,
        dtype=joint_positions.dtype,
        device=joint_positions.device,
    )
    upper = torch.as_tensor(
        environment.effector.pos_upper_bound,
        dtype=joint_positions.dtype,
        device=joint_positions.device,
    )
    joint_margin = torch.minimum(joint_positions - lower, upper - joint_positions)
    return {
        "min_inner_radius_margin_m": float((radius - inner_radius).min()),
        "min_outer_radius_margin_m": float((outer_radius - radius).min()),
        "min_joint_margin_rad": float(joint_margin.min()),
    }


def _save_overlay(output: Path, actual, target) -> None:
    actual_mean = actual.detach().mean(dim=0).cpu().numpy()
    target_mean = target.detach().mean(dim=0).cpu().numpy()
    with (output / "movement_overlay.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("movement_index", "actual_x_m", "actual_y_m", "target_x_m", "target_y_m"))
        for index, (actual_xy, target_xy) in enumerate(
            zip(actual_mean, target_mean)
        ):
            writer.writerow((index, *actual_xy.tolist(), *target_xy.tolist()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(target_mean[:, 0], target_mean[:, 1], "k--", label="target")
    axes[0].plot(actual_mean[:, 0], actual_mean[:, 1], label="actual")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend()
    axes[0].set_title("movement overlay")
    axes[1].plot(target_mean[:, 0], "k--", label="target x")
    axes[1].plot(actual_mean[:, 0], label="actual x")
    axes[1].legend()
    axes[1].set_title("x(t)")
    axes[2].plot(target_mean[:, 1], "k--", label="target y")
    axes[2].plot(actual_mean[:, 1], label="actual y")
    axes[2].legend()
    axes[2].set_title("y(t)")
    figure.tight_layout()
    figure.savefig(output / "movement_overlay.png", dpi=180)
    plt.close(figure)


def audit_gate2_case(
    policy,
    hp: Mapping[str, Any],
    environment_class,
    case: Mapping[str, Any],
    output: str | Path,
    workspace_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one trained candidate without taking an optimizer step."""

    import motornet as mn

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before_hash = state_dict_sha256(policy.state_dict())
    policy.eval()
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    environment = environment_class(
        effector=effector,
        **hp.get("env_kwargs", {}),
    )
    batch_size = int(hp["batch_size"])
    observation, info = environment.reset(
        testing=False,
        seed=int(hp["validation_seed"]),
        options={
            "batch_size": batch_size,
            "reach_conds": np.full(
                batch_size, int(case["direction_index"]), dtype=np.int64
            ),
            "speed_cond": 0,
            "delay_cond": int(hp["gate2_delay_index"]),
            "deterministic": True,
        },
    )
    x = torch.zeros((batch_size, hp["hid_size"]))
    h = torch.zeros_like(x)
    positions = []
    targets = []
    actions = []
    muscles = [info["states"]["muscle"][:, 0].unsqueeze(1)]
    hidden = [h.unsqueeze(1)]
    joint_positions = []
    finite = _all_finite(observation) and _all_finite(info)
    terminated = False
    timestep = 0
    while not terminated:
        x, h, action = policy(observation, x, h, noise=False)
        observation, _, terminated, info = environment.step(timestep, action)
        positions.append(info["states"]["fingertip"][:, None, :])
        targets.append(info["goal"][:, None, :])
        actions.append(action[:, None, :])
        muscles.append(info["states"]["muscle"][:, 0].unsqueeze(1))
        hidden.append(h.unsqueeze(1))
        joint_positions.append(
            info["states"]["joint"][:, : environment.effector.dof].unsqueeze(1)
        )
        finite &= _all_finite(observation) and _all_finite(info)
        timestep += 1

    positions = torch.cat(positions, dim=1)
    targets = torch.cat(targets, dim=1)
    actions = torch.cat(actions, dim=1)
    muscles = torch.cat(muscles, dim=1)
    hidden = torch.cat(hidden, dim=1)
    joint_positions = torch.cat(joint_positions, dim=1)
    movement_actual, movement_target, movement_metrics = _movement_metrics(
        positions, targets, environment.epoch_bounds
    )
    phase_metrics = position_l1_metrics(
        positions, targets, environment.epoch_bounds
    )
    position_loss = phase_metrics["phase_normalized_position_l1"]
    regularization_terms = {
        "weighted_l1_rate": l1_rate(hidden, hp["l1_rate"]),
        "weighted_l1_weight": l1_weight(policy, hp["l1_weight"]),
        "weighted_l1_muscle_act": l1_muscle_act(
            muscles, hp["l1_muscle_act"]
        ),
        "weighted_simple_dynamics": simple_dynamics(
            hidden, policy.mrnn, weight=hp["simple_dynamics_weight"]
        ),
    }
    regularization_loss = sum(regularization_terms.values())
    total_loss = position_loss + regularization_loss
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    position_gradient = _gradient_vector(
        position_loss, parameters, retain_graph=True
    )
    regularization_gradient = _gradient_vector(
        regularization_loss, parameters, retain_graph=True
    )
    total_gradient = _gradient_vector(total_loss, parameters, retain_graph=False)
    position_norm = float(torch.linalg.vector_norm(position_gradient))
    regularization_norm = float(torch.linalg.vector_norm(regularization_gradient))
    total_norm = float(torch.linalg.vector_norm(total_gradient))
    small_ratio_denominator = position_norm < GRADIENT_EPSILON
    cosine_undefined = (
        position_norm < GRADIENT_EPSILON or total_norm < GRADIENT_EPSILON
    )
    gradient_metrics = {
        "position_gradient_norm": position_norm,
        "regularization_gradient_norm": regularization_norm,
        "total_gradient_norm": total_norm,
        "regularization_to_position_gradient_ratio": (
            None if small_ratio_denominator else regularization_norm / position_norm
        ),
        "regularization_ratio_small_denominator": small_ratio_denominator,
        "position_total_gradient_cosine": (
            None
            if cosine_undefined
            else float(
                torch.dot(position_gradient, total_gradient)
                / (torch.linalg.vector_norm(position_gradient) * torch.linalg.vector_norm(total_gradient))
            )
        ),
        "position_total_gradient_cosine_undefined": cosine_undefined,
        "gradient_epsilon": GRADIENT_EPSILON,
        "hard_stop_from_gradient_metrics": False,
    }
    dynamic_safety = _dynamic_safety_metrics(
        environment, positions, joint_positions
    )
    workspace_minima = {
        "target_min_inner_radius_margin_m": workspace_audit[
            "radial_reach_margin_m"
        ]["minimum_inner"],
        "target_min_outer_radius_margin_m": workspace_audit[
            "radial_reach_margin_m"
        ]["minimum_outer"],
        "target_min_joint_margin_rad": workspace_audit[
            "joint_angle_margin_rad"
        ]["overall_minimum"],
    }
    boundary_metrics = {
        "muscle_activation_max": float(muscles.max()),
        "muscle_activation_fraction_ge_0_99": float(
            (muscles >= 0.99).float().mean()
        ),
        "muscle_excitation_fraction_le_0_01": float(
            (actions <= 0.01).float().mean()
        ),
        "muscle_excitation_fraction_ge_0_99": float(
            (actions >= 0.99).float().mean()
        ),
    }
    after_hash = state_dict_sha256(policy.state_dict())
    loss_metrics = {
        "position_loss": float(position_loss.detach()),
        **{
            name: float(value.detach())
            for name, value in regularization_terms.items()
        },
        "weighted_regularization_to_position_ratio": (
            None
            if abs(float(position_loss.detach())) < GRADIENT_EPSILON
            else float(regularization_loss.detach() / position_loss.detach())
        ),
    }
    finite &= all(
        math.isfinite(value)
        for group in (
            movement_metrics,
            dynamic_safety,
            boundary_metrics,
            loss_metrics,
            gradient_metrics,
        )
        for value in group.values()
        if value is not None and not isinstance(value, bool)
    )
    safety_passed = bool(
        workspace_audit["passed"]
        and min(
            workspace_minima["target_min_inner_radius_margin_m"],
            workspace_minima["target_min_outer_radius_margin_m"],
            dynamic_safety["min_inner_radius_margin_m"],
            dynamic_safety["min_outer_radius_margin_m"],
        ) >= 0.02
        and min(
            workspace_minima["target_min_joint_margin_rad"],
            dynamic_safety["min_joint_margin_rad"],
        ) >= 0.10
    )
    behavior_passed = bool(
        movement_metrics["normalized_mean_error"] <= 0.08
        and movement_metrics["normalized_endpoint_error"] <= 0.05
        and 0.85 <= movement_metrics["path_length_ratio"] <= 1.15
    )
    engineering_passed = bool(
        finite and safety_passed and before_hash == after_hash
    )
    _save_overlay(output, movement_actual, movement_target)
    result = {
        "case": dict(case),
        "optimizer_steps_during_audit": 0,
        "all_values_finite": finite,
        "state_sha256_before": before_hash,
        "state_sha256_after": after_hash,
        "state_unchanged": before_hash == after_hash,
        "movement_metrics": movement_metrics,
        "phase_metrics": detached_position_metrics(phase_metrics),
        "loss_metrics": loss_metrics,
        "gradient_metrics": gradient_metrics,
        "workspace_minima": workspace_minima,
        "dynamic_safety": dynamic_safety,
        "boundary_metrics": boundary_metrics,
        "safety_passed": safety_passed,
        "engineering_passed": engineering_passed,
        "behavior_passed": behavior_passed,
        "qualitative_overlay_review_required": True,
        "automatic_gate2_pass": bool(engineering_passed and behavior_passed),
    }
    _write_json(output / "gate2_candidate_audit.json", result)
    return result
