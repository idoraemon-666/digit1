"""Project-2 entries for rule composition and digit-5 transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from losses import detached_position_metrics, position_l1_metrics
from train import (
    DIGIT_ENV_CLASSES,
    load_digit_policy_checkpoint,
    train_digit_transfer5,
)


def replace_rule_input(observation, rule_input):
    """Replace only the ten rule columns in the 28-dimensional observation."""
    if observation.shape[-1] != 28 or rule_input.shape[-1] != 10:
        raise ValueError("composition requires 28-D observations and 10-D rules")
    if rule_input.shape[0] != observation.shape[0]:
        raise ValueError("each batch row requires its own composition coefficients")
    return torch.cat((rule_input, observation[:, 10:]), dim=-1)


def _full_rule(coefficients, source_digits):
    if coefficients.shape[1] != len(source_digits):
        raise ValueError("coefficient columns must match the configured source digits")
    basis = torch.eye(10, dtype=coefficients.dtype, device=coefficients.device)[
        list(source_digits)
    ]
    return coefficients @ basis


def prepare_composition(
    policy, target_digit, source_digits, batch_size, learning_rate=0.1
):
    """Freeze a full10 policy and create independent per-direction coefficients."""
    if not 0 <= target_digit < 10:
        raise ValueError("target_digit must be in [0, 9]")
    if target_digit in source_digits or len(source_digits) != 4:
        raise ValueError("composition requires four non-target source digits")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    coefficients = torch.nn.Parameter(torch.ones((batch_size, len(source_digits))))
    optimizer = torch.optim.Adam((coefficients,), lr=learning_rate)
    return coefficients, optimizer


def optimize_rule_composition(policy, config, target_digit):
    """Optimize four external rule coefficients using the final position loss."""
    batch_size = config["batch_size"]
    source_digits = tuple(
        digit for digit in config["digit_group"] if digit != target_digit
    )
    coefficients, optimizer = prepare_composition(
        policy,
        target_digit,
        source_digits,
        batch_size,
        learning_rate=config["coefficient_learning_rate"],
    )
    effector = config["effector_factory"]()
    best_loss = np.inf
    best_trial = None
    training_losses = []

    for _ in range(config["iterations"]):
        environment = DIGIT_ENV_CLASSES[target_digit](
            effector=effector,
            action_frame_stacking=0,
            geometry_config_path=config["geometry_config"],
        )
        options = {
            "batch_size": batch_size,
            "reach_conds": np.asarray(config["direction_indices"], dtype=np.int64),
            "speed_cond": config["validation_speed_index"],
            "custom_delay": config["custom_delay"],
        }
        observation, info = environment.reset(testing=True, options=options)
        x = torch.zeros((batch_size, policy.mrnn.total_num_units))
        h = torch.zeros_like(x)
        xy = []
        targets = []
        actions = []
        terminated = False
        timestep = 0
        rule_input = _full_rule(coefficients, source_digits)

        while not terminated:
            observation = replace_rule_input(observation, rule_input)
            x, h, action = policy(observation, x, h, noise=False)
            observation, _, terminated, info = environment.step(timestep, action=action)
            xy.append(info["states"]["fingertip"][:, None, :])
            targets.append(info["goal"][:, None, :])
            actions.append(action[:, None, :])
            timestep += 1

        trial_xy = torch.cat(xy, dim=1)
        trial_target = torch.cat(targets, dim=1)
        position_metrics = position_l1_metrics(
            trial_xy, trial_target, environment.epoch_bounds
        )
        loss = position_metrics["phase_normalized_position_l1"]
        training_losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_trial = {
                "rule_input": rule_input.detach().clone(),
                "xy": trial_xy.detach().clone(),
                "target": trial_target.detach().clone(),
                "action": torch.cat(actions, dim=1).detach().clone(),
                "epoch_bounds": environment.epoch_bounds,
                "source_digits": source_digits,
                "position_metrics": detached_position_metrics(position_metrics),
            }
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_trial["training_losses"] = training_losses
    best_trial["best_phase_normalized_position_l1"] = best_loss
    best_trial["target_digit"] = target_digit
    best_trial["final_coefficients"] = coefficients.detach().clone()
    best_trial["optimizer_state_dict"] = optimizer.state_dict()
    best_trial["iterations"] = config["iterations"]
    return best_trial


def _validate_composition_config(config):
    if config["run_kind"] != "composition":
        raise ValueError("composition requires run_kind=composition")
    if config["device"] != "cpu" or config["network_noise"]:
        raise ValueError("composition requires the CPU baseline with network_noise=false")
    if config["batch_size"] != 8:
        raise ValueError("composition requires batch_size=8")
    if config["direction_indices"] != [0, 4, 8, 12, 16, 20, 24, 28]:
        raise ValueError("composition directions must reproduce the original evaluation grid")
    if config["validation_speed_index"] != 9 or config["custom_delay"] != 150:
        raise ValueError("composition requires validation speed 9 and custom delay 150")
    if config["iterations"] != 250 or config["coefficient_learning_rate"] != 0.1:
        raise ValueError("composition requires 250 Adam iterations at learning rate 0.1")
    if config["digit_group"] != [0, 4, 6, 9, 8] or not config["leave_one_out"]:
        raise ValueError("composition requires the frozen five-digit leave-one-out group")
    if config["position_loss"] != {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }:
        raise ValueError("composition position loss does not match the final protocol")


def run_composition_config(config):
    """Run every configured target using only the independent full10 checkpoint."""
    _validate_composition_config(config)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    policy, checkpoint = load_digit_policy_checkpoint(
        config["source_checkpoint"], expected_variant="full10"
    )
    source_config = checkpoint.get("protocol_config")
    if source_config is None:
        raise ValueError("full10 checkpoint is missing its protocol config")
    if source_config.get("protocol") != "digit_writing_original_protocol2":
        raise ValueError("composition requires a project-2 full10 checkpoint")
    if source_config.get("position_loss") != config["position_loss"]:
        raise ValueError("composition position loss must match the full10 checkpoint")
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    output_dir = Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    runtime_config = dict(config)
    runtime_config["geometry_config"] = source_config["geometry_config"]
    import motornet as mn

    runtime_config["effector_factory"] = lambda: mn.effector.RigidTendonArm26(
        mn.muscle.MujocoHillMuscle()
    )
    summary = {}
    for target_digit in config["digit_group"]:
        result = optimize_rule_composition(policy, runtime_config, target_digit)
        torch.save(result, output_dir / f"digit_{target_digit}.pt")
        summary[str(target_digit)] = result["best_phase_normalized_position_l1"]

    after = policy.state_dict()
    for name, tensor in before.items():
        if not torch.equal(tensor, after[name]):
            raise RuntimeError(f"composition changed frozen network state: {name}")
    with open(output_dir / "composition_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "digit_group": config["digit_group"],
                "per_digit_best_phase_normalized_position_l1": summary,
                "network_state_bitwise_unchanged": True,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return summary


def run_transfer_config(config):
    """Delegate the digit-5-only transfer loop to the original training module."""
    return train_digit_transfer5(config)
