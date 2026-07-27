"""Necessary no-training audits for digit_writing_original_protocol2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from digit_writing.geometry import load_geometry_config
from digit_writing.geometry_audit import build_workspace_audit


HIGH_RISK_CASES = (
    (1, 0, "shortest_fast"),
    (4, 0, "corners_fast"),
    (6, 0, "tangent_loop_fast"),
    (8, 0, "longest_fast"),
    (8, 2, "longest_slow"),
    (9, 0, "loop_tail_fast"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if hasattr(value, "detach"):
        return bool(np.isfinite(value.detach().cpu().numpy()).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def run_high_risk_closed_loop_smoke(
    geometry_config_path: str | Path,
    base_config_path: str | Path,
) -> dict[str, Any]:
    """Run one deterministic untrained policy without optimization."""

    import motornet as mn
    import torch

    from losses import (
        detached_position_metrics,
        l1_muscle_act,
        l1_rate,
        l1_weight,
        position_l1_metrics,
        simple_dynamics,
    )
    from train import (
        DIGIT_ENV_CLASSES,
        _base_hp_from_config,
        _build_policy,
        _validate_base_config,
    )

    with Path(base_config_path).open("r", encoding="utf-8") as handle:
        base_config = json.load(handle)
    _validate_base_config(base_config)
    hp = _base_hp_from_config(base_config)
    torch.manual_seed(0)
    policy = _build_policy(hp, 6, torch.device("cpu"))
    policy.eval()

    rows = []
    for digit, speed_index, label in HIGH_RISK_CASES:
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        environment = DIGIT_ENV_CLASSES[digit](
            effector=effector,
            action_frame_stacking=0,
            geometry_config_path=geometry_config_path,
        )
        batch_size = 8
        observation, info = environment.reset(
            testing=False,
            options={
                "batch_size": batch_size,
                "reach_conds": np.arange(batch_size, dtype=np.int64),
                "speed_cond": speed_index,
                "delay_cond": 0,
                "deterministic": True,
            },
        )
        x = torch.zeros((batch_size, hp["hid_size"]))
        h = torch.zeros_like(x)
        positions = []
        targets = []
        muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hidden = [h.unsqueeze(1)]
        finite = _all_finite(observation) and _all_finite(info)
        timestep = 0
        terminated = False
        while not terminated:
            with torch.no_grad():
                x, h, action = policy(observation, x, h, noise=False)
            observation, _, terminated, info = environment.step(
                timestep, action=action
            )
            positions.append(info["states"]["fingertip"][:, None, :])
            targets.append(info["goal"][:, None, :])
            muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hidden.append(h.unsqueeze(1))
            finite &= _all_finite(observation) and _all_finite(info)
            timestep += 1

        positions_tensor = torch.cat(positions, dim=1)
        targets_tensor = torch.cat(targets, dim=1)
        muscle_tensor = torch.cat(muscle_acts, dim=1)
        hidden_tensor = torch.cat(hidden, dim=1)
        position_metrics = position_l1_metrics(
            positions_tensor, targets_tensor, environment.epoch_bounds
        )
        loss_terms = {
            **detached_position_metrics(position_metrics),
            "l1_rate": float(l1_rate(hidden_tensor, hp["l1_rate"]).detach()),
            "l1_weight": float(l1_weight(policy, hp["l1_weight"]).detach()),
            "l1_muscle_act": float(
                l1_muscle_act(muscle_tensor, hp["l1_muscle_act"]).detach()
            ),
            "simple_dynamics": float(
                simple_dynamics(
                    hidden_tensor,
                    policy.mrnn,
                    weight=hp["simple_dynamics_weight"],
                ).detach()
            ),
        }
        total_loss = (
            loss_terms["phase_normalized_position_l1"]
            + loss_terms["l1_rate"]
            + loss_terms["l1_weight"]
            + loss_terms["l1_muscle_act"]
            + loss_terms["simple_dynamics"]
        )
        movement_start, movement_end = environment.epoch_bounds["movement"]
        hold_start, hold_end = environment.epoch_bounds["hold"]
        timing_valid = bool(
            movement_end - movement_start == environment.traj.shape[1]
            and hold_start == movement_end
            and hold_end == timestep
        )
        endpoint_valid = bool(
            torch.allclose(
                environment._target_at(movement_start), environment.traj[:, 0]
            )
            and torch.allclose(
                environment._target_at(movement_end - 1), environment.traj[:, -1]
            )
        )
        passed = bool(
            finite
            and timing_valid
            and endpoint_valid
            and math.isfinite(total_loss)
            and all(math.isfinite(value) for value in loss_terms.values())
        )
        rows.append(
            {
                "case": label,
                "digit": digit,
                "speed_index": speed_index,
                "direction_count": batch_size,
                "movement_intervals": environment.movement_intervals,
                "movement_samples": environment.traj.shape[1],
                "episode_steps": timestep,
                "timing_valid": timing_valid,
                "movement_endpoints_supervised": endpoint_valid,
                "all_values_finite": finite,
                "loss_terms": loss_terms,
                "total_loss": total_loss,
                "passed": passed,
            }
        )
    return {
        "policy": "deterministic untrained RNN smoke only",
        "optimizer_used": False,
        "formal_training_started": False,
        "cases": rows,
        "passed": all(row["passed"] for row in rows),
    }


def run_audit(
    geometry_config_path: str | Path,
    base_config_path: str | Path,
) -> dict[str, Any]:
    config = load_geometry_config(geometry_config_path)
    workspace = build_workspace_audit(config)
    smoke = run_high_risk_closed_loop_smoke(
        geometry_config_path, base_config_path
    )
    return {
        "protocol": "digit_writing_original_protocol2",
        "workspace": workspace,
        "closed_loop_smoke": smoke,
        "passed": bool(workspace["passed"] and smoke["passed"]),
        "formal_training_started": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    audit = run_audit(args.geometry_config, args.base_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(audit), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"FINAL_PROTOCOL_AUDIT_PASSED={int(audit['passed'])}")
    print("FORMAL_TRAINING_STARTED=0")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
