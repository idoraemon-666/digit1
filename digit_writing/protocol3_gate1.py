"""Static, safety, and untrained closed-loop Gate 1 for Protocol3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from digit_writing.final_protocol_audit import _all_finite, _json_safe
from digit_writing.geometry import build_digit_trajectory, load_geometry_config
from digit_writing.geometry_audit import build_workspace_audit


EXPECTED_INTERVALS = {
    50: (85, 30, 100, 85, 90, 105, 100, 60, 100, 100),
    100: (170, 60, 200, 170, 180, 210, 200, 120, 200, 200),
}
STRICT_SHARED_IDS = {"curve_A", "curve_B"}
HIGH_RISK_CASES = (
    (1, 50, "digit1_fast"),
    (4, 50, "digit4_fast"),
    (5, 50, "digit5_fast"),
    (6, 50, "digit6_fast"),
    (8, 50, "digit8_fast"),
    (9, 50, "digit9_fast"),
    (8, 100, "digit8_medium"),
)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _minimum_finite_curvature_radius(points: np.ndarray) -> float | None:
    radii = []
    for first, middle, last in zip(points[:-2], points[1:-1], points[2:]):
        a = float(np.linalg.norm(middle - first))
        b = float(np.linalg.norm(last - middle))
        c = float(np.linalg.norm(last - first))
        first_delta = middle - first
        second_delta = last - first
        cross = abs(
            float(
                first_delta[0] * second_delta[1]
                - first_delta[1] * second_delta[0]
            )
        )
        if a > 0.0 and b > 0.0 and c > 0.0 and cross > 1e-15:
            radii.append(a * b * c / (2.0 * cross))
    return min(radii) if radii else None


def _junction_angles(trajectory) -> list[float]:
    angles = []
    points = trajectory.points
    for boundary in trajectory.boundaries[:-1]:
        index = boundary.end_index
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator <= 0.0:
            raise RuntimeError("zero-length junction step")
        cosine = float(np.dot(incoming, outgoing) / denominator)
        angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
    return angles


def target_kinematics_rows(config) -> list[dict[str, Any]]:
    rows = []
    reference = int(config.selected_reference_steps)
    for digit in range(10):
        trajectory = build_digit_trajectory(digit, config, reference)
        velocity = np.diff(trajectory.points, axis=0) / config.dt_seconds
        speed = np.linalg.norm(velocity, axis=1)
        acceleration = np.diff(velocity, axis=0) / config.dt_seconds
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        jerk = np.diff(acceleration, axis=0) / config.dt_seconds
        jerk_norm = np.linalg.norm(jerk, axis=1)
        junction_angles = _junction_angles(trajectory)
        rows.append(
            {
                "digit": digit,
                "selected_reference_steps": reference,
                "arc_length_m": trajectory.arc_length_m,
                "movement_intervals": trajectory.movement_intervals,
                "movement_duration_s": trajectory.movement_duration_s,
                "actual_mean_speed_m_s": trajectory.actual_mean_speed_m_s,
                "p95_speed_m_s": _percentile_or_zero(speed, 95.0),
                "peak_speed_m_s": float(speed.max()),
                "p95_acceleration_m_s2": _percentile_or_zero(
                    acceleration_norm, 95.0
                ),
                "peak_acceleration_m_s2": (
                    float(acceleration_norm.max())
                    if acceleration_norm.size
                    else 0.0
                ),
                "p95_jerk_m_s3": _percentile_or_zero(jerk_norm, 95.0),
                "peak_jerk_m_s3": float(jerk_norm.max()) if jerk_norm.size else 0.0,
                "minimum_finite_curvature_radius_m": (
                    _minimum_finite_curvature_radius(trajectory.points)
                ),
                "maximum_junction_angle_deg": max(junction_angles, default=0.0),
            }
        )
    return rows


def primitive_manifest_rows(config) -> list[dict[str, Any]]:
    rows = []
    reference = int(config.selected_reference_steps)
    for digit in range(10):
        trajectory = build_digit_trajectory(digit, config, reference)
        for boundary in trajectory.boundaries:
            strict_shared_id = (
                boundary.template_key
                if boundary.template_key in STRICT_SHARED_IDS
                else ""
            )
            rows.append(
                {
                    "digit": digit,
                    "segment_name": boundary.name,
                    "strict_shared_id": strict_shared_id,
                    "timing_key": boundary.timing_key,
                    "reference_steps": reference,
                    "arc_length_m": boundary.arc_length_m,
                    "intervals": boundary.intervals,
                    "samples": boundary.intervals + 1,
                    "actual_mean_speed_m_s": boundary.actual_mean_speed_m_s,
                    "canonical_template_sha256": (
                        boundary.canonical_template_sha256
                        if strict_shared_id
                        else ""
                    ),
                    "ordered_instance_sha256": boundary.ordered_instance_sha256,
                }
            )
    return rows


def _shared_assertions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for reference in (50, 100):
        for shared_id, expected_digits in (
            ("curve_A", {2, 3}),
            ("curve_B", {3, 5}),
        ):
            shared = [
                row
                for row in rows
                if row["reference_steps"] == reference
                and row["strict_shared_id"] == shared_id
            ]
            passed = bool(
                {row["digit"] for row in shared} == expected_digits
                and len({row["intervals"] for row in shared}) == 1
                and len(
                    {row["canonical_template_sha256"] for row in shared}
                )
                == 1
            )
            result[f"{shared_id}_ref{reference}"] = {
                "digits": sorted(row["digit"] for row in shared),
                "passed": passed,
            }
    ellipse_rows = [row for row in rows if row["timing_key"] == "ellipse_5_4"]
    result["ellipse_5_4_is_not_strict_shared"] = {
        "digits": sorted(row["digit"] for row in ellipse_rows),
        "strict_shared_ids": sorted(
            {row["strict_shared_id"] for row in ellipse_rows}
        ),
        "passed": bool(
            {row["digit"] for row in ellipse_rows} == {6, 9}
            and all(not row["strict_shared_id"] for row in ellipse_rows)
        ),
    }
    result["passed"] = all(item["passed"] for item in result.values())
    return result


def run_closed_loop_smoke(fast_config, medium_config, model_config_path: Path):
    import motornet as mn
    import torch

    from train import DIGIT_ENV_CLASSES, _base_hp_from_config, _build_policy

    with model_config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    hp = _base_hp_from_config(model_config)
    torch.manual_seed(0)
    policy = _build_policy(hp, 6, torch.device("cpu"))
    policy.eval()
    rows = []
    for digit, reference, label in HIGH_RISK_CASES:
        geometry_config = fast_config if reference == 50 else medium_config
        geometry_path = (
            model_config_path.parent
            / Path(model_config["geometry_config"]).name.replace(
                f"ref{model_config['selected_reference_steps']}",
                f"ref{reference}",
            )
        )
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        environment = DIGIT_ENV_CLASSES[digit](
            effector=effector,
            action_frame_stacking=0,
            geometry_config_path=geometry_path,
        )
        batch_size = 8
        observation, info = environment.reset(
            testing=False,
            seed=0,
            options={
                "batch_size": batch_size,
                "reach_conds": np.arange(8, dtype=np.int64),
                "speed_cond": 0,
                "delay_cond": 0,
                "deterministic": True,
            },
        )
        x = torch.zeros((batch_size, hp["hid_size"]))
        h = torch.zeros_like(x)
        actions = []
        muscle_activations = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        finite = _all_finite(observation) and _all_finite(info)
        timestep = 0
        terminated = False
        while not terminated:
            with torch.no_grad():
                x, h, action = policy(observation, x, h, noise=False)
            observation, _, terminated, info = environment.step(timestep, action)
            actions.append(action[:, None, :])
            muscle_activations.append(
                info["states"]["muscle"][:, 0].unsqueeze(1)
            )
            finite &= _all_finite(observation) and _all_finite(info)
            timestep += 1
        action_tensor = torch.cat(actions, dim=1)
        muscle_tensor = torch.cat(muscle_activations, dim=1)
        row = {
            "case": label,
            "digit": digit,
            "selected_reference_steps": reference,
            "movement_intervals": environment.movement_intervals,
            "episode_steps": timestep,
            "all_values_finite": finite,
            "muscle_activation_max": float(muscle_tensor.max()),
            "muscle_activation_fraction_ge_0_99": float(
                (muscle_tensor >= 0.99).float().mean()
            ),
            "muscle_excitation_fraction_le_0_01": float(
                (action_tensor <= 0.01).float().mean()
            ),
            "muscle_excitation_fraction_ge_0_99": float(
                (action_tensor >= 0.99).float().mean()
            ),
            "passed": bool(finite and timestep == environment.epoch_bounds["hold"][1]),
        }
        rows.append(row)
    return {
        "optimizer_used": False,
        "training_updates": 0,
        "cases": rows,
        "passed": all(row["passed"] for row in rows),
    }


def run_gate1(scale_token: str, output: Path) -> dict[str, Any]:
    from digit_writing.protocol3_checkpoint import current_git_identity

    if scale_token not in {"2p50", "2p25"}:
        raise ValueError("scale_token must be 2p50 or 2p25")
    if output.exists():
        raise FileExistsError(f"Gate 1 output already exists: {output}")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    config_root = root / "configurations"
    fast_path = config_root / (
        "digit_writing_original_protocol3_geometry_"
        f"scale{scale_token}_ref50.json"
    )
    medium_path = config_root / (
        "digit_writing_original_protocol3_geometry_"
        f"scale{scale_token}_ref100.json"
    )
    model_path = config_root / (
        "digit_writing_original_protocol3_gate2_"
        f"scale{scale_token}_ref50.json"
    )
    fast_config = load_geometry_config(fast_path)
    medium_config = load_geometry_config(medium_path)
    fast_manifest = primitive_manifest_rows(fast_config)
    medium_manifest = primitive_manifest_rows(medium_config)
    fast_kinematics = target_kinematics_rows(fast_config)
    medium_kinematics = target_kinematics_rows(medium_config)
    _write_csv(output / "primitive_manifest.csv", fast_manifest + medium_manifest)
    _write_csv(output / "digit_timing_fast.csv", fast_kinematics)
    _write_csv(output / "digit_timing_medium.csv", medium_kinematics)
    _write_csv(output / "target_kinematics_fast.csv", fast_kinematics)
    _write_csv(output / "target_kinematics_medium.csv", medium_kinematics)

    fast_intervals = tuple(row["movement_intervals"] for row in fast_kinematics)
    medium_intervals = tuple(row["movement_intervals"] for row in medium_kinematics)
    timing_passed = bool(
        fast_intervals == EXPECTED_INTERVALS[50]
        and medium_intervals == EXPECTED_INTERVALS[100]
    )
    shared = _shared_assertions(fast_manifest + medium_manifest)
    workspace = build_workspace_audit(fast_config)
    _write_json(output / "workspace_audit.json", workspace)
    minimum_inner = workspace["radial_reach_margin_m"]["minimum_inner"]
    minimum_outer = workspace["radial_reach_margin_m"]["minimum_outer"]
    minimum_joint = workspace["joint_angle_margin_rad"]["overall_minimum"]
    safety_warning = bool(
        min(minimum_inner, minimum_outer) < 0.02
        or minimum_joint < 0.10
    )
    smoke = run_closed_loop_smoke(fast_config, medium_config, model_path)
    _write_json(output / "closed_loop_smoke.json", smoke)
    requires_scale_fallback = bool(scale_token == "2p50" and safety_warning)
    passed = bool(
        timing_passed
        and shared["passed"]
        and workspace["passed"]
        and not safety_warning
        and smoke["passed"]
    )
    summary = {
        "protocol": "digit_writing_original_protocol3",
        "gate": "gate1",
        "git_identity": current_git_identity(root),
        "scale_token": scale_token,
        "timing_passed": timing_passed,
        "shared_segment_assertions": shared,
        "workspace_passed": workspace["passed"],
        "safety_warning": safety_warning,
        "requires_scale_fallback": requires_scale_fallback,
        "closed_loop_smoke_passed": smoke["passed"],
        "training_updates": 0,
        "optimizer_used": False,
        "passed": passed,
    }
    _write_json(output / "preflight_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-token", required=True, choices=("2p50", "2p25"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_gate1(args.scale_token, args.output)
    print(f"GATE1_PASS={int(summary['passed'])}")
    print(
        "GATE1_REQUIRES_SCALE_FALLBACK="
        f"{int(summary['requires_scale_fallback'])}"
    )
    print("TRAINING_UPDATES=0")
    return 0 if summary["passed"] or summary["requires_scale_fallback"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
