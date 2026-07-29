"""Figures and MotorNet workspace audit for the final digit geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from digit_writing.geometry import (
    DigitTrajectory,
    GeometryConfig,
    build_digit_trajectory,
    build_time_audit,
    load_geometry_config,
    training_angles,
    validation_angles,
)


FK_TOLERANCE_M = 2e-6


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inverse_kinematics(
    points: np.ndarray, link1: float, link2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the positive-elbow branch used by RigidTendonArm26 limits."""

    x = points[:, 0]
    y = points[:, 1]
    cosine_elbow = (
        x * x + y * y - link1 * link1 - link2 * link2
    ) / (2.0 * link1 * link2)
    reachable = (cosine_elbow >= -1.0) & (cosine_elbow <= 1.0)
    elbow = np.arccos(np.clip(cosine_elbow, -1.0, 1.0))
    shoulder = np.arctan2(y, x) - np.arctan2(
        link2 * np.sin(elbow), link1 + link2 * np.cos(elbow)
    )
    shoulder = np.where(shoulder < 0.0, shoulder + 2.0 * math.pi, shoulder)
    return np.column_stack((shoulder, elbow)), reachable


def _motor_forward_kinematics(skeleton, joint_positions: np.ndarray) -> np.ndarray:
    import torch

    zeros = np.zeros_like(joint_positions)
    joint_state = np.column_stack((joint_positions, zeros))
    with torch.no_grad():
        cartesian = skeleton.joint2cartesian(
            torch.as_tensor(joint_state, dtype=torch.float32)
        )
    return cartesian.detach().cpu().numpy()[:, :2]


def _kinematic_metrics(
    points: np.ndarray,
    skeleton,
    link1: float,
    link2: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    joint_positions, reachable = _inverse_kinematics(points, link1, link2)
    reconstructed = _motor_forward_kinematics(skeleton, joint_positions)
    reconstruction_error = np.linalg.norm(reconstructed - points, axis=1)

    radius = np.linalg.norm(points, axis=1)
    inner_radius = abs(link1 - link2)
    outer_radius = link1 + link2
    inner_margin = radius - inner_radius
    outer_margin = outer_radius - radius
    joint_margins = np.minimum(joint_positions - lower, upper - joint_positions)

    metrics = {
        "cartesian_bbox_m": {
            "min_x": float(points[:, 0].min()),
            "max_x": float(points[:, 0].max()),
            "min_y": float(points[:, 1].min()),
            "max_y": float(points[:, 1].max()),
        },
        "all_inverse_solutions_reachable": bool(reachable.all()),
        "minimum_inner_radial_margin_m": float(inner_margin.min()),
        "minimum_outer_radial_margin_m": float(outer_margin.min()),
        "minimum_joint_margin_rad": float(joint_margins.min()),
        "minimum_joint0_margin_rad": float(joint_margins[:, 0].min()),
        "minimum_joint1_margin_rad": float(joint_margins[:, 1].min()),
        "maximum_motor_fk_error_m": float(reconstruction_error.max()),
    }
    return metrics, joint_positions


def _baseline_joint_state(effector) -> np.ndarray:
    position = np.array(
        (
            float(effector.pos_range_bound[0]) * 0.5
            + float(effector.pos_upper_bound[0])
            + 0.1,
            float(effector.pos_range_bound[1]) * 0.5
            + float(effector.pos_upper_bound[1])
            + 0.5,
        ),
        dtype=np.float64,
    )
    return np.concatenate((position, np.zeros(2, dtype=np.float64)))


def build_workspace_audit(config: GeometryConfig) -> dict[str, object]:
    """Audit all 10 digits by 32 directions against MotorNet kinematics."""

    import motornet as mn
    import torch

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    skeleton = effector.skeleton
    link1 = float(skeleton.L1)
    link2 = float(skeleton.L2)
    lower = np.asarray(effector.pos_lower_bound, dtype=np.float64)
    upper = np.asarray(effector.pos_upper_bound, dtype=np.float64)
    velocity_lower = np.asarray(effector.vel_lower_bound, dtype=np.float64)
    velocity_upper = np.asarray(effector.vel_upper_bound, dtype=np.float64)

    initial_joint_state = _baseline_joint_state(effector)
    with torch.no_grad():
        initial_cartesian_state = skeleton.joint2cartesian(
            torch.as_tensor(initial_joint_state[None, :], dtype=torch.float32)
        )
    anchor = initial_cartesian_state.detach().cpu().numpy()[0, :2].astype(np.float64)
    audit_reference = (
        int(config.selected_reference_steps)
        if config.selected_reference_steps is not None
        else 100
    )

    conditions = []
    all_points = []
    most_dangerous = None
    touches_limit = False
    for digit in range(10):
        for direction_index, angle in enumerate(validation_angles()):
            trajectory = build_digit_trajectory(
                digit,
                config,
                audit_reference,
                spatial_angle_rad=float(angle),
                anchor=anchor,
            )
            metrics, _ = _kinematic_metrics(
                trajectory.points, skeleton, link1, link2, lower, upper
            )
            radial_margin = min(
                metrics["minimum_inner_radial_margin_m"],
                metrics["minimum_outer_radial_margin_m"],
            )
            radial_scale = link1 + link2 - abs(link1 - link2)
            joint_scale = float(np.min(upper - lower))
            normalized_safety = min(
                radial_margin / radial_scale,
                metrics["minimum_joint_margin_rad"] / joint_scale,
            )
            row = {
                "digit": digit,
                "direction_index": direction_index,
                "angle_deg": math.degrees(float(angle)),
                "sample_count": len(trajectory.points),
                "normalized_safety_margin": float(normalized_safety),
                **metrics,
            }
            conditions.append(row)
            all_points.append(trajectory.points)
            if most_dangerous is None or normalized_safety < most_dangerous[0]:
                most_dangerous = (normalized_safety, row)
            if (
                not metrics["all_inverse_solutions_reachable"]
                or radial_margin <= 0.0
                or metrics["minimum_joint_margin_rad"] <= 0.0
            ):
                touches_limit = True

    combined_points = np.concatenate(all_points, axis=0)
    minimum_inner = min(row["minimum_inner_radial_margin_m"] for row in conditions)
    minimum_outer = min(row["minimum_outer_radial_margin_m"] for row in conditions)
    minimum_joint0 = min(row["minimum_joint0_margin_rad"] for row in conditions)
    minimum_joint1 = min(row["minimum_joint1_margin_rad"] for row in conditions)
    maximum_fk_error = max(row["maximum_motor_fk_error_m"] for row in conditions)

    digit_lengths = {
        digit: build_digit_trajectory(digit, config, audit_reference).arc_length_m
        for digit in range(10)
    }
    longest_digit = max(digit_lengths, key=digit_lengths.get)
    dynamic_smoke = []
    for reference_steps in sorted(
        {
            min(config.training_reference_steps),
            max(config.training_reference_steps),
        }
    ):
        for direction_index, angle in enumerate(validation_angles()):
            trajectory = build_digit_trajectory(
                longest_digit,
                config,
                reference_steps,
                spatial_angle_rad=float(angle),
                anchor=anchor,
            )
            metrics, joint_positions = _kinematic_metrics(
                trajectory.points, skeleton, link1, link2, lower, upper
            )
            joint_velocity = np.diff(joint_positions, axis=0) / config.dt_seconds
            velocity_margin = np.minimum(
                joint_velocity - velocity_lower,
                velocity_upper - joint_velocity,
            )
            dynamic_smoke.append(
                {
                    "digit": longest_digit,
                    "direction_index": direction_index,
                    "angle_deg": math.degrees(float(angle)),
                    "reference_steps": reference_steps,
                    "movement_intervals": trajectory.movement_intervals,
                    "episode_steps_at_max_delay": (
                        config.stable_steps
                        + max(config.delay_steps)
                        + trajectory.movement_intervals
                        + 1
                        + config.hold_steps
                    ),
                    "maximum_abs_joint_velocity_rad_s": float(
                        np.abs(joint_velocity).max()
                    ),
                    "minimum_velocity_limit_margin_rad_s": float(
                        velocity_margin.min()
                    ),
                    "all_values_finite": bool(
                        np.isfinite(joint_positions).all()
                        and np.isfinite(joint_velocity).all()
                    ),
                    **metrics,
                }
            )

    dynamic_passed = all(
        row["all_values_finite"]
        and row["all_inverse_solutions_reachable"]
        and row["minimum_inner_radial_margin_m"] > 0.0
        and row["minimum_outer_radial_margin_m"] > 0.0
        and row["minimum_joint_margin_rad"] > 0.0
        and row["minimum_velocity_limit_margin_rad_s"] > 0.0
        and row["maximum_motor_fk_error_m"] <= FK_TOLERANCE_M
        for row in dynamic_smoke
    )
    static_passed = (
        not touches_limit
        and maximum_fk_error <= FK_TOLERANCE_M
        and all(row["all_inverse_solutions_reachable"] for row in conditions)
    )

    return {
        "motornet_version": mn.__version__,
        "device": "cpu",
        "effector": "RigidTendonArm26(MujocoHillMuscle)",
        "link_lengths_m": [link1, link2],
        "joint_position_lower_bounds_rad": lower.tolist(),
        "joint_position_upper_bounds_rad": upper.tolist(),
        "joint_velocity_lower_bounds_rad_s": velocity_lower.tolist(),
        "joint_velocity_upper_bounds_rad_s": velocity_upper.tolist(),
        "baseline_initial_joint_state": initial_joint_state.tolist(),
        "anchor_m": anchor.tolist(),
        "condition_count": len(conditions),
        "cartesian_bbox_m": {
            "min_x": float(combined_points[:, 0].min()),
            "max_x": float(combined_points[:, 0].max()),
            "min_y": float(combined_points[:, 1].min()),
            "max_y": float(combined_points[:, 1].max()),
        },
        "radial_reach_margin_m": {
            "minimum_inner": float(minimum_inner),
            "minimum_outer": float(minimum_outer),
        },
        "joint_angle_margin_rad": {
            "joint0_minimum": float(minimum_joint0),
            "joint1_minimum": float(minimum_joint1),
            "overall_minimum": float(min(minimum_joint0, minimum_joint1)),
        },
        "inverse_forward_consistency": {
            "maximum_error_m": float(maximum_fk_error),
            "tolerance_m": FK_TOLERANCE_M,
            "passed": bool(maximum_fk_error <= FK_TOLERANCE_M),
        },
        "most_dangerous_condition": {
            "digit": most_dangerous[1]["digit"],
            "direction_index": most_dangerous[1]["direction_index"],
            "angle_deg": most_dangerous[1]["angle_deg"],
            "normalized_safety_margin": float(most_dangerous[0]),
        },
        "touches_limit": touches_limit,
        "conditions": conditions,
        "dynamic_smoke": dynamic_smoke,
        "dynamic_smoke_passed": dynamic_passed,
        "passed": bool(static_passed and dynamic_passed),
    }


def _plot_trajectory(axis, trajectory: DigitTrajectory, axis_limit: float) -> None:
    points = trajectory.points
    axis.plot(points[:, 0], points[:, 1], color="black", linewidth=1.5)

    boundary_indices = sorted(
        {0}
        | {boundary.start_index for boundary in trajectory.boundaries}
        | {boundary.end_index for boundary in trajectory.boundaries}
    )
    boundary_points = points[boundary_indices]
    axis.scatter(
        boundary_points[:, 0],
        boundary_points[:, 1],
        marker="s",
        s=24,
        color="#d97706",
        zorder=3,
        label="primitive boundary",
    )
    axis.scatter(
        points[0, 0],
        points[0, 1],
        marker="*",
        s=90,
        color="#15803d",
        zorder=4,
        label="start",
    )
    for boundary in trajectory.boundaries:
        arrow_start = boundary.start_index + boundary.intervals // 2
        arrow_start = min(arrow_start, boundary.end_index - 1)
        arrow_end = arrow_start + 1
        axis.annotate(
            "",
            xy=points[arrow_end],
            xytext=points[arrow_start],
            arrowprops={"arrowstyle": "-|>", "color": "#2563eb", "lw": 1.2},
        )

    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-axis_limit, axis_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")


def generate_audit_figures(
    config: GeometryConfig, output_directory: str | Path
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"figure directory is not empty: {output}")
    else:
        output.mkdir(parents=True)

    audit_reference = (
        int(config.selected_reference_steps)
        if config.selected_reference_steps is not None
        else 100
    )
    axis_limit = 0.0
    for digit in range(10):
        for angle in validation_angles():
            points = build_digit_trajectory(
                digit, config, audit_reference, spatial_angle_rad=float(angle)
            ).points
            axis_limit = max(axis_limit, float(np.linalg.norm(points, axis=1).max()))
    axis_limit *= 1.08

    manifest = []
    for digit in range(10):
        trajectory = build_digit_trajectory(digit, config, audit_reference)
        figure, axis = plt.subplots(figsize=(5, 5))
        _plot_trajectory(axis, trajectory, axis_limit)
        axis.set_title(f"Digit {digit}: prescribed direction")
        figure.text(
            0.5,
            0.01,
            "start: green star | primitive boundary: orange square | motion: blue arrow",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        path = output / f"prescribed_digit_{digit}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        manifest.append(
            {
                "file": path.name,
                "kind": "prescribed_direction",
                "digits": [digit],
                "angle_deg": 0.0,
                "sha256": _file_sha256(path),
            }
        )

    for direction_index, angle in enumerate(training_angles()):
        figure, axes = plt.subplots(2, 5, figsize=(16, 7))
        for digit, axis in enumerate(axes.flat):
            trajectory = build_digit_trajectory(
                digit,
                config,
                audit_reference,
                spatial_angle_rad=float(angle),
            )
            _plot_trajectory(axis, trajectory, axis_limit)
            axis.set_title(f"Digit {digit}")
        angle_deg = math.degrees(float(angle))
        figure.suptitle(
            f"Training direction {direction_index}: {angle_deg:.1f} degrees\n"
            "start: green star | primitive boundary: orange square | motion: blue arrow"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        path = output / f"training_direction_{direction_index}_{angle_deg:05.1f}deg.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        manifest.append(
            {
                "file": path.name,
                "kind": "training_direction_2x5",
                "digits": list(range(10)),
                "direction_index": direction_index,
                "angle_deg": angle_deg,
                "sha256": _file_sha256(path),
            }
        )

    return {
        "axis_limit_m": axis_limit,
        "figure_count": len(manifest),
        "prescribed_figure_count": 10,
        "training_direction_figure_count": 8,
        "figures": manifest,
    }


def run_geometry_audit(config_path: str | Path, output_directory: str | Path) -> dict:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    output.mkdir(parents=True)

    config_file = Path(config_path)
    config = load_geometry_config(config_file)
    time_audit = build_time_audit(config)
    workspace_audit = build_workspace_audit(config)
    figure_manifest = generate_audit_figures(config, output / "figures")

    _write_json(output / "time_audit.json", time_audit)
    _write_json(output / "workspace_audit.json", workspace_audit)
    _write_json(output / "figure_manifest.json", figure_manifest)

    passed = (
        not time_audit["segments_with_fewer_than_two_intervals"]
        and workspace_audit["passed"]
        and figure_manifest["figure_count"] == 18
    )
    summary = {
        "config_path": str(config_file),
        "config_sha256": _file_sha256(config_file),
        "geometry_time_passed": not bool(
            time_audit["segments_with_fewer_than_two_intervals"]
        ),
        "workspace_passed": workspace_audit["passed"],
        "figures_passed": figure_manifest["figure_count"] == 18,
        "formal_training_started": False,
        "passed": bool(passed),
    }
    _write_json(output / "audit_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    summary = run_geometry_audit(args.config, args.output)
    print(json.dumps(summary, sort_keys=True))
    print(f"GEOMETRY_AUDIT_PASSED={int(summary['passed'])}")
    print("FORMAL_TRAINING_STARTED=0")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
