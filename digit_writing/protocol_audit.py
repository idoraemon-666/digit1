"""No-training geometry and timing audit against the original task suite.

The frozen digit generator and the baseline ``105cd0c...:envs.py`` are the two
trajectory sources.  Every trajectory is passed through the same metric
functions.  MotorNet is imported only by the server-only workspace and rollout
stages so that the numerical metric tests remain runnable without MotorNet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import types
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from digit_writing.geometry import (
    DigitTrajectory,
    GeometryConfig,
    build_digit_trajectory,
    load_geometry_config,
    prescribed_digit_primitives,
    primitive_library,
    training_angles,
    validation_angles,
)
from digit_writing.geometry_audit import (
    FK_TOLERANCE_M,
    _baseline_joint_state,
    _inverse_kinematics,
    _motor_forward_kinematics,
)


ORIGINAL_BASELINE = "105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33"
ORIGINAL_TASK_NAMES = (
    "DlyHalfReach",
    "DlyHalfCircleClk",
    "DlyHalfCircleCClk",
    "DlySinusoid",
    "DlySinusoidInv",
    "DlyFullReach",
    "DlyFullCircleClk",
    "DlyFullCircleCClk",
    "DlyFigure8",
    "DlyFigure8Inv",
)
TANGENT_DIGIT_JOINS = {(2, 0), (6, 0), (8, 0), (8, 1), (9, 0)}
REVERSAL_DIGIT_JOINS = {(3, 0)}


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_tolerance(points: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(points))))
    return 128.0 * np.finfo(np.float64).eps * scale


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _direction_before(points: np.ndarray, index: int, tolerance: float) -> np.ndarray:
    for cursor in range(index, 0, -1):
        vector = points[cursor] - points[cursor - 1]
        if np.linalg.norm(vector) > tolerance:
            return vector
    return np.zeros(2, dtype=np.float64)


def _direction_after(points: np.ndarray, index: int, tolerance: float) -> np.ndarray:
    for cursor in range(index, len(points) - 1):
        vector = points[cursor + 1] - points[cursor]
        if np.linalg.norm(vector) > tolerance:
            return vector
    return np.zeros(2, dtype=np.float64)


def trajectory_metrics(
    points: np.ndarray,
    *,
    dt_seconds: float,
    movement_steps: int,
    joins: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compute one common set of metrics for original and digit trajectories."""

    samples = np.asarray(points, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 2:
        raise ValueError("points must have shape (n>=2, 2)")
    if dt_seconds <= 0.0 or movement_steps <= 0:
        raise ValueError("dt_seconds and movement_steps must be positive")

    tolerance = _numeric_tolerance(samples)
    segments = np.diff(samples, axis=0)
    displacements = np.linalg.norm(segments, axis=1)
    velocities = segments / dt_seconds
    accelerations = np.diff(velocities, axis=0) / dt_seconds
    acceleration_norms = np.linalg.norm(accelerations, axis=1)
    acceleration_roundoff = (
        128.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.norm(velocities, axis=1).max()))
        / dt_seconds
    )
    acceleration_norms[acceleration_norms <= acceleration_roundoff] = 0.0
    center_indices = np.arange(1, len(samples) - 1, dtype=np.int64)

    first = samples[1:-1] - samples[:-2]
    second = samples[2:] - samples[1:-1]
    chord = samples[2:] - samples[:-2]
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
    )
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    curvature = np.full(len(center_indices), np.nan, dtype=np.float64)
    valid_curvature = denominator > tolerance**3
    curvature[valid_curvature] = 2.0 * cross[valid_curvature] / denominator[valid_curvature]
    roundoff_cross = (
        128.0
        * np.finfo(np.float64).eps
        * np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
    )
    curvature[valid_curvature & (cross <= roundoff_cross)] = 0.0

    join_rows = []
    excluded_centers: set[int] = set()
    connection_continuous = True
    for join_index, join in enumerate(joins):
        left_index = int(join["left_index"])
        right_index = int(join.get("right_index", left_index))
        if not (0 <= left_index < len(samples) and 0 <= right_index < len(samples)):
            raise ValueError("join index is outside the trajectory")
        error = float(np.linalg.norm(samples[left_index] - samples[right_index]))
        before = _direction_before(samples, left_index, tolerance)
        after = _direction_after(samples, right_index, tolerance)
        angle = _angle_degrees(before, after)
        connection_continuous &= error <= tolerance
        excluded_centers.update(range(max(1, left_index - 1), min(len(samples) - 1, right_index + 2)))
        join_rows.append(
            {
                "join_index": join_index,
                "label": str(join.get("label", f"join_{join_index}")),
                "classification": str(join.get("classification", "ordinary_corner")),
                "left_index": left_index,
                "right_index": right_index,
                "position_error_m": error,
                "direction_change_deg": angle,
            }
        )

    smooth_mask = valid_curvature.copy()
    if excluded_centers:
        smooth_mask &= ~np.isin(center_indices, np.fromiter(excluded_centers, dtype=np.int64))
    smooth_curvature = curvature[smooth_mask]
    maximum_curvature = float(np.max(smooth_curvature)) if smooth_curvature.size else 0.0
    minimum_radius = 1.0 / maximum_curvature if maximum_curvature > 0.0 else None
    if smooth_curvature.size:
        smooth_center_indices = center_indices[smooth_mask]
        curvature_peak_offset = int(np.argmax(smooth_curvature))
        peak_curvature_index = int(smooth_center_indices[curvature_peak_offset])
    else:
        peak_curvature_index = None

    central_speed = 0.5 * (
        np.linalg.norm(velocities[:-1], axis=1)
        + np.linalg.norm(velocities[1:], axis=1)
    )
    normal_acceleration = central_speed * central_speed * curvature
    smooth_normal = normal_acceleration[smooth_mask]

    smooth_acceleration_mask = ~np.isin(
        center_indices, np.fromiter(excluded_centers, dtype=np.int64)
    ) if excluded_centers else np.ones(len(center_indices), dtype=bool)
    smooth_acceleration = acceleration_norms[smooth_acceleration_mask]
    join_acceleration = acceleration_norms[~smooth_acceleration_mask]

    finite = bool(
        np.isfinite(samples).all()
        and np.isfinite(displacements).all()
        and np.isfinite(velocities).all()
        and np.isfinite(accelerations).all()
    )
    nonzero = displacements[displacements > tolerance]
    zero_indices = np.flatnonzero(displacements <= tolerance)
    peak_acceleration_index = int(np.argmax(acceleration_norms) + 1) if len(acceleration_norms) else None
    peak_speed_index = int(np.argmax(np.linalg.norm(velocities, axis=1))) if len(velocities) else None

    return {
        "target_sample_count": int(len(samples)),
        "kinematic_interval_count": int(len(samples) - 1),
        "movement_steps": int(movement_steps),
        "movement_epoch_duration_s": float(movement_steps * dt_seconds),
        "sampled_kinematic_duration_s": float((len(samples) - 1) * dt_seconds),
        "discrete_arc_length_m": float(displacements.sum()),
        "mean_step_displacement_m": float(displacements.mean()),
        "minimum_step_displacement_m": float(displacements.min()),
        "maximum_step_displacement_m": float(displacements.max()),
        "median_nonzero_step_displacement_m": float(np.median(nonzero)) if nonzero.size else 0.0,
        "maximum_to_median_step_ratio": (
            float(displacements.max() / np.median(nonzero)) if nonzero.size else None
        ),
        "bbox_width_m": float(np.ptp(samples[:, 0])),
        "bbox_height_m": float(np.ptp(samples[:, 1])),
        "start_to_farthest_point_m": float(np.linalg.norm(samples - samples[0], axis=1).max()),
        "maximum_discrete_speed_m_s": float(np.linalg.norm(velocities, axis=1).max()),
        "mean_discrete_speed_m_s": float(np.linalg.norm(velocities, axis=1).mean()),
        "maximum_discrete_acceleration_m_s2": float(acceleration_norms.max()) if len(acceleration_norms) else 0.0,
        "maximum_smooth_discrete_acceleration_m_s2": float(smooth_acceleration.max()) if smooth_acceleration.size else 0.0,
        "maximum_join_discrete_acceleration_m_s2": float(join_acceleration.max()) if join_acceleration.size else None,
        "maximum_smooth_curvature_1_m": maximum_curvature,
        "minimum_smooth_curvature_radius_m": minimum_radius,
        "peak_smooth_curvature_sample_index": peak_curvature_index,
        "peak_smooth_curvature_point_m": samples[peak_curvature_index].tolist() if peak_curvature_index is not None else None,
        "maximum_smooth_normal_acceleration_m_s2": float(np.nanmax(smooth_normal)) if smooth_normal.size else 0.0,
        "peak_speed_segment_index": peak_speed_index,
        "peak_acceleration_sample_index": peak_acceleration_index,
        "peak_acceleration_point_m": samples[peak_acceleration_index].tolist() if peak_acceleration_index is not None else None,
        "join_count": len(join_rows),
        "maximum_join_direction_change_deg": max(
            (row["direction_change_deg"] for row in join_rows if row["direction_change_deg"] is not None),
            default=None,
        ),
        "joins": join_rows,
        "position_continuous": bool(connection_continuous),
        "all_values_finite": finite,
        "near_duplicate_segment_count": int(len(zero_indices)),
        "near_duplicate_segment_indices": zero_indices.astype(int).tolist(),
        "numerical_position_tolerance_m": tolerance,
    }


def _digit_join_descriptors(trajectory: DigitTrajectory) -> list[dict[str, Any]]:
    joins = []
    for index, (left, right) in enumerate(zip(trajectory.boundaries, trajectory.boundaries[1:])):
        key = (trajectory.digit, index)
        if key in TANGENT_DIGIT_JOINS:
            classification = "theoretical_tangent"
        elif key in REVERSAL_DIGIT_JOINS:
            classification = "explicit_reversal"
        else:
            classification = "ordinary_corner"
        joins.append(
            {
                "label": f"{left.name}_to_{right.name}",
                "classification": classification,
                "left_index": left.end_index,
                "right_index": right.start_index,
            }
        )
    return joins


def _original_join_descriptors(task_index: int, movement_steps: int) -> list[dict[str, Any]]:
    if task_index not in {5, 8, 9}:
        return []
    half = movement_steps // 2
    return [
        {
            "label": "forward_half_to_backward_half",
            "classification": "original_concatenation_boundary",
            "left_index": half - 1,
            "right_index": half,
        }
    ]


def _rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return points @ np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64).T


def _placed_dense_digit_primitives(
    digit: int, config: GeometryConfig
) -> list[tuple[str, str, np.ndarray]]:
    library = primitive_library(config)
    current = np.zeros(2, dtype=np.float64)
    blocks = []
    for occurrence in prescribed_digit_primitives(digit, config):
        placed = _rotate(library[occurrence.template_key], occurrence.rotation_rad) + current
        blocks.append((occurrence.name, occurrence.template_key, placed))
        current = placed[-1]
    return blocks


def _high_resolution_join_angle(
    digit: int, join_index: int, config: GeometryConfig
) -> float | None:
    blocks = _placed_dense_digit_primitives(digit, config)
    return _angle_degrees(
        blocks[join_index][2][-1] - blocks[join_index][2][-2],
        blocks[join_index + 1][2][1] - blocks[join_index + 1][2][0],
    )


def _same_direction(first: np.ndarray, second: np.ndarray) -> tuple[bool, float | None]:
    first_unit = first / np.linalg.norm(first)
    second_unit = second / np.linalg.norm(second)
    tolerance = 128.0 * np.finfo(np.float64).eps
    return bool(np.allclose(first_unit, second_unit, rtol=0.0, atol=tolerance)), _angle_degrees(first, second)


def build_shared_constraint_audit(config: GeometryConfig) -> dict[str, Any]:
    """Verify frozen shared primitives and the protocol's analytic tangencies."""

    library = primitive_library(config)
    lengths = {
        name: float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        for name, points in library.items()
    }
    horizontal_equal = math.isclose(lengths["H_right"], lengths["H_left"], rel_tol=0.0, abs_tol=1e-14)
    vertical_equal = math.isclose(lengths["V_down"], lengths["V_up"], rel_tol=0.0, abs_tol=1e-14)
    diagonal_equal = math.isclose(lengths["D_up"], lengths["D_down"], rel_tol=0.0, abs_tol=1e-14)

    angle2 = math.atan2(config.D_dy, config.D_dx)
    digit2_ellipse_tangent = _rotate(np.array((-config.a, 0.0)), angle2)
    digit2_diagonal = np.array((-config.D_dx, -config.D_dy))
    digit2_tangent, digit2_analytic_angle = _same_direction(digit2_ellipse_tangent, digit2_diagonal)

    phase6 = math.atan2(config.D_dx / config.b, -config.D_dy / config.a)
    digit6_diagonal = np.array((-config.D_dx, -config.D_dy))
    digit6_ellipse_tangent = np.array(
        (-config.b * math.sin(phase6), config.a * math.cos(phase6))
    )
    digit6_tangent, digit6_analytic_angle = _same_direction(digit6_diagonal, digit6_ellipse_tangent)

    diagonal_occurrences = {
        digit: [
            occurrence.template_key
            for occurrence in prescribed_digit_primitives(digit, config)
            if occurrence.template_key in {"D_up", "D_down"}
        ]
        for digit in (2, 4, 6, 7)
    }
    digit2_half = prescribed_digit_primitives(2, config)[0]
    digit5_half = prescribed_digit_primitives(5, config)[2]
    digit8 = prescribed_digit_primitives(8, config)
    digit9 = prescribed_digit_primitives(9, config)
    digit3_first = _placed_dense_digit_primitives(3, config)[0][2]
    digit8_first = _placed_dense_digit_primitives(8, config)[0][2]
    digit0_first = _placed_dense_digit_primitives(0, config)[0][2]
    digit9_first = _placed_dense_digit_primitives(9, config)[0][2]
    ellipse_expected = {
        "half_ellipse_right_top_to_bottom": (config.a * config.global_scale, config.b * config.global_scale),
        "half_ellipse_right_bottom_to_top": (config.a * config.global_scale, config.b * config.global_scale),
        "half_ellipse_left_top_to_bottom": (config.a * config.global_scale, config.b * config.global_scale),
        "ellipse_vertical_ccw_top": (config.b * config.global_scale, config.a * config.global_scale),
        "ellipse_vertical_ccw_tangent6": (config.b * config.global_scale, config.a * config.global_scale),
        "ellipse_vertical_cw_right": (config.b * config.global_scale, config.a * config.global_scale),
        "ellipse_horizontal_cw_top": (config.a * config.global_scale, config.b * config.global_scale),
    }
    ellipse_measured = {}
    ellipse_axes_match = True
    angular_step = 2.0 * math.pi / config.arc_table_intervals
    sampling_tolerance = (
        max(config.a, config.b)
        * config.global_scale
        * (1.0 - math.cos(angular_step))
        + 128.0 * np.finfo(np.float64).eps
    )
    for key, expected_axes in ellipse_expected.items():
        points = library[key]
        x_radius = 0.5 * float(np.ptp(points[:, 0]))
        y_radius = 0.5 * float(np.ptp(points[:, 1]))
        if key.startswith("half_ellipse"):
            x_radius *= 2.0
        ellipse_measured[key] = [x_radius, y_radius]
        ellipse_axes_match &= bool(
            np.allclose(
                (x_radius, y_radius), expected_axes, rtol=0.0, atol=sampling_tolerance
            )
        )

    checks = {
        "all_horizontal_lengths_equal": horizontal_equal,
        "all_vertical_lengths_equal": vertical_equal,
        "all_diagonal_lengths_equal": diagonal_equal,
        "digit_2_and_5_share_half_ellipse_template": digit2_half.template_key == digit5_half.template_key,
        "digits_2_4_6_7_use_only_standard_diagonals": all(diagonal_occurrences.values()) and all(
            key in {"D_up", "D_down"}
            for values in diagonal_occurrences.values()
            for key in values
        ),
        "digit_2_analytic_tangency": digit2_tangent,
        "digit_6_analytic_tangency": digit6_tangent,
        "all_full_and_half_ellipses_use_shared_axes": ellipse_axes_match,
        "digit_8_uses_shared_ellipse_axes": [item.template_key for item in digit8] == [
            "half_ellipse_left_top_to_bottom",
            "ellipse_horizontal_cw_top",
            "half_ellipse_right_bottom_to_top",
        ],
        "digit_9_right_start_cw_then_v_down": [item.template_key for item in digit9] == [
            "ellipse_vertical_cw_right",
            "V_down",
        ],
        "digit_3_is_not_digit_8_prefix": bool(
            digit3_first[1, 0] > 0.0 and digit8_first[1, 0] < 0.0
        ),
        "digit_0_is_not_digit_9_prefix": not bool(
            np.allclose(digit0_first[1], digit9_first[1], rtol=0.0, atol=1e-14)
        ),
        "lobe_parameters_absent": not hasattr(config, "lobe_rx") and not hasattr(config, "lobe_ry"),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "executed_lengths_m": {
            "horizontal": lengths["H_right"],
            "vertical": lengths["V_down"],
            "diagonal": lengths["D_up"],
        },
        "diagonal_occurrences": diagonal_occurrences,
        "ellipse_axes": {
            "expected_xy_radii_m": ellipse_expected,
            "measured_xy_radii_m": ellipse_measured,
            "sampling_tolerance_m": sampling_tolerance,
        },
        "digit_2_5_half_ellipse_rigid_alignment_max_error_m": float(
            np.max(
                np.abs(
                    _rotate(
                        _rotate(library[digit2_half.template_key], digit2_half.rotation_rad),
                        -digit2_half.rotation_rad,
                    )
                    - library[digit5_half.template_key]
                )
            )
        ),
        "tangency": {
            "digit_2": {
                "analytic_angle_error_deg": digit2_analytic_angle,
                "high_resolution_finite_difference_angle_error_deg": _high_resolution_join_angle(2, 0, config),
            },
            "digit_6": {
                "analytic_angle_error_deg": digit6_analytic_angle,
                "high_resolution_finite_difference_angle_error_deg": _high_resolution_join_angle(6, 0, config),
            },
        },
    }


def build_digit_component_audit(config: GeometryConfig, reference_steps: int = 100) -> list[dict[str, Any]]:
    rows = []
    for digit in range(10):
        trajectory = build_digit_trajectory(digit, config, reference_steps)
        primitives = []
        for boundary in trajectory.boundaries:
            points = trajectory.points[boundary.start_index : boundary.end_index + 1]
            primitives.append(
                {
                    "name": boundary.name,
                    "template_key": boundary.template_key,
                    "start_index": boundary.start_index,
                    "end_index": boundary.end_index,
                    "theoretical_length_m": boundary.arc_length_m,
                    "discrete_arc_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                    "start_point_m": points[0].tolist(),
                    "end_point_m": points[-1].tolist(),
                }
            )
        metrics = trajectory_metrics(
            trajectory.points,
            dt_seconds=config.dt_seconds,
            movement_steps=trajectory.movement_intervals,
            joins=_digit_join_descriptors(trajectory),
        )
        rows.append(
            {
                "digit": digit,
                "reference_steps": reference_steps,
                "primitive_count": len(primitives),
                "primitives": primitives,
                "joins": metrics["joins"],
                "position_continuous": metrics["position_continuous"],
            }
        )
    return rows


def _load_original_env_module(repository: Path) -> tuple[types.ModuleType, str]:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{ORIGINAL_BASELINE}:envs.py"],
        check=True,
        capture_output=True,
    )
    source = result.stdout
    module = types.ModuleType("original_protocol_baseline_envs")
    module.__file__ = f"git:{ORIGINAL_BASELINE}:envs.py"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def _original_theoretical_lengths() -> dict[int, float]:
    parameter = np.linspace(0.0, 1.0, 200_001, dtype=np.float64)
    dx = np.full_like(parameter, 0.25)
    dy = 0.25 * math.pi * np.cos(2.0 * math.pi * parameter)
    sinusoid_length = float(np.trapezoid(np.sqrt(dx * dx + dy * dy), parameter))
    return {
        0: 0.25,
        1: math.pi * 0.125,
        2: math.pi * 0.125,
        3: sinusoid_length,
        4: sinusoid_length,
        5: 0.5,
        6: 2.0 * math.pi * 0.125,
        7: 2.0 * math.pi * 0.125,
        8: 2.0 * sinusoid_length,
        9: 2.0 * sinusoid_length,
    }


def _condition_grids(config: GeometryConfig) -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "training",
            "angles": training_angles(),
            "reference_steps": config.training_reference_steps,
            "testing": False,
        },
        {
            "name": "validation",
            "angles": validation_angles(),
            "reference_steps": config.validation_reference_steps,
            "testing": True,
        },
    )


def _flatten_metric_row(
    *,
    group: str,
    task_index: int,
    task_name: str,
    grid_name: str,
    direction_index: int,
    angle_rad: float,
    speed_index: int,
    reference_steps: int,
    speed_scalar_value: float,
    theoretical_arc_length_m: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    scalar_keys = (
        "target_sample_count",
        "kinematic_interval_count",
        "movement_steps",
        "movement_epoch_duration_s",
        "sampled_kinematic_duration_s",
        "discrete_arc_length_m",
        "mean_step_displacement_m",
        "minimum_step_displacement_m",
        "maximum_step_displacement_m",
        "median_nonzero_step_displacement_m",
        "maximum_to_median_step_ratio",
        "bbox_width_m",
        "bbox_height_m",
        "start_to_farthest_point_m",
        "maximum_discrete_speed_m_s",
        "mean_discrete_speed_m_s",
        "maximum_discrete_acceleration_m_s2",
        "maximum_smooth_discrete_acceleration_m_s2",
        "maximum_join_discrete_acceleration_m_s2",
        "maximum_smooth_curvature_1_m",
        "minimum_smooth_curvature_radius_m",
        "peak_smooth_curvature_sample_index",
        "maximum_smooth_normal_acceleration_m_s2",
        "peak_speed_segment_index",
        "peak_acceleration_sample_index",
        "join_count",
        "maximum_join_direction_change_deg",
        "position_continuous",
        "all_values_finite",
        "near_duplicate_segment_count",
    )
    row = {
        "group": group,
        "task_index": task_index,
        "task_name": task_name,
        "grid": grid_name,
        "direction_index": direction_index,
        "angle_deg": math.degrees(angle_rad),
        "speed_index": speed_index,
        "reference_steps": reference_steps,
        "speed_scalar": speed_scalar_value,
        "theoretical_arc_length_m": theoretical_arc_length_m,
    }
    row.update({key: metrics[key] for key in scalar_keys})
    peak_acceleration = metrics["peak_acceleration_point_m"]
    peak_curvature = metrics["peak_smooth_curvature_point_m"]
    row.update(
        {
            "peak_acceleration_x_m": peak_acceleration[0] if peak_acceleration is not None else None,
            "peak_acceleration_y_m": peak_acceleration[1] if peak_acceleration is not None else None,
            "peak_smooth_curvature_x_m": peak_curvature[0] if peak_curvature is not None else None,
            "peak_smooth_curvature_y_m": peak_curvature[1] if peak_curvature is not None else None,
            "join_classifications": ";".join(item["classification"] for item in metrics["joins"]),
            "join_direction_changes_deg": ";".join(
                "" if item["direction_change_deg"] is None else f"{item['direction_change_deg']:.12g}"
                for item in metrics["joins"]
            ),
            "near_duplicate_segment_indices": ";".join(
                str(index) for index in metrics["near_duplicate_segment_indices"]
            ),
        }
    )
    return row


def collect_geometry_condition_metrics(
    repository: Path, config: GeometryConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect the common metric table from exact original and digit sources."""

    import motornet as mn

    original_module, original_source_sha256 = _load_original_env_module(repository)
    original_lengths = _original_theoretical_lengths()
    rows: list[dict[str, Any]] = []

    for task_index, class_name in enumerate(ORIGINAL_TASK_NAMES):
        environment_class = getattr(original_module, class_name)
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        environment = environment_class(effector=effector, action_frame_stacking=0)
        for grid in _condition_grids(config):
            angles = grid["angles"]
            for speed_index, reference_steps in enumerate(grid["reference_steps"]):
                movement_steps = reference_steps if task_index < 5 else 2 * reference_steps
                environment.reset(
                    testing=grid["testing"],
                    options={
                        "batch_size": len(angles),
                        "reach_conds": np.arange(len(angles), dtype=np.int64),
                        "speed_cond": speed_index,
                        "delay_cond": 0,
                        "deterministic": True,
                    },
                )
                if environment.movement_time != movement_steps:
                    raise RuntimeError("original movement grid no longer matches baseline source")
                trajectories = environment.traj.detach().cpu().numpy()
                for direction_index, angle in enumerate(angles):
                    metrics = trajectory_metrics(
                        trajectories[direction_index],
                        dt_seconds=config.dt_seconds,
                        movement_steps=movement_steps,
                        joins=_original_join_descriptors(task_index, movement_steps),
                    )
                    rows.append(
                        _flatten_metric_row(
                            group="original",
                            task_index=task_index,
                            task_name=class_name,
                            grid_name=grid["name"],
                            direction_index=direction_index,
                            angle_rad=float(angle),
                            speed_index=speed_index,
                            reference_steps=reference_steps,
                            speed_scalar_value=1.0 - reference_steps / 150.0,
                            theoretical_arc_length_m=original_lengths[task_index],
                            metrics=metrics,
                        )
                    )

    for grid in _condition_grids(config):
        for digit in range(10):
            for speed_index, reference_steps in enumerate(grid["reference_steps"]):
                for direction_index, angle in enumerate(grid["angles"]):
                    trajectory = build_digit_trajectory(
                        digit,
                        config,
                        reference_steps,
                        spatial_angle_rad=float(angle),
                    )
                    metrics = trajectory_metrics(
                        trajectory.points,
                        dt_seconds=config.dt_seconds,
                        movement_steps=trajectory.movement_intervals,
                        joins=_digit_join_descriptors(trajectory),
                    )
                    rows.append(
                        _flatten_metric_row(
                            group="digit",
                            task_index=digit,
                            task_name=f"digit_{digit}",
                            grid_name=grid["name"],
                            direction_index=direction_index,
                            angle_rad=float(angle),
                            speed_index=speed_index,
                            reference_steps=reference_steps,
                            speed_scalar_value=trajectory.speed_scalar,
                            theoretical_arc_length_m=trajectory.arc_length_m,
                            metrics=metrics,
                        )
                    )

    provenance = {
        "original_baseline_commit": ORIGINAL_BASELINE,
        "original_envs_source_sha256": original_source_sha256,
        "original_trajectory_source": "baseline env classes executed through MotorNet reset",
        "digit_trajectory_source": "digit_writing.geometry.build_digit_trajectory",
        "shared_metric_function": "digit_writing.protocol_audit.trajectory_metrics",
    }
    return rows, provenance


def _workspace_metrics_from_arrays(
    points: np.ndarray,
    joint_positions: np.ndarray,
    reachable: np.ndarray,
    reconstruction_error: np.ndarray,
    *,
    dt_seconds: float,
    link1: float,
    link2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
) -> dict[str, Any]:
    radius = np.linalg.norm(points, axis=1)
    inner_margin = radius - abs(link1 - link2)
    outer_margin = link1 + link2 - radius
    lower_margin = joint_positions - lower
    upper_margin = upper - joint_positions
    all_joint_margins = np.stack((lower_margin, upper_margin), axis=-1)
    flat_margin_index = int(np.argmin(all_joint_margins))
    sample_index, joint_index, side_index = np.unravel_index(
        flat_margin_index, all_joint_margins.shape
    )

    joint_steps = np.diff(joint_positions, axis=0)
    joint_velocity = joint_steps / dt_seconds
    velocity_margin = np.minimum(
        joint_velocity - velocity_lower,
        velocity_upper - joint_velocity,
    )
    max_step_flat = int(np.argmax(np.abs(joint_steps))) if joint_steps.size else 0
    step_sample_index, step_joint_index = (
        np.unravel_index(max_step_flat, joint_steps.shape) if joint_steps.size else (0, 0)
    )
    finite = bool(
        np.isfinite(points).all()
        and np.isfinite(joint_positions).all()
        and np.isfinite(joint_velocity).all()
        and np.isfinite(reconstruction_error).all()
    )
    minimum_joint_margin = float(all_joint_margins[sample_index, joint_index, side_index])
    branch_jump = bool(np.any(np.abs(joint_steps) > math.pi))
    velocity_within_limits = bool(velocity_margin.min() >= 0.0) if velocity_margin.size else True
    passed = bool(
        finite
        and reachable.all()
        and inner_margin.min() >= 0.0
        and outer_margin.min() >= 0.0
        and minimum_joint_margin >= 0.0
        and not branch_jump
        and velocity_within_limits
        and reconstruction_error.max() <= FK_TOLERANCE_M
    )
    return {
        "sample_count": len(points),
        "all_inverse_solutions_reachable": bool(reachable.all()),
        "minimum_inner_radial_margin_m": float(inner_margin.min()),
        "minimum_outer_radial_margin_m": float(outer_margin.min()),
        "minimum_joint_margin_rad": minimum_joint_margin,
        "limiting_joint_index": int(joint_index),
        "limiting_joint_side": "lower" if side_index == 0 else "upper",
        "limiting_sample_index": int(sample_index),
        "limiting_time_s": float(sample_index * dt_seconds),
        "limiting_hand_x_m": float(points[sample_index, 0]),
        "limiting_hand_y_m": float(points[sample_index, 1]),
        "limiting_joint0_rad": float(joint_positions[sample_index, 0]),
        "limiting_joint1_rad": float(joint_positions[sample_index, 1]),
        "maximum_motor_fk_error_m": float(reconstruction_error.max()),
        "maximum_abs_joint_step_rad": float(np.abs(joint_steps).max()) if joint_steps.size else 0.0,
        "maximum_joint_step_sample_index": int(step_sample_index),
        "maximum_joint_step_joint_index": int(step_joint_index),
        "joint_branch_jump_gt_pi": branch_jump,
        "maximum_abs_joint_velocity_rad_s": float(np.abs(joint_velocity).max()) if joint_velocity.size else 0.0,
        "minimum_joint_velocity_limit_margin_rad_s": float(velocity_margin.min()) if velocity_margin.size else None,
        "joint_velocity_within_motornet_limits": velocity_within_limits,
        "all_values_finite": finite,
        "passed_physical_and_numerical_constraints": passed,
    }


def build_digit_workspace_audit(
    config: GeometryConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit all digit direction/speed rows using the actual MotorNet arm."""

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
        initial_cartesian = skeleton.joint2cartesian(
            torch.as_tensor(initial_joint_state[None, :], dtype=torch.float32)
        )
    anchor = initial_cartesian.detach().cpu().numpy()[0, :2].astype(np.float64)

    rows: list[dict[str, Any]] = []
    for grid in _condition_grids(config):
        angles = grid["angles"]
        for digit in range(10):
            for speed_index, reference_steps in enumerate(grid["reference_steps"]):
                trajectories = [
                    build_digit_trajectory(
                        digit,
                        config,
                        reference_steps,
                        spatial_angle_rad=float(angle),
                        anchor=anchor,
                    )
                    for angle in angles
                ]
                point_count = len(trajectories[0].points)
                combined = np.concatenate([item.points for item in trajectories], axis=0)
                joint_positions, reachable = _inverse_kinematics(combined, link1, link2)
                reconstructed = _motor_forward_kinematics(skeleton, joint_positions)
                reconstruction_error = np.linalg.norm(reconstructed - combined, axis=1)
                for direction_index, angle in enumerate(angles):
                    start = direction_index * point_count
                    stop = start + point_count
                    metrics = _workspace_metrics_from_arrays(
                        combined[start:stop],
                        joint_positions[start:stop],
                        reachable[start:stop],
                        reconstruction_error[start:stop],
                        dt_seconds=config.dt_seconds,
                        link1=link1,
                        link2=link2,
                        lower=lower,
                        upper=upper,
                        velocity_lower=velocity_lower,
                        velocity_upper=velocity_upper,
                    )
                    rows.append(
                        {
                            "digit": digit,
                            "grid": grid["name"],
                            "direction_index": direction_index,
                            "angle_deg": math.degrees(float(angle)),
                            "speed_index": speed_index,
                            "reference_steps": reference_steps,
                            "speed_scalar": trajectories[direction_index].speed_scalar,
                            "movement_steps": trajectories[direction_index].movement_intervals,
                            **metrics,
                        }
                    )

    dangerous = min(rows, key=lambda row: row["minimum_joint_margin_rad"])
    by_direction = []
    for direction_index, angle in enumerate(validation_angles()):
        selected = [
            row for row in rows
            if row["grid"] == "validation" and row["direction_index"] == direction_index
        ]
        by_direction.append(
            {
                "direction_index": direction_index,
                "angle_deg": math.degrees(float(angle)),
                "minimum_joint_margin_rad": min(row["minimum_joint_margin_rad"] for row in selected),
                "minimum_inner_radial_margin_m": min(row["minimum_inner_radial_margin_m"] for row in selected),
                "minimum_outer_radial_margin_m": min(row["minimum_outer_radial_margin_m"] for row in selected),
            }
        )
    summary = {
        "motornet_version": mn.__version__,
        "device": "cpu",
        "effector": "RigidTendonArm26(MujocoHillMuscle)",
        "link_lengths_m": [link1, link2],
        "joint_position_lower_bounds_rad": lower.tolist(),
        "joint_position_upper_bounds_rad": upper.tolist(),
        "joint_velocity_lower_bounds_rad_s": velocity_lower.tolist(),
        "joint_velocity_upper_bounds_rad_s": velocity_upper.tolist(),
        "initial_joint_state": initial_joint_state.tolist(),
        "anchor_m": anchor.tolist(),
        "condition_count": len(rows),
        "training_condition_count": sum(row["grid"] == "training" for row in rows),
        "validation_condition_count": sum(row["grid"] == "validation" for row in rows),
        "most_dangerous_condition": dangerous,
        "direction_summary": by_direction,
        "all_conditions_passed": bool(
            all(row["passed_physical_and_numerical_constraints"] for row in rows)
        ),
    }
    return rows, summary


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
        return bool(np.isfinite(array).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def run_digit_environment_rollout_audit(config: GeometryConfig) -> dict[str, Any]:
    """Run every digit/direction/speed group without a policy or optimization."""

    import motornet as mn
    import torch

    from envs import (
        DlyFigure8,
        DlyFigure8Inv,
        DlyFullCircleCClk,
        DlyFullCircleClk,
        DlyFullReach,
        DlyHalfCircleCClk,
        DlyHalfCircleClk,
        DlyHalfReach,
        DlySinusoid,
        DlySinusoidInv,
    )
    from losses import l1_dist

    environment_classes = (
        DlyHalfReach,
        DlyHalfCircleClk,
        DlyHalfCircleCClk,
        DlySinusoid,
        DlySinusoidInv,
        DlyFullReach,
        DlyFullCircleClk,
        DlyFullCircleCClk,
        DlyFigure8,
        DlyFigure8Inv,
    )
    rows = []
    all_passed = True
    for digit, environment_class in enumerate(environment_classes):
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        environment = environment_class(effector=effector, action_frame_stacking=0)
        for grid in _condition_grids(config):
            directions = np.arange(len(grid["angles"]), dtype=np.int64)
            for speed_index, reference_steps in enumerate(grid["reference_steps"]):
                obs, info = environment.reset(
                    testing=grid["testing"],
                    options={
                        "batch_size": len(directions),
                        "reach_conds": directions,
                        "speed_cond": speed_index,
                        "delay_cond": 0,
                        "deterministic": True,
                    },
                )
                movement_start, movement_end = environment.epoch_bounds["movement"]
                hold_start, hold_end = environment.epoch_bounds["hold"]
                observations_valid = tuple(obs.shape) == (len(directions), 28)
                trajectory_length_valid = environment.traj.shape[1] == environment.movement_intervals + 1
                timing_valid = bool(
                    movement_end - movement_start == environment.movement_intervals
                    and hold_start == movement_end
                    and hold_end == environment.max_ep_duration + 1
                )
                contract_valid = bool(
                    environment.current_digit == digit
                    and environment.speed_cond == speed_index
                    and environment.delay_time == config.delay_steps[0]
                    and np.array_equal(environment.direction_indices, directions)
                    and torch.all(environment.rule_input[:, digit] == 1.0)
                    and int(torch.count_nonzero(environment.rule_input).item()) == len(directions)
                )
                actual_positions = [info["states"]["fingertip"][:, None, :]]
                targets = [info["goal"][:, None, :]]
                finite = _all_finite(obs) and _all_finite(info)
                terminated = False
                timestep = 0
                reward_contract = True
                while not terminated:
                    action = torch.zeros(
                        (len(directions), environment.action_space.shape[0]),
                        dtype=torch.float32,
                    )
                    obs, reward, terminated, info = environment.step(timestep, action)
                    observations_valid &= tuple(obs.shape) == (len(directions), 28)
                    reward_contract &= reward is None or _all_finite(reward)
                    finite &= _all_finite(obs) and _all_finite(info)
                    actual_positions.append(info["states"]["fingertip"][:, None, :])
                    targets.append(info["goal"][:, None, :])
                    timestep += 1

                actual_tensor = torch.cat(actual_positions, dim=1)
                target_tensor = torch.cat(targets, dim=1)
                diagnostic_l1 = l1_dist(actual_tensor, target_tensor)
                loss_inputs_finite = bool(
                    torch.isfinite(actual_tensor).all()
                    and torch.isfinite(target_tensor).all()
                    and torch.isfinite(diagnostic_l1)
                )
                rollout_length_valid = timestep == hold_end
                passed = bool(
                    observations_valid
                    and trajectory_length_valid
                    and timing_valid
                    and contract_valid
                    and reward_contract
                    and finite
                    and loss_inputs_finite
                    and rollout_length_valid
                )
                all_passed &= passed
                rows.append(
                    {
                        "digit": digit,
                        "grid": grid["name"],
                        "speed_index": speed_index,
                        "reference_steps": reference_steps,
                        "direction_count": len(directions),
                        "movement_steps": environment.movement_intervals,
                        "episode_steps": hold_end,
                        "observation_shape_valid": observations_valid,
                        "trajectory_length_valid": trajectory_length_valid,
                        "epoch_timing_valid": timing_valid,
                        "input_contract_valid": contract_valid,
                        "reward_contract_valid": reward_contract,
                        "all_rollout_values_finite": finite,
                        "loss_inputs_finite": loss_inputs_finite,
                        "diagnostic_zero_action_l1": float(diagnostic_l1.detach().cpu()),
                        "rollout_length_valid": rollout_length_valid,
                        "passed": passed,
                    }
                )

    shortest = min(rows, key=lambda row: row["movement_steps"])
    longest = max(rows, key=lambda row: row["movement_steps"])
    return {
        "group_count": len(rows),
        "condition_count": sum(row["direction_count"] for row in rows),
        "batch_padding": {
            "status": "not_applicable",
            "reason": "digit, speed, and delay are shared within each batch, so every sample has one episode length",
        },
        "policy_or_network_used": False,
        "optimizer_used": False,
        "checkpoint_loaded": False,
        "formal_training_started": False,
        "shortest_group": shortest,
        "longest_group": longest,
        "groups": rows,
        "passed": bool(all_passed),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fields})


def _matched_medium_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row["grid"] == "training"
        and row["direction_index"] == 0
        and row["speed_index"] == 1
    ]


def build_comparison_summary(
    rows: Sequence[Mapping[str, Any]], config: GeometryConfig
) -> dict[str, Any]:
    medium = _matched_medium_rows(rows)
    original = [row for row in medium if row["group"] == "original"]
    digits = [row for row in medium if row["group"] == "digit"]
    metrics = (
        "theoretical_arc_length_m",
        "movement_steps",
        "bbox_width_m",
        "bbox_height_m",
        "start_to_farthest_point_m",
        "maximum_discrete_speed_m_s",
        "maximum_smooth_discrete_acceleration_m_s2",
        "maximum_smooth_curvature_1_m",
        "maximum_smooth_normal_acceleration_m_s2",
        "maximum_join_direction_change_deg",
    )
    ranges = {}
    outside = []
    inside = []
    for metric in metrics:
        original_values = [float(row[metric]) for row in original if row[metric] is not None]
        digit_values = [float(row[metric]) for row in digits if row[metric] is not None]
        original_min = min(original_values) if original_values else None
        original_max = max(original_values) if original_values else None
        ranges[metric] = {
            "original_min": original_min,
            "original_max": original_max,
            "digit_min": min(digit_values) if digit_values else None,
            "digit_max": max(digit_values) if digit_values else None,
        }
        if original_values:
            for row in digits:
                value = row[metric]
                if value is None:
                    continue
                item = {
                    "digit": row["task_index"],
                    "metric": metric,
                    "value": value,
                    "original_min": original_min,
                    "original_max": original_max,
                }
                (inside if original_min <= float(value) <= original_max else outside).append(item)

    original_steps = [float(row["movement_steps"]) for row in original]
    digit_steps = [float(row["movement_steps"]) for row in digits]
    duration_ratios = {
        "original_longest_to_shortest": max(original_steps) / min(original_steps),
        "digit_longest_to_shortest": max(digit_steps) / min(digit_steps),
    }
    training_step_summary = []
    for speed_index in range(3):
        selected = [
            row for row in rows
            if row["grid"] == "training"
            and row["direction_index"] == 0
            and row["speed_index"] == speed_index
        ]
        for group in ("original", "digit"):
            group_rows = [row for row in selected if row["group"] == group]
            steps = np.asarray([row["movement_steps"] for row in group_rows], dtype=np.float64)
            shortest = min(group_rows, key=lambda row: row["movement_steps"])
            longest = max(group_rows, key=lambda row: row["movement_steps"])
            training_step_summary.append(
                {
                    "group": group,
                    "speed_index": speed_index,
                    "reference_steps": int(group_rows[0]["reference_steps"]),
                    "minimum_movement_steps": int(steps.min()),
                    "median_movement_steps": float(np.median(steps)),
                    "maximum_movement_steps": int(steps.max()),
                    "longest_to_shortest_ratio": float(steps.max() / steps.min()),
                    "shortest_task": shortest["task_name"],
                    "longest_task": longest["task_name"],
                    "shortest_trial_steps_min_delay": int(
                        config.stable_steps + min(config.delay_steps) + steps.min() + config.hold_steps
                    ),
                    "longest_trial_steps_max_delay": int(
                        config.stable_steps + max(config.delay_steps) + steps.max() + config.hold_steps
                    ),
                }
            )
    peak_acceleration = max(
        medium, key=lambda row: float(row["maximum_discrete_acceleration_m_s2"])
    )
    peak_smooth_curvature = max(
        medium, key=lambda row: float(row["maximum_smooth_curvature_1_m"])
    )
    return {
        "comparison_condition": {
            "grid": "training",
            "direction_index": 0,
            "speed_index": 1,
            "speed_scalar": 1.0 / 3.0,
        },
        "ranges": ranges,
        "digit_values_inside_original_observed_range": inside,
        "digit_values_outside_original_observed_range": outside,
        "movement_step_ratios": duration_ratios,
        "training_step_summary": training_step_summary,
        "overall_peak_discrete_acceleration_medium": peak_acceleration,
        "overall_peak_smooth_curvature_medium": peak_smooth_curvature,
        "medium_rows": medium,
    }


def build_conclusions(
    *,
    shared: Mapping[str, Any],
    component_rows: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    workspace: Mapping[str, Any],
    rollout: Mapping[str, Any],
    geometry_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    confirmed = []
    relative_normal = []
    risks = []
    blockers = []

    if shared["passed"]:
        confirmed.append("共享横线、竖线、斜线和椭圆参数一致，数字 2/6 的解析相切约束成立。")
    else:
        failed = [name for name, passed in shared["checks"].items() if not passed]
        blockers.append(f"共享几何约束失败：{', '.join(failed)}。")

    if all(row["position_continuous"] for row in component_rows):
        confirmed.append("十个数字的所有基元连接位置连续。")
    else:
        blockers.append("至少一个数字存在基元连接位置不连续。")

    digit_geometry = [row for row in geometry_rows if row["group"] == "digit"]
    if all(row["all_values_finite"] for row in digit_geometry):
        confirmed.append("全部数字几何条件均无 NaN 或 Inf。")
    else:
        blockers.append("至少一个数字几何条件包含 NaN 或 Inf。")
    if any(row["near_duplicate_segment_count"] for row in digit_geometry):
        blockers.append("至少一个数字 movement 轨迹含非预期连续重复点。")

    if workspace["all_conditions_passed"]:
        confirmed.append("全部数字训练与验证方向/速度条件均满足 MotorNet 可达性、关节和数值约束。")
    else:
        blockers.append("至少一个数字条件不可达、关节越界、关节解跳变或违反 MotorNet 数值约束。")

    if rollout["passed"]:
        confirmed.append("所有数字方向/速度组均可无训练完整 rollout，28 维输入和 loss 输入保持有限。")
    else:
        blockers.append("至少一个数字环境条件无法完整 rollout 或违反 28 维输入合同。")

    ranges = comparison["ranges"]
    for metric, values in ranges.items():
        if values["digit_min"] is None or values["original_min"] is None:
            continue
        if (
            values["original_min"] <= values["digit_min"] <= values["original_max"]
            and values["original_min"] <= values["digit_max"] <= values["original_max"]
        ):
            relative_normal.append(
                f"{metric} 的数字范围 [{values['digit_min']:.6g}, {values['digit_max']:.6g}] "
                f"位于原任务观测范围 [{values['original_min']:.6g}, {values['original_max']:.6g}] 内。"
            )

    outside_by_metric: dict[str, list[Mapping[str, Any]]] = {}
    for item in comparison["digit_values_outside_original_observed_range"]:
        outside_by_metric.setdefault(item["metric"], []).append(item)
    for metric, items in outside_by_metric.items():
        digits = sorted({int(item["digit"]) for item in items})
        values = ranges[metric]
        risks.append(
            f"数字 {digits} 的 {metric} 超出原任务中速规定方向观测范围 "
            f"[{values['original_min']:.6g}, {values['original_max']:.6g}]；这只标记相对风险，不构成物理失败。"
        )
    ratios = comparison["movement_step_ratios"]
    if ratios["digit_longest_to_shortest"] > ratios["original_longest_to_shortest"]:
        risks.append(
            "数字最长/最短 movement 步数比 "
            f"{ratios['digit_longest_to_shortest']:.6g} 高于原任务的 "
            f"{ratios['original_longest_to_shortest']:.6g}；不设置人为失败阈值。"
        )
    if not risks:
        relative_normal.append("在已比较指标中，没有数字值超出 original_repo 的观测范围。")

    return {
        "A_confirmed_no_problem": confirmed,
        "B_no_obvious_anomaly_relative_to_original": relative_normal,
        "C_risk_not_yet_unreasonable": risks,
        "D_blocking_must_fix": blockers,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.9g}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_format_number(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown_report(path: Path, audit: Mapping[str, Any]) -> None:
    fixed = audit["fixed_parameters"]
    comparison = audit["comparison"]
    workspace = audit["workspace_summary"]
    rollout = audit["environment_rollout"]
    components = audit["digit_components"]
    conclusions = audit["conclusions"]

    medium_rows = comparison["medium_rows"]
    comparison_table = _markdown_table(
        (
            "组",
            "任务",
            "理论弧长 m",
            "离散弧长 m",
            "movement 步",
            "最大平滑曲率 1/m",
            "最大平滑目标加速度 m/s²",
            "最大连接转角 °",
        ),
        (
            (
                row["group"],
                row["task_name"],
                row["theoretical_arc_length_m"],
                row["discrete_arc_length_m"],
                row["movement_steps"],
                row["maximum_smooth_curvature_1_m"],
                row["maximum_smooth_discrete_acceleration_m_s2"],
                row["maximum_join_direction_change_deg"],
            )
            for row in medium_rows
        ),
    )

    component_table = _markdown_table(
        ("数字", "顺序", "理论长度 m", "离散长度 m", "连接最大误差 m"),
        (
            (
                item["digit"],
                " → ".join(primitive["name"] for primitive in item["primitives"]),
                sum(primitive["theoretical_length_m"] for primitive in item["primitives"]),
                sum(primitive["discrete_arc_length_m"] for primitive in item["primitives"]),
                max((join["position_error_m"] for join in item["joins"]), default=0.0),
            )
            for item in components
        ),
    )
    component_details = []
    for item in components:
        primitive_rows = []
        for order, primitive in enumerate(item["primitives"], start=1):
            primitive_rows.append(
                (
                    order,
                    primitive["name"],
                    primitive["template_key"],
                    primitive["theoretical_length_m"],
                    primitive["discrete_arc_length_m"],
                    primitive["start_point_m"],
                    primitive["end_point_m"],
                )
            )
        detail = [
            f"### 数字 {item['digit']}",
            "",
            _markdown_table(
                ("顺序", "基元", "模板", "理论长度 m", "离散长度 m", "起点 m", "终点 m"),
                primitive_rows,
            ),
        ]
        if item["joins"]:
            detail.extend(
                (
                    "",
                    _markdown_table(
                        ("连接", "分类", "位置误差 m", "方向变化 °"),
                        (
                            (
                                join["label"],
                                join["classification"],
                                join["position_error_m"],
                                join["direction_change_deg"],
                            )
                            for join in item["joins"]
                        ),
                    ),
                )
            )
        component_details.append("\n".join(detail))

    conclusion_sections = []
    labels = (
        ("A_confirmed_no_problem", "A. 已确认没有问题"),
        ("B_no_obvious_anomaly_relative_to_original", "B. 相对 original_repo 没有明显异常"),
        ("C_risk_not_yet_unreasonable", "C. 存在风险但尚不能判定不合理"),
        ("D_blocking_must_fix", "D. 必须修改的阻断性问题"),
    )
    for key, title in labels:
        items = conclusions[key]
        body = "\n".join(f"- {item}" for item in items) if items else "- 无。"
        conclusion_sections.append(f"### {title}\n\n{body}")

    report = f"""# 当前数字轨迹几何与时间尺度合理性审计

## 审计边界

本审计没有修改任何轨迹、基元、速度、损失或训练协议，没有加载 checkpoint，
没有运行开发 seed 或正式训练。原任务直接执行
`{ORIGINAL_BASELINE}:envs.py`；数字任务直接调用当前冻结生成器。两组轨迹统一
进入 `trajectory_metrics`。

开发 seed 状态：**未运行，等待 Phase D 完成和 Phase E 人工授权**。

## 固定参数（从代码读取）

{_markdown_table(("参数", "原始值", "执行值/说明"), (
    ("dt", fixed['dt_seconds'], "second"),
    ("global_scale", fixed['global_scale'], "uniform"),
    ("H", fixed['H_raw_m'], fixed['H_executed_m']),
    ("V", fixed['V_raw_m'], fixed['V_executed_m']),
    ("D_dx", fixed['D_dx_raw_m'], fixed['D_dx_executed_m']),
    ("D_dy", fixed['D_dy_raw_m'], fixed['D_dy_executed_m']),
    ("D length", fixed['diagonal_length_raw_m'], fixed['diagonal_length_executed_m']),
    ("D unoriented angle", fixed['diagonal_unoriented_angle_deg'], "degrees"),
    ("ellipse a", fixed['ellipse_a_raw_m'], fixed['ellipse_a_executed_m']),
    ("ellipse b", fixed['ellipse_b_raw_m'], fixed['ellipse_b_executed_m']),
    ("a:b", fixed['ellipse_axis_ratio'], "major:minor"),
    ("training reference steps", fixed['digit_training_reference_steps'], "digit speed reference"),
    ("validation reference steps", fixed['digit_validation_reference_steps'], "digit speed reference"),
))}

数字训练物理速度：`{fixed['digit_training_physical_speed_m_s']}` m/s。原仓库训练方向为
8 个、验证方向为 32 个；原任务前五类的训练 movement 步数为 `[50,100,150]`，
后五类为 `[100,200,300]`，验证分别为 `50..140` 和 `100..280`。
执行后椭圆的解析最大曲率为 `{fixed['ellipse_analytic_maximum_curvature_1_m']}` 1/m，
最小曲率半径为 `{fixed['ellipse_analytic_minimum_curvature_radius_m']}` m；三个训练速度的
解析最大法向加速度为
`{fixed['ellipse_training_theoretical_maximum_normal_acceleration_m_s2']}` m/s²。

## 方法说明

- `movement_steps` 是环境 movement epoch 的实际步数；
- `kinematic_interval_count` 是相邻目标样本间隔数，两者分别记录；
- 原 full-reach/Figure-8 的半程拼接重复点按原代码保留并标注；
- 平滑曲率排除基元连接附近样本，连接转角和连接加速度单独报告；
- 相切约束同时使用解析方向和 20,000 区间模板的高分辨率有限差分；
- 工作空间使用 MotorNet `RigidTendonArm26(MujocoHillMuscle)`、实际关节限位和正肘 IK；
- 只以物理/数值约束判定阻断，超出原任务观测范围只进入风险项。

## 数字基元组成

{component_table}

{chr(10).join(component_details)}

数字 2 解析相切误差为
`{audit['shared_constraints']['tangency']['digit_2']['analytic_angle_error_deg']}`°，
高分辨率有限差分误差为
`{audit['shared_constraints']['tangency']['digit_2']['high_resolution_finite_difference_angle_error_deg']}`°。
数字 6 对应值分别为
`{audit['shared_constraints']['tangency']['digit_6']['analytic_angle_error_deg']}`° 和
`{audit['shared_constraints']['tangency']['digit_6']['high_resolution_finite_difference_angle_error_deg']}`°。

## 原任务与数字任务统一比较

下表为训练网格、规定方向、中速条件。完整的 6,880 条方向/速度记录位于
`digit_vs_original_geometry_metrics.csv`。

{comparison_table}

### 三个训练速度的时间尺度

{_markdown_table(
    ("组", "速度索引", "参考步", "最短 movement", "中位 movement", "最长 movement", "最长/最短", "最短 trial（最短 delay）", "最长 trial（最长 delay）"),
    (
        (
            row['group'], row['speed_index'], row['reference_steps'],
            row['minimum_movement_steps'], row['median_movement_steps'],
            row['maximum_movement_steps'], row['longest_to_shortest_ratio'],
            row['shortest_trial_steps_min_delay'], row['longest_trial_steps_max_delay'],
        )
        for row in comparison['training_step_summary']
    ),
)}

原任务最长/最短 movement 步数比为
`{comparison['movement_step_ratios']['original_longest_to_shortest']}`，数字任务为
`{comparison['movement_step_ratios']['digit_longest_to_shortest']}`。
中速规定方向的全体最大离散目标加速度出现在
`{comparison['overall_peak_discrete_acceleration_medium']['group']}` 的
`{comparison['overall_peak_discrete_acceleration_medium']['task_name']}`，样本
`{comparison['overall_peak_discrete_acceleration_medium']['peak_acceleration_sample_index']}`；
最大平滑曲率出现在
`{comparison['overall_peak_smooth_curvature_medium']['group']}` 的
`{comparison['overall_peak_smooth_curvature_medium']['task_name']}`，样本
`{comparison['overall_peak_smooth_curvature_medium']['peak_smooth_curvature_sample_index']}`。

## MotorNet 工作空间

- 条件数：{workspace['condition_count']}（训练 {workspace['training_condition_count']}，验证 {workspace['validation_condition_count']}）；
- 全部条件通过：{workspace['all_conditions_passed']}；
- 最危险条件：数字 {workspace['most_dangerous_condition']['digit']}，
  {workspace['most_dangerous_condition']['grid']} 方向
  {workspace['most_dangerous_condition']['direction_index']}，速度
  {workspace['most_dangerous_condition']['speed_index']}，movement 样本
  {workspace['most_dangerous_condition']['limiting_sample_index']}；
- 最小关节余量：{workspace['most_dangerous_condition']['minimum_joint_margin_rad']} rad；
- 手部坐标：({workspace['most_dangerous_condition']['limiting_hand_x_m']},
  {workspace['most_dangerous_condition']['limiting_hand_y_m']}) m；
- 关节角：({workspace['most_dangerous_condition']['limiting_joint0_rad']},
  {workspace['most_dangerous_condition']['limiting_joint1_rad']}) rad。

## 无训练环境 rollout

- 组数：{rollout['group_count']}；方向条件总数：{rollout['condition_count']}；
- 全部通过：{rollout['passed']}；
- 网络/优化器/checkpoint：均未使用；
- batch padding：不适用，同一 batch 的 digit/speed/delay 一致，因此 episode 等长；
- loss 检查只计算零动作 fingertip/target 的有限性和诊断 L1，不执行反向传播。

## 结论

{chr(10).join(conclusion_sections)}

## 文件

- `digit_geometry_protocol_audit.json`：方法、固定参数、约束、汇总和结论；
- `digit_vs_original_geometry_metrics.csv`：原任务与数字任务统一指标表；
- `digit_condition_workspace_audit.csv`：数字条件逐行 MotorNet 工作空间结果；
- `digit_geometry_protocol_audit.png`：审计总览；开发 seed 面板明确标记未运行。
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def generate_summary_figure(
    path: Path,
    config: GeometryConfig,
    geometry_rows: Sequence[Mapping[str, Any]],
    workspace_summary: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    medium = _matched_medium_rows(geometry_rows)
    original = [row for row in medium if row["group"] == "original"]
    digits = [row for row in medium if row["group"] == "digit"]

    figure = plt.figure(figsize=(20, 22), constrained_layout=True)
    outer = figure.add_gridspec(4, 1, height_ratios=(2.3, 1.0, 1.0, 0.8))
    digit_grid = outer[0].subgridspec(2, 5)
    digit_trajectories = [build_digit_trajectory(digit, config, 100) for digit in range(10)]
    all_digit_points = np.concatenate([item.points for item in digit_trajectories], axis=0)
    axis_limit = 1.08 * float(np.max(np.abs(all_digit_points)))
    for digit in range(10):
        axis = figure.add_subplot(digit_grid[digit // 5, digit % 5])
        trajectory = digit_trajectories[digit]
        axis.plot(trajectory.points[:, 0], trajectory.points[:, 1], color="black", linewidth=1.4)
        axis.scatter(*trajectory.points[0], marker="*", color="#15803d", s=60, zorder=3)
        axis.set_title(f"Digit {digit}")
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")

    first_metrics = outer[1].subgridspec(1, 3)
    axis = figure.add_subplot(first_metrics[0, 0])
    axis.boxplot(
        ([row["theoretical_arc_length_m"] for row in original],
         [row["theoretical_arc_length_m"] for row in digits]),
        tick_labels=("original", "digits"),
    )
    axis.set_title("Arc length distribution (medium)")
    axis.set_ylabel("m")
    axis.grid(axis="y", alpha=0.25)

    axis = figure.add_subplot(first_metrics[0, 1])
    training = [row for row in geometry_rows if row["grid"] == "training" and row["direction_index"] == 0]
    axis.boxplot(
        ([row["movement_steps"] for row in training if row["group"] == "original"],
         [row["movement_steps"] for row in training if row["group"] == "digit"]),
        tick_labels=("original", "digits"),
    )
    axis.set_title("Movement-step distribution (3 speeds)")
    axis.set_ylabel("steps")
    axis.grid(axis="y", alpha=0.25)

    axis = figure.add_subplot(first_metrics[0, 2])
    width = 0.38
    indices = np.arange(10)
    axis.bar(
        indices - width / 2,
        [row["minimum_smooth_curvature_radius_m"] or 0.0 for row in original],
        width,
        label="original",
    )
    axis.bar(
        indices + width / 2,
        [row["minimum_smooth_curvature_radius_m"] or 0.0 for row in digits],
        width,
        label="digits",
    )
    axis.set_title("Minimum smooth curvature radius")
    axis.set_xlabel("task/digit index")
    axis.set_ylabel("m (0 means straight-only)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    second_metrics = outer[2].subgridspec(1, 3)
    axis = figure.add_subplot(second_metrics[0, 0])
    axis.bar(
        indices - width / 2,
        [row["maximum_smooth_discrete_acceleration_m_s2"] for row in original],
        width,
        label="original",
    )
    axis.bar(
        indices + width / 2,
        [row["maximum_smooth_discrete_acceleration_m_s2"] for row in digits],
        width,
        label="digits",
    )
    axis.set_title("Maximum smooth target acceleration")
    axis.set_xlabel("task/digit index")
    axis.set_ylabel("m/s²")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    axis = figure.add_subplot(second_metrics[0, 1])
    directions = workspace_summary["direction_summary"]
    axis.plot(
        [row["angle_deg"] for row in directions],
        [row["minimum_joint_margin_rad"] for row in directions],
        marker="o",
        linewidth=1.2,
    )
    axis.set_title("Minimum joint margin by validation direction")
    axis.set_xlabel("direction (deg)")
    axis.set_ylabel("rad")
    axis.grid(alpha=0.25)

    axis = figure.add_subplot(second_metrics[0, 2])
    axis.axis("off")
    axis.text(
        0.5,
        0.55,
        "Development seed: NOT RUN",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
    )
    axis.text(
        0.5,
        0.38,
        "Await Phase D completion and Phase E human authorization",
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )

    final_axis = figure.add_subplot(outer[3])
    final_axis.axis("off")
    final_axis.text(
        0.0,
        0.9,
        "Interpretation rules",
        fontsize=15,
        weight="bold",
        va="top",
    )
    final_axis.text(
        0.0,
        0.72,
        "• Smooth curvature excludes primitive joins.  Join direction changes and join acceleration are reported separately.\n"
        "• Relative risk means outside the original observed range; it is not an invented pass/fail threshold.\n"
        "• Workspace conclusions use the actual MotorNet arm and joint limits.",
        fontsize=12,
        va="top",
    )
    figure.suptitle("Digit geometry protocol audit: original tasks vs digits", fontsize=20)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _fixed_parameters(config: GeometryConfig) -> dict[str, Any]:
    diagonal_raw = math.hypot(config.D_dx, config.D_dy)
    ellipse_maximum_curvature = config.a / (
        config.b * config.b * config.global_scale
    )
    training_speeds = [
        config.reach_distance_m / (steps * config.dt_seconds)
        for steps in config.training_reference_steps
    ]
    return {
        "dt_seconds": config.dt_seconds,
        "global_scale": config.global_scale,
        "H_raw_m": config.H,
        "H_executed_m": config.H * config.global_scale,
        "V_raw_m": config.V,
        "V_executed_m": config.V * config.global_scale,
        "D_dx_raw_m": config.D_dx,
        "D_dx_executed_m": config.D_dx * config.global_scale,
        "D_dy_raw_m": config.D_dy,
        "D_dy_executed_m": config.D_dy * config.global_scale,
        "diagonal_length_raw_m": diagonal_raw,
        "diagonal_length_executed_m": diagonal_raw * config.global_scale,
        "diagonal_unoriented_angle_deg": math.degrees(math.atan2(config.D_dy, config.D_dx)),
        "ellipse_a_raw_m": config.a,
        "ellipse_a_executed_m": config.a * config.global_scale,
        "ellipse_b_raw_m": config.b,
        "ellipse_b_executed_m": config.b * config.global_scale,
        "ellipse_axis_ratio": config.a / config.b,
        "ellipse_analytic_maximum_curvature_1_m": ellipse_maximum_curvature,
        "ellipse_analytic_minimum_curvature_radius_m": (
            config.b * config.b * config.global_scale / config.a
        ),
        "digit_training_reference_steps": list(config.training_reference_steps),
        "digit_validation_reference_steps": list(config.validation_reference_steps),
        "digit_training_physical_speed_m_s": training_speeds,
        "ellipse_training_theoretical_maximum_normal_acceleration_m_s2": [
            speed * speed * ellipse_maximum_curvature for speed in training_speeds
        ],
        "original_training_directions": len(training_angles()),
        "original_validation_directions": len(validation_angles()),
        "original_training_movement_steps_first_five": [50, 100, 150],
        "original_training_movement_steps_last_five": [100, 200, 300],
        "original_validation_movement_steps_first_five": list(range(50, 150, 10)),
        "original_validation_movement_steps_last_five": list(range(100, 300, 20)),
        "training_speed_scalar": [1.0 - steps / 150.0 for steps in config.training_reference_steps],
    }


def run_protocol_audit(
    repository: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    output.mkdir(parents=True)

    repository_path = Path(repository).resolve()
    config_file = Path(config_path).resolve()
    config = load_geometry_config(config_file)
    shared = build_shared_constraint_audit(config)
    components = build_digit_component_audit(config)
    geometry_rows, provenance = collect_geometry_condition_metrics(repository_path, config)
    workspace_rows, workspace_summary = build_digit_workspace_audit(config)
    rollout = run_digit_environment_rollout_audit(config)
    comparison = build_comparison_summary(geometry_rows, config)
    conclusions = build_conclusions(
        shared=shared,
        component_rows=components,
        comparison=comparison,
        workspace=workspace_summary,
        rollout=rollout,
        geometry_rows=geometry_rows,
    )

    geometry_csv = output / "digit_vs_original_geometry_metrics.csv"
    workspace_csv = output / "digit_condition_workspace_audit.csv"
    _write_csv(geometry_csv, geometry_rows)
    _write_csv(workspace_csv, workspace_rows)

    audit = {
        "scope": {
            "geometry_or_protocol_modified": False,
            "development_seed_run": False,
            "formal_training_started": False,
            "checkpoint_loaded": False,
        },
        "provenance": provenance,
        "config_path": str(config_file),
        "config_sha256": _file_sha256(config_file),
        "fixed_parameters": _fixed_parameters(config),
        "metric_definitions": {
            "same_metric_code_for_both_groups": True,
            "movement_steps": "environment movement epoch length",
            "kinematic_interval_count": "target sample count minus one",
            "smooth_curvature": "three-point circumcircle curvature excluding join neighborhoods",
            "join_direction_change": "angle between nearest nonzero incoming and outgoing segments",
            "absolute_thresholds": {
                "motor_fk_tolerance_m": FK_TOLERANCE_M,
                "joint_branch_jump_rad": math.pi,
                "other_relative_metrics": "observed original range only; no invented pass/fail threshold",
            },
        },
        "shared_constraints": shared,
        "digit_components": components,
        "comparison": comparison,
        "geometry_condition_row_count": len(geometry_rows),
        "workspace_summary": workspace_summary,
        "environment_rollout": rollout,
        "conclusions": conclusions,
        "passed_without_blockers": not conclusions["D_blocking_must_fix"],
        "generated_files": {
            "geometry_metrics_csv": geometry_csv.name,
            "workspace_audit_csv": workspace_csv.name,
            "markdown_report": "DIGIT_GEOMETRY_PROTOCOL_AUDIT.md",
            "summary_figure": "digit_geometry_protocol_audit.png",
        },
    }

    json_path = output / "digit_geometry_protocol_audit.json"
    report_path = output / "DIGIT_GEOMETRY_PROTOCOL_AUDIT.md"
    figure_path = output / "digit_geometry_protocol_audit.png"
    _write_json(json_path, audit)
    write_markdown_report(report_path, audit)
    generate_summary_figure(figure_path, config, geometry_rows, workspace_summary)

    file_manifest = {
        path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
        for path in (report_path, json_path, geometry_csv, workspace_csv, figure_path)
    }
    _write_json(output / "OUTPUT_SHA256.json", file_manifest)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    audit = run_protocol_audit(arguments.repository, arguments.config, arguments.output)
    print("GEOMETRY_CONDITION_ROWS", audit["geometry_condition_row_count"])
    print("WORKSPACE_CONDITIONS", audit["workspace_summary"]["condition_count"])
    print("ROLLOUT_CONDITIONS", audit["environment_rollout"]["condition_count"])
    print("BLOCKING_ISSUES", len(audit["conclusions"]["D_blocking_must_fix"]))
    print(f"DIGIT_GEOMETRY_PROTOCOL_AUDIT_PASSED={int(audit['passed_without_blockers'])}")
    print("DEVELOPMENT_SEED_RUN=0")
    print("FORMAL_TRAINING_STARTED=0")
    return 0 if audit["passed_without_blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
