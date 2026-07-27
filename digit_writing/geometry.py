"""Canonical digit geometry and linear arc-length timing.

This module implements PROJECT_PROTOCOL.md sections 5 and 6.  Coordinates are
created from the frozen primitive definitions, uniformly scaled, and only then
sampled at a requested physical speed.  It intentionally contains no MotorNet
environment, model, loss, or training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    a: float
    b: float
    H: float
    V: float
    D_dx: float
    D_dy: float
    global_scale: float
    arc_table_intervals: int
    dt_seconds: float
    stable_steps: int
    delay_steps: tuple[int, ...]
    hold_steps: int
    reach_distance_m: float
    training_reference_steps: tuple[int, ...]
    validation_reference_steps: tuple[int, ...]


@dataclass(frozen=True)
class PrimitiveOccurrence:
    name: str
    template_key: str
    rotation_rad: float = 0.0


@dataclass(frozen=True)
class PrimitiveBoundary:
    name: str
    template_key: str
    start_index: int
    end_index: int
    intervals: int
    arc_length_m: float
    canonical_sample_sha256: str


@dataclass(frozen=True)
class DigitTrajectory:
    digit: int
    reference_steps: int
    physical_speed_m_s: float
    speed_scalar: float
    movement_intervals: int
    movement_duration_s: float
    arc_length_m: float
    points: np.ndarray
    boundaries: tuple[PrimitiveBoundary, ...]


@dataclass(frozen=True)
class _TemplateSample:
    points: np.ndarray
    intervals: int
    arc_length_m: float
    sha256: str


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


def load_geometry_config(path: str | Path) -> GeometryConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("geometry configuration must be a JSON object")

    _require_exact_keys(raw, {"geometry", "timing"}, "top-level")
    geometry = raw["geometry"]
    timing = raw["timing"]
    if not isinstance(geometry, dict) or not isinstance(timing, dict):
        raise ValueError("geometry and timing must be JSON objects")

    geometry_keys = {
        "a",
        "b",
        "H",
        "V",
        "D_dx",
        "D_dy",
        "global_scale",
        "arc_table_intervals",
    }
    timing_keys = {
        "dt_seconds",
        "stable_steps",
        "delay_steps",
        "hold_steps",
        "reach_distance_m",
        "training_reference_steps",
        "validation_reference_steps",
    }
    _require_exact_keys(geometry, geometry_keys, "geometry")
    _require_exact_keys(timing, timing_keys, "timing")

    config = GeometryConfig(
        a=float(geometry["a"]),
        b=float(geometry["b"]),
        H=float(geometry["H"]),
        V=float(geometry["V"]),
        D_dx=float(geometry["D_dx"]),
        D_dy=float(geometry["D_dy"]),
        global_scale=float(geometry["global_scale"]),
        arc_table_intervals=int(geometry["arc_table_intervals"]),
        dt_seconds=float(timing["dt_seconds"]),
        stable_steps=int(timing["stable_steps"]),
        delay_steps=tuple(int(value) for value in timing["delay_steps"]),
        hold_steps=int(timing["hold_steps"]),
        reach_distance_m=float(timing["reach_distance_m"]),
        training_reference_steps=tuple(
            int(value) for value in timing["training_reference_steps"]
        ),
        validation_reference_steps=tuple(
            int(value) for value in timing["validation_reference_steps"]
        ),
    )
    positive_values = (
        config.a,
        config.b,
        config.H,
        config.V,
        config.D_dx,
        config.D_dy,
        config.global_scale,
        config.dt_seconds,
        config.reach_distance_m,
    )
    if not all(value > 0 for value in positive_values):
        raise ValueError("all geometry, scale, time, and distance values must be positive")
    if config.arc_table_intervals < 1024:
        raise ValueError("arc_table_intervals must be at least 1024")
    if config.stable_steps < 0 or config.hold_steps < 0:
        raise ValueError("stable_steps and hold_steps must be non-negative")
    if not config.delay_steps or any(value < 0 for value in config.delay_steps):
        raise ValueError("delay_steps must contain non-negative values")
    if config.training_reference_steps != (50, 100, 150):
        raise ValueError("training_reference_steps must be [50, 100, 150]")
    if config.validation_reference_steps != tuple(range(50, 150, 10)):
        raise ValueError("validation_reference_steps must be 50 through 140 by 10")
    return config


def _phase_values(
    start_phase: float, end_phase: float, direction: str, intervals: int
) -> np.ndarray:
    delta = end_phase - start_phase
    if direction == "increasing" and delta <= 0:
        raise ValueError("increasing phase requires end_phase > start_phase")
    if direction == "decreasing" and delta >= 0:
        raise ValueError("decreasing phase requires end_phase < start_phase")
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    return np.linspace(start_phase, end_phase, intervals + 1, dtype=np.float64)


def standard_ellipse(
    config: GeometryConfig,
    orientation: str | Sequence[float],
    start_phase: float,
    end_phase: float,
    direction: str,
) -> np.ndarray:
    """Return the unscaled standard ellipse arc required by section 5.3."""

    phase = _phase_values(
        start_phase, end_phase, direction, config.arc_table_intervals
    )
    if isinstance(orientation, str):
        if orientation == "horizontal":
            return np.column_stack(
                (config.a * np.cos(phase), config.b * np.sin(phase))
            )
        if orientation == "vertical":
            return np.column_stack(
                (config.b * np.cos(phase), config.a * np.sin(phase))
            )
        raise ValueError("string orientation must be 'horizontal' or 'vertical'")

    u = np.asarray(orientation, dtype=np.float64)
    if u.shape != (2,):
        raise ValueError("rotated ellipse orientation must be a two-vector")
    norm = float(np.linalg.norm(u))
    if norm == 0:
        raise ValueError("rotated ellipse orientation cannot be zero")
    u = u / norm
    v = np.array((u[1], -u[0]), dtype=np.float64)
    return (
        config.a * np.cos(phase)[:, None] * u
        + config.b * np.sin(phase)[:, None] * v
    )


def standard_half_ellipse(
    config: GeometryConfig,
    orientation: str | Sequence[float],
    start_phase: float,
    end_phase: float,
    direction: str,
) -> np.ndarray:
    if not math.isclose(abs(end_phase - start_phase), math.pi, abs_tol=1e-12):
        raise ValueError("a standard half ellipse must span exactly pi radians")
    return standard_ellipse(config, orientation, start_phase, end_phase, direction)


def _freeze_template(points: np.ndarray, scale: float, *, closed: bool = False) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    result -= result[0]
    result *= scale
    result[0] = 0.0
    if closed:
        result[-1] = 0.0
    result.setflags(write=False)
    return result


def _reverse_template(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points[::-1] - points[-1], dtype=np.float64)
    result[0] = 0.0
    result.setflags(write=False)
    return result


@lru_cache(maxsize=8)
def primitive_library(config: GeometryConfig) -> Mapping[str, np.ndarray]:
    """Build the only canonical primitive template library."""

    scale = config.global_scale
    h_right = _freeze_template(np.array(((0.0, 0.0), (config.H, 0.0))), scale)
    v_down = _freeze_template(np.array(((0.0, 0.0), (0.0, -config.V))), scale)
    d_up = _freeze_template(
        np.array(((0.0, 0.0), (config.D_dx, config.D_dy))), scale
    )

    half_right = _freeze_template(
        standard_half_ellipse(
            config, "horizontal", math.pi / 2, -math.pi / 2, "decreasing"
        ),
        scale,
    )
    half_left = _freeze_template(
        standard_half_ellipse(
            config, "horizontal", math.pi / 2, 3 * math.pi / 2, "increasing"
        ),
        scale,
    )
    phase6 = math.atan2(config.D_dx / config.b, -config.D_dy / config.a)

    library = {
        "H_right": h_right,
        "H_left": _reverse_template(h_right),
        "V_down": v_down,
        "V_up": _reverse_template(v_down),
        "D_up": d_up,
        "D_down": _reverse_template(d_up),
        "half_ellipse_right_top_to_bottom": half_right,
        "half_ellipse_right_bottom_to_top": _reverse_template(half_right),
        "half_ellipse_left_top_to_bottom": half_left,
        "ellipse_vertical_ccw_top": _freeze_template(
            standard_ellipse(
                config,
                "vertical",
                math.pi / 2,
                5 * math.pi / 2,
                "increasing",
            ),
            scale,
            closed=True,
        ),
        "ellipse_vertical_ccw_tangent6": _freeze_template(
            standard_ellipse(
                config, "vertical", phase6, phase6 + 2 * math.pi, "increasing"
            ),
            scale,
            closed=True,
        ),
        "ellipse_vertical_cw_right": _freeze_template(
            standard_ellipse(config, "vertical", 0.0, -2 * math.pi, "decreasing"),
            scale,
            closed=True,
        ),
        "ellipse_horizontal_cw_top": _freeze_template(
            standard_ellipse(
                config,
                "horizontal",
                math.pi / 2,
                -3 * math.pi / 2,
                "decreasing",
            ),
            scale,
            closed=True,
        ),
    }
    return library


def prescribed_digit_primitives(
    digit: int, config: GeometryConfig
) -> tuple[PrimitiveOccurrence, ...]:
    if digit == 0:
        return (PrimitiveOccurrence("ellipse", "ellipse_vertical_ccw_top"),)
    if digit == 1:
        return (PrimitiveOccurrence("vertical", "V_down"),)
    if digit == 2:
        angle = math.atan2(config.D_dy, config.D_dx)
        return (
            PrimitiveOccurrence(
                "half_ellipse", "half_ellipse_right_top_to_bottom", angle
            ),
            PrimitiveOccurrence("diagonal", "D_down"),
            PrimitiveOccurrence("horizontal", "H_right"),
        )
    if digit == 3:
        return (
            PrimitiveOccurrence("upper_half", "half_ellipse_right_top_to_bottom"),
            PrimitiveOccurrence("lower_half", "half_ellipse_right_top_to_bottom"),
        )
    if digit == 4:
        return (
            PrimitiveOccurrence("horizontal", "H_left"),
            PrimitiveOccurrence("diagonal", "D_up"),
            PrimitiveOccurrence("vertical", "V_down"),
        )
    if digit == 5:
        return (
            PrimitiveOccurrence("horizontal", "H_left"),
            PrimitiveOccurrence("vertical", "V_down"),
            PrimitiveOccurrence(
                "half_ellipse", "half_ellipse_right_top_to_bottom"
            ),
        )
    if digit == 6:
        return (
            PrimitiveOccurrence("diagonal", "D_down"),
            PrimitiveOccurrence("ellipse", "ellipse_vertical_ccw_tangent6"),
        )
    if digit == 7:
        return (
            PrimitiveOccurrence("horizontal", "H_right"),
            PrimitiveOccurrence("diagonal", "D_down"),
        )
    if digit == 8:
        return (
            PrimitiveOccurrence(
                "upper_left_half", "half_ellipse_left_top_to_bottom"
            ),
            PrimitiveOccurrence("lower_loop", "ellipse_horizontal_cw_top"),
            PrimitiveOccurrence(
                "upper_right_half", "half_ellipse_right_bottom_to_top"
            ),
        )
    if digit == 9:
        return (
            PrimitiveOccurrence("ellipse", "ellipse_vertical_cw_right"),
            PrimitiveOccurrence("vertical", "V_down"),
        )
    raise ValueError("digit must be an integer from 0 through 9")


def _arc_table(points: np.ndarray) -> tuple[np.ndarray, float]:
    cumulative = np.concatenate(
        (np.array((0.0,)), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    return cumulative, float(cumulative[-1])


def _resample_linear_arc(
    points: np.ndarray, cumulative: np.ndarray, arc_length: float, intervals: int
) -> np.ndarray:
    target_arc = np.linspace(0.0, arc_length, intervals + 1, dtype=np.float64)
    result = np.column_stack(
        (
            np.interp(target_arc, cumulative, points[:, 0]),
            np.interp(target_arc, cumulative, points[:, 1]),
        )
    )
    result[0] = points[0]
    result[-1] = points[-1]
    result.setflags(write=False)
    return result


def physical_speed(config: GeometryConfig, reference_steps: int) -> float:
    if reference_steps <= 0:
        raise ValueError("reference_steps must be positive")
    return config.reach_distance_m / (reference_steps * config.dt_seconds)


def speed_scalar(reference_steps: int) -> float:
    return 1.0 - reference_steps / 150.0


def _sample_hash(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@lru_cache(maxsize=256)
def _sample_template(
    config: GeometryConfig, template_key: str, reference_steps: int
) -> _TemplateSample:
    try:
        points = primitive_library(config)[template_key]
    except KeyError as exc:
        raise ValueError(f"unknown primitive template: {template_key}") from exc
    cumulative, arc_length = _arc_table(points)
    intervals = math.ceil(
        arc_length / (physical_speed(config, reference_steps) * config.dt_seconds)
    )
    sampled = _resample_linear_arc(points, cumulative, arc_length, intervals)
    return _TemplateSample(sampled, intervals, arc_length, _sample_hash(sampled))


def _rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return points @ rotation.T


def build_digit_trajectory(
    digit: int,
    config: GeometryConfig,
    reference_steps: int,
    *,
    spatial_angle_rad: float = 0.0,
    anchor: Sequence[float] = (0.0, 0.0),
) -> DigitTrajectory:
    anchor_array = np.asarray(anchor, dtype=np.float64)
    if anchor_array.shape != (2,):
        raise ValueError("anchor must be a two-vector")

    point_blocks: list[np.ndarray] = []
    boundaries: list[PrimitiveBoundary] = []
    current = np.zeros(2, dtype=np.float64)
    sample_count = 0
    total_arc = 0.0
    for occurrence in prescribed_digit_primitives(digit, config):
        sampled = _sample_template(config, occurrence.template_key, reference_steps)
        placed = _rotate(sampled.points, occurrence.rotation_rad) + current
        start_index = 0 if not point_blocks else sample_count - 1
        point_blocks.append(placed if not point_blocks else placed[1:])
        sample_count += sampled.intervals if sample_count else sampled.intervals + 1
        end_index = start_index + sampled.intervals
        boundaries.append(
            PrimitiveBoundary(
                name=occurrence.name,
                template_key=occurrence.template_key,
                start_index=start_index,
                end_index=end_index,
                intervals=sampled.intervals,
                arc_length_m=sampled.arc_length_m,
                canonical_sample_sha256=sampled.sha256,
            )
        )
        current = placed[-1]
        total_arc += sampled.arc_length_m

    local_points = np.concatenate(point_blocks, axis=0)
    points = _rotate(local_points, spatial_angle_rad) + anchor_array
    points[0] = anchor_array
    points.setflags(write=False)
    movement_intervals = sum(boundary.intervals for boundary in boundaries)
    if len(points) != movement_intervals + 1:
        raise RuntimeError("trajectory concatenation violated the one-join-sample rule")
    return DigitTrajectory(
        digit=digit,
        reference_steps=reference_steps,
        physical_speed_m_s=physical_speed(config, reference_steps),
        speed_scalar=speed_scalar(reference_steps),
        movement_intervals=movement_intervals,
        movement_duration_s=movement_intervals * config.dt_seconds,
        arc_length_m=total_arc,
        points=points,
        boundaries=tuple(boundaries),
    )


def training_angles() -> np.ndarray:
    return np.arange(8, dtype=np.float64) * (2 * math.pi / 8)


def validation_angles() -> np.ndarray:
    return np.arange(32, dtype=np.float64) * (2 * math.pi / 32)


def build_time_audit(config: GeometryConfig) -> dict[str, object]:
    primitive_lengths = {
        key: _arc_table(points)[1] for key, points in primitive_library(config).items()
    }
    digit_lengths = {
        str(digit): build_digit_trajectory(digit, config, 100).arc_length_m
        for digit in range(10)
    }

    def timing_rows(reference_steps_values: Sequence[int]) -> list[dict[str, object]]:
        rows = []
        for digit in range(10):
            for reference_steps in reference_steps_values:
                trajectory = build_digit_trajectory(digit, config, reference_steps)
                rows.append(
                    {
                        "digit": digit,
                        "reference_steps": reference_steps,
                        "physical_speed_m_s": trajectory.physical_speed_m_s,
                        "speed_scalar": trajectory.speed_scalar,
                        "movement_intervals": trajectory.movement_intervals,
                        "movement_duration_s": trajectory.movement_duration_s,
                    }
                )
        return rows

    sample_hashes = []
    fewer_than_two = []
    for digit in range(10):
        for reference_steps in config.training_reference_steps:
            trajectory = build_digit_trajectory(digit, config, reference_steps)
            for boundary in trajectory.boundaries:
                row = {
                    "digit": digit,
                    "reference_steps": reference_steps,
                    "name": boundary.name,
                    "template_key": boundary.template_key,
                    "intervals": boundary.intervals,
                    "canonical_sample_sha256": boundary.canonical_sample_sha256,
                }
                sample_hashes.append(row)
                if boundary.intervals < 2:
                    fewer_than_two.append(row)

    episodes = []
    all_reference_steps = sorted(
        set(config.training_reference_steps) | set(config.validation_reference_steps)
    )
    for digit in range(10):
        for reference_steps in all_reference_steps:
            trajectory = build_digit_trajectory(digit, config, reference_steps)
            for delay_steps in config.delay_steps:
                episode_steps = (
                    config.stable_steps
                    + delay_steps
                    + trajectory.movement_intervals
                    + config.hold_steps
                )
                episodes.append(
                    {
                        "digit": digit,
                        "reference_steps": reference_steps,
                        "delay_steps": delay_steps,
                        "episode_steps": episode_steps,
                        "episode_duration_s": episode_steps * config.dt_seconds,
                    }
                )

    return {
        "primitive_arc_lengths_m": primitive_lengths,
        "digit_arc_lengths_m": digit_lengths,
        "training_timing": timing_rows(config.training_reference_steps),
        "validation_timing": timing_rows(config.validation_reference_steps),
        "shortest_episode": min(episodes, key=lambda row: row["episode_steps"]),
        "longest_episode": max(episodes, key=lambda row: row["episode_steps"]),
        "shared_primitive_sample_hashes": sample_hashes,
        "primitives_with_fewer_than_two_intervals": fewer_than_two,
    }
