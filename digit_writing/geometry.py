"""Runtime adapter for the final digit geometry and timing authority.

All digit coordinates and segment sampling come from
``digit_writing.digit_geometry_final``.  This module only preserves the small
interface consumed by the MotorNet environment and audits.
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

from digit_writing.digit_geometry_final import (
    DT_S,
    GLOBAL_SCALE_M_PER_UNIT,
    canonical_curve_a,
    canonical_curve_b,
    resample_linear_arclength,
    sample_digit,
)


FINAL_GEOMETRY_SOURCE = "digit_writing/digit_geometry_final.py"


@dataclass(frozen=True)
class GeometryConfig:
    geometry_source: str
    global_scale_m_per_unit: float
    dt_seconds: float
    stable_steps: int
    delay_steps: tuple[int, ...]
    hold_steps: int
    reach_distance_m: float
    training_reference_steps: tuple[int, ...]
    validation_reference_steps: tuple[int, ...]


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
    _require_exact_keys(
        raw,
        {"geometry_source", "global_scale_m_per_unit", "timing"},
        "top-level",
    )
    timing = raw["timing"]
    if not isinstance(timing, dict):
        raise ValueError("timing must be a JSON object")
    _require_exact_keys(
        timing,
        {
            "dt_seconds",
            "stable_steps",
            "delay_steps",
            "hold_steps",
            "reach_distance_m",
            "training_reference_steps",
            "validation_reference_steps",
        },
        "timing",
    )

    config = GeometryConfig(
        geometry_source=str(raw["geometry_source"]),
        global_scale_m_per_unit=float(raw["global_scale_m_per_unit"]),
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
    if config.geometry_source != FINAL_GEOMETRY_SOURCE:
        raise ValueError("geometry_source must name the final geometry authority")
    if not math.isclose(
        config.global_scale_m_per_unit,
        GLOBAL_SCALE_M_PER_UNIT,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("configured global scale differs from the final geometry authority")
    if not math.isclose(config.dt_seconds, DT_S, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("configured dt differs from the final geometry authority")
    if config.stable_steps < 0 or config.hold_steps < 0:
        raise ValueError("stable_steps and hold_steps must be non-negative")
    if not config.delay_steps or any(value < 0 for value in config.delay_steps):
        raise ValueError("delay_steps must contain non-negative values")
    if config.reach_distance_m <= 0.0:
        raise ValueError("reach_distance_m must be positive")
    if config.training_reference_steps != (50, 100, 150):
        raise ValueError("training_reference_steps must be [50, 100, 150]")
    if config.validation_reference_steps != tuple(range(50, 150, 10)):
        raise ValueError("validation_reference_steps must be 50 through 140 by 10")
    return config


def physical_speed(config: GeometryConfig, reference_steps: int) -> float:
    if reference_steps <= 0:
        raise ValueError("reference_steps must be positive")
    return config.reach_distance_m / (reference_steps * config.dt_seconds)


def speed_scalar(reference_steps: int) -> float:
    return 1.0 - reference_steps / 150.0


def training_angles() -> np.ndarray:
    return np.arange(8, dtype=np.float64) * (2.0 * math.pi / 8.0)


def validation_angles() -> np.ndarray:
    return np.arange(32, dtype=np.float64) * (2.0 * math.pi / 32.0)


def _rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ rotation.T


def _sample_hash(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@lru_cache(maxsize=256)
def _sample_final(
    digit: int,
    reference_steps: int,
    dt_seconds: float,
    reach_distance_m: float,
) -> dict[str, object]:
    result = sample_digit(
        digit,
        reach_distance_m / (reference_steps * dt_seconds),
        dt_s=dt_seconds,
    )
    path = np.asarray(result["path_m"], dtype=np.float64)
    path.setflags(write=False)
    result["path_m"] = path
    return result


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
    if not 0 <= digit <= 9:
        raise ValueError("digit must be an integer from 0 through 9")

    sampled = _sample_final(
        digit,
        int(reference_steps),
        config.dt_seconds,
        config.reach_distance_m,
    )
    local_points = np.asarray(sampled["path_m"], dtype=np.float64)
    points = _rotate(local_points, spatial_angle_rad) + anchor_array
    points[0] = anchor_array
    points.setflags(write=False)

    boundary_indices = tuple(int(value) for value in sampled["segment_boundaries"])
    segment_records = sampled["segments"]
    boundaries = []
    for index, record in enumerate(segment_records):
        start_index = boundary_indices[index]
        end_index = boundary_indices[index + 1]
        shared_id = record["shared_id"]
        if shared_id == "curve_A":
            canonical_dense = canonical_curve_a() * GLOBAL_SCALE_M_PER_UNIT
            canonical_sample = resample_linear_arclength(
                canonical_dense, int(record["intervals"])
            )
        elif shared_id == "curve_B":
            canonical_dense = canonical_curve_b() * GLOBAL_SCALE_M_PER_UNIT
            canonical_sample = resample_linear_arclength(
                canonical_dense, int(record["intervals"])
            )
        else:
            canonical_sample = local_points[start_index : end_index + 1]
        boundaries.append(
            PrimitiveBoundary(
                name=str(record["name"]),
                template_key=str(shared_id or record["name"]),
                start_index=start_index,
                end_index=end_index,
                intervals=int(record["intervals"]),
                arc_length_m=float(record["arc_length_m"]),
                canonical_sample_sha256=_sample_hash(canonical_sample),
            )
        )

    movement_intervals = int(sampled["movement_intervals"])
    if len(points) != movement_intervals + 1:
        raise RuntimeError("trajectory concatenation violated the one-join-sample rule")
    return DigitTrajectory(
        digit=digit,
        reference_steps=int(reference_steps),
        physical_speed_m_s=physical_speed(config, int(reference_steps)),
        speed_scalar=speed_scalar(int(reference_steps)),
        movement_intervals=movement_intervals,
        movement_duration_s=float(sampled["duration_s"]),
        arc_length_m=sum(boundary.arc_length_m for boundary in boundaries),
        points=points,
        boundaries=tuple(boundaries),
    )


def build_time_audit(config: GeometryConfig) -> dict[str, object]:
    digit_lengths = {
        str(digit): build_digit_trajectory(digit, config, 100).arc_length_m
        for digit in range(10)
    }

    def timing_rows(reference_values: Sequence[int]) -> list[dict[str, object]]:
        rows = []
        for digit in range(10):
            for reference_steps in reference_values:
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
    reference_values = sorted(
        set(config.training_reference_steps) | set(config.validation_reference_steps)
    )
    for digit in range(10):
        for reference_steps in reference_values:
            trajectory = build_digit_trajectory(digit, config, reference_steps)
            for delay_steps in config.delay_steps:
                episode_steps = (
                    config.stable_steps
                    + delay_steps
                    + trajectory.movement_intervals
                    + 1
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
        "digit_arc_lengths_m": digit_lengths,
        "training_timing": timing_rows(config.training_reference_steps),
        "validation_timing": timing_rows(config.validation_reference_steps),
        "shortest_episode": min(episodes, key=lambda row: row["episode_steps"]),
        "longest_episode": max(episodes, key=lambda row: row["episode_steps"]),
        "shared_segment_sample_hashes": sample_hashes,
        "segments_with_fewer_than_two_intervals": fewer_than_two,
    }
