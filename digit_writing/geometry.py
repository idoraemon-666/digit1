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
    FIXED_SEGMENT_TIMING,
    GLOBAL_SCALE_M_PER_UNIT,
    PHYSICAL_SPEED_ARCLENGTH,
    arc_length,
    build_digit_segments_units,
    canonical_curve_a,
    canonical_curve_b,
    resample_linear_arclength,
    sample_digit,
    segment_intervals,
    timing_key,
)
from digit_writing.corner_time_reparameterization import (
    CORNER_EASE_TIMING,
    detect_sharp_boundaries,
    resample_segment_with_corner_easing,
)


FINAL_GEOMETRY_SOURCE = "digit_writing/digit_geometry_final.py"


@dataclass(frozen=True)
class GeometryConfig:
    protocol: str
    geometry_source: str
    global_scale_m_per_unit: float
    scale_multiplier: float
    timing_mode: str
    selected_reference_steps: int | None
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
    timing_key: str
    start_index: int
    end_index: int
    intervals: int
    arc_length_m: float
    actual_mean_speed_m_s: float
    canonical_sample_sha256: str
    canonical_template_sha256: str
    ordered_instance_sha256: str
    source_geometry_sha256: str
    derived_path_geometry_sha256: str
    temporal_sampling_sha256: str


@dataclass(frozen=True)
class DigitTrajectory:
    digit: int
    reference_steps: int
    physical_speed_m_s: float
    speed_scalar: float
    movement_intervals: int
    movement_duration_s: float
    arc_length_m: float
    actual_mean_speed_m_s: float
    points: np.ndarray
    boundaries: tuple[PrimitiveBoundary, ...]
    corner_ease_boundaries: tuple[Mapping[str, object], ...]


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
    protocol = str(raw.get("protocol", "digit_writing_original_protocol2"))
    if protocol == "digit_writing_original_protocol2":
        _require_exact_keys(
            raw,
            {"geometry_source", "global_scale_m_per_unit", "timing"},
            "top-level",
        )
        scale_multiplier = 1.0
        timing_mode = PHYSICAL_SPEED_ARCLENGTH
        selected_reference_steps = None
    elif protocol == "digit_writing_original_protocol3":
        _require_exact_keys(
            raw,
            {
                "protocol",
                "geometry_source",
                "global_scale_m_per_unit",
                "scale_multiplier",
                "timing_mode",
                "selected_reference_steps",
                "timing",
            },
            "top-level",
        )
        scale_multiplier = float(raw["scale_multiplier"])
        timing_mode = str(raw["timing_mode"])
        selected_reference_steps = int(raw["selected_reference_steps"])
    else:
        raise ValueError(f"unsupported geometry protocol: {protocol}")
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
        protocol=protocol,
        geometry_source=str(raw["geometry_source"]),
        global_scale_m_per_unit=float(raw["global_scale_m_per_unit"]),
        scale_multiplier=scale_multiplier,
        timing_mode=timing_mode,
        selected_reference_steps=selected_reference_steps,
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
    expected_scale = GLOBAL_SCALE_M_PER_UNIT * config.scale_multiplier
    if not math.isclose(
        config.global_scale_m_per_unit,
        expected_scale,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("configured global scale differs from the protocol scale")
    if not math.isclose(config.dt_seconds, DT_S, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("configured dt differs from the final geometry authority")
    if config.stable_steps < 0 or config.hold_steps < 0:
        raise ValueError("stable_steps and hold_steps must be non-negative")
    if not config.delay_steps or any(value < 0 for value in config.delay_steps):
        raise ValueError("delay_steps must contain non-negative values")
    if config.reach_distance_m <= 0.0:
        raise ValueError("reach_distance_m must be positive")
    if config.protocol == "digit_writing_original_protocol2":
        if config.training_reference_steps != (50, 100, 150):
            raise ValueError("training_reference_steps must be [50, 100, 150]")
        if config.validation_reference_steps != tuple(range(50, 150, 10)):
            raise ValueError("validation_reference_steps must be 50 through 140 by 10")
    else:
        if config.scale_multiplier not in {2.5, 2.25}:
            raise ValueError("protocol3 scale_multiplier must be 2.5 or 2.25")
        if config.timing_mode not in {
            FIXED_SEGMENT_TIMING,
            CORNER_EASE_TIMING,
        }:
            raise ValueError("protocol3 timing_mode is not supported")
        if config.selected_reference_steps not in {50, 100}:
            raise ValueError("protocol3 selected_reference_steps must be 50 or 100")
        expected_references = (int(config.selected_reference_steps),)
        if config.training_reference_steps != expected_references:
            raise ValueError("protocol3 training reference must match the selected value")
        if config.validation_reference_steps != expected_references:
            raise ValueError("protocol3 validation reference must match the selected value")
        if config.timing_mode == CORNER_EASE_TIMING and (
            config.scale_multiplier != 2.5
            or config.selected_reference_steps != 100
        ):
            raise ValueError("corner easing is frozen to scale2p50/ref100")
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


def _identified_sample_hash(points: np.ndarray) -> str:
    canonical = np.ascontiguousarray(points, dtype="<f8")
    header = f"shape={canonical.shape};dtype=<f8;".encode("ascii")
    return hashlib.sha256(header + canonical.tobytes()).hexdigest()


def _identified_json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"encoding=canonical-json-v1;" + payload).hexdigest()


def _sample_corner_ease_digit(
    digit: int,
    *,
    dt_seconds: float,
    scale_m_per_unit: float,
    selected_reference_steps: int,
    movement_intervals_override: int | None = None,
) -> dict[str, object]:
    digits = build_digit_segments_units()
    if digit not in digits:
        raise ValueError(f"unknown digit: {digit}")
    segments = digits[digit]
    base_intervals = [
        segment_intervals(segment, selected_reference_steps)
        for segment in segments
    ]
    if movement_intervals_override is not None:
        if (
            isinstance(movement_intervals_override, bool)
            or not isinstance(movement_intervals_override, int)
        ):
            raise TypeError("movement_intervals_override must be an integer")
        if digit not in {0, 8} or len(segments) != 1:
            raise ValueError(
                "movement interval override is restricted to the one-primitive "
                "Protocol3 digit0/digit8 timing diagnostic"
            )
        allowed_intervals = {
            0: {170, 200, 220},
            8: {200, 220, 240},
        }
        if movement_intervals_override not in allowed_intervals[digit]:
            raise ValueError(
                "movement interval override is outside the frozen digit arm"
            )
        base_intervals = [movement_intervals_override]
    dense_segments_m = [
        segment.points_m(scale_m_per_unit) for segment in segments
    ]
    baseline_samples = [
        resample_linear_arclength(points, intervals)
        for points, intervals in zip(dense_segments_m, base_intervals)
    ]
    corners = detect_sharp_boundaries(
        baseline_samples,
        [segment.name for segment in segments],
    )
    ease_at_end = {corner.boundary_index for corner in corners if corner.qualifies}
    ease_at_start = {
        corner.boundary_index + 1 for corner in corners if corner.qualifies
    }

    sampled_segments = []
    segment_records = []
    boundaries = [0]
    for index, (segment, dense_m, intervals) in enumerate(
        zip(segments, dense_segments_m, base_intervals)
    ):
        ease_start = index in ease_at_start
        ease_end = index in ease_at_end
        sampled = (
            resample_segment_with_corner_easing(
                dense_m,
                intervals,
                ease_start=ease_start,
                ease_end=ease_end,
            )
            if ease_start or ease_end
            else baseline_samples[index]
        )
        actual_intervals = len(sampled) - 1
        length_m = arc_length(dense_m)
        sampled_segments.append(sampled)
        segment_records.append(
            {
                "name": segment.name,
                "shared_id": segment.shared_id,
                "timing_key": timing_key(segment),
                "source_geometry": segment.source_geometry,
                "derived_path_units": np.asarray(
                    segment.points_units, dtype=np.float64
                ),
                "arc_length_m": length_m,
                "base_intervals": intervals,
                "intervals": actual_intervals,
                "samples": actual_intervals + 1,
                "ease_at_start": ease_start,
                "ease_at_end": ease_end,
                "actual_mean_speed_m_s": (
                    length_m / (actual_intervals * dt_seconds)
                ),
            }
        )
        boundaries.append(boundaries[-1] + actual_intervals)

    path = np.vstack(
        [
            sampled if index == 0 else sampled[1:]
            for index, sampled in enumerate(sampled_segments)
        ]
    )
    return {
        "digit": digit,
        "speed_mps": None,
        "timing_mode": CORNER_EASE_TIMING,
        "selected_reference_steps": selected_reference_steps,
        "scale_m_per_unit": scale_m_per_unit,
        "dt_s": dt_seconds,
        "path_m": path,
        "movement_intervals": len(path) - 1,
        "movement_samples": len(path),
        "duration_s": (len(path) - 1) * dt_seconds,
        "segment_boundaries": boundaries,
        "segments": segment_records,
        "corner_ease_boundaries": [
            {
                "boundary_index": corner.boundary_index,
                "previous_segment": corner.previous_segment,
                "next_segment": corner.next_segment,
                "turn_angle_deg": corner.turn_angle_deg,
                "qualifies": corner.qualifies,
                "corner_sample_index": boundaries[corner.boundary_index + 1],
            }
            for corner in corners
        ],
    }


@lru_cache(maxsize=256)
def _sample_final(
    digit: int,
    reference_steps: int,
    dt_seconds: float,
    reach_distance_m: float,
    scale_m_per_unit: float,
    timing_mode: str,
    selected_reference_steps: int | None,
    movement_intervals_override: int | None,
) -> dict[str, object]:
    if movement_intervals_override is not None and timing_mode != CORNER_EASE_TIMING:
        raise ValueError(
            "movement interval override requires Protocol3 corner-ease timing"
        )
    if timing_mode == PHYSICAL_SPEED_ARCLENGTH:
        result = sample_digit(
            digit,
            reach_distance_m / (reference_steps * dt_seconds),
            dt_s=dt_seconds,
            scale_m_per_unit=scale_m_per_unit,
            timing_mode=timing_mode,
        )
    elif timing_mode == FIXED_SEGMENT_TIMING:
        result = sample_digit(
            digit,
            None,
            dt_s=dt_seconds,
            scale_m_per_unit=scale_m_per_unit,
            timing_mode=timing_mode,
            selected_reference_steps=selected_reference_steps,
        )
    elif timing_mode == CORNER_EASE_TIMING:
        if selected_reference_steps != 100:
            raise ValueError("corner easing is frozen to reference_steps=100")
        result = _sample_corner_ease_digit(
            digit,
            dt_seconds=dt_seconds,
            scale_m_per_unit=scale_m_per_unit,
            selected_reference_steps=selected_reference_steps,
            movement_intervals_override=movement_intervals_override,
        )
    else:
        raise ValueError(f"unsupported timing_mode: {timing_mode}")
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
    movement_intervals_override: int | None = None,
) -> DigitTrajectory:
    anchor_array = np.asarray(anchor, dtype=np.float64)
    if anchor_array.shape != (2,):
        raise ValueError("anchor must be a two-vector")
    if not 0 <= digit <= 9:
        raise ValueError("digit must be an integer from 0 through 9")
    if (
        config.protocol == "digit_writing_original_protocol3"
        and reference_steps != config.selected_reference_steps
    ):
        raise ValueError("protocol3 reference_steps must match the selected value")

    sampled = _sample_final(
        digit,
        int(reference_steps),
        config.dt_seconds,
        config.reach_distance_m,
        config.global_scale_m_per_unit,
        config.timing_mode,
        config.selected_reference_steps,
        movement_intervals_override,
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
            canonical_dense = canonical_curve_a() * config.global_scale_m_per_unit
            canonical_sample = resample_linear_arclength(
                canonical_dense, int(record["intervals"])
            )
        elif shared_id == "curve_B":
            canonical_dense = canonical_curve_b() * config.global_scale_m_per_unit
            canonical_sample = resample_linear_arclength(
                canonical_dense, int(record["intervals"])
            )
        else:
            canonical_sample = local_points[start_index : end_index + 1]
        ordered_instance = (
            local_points[start_index : end_index + 1]
            / config.global_scale_m_per_unit
        )
        if shared_id == "curve_A":
            canonical_template = resample_linear_arclength(
                canonical_curve_a(), int(record["intervals"])
            )
        elif shared_id == "curve_B":
            canonical_template = resample_linear_arclength(
                canonical_curve_b(), int(record["intervals"])
            )
        else:
            canonical_template = ordered_instance
        boundaries.append(
            PrimitiveBoundary(
                name=str(record["name"]),
                template_key=str(shared_id or record["name"]),
                timing_key=str(record["timing_key"]),
                start_index=start_index,
                end_index=end_index,
                intervals=int(record["intervals"]),
                arc_length_m=float(record["arc_length_m"]),
                actual_mean_speed_m_s=float(record["actual_mean_speed_m_s"]),
                canonical_sample_sha256=_sample_hash(canonical_sample),
                canonical_template_sha256=_identified_sample_hash(
                    canonical_template
                ),
                ordered_instance_sha256=_identified_sample_hash(
                    ordered_instance
                ),
                source_geometry_sha256=_identified_json_hash(
                    record["source_geometry"]
                ),
                derived_path_geometry_sha256=_identified_sample_hash(
                    np.asarray(record["derived_path_units"], dtype=np.float64)
                ),
                temporal_sampling_sha256=_identified_sample_hash(
                    ordered_instance
                ),
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
        actual_mean_speed_m_s=(
            sum(boundary.arc_length_m for boundary in boundaries)
            / float(sampled["duration_s"])
        ),
        points=points,
        boundaries=tuple(boundaries),
        corner_ease_boundaries=tuple(
            dict(row) for row in sampled.get("corner_ease_boundaries", ())
        ),
    )


def build_time_audit(config: GeometryConfig) -> dict[str, object]:
    audit_reference = (
        int(config.selected_reference_steps)
        if config.selected_reference_steps is not None
        else 100
    )
    digit_lengths = {
        str(digit): build_digit_trajectory(
            digit, config, audit_reference
        ).arc_length_m
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
