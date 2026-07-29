#!/usr/bin/env python3
"""
Final parametric digit geometry for the motor-compositionality project.

This file is the sole geometry/time source for digits 0-9.

Scientific choices frozen here
------------------------------
1. Coordinates are first defined in normalized design units, then converted
   to metres with one global scale. No digit-specific scaling is allowed.
2. Digit 0 keeps the original 4:3 vertical ellipse and anchors the physical
   scale at 0.06153846153846154 m total height.
3. Curves A and B are exact axis-symmetric Bézier curves:
      A is shared by digit 2 upper and digit 3 upper.
      B is shared by digit 3 lower and digit 5 lower.
4. Digit 8 is an independent, exactly axis-symmetric Gerono-style path,
   with width/height ratio 0.5981519742883377 derived previously from the
   one-stroke UJI digit-8 samples. It shares no fragment with other digits.
5. Digits 6 and 9 use the same 5:4 ellipse dimensions, but different start
   phases/directions and different attached line segments.
6. Nominal physical speed conditions remain reach-referenced:
      fast   0.5000000000 m/s
      medium 0.2500000000 m/s
      slow   0.1666666667 m/s
   These are not claimed to be human handwriting speeds.
7. Every segment is sampled by linear arc length. The interval count is
   ceil(segment_length / (speed * dt)); adjacent segment junctions are
   retained once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Iterable

import numpy as np


DT_S = 0.01
DIGIT0_HEIGHT_M = 2.0 * 0.035 * 0.8791208791208792
DIGIT0_HEIGHT_UNITS = 0.96
GLOBAL_SCALE_M_PER_UNIT = DIGIT0_HEIGHT_M / DIGIT0_HEIGHT_UNITS

PHYSICAL_SPEED_ARCLENGTH = "physical_speed_arclength"
FIXED_SEGMENT_TIMING = "fixed_segment_timing"
FAST_BASE_INTERVALS = {
    "line": 30,
    "curve_A": 40,
    "curve_B": 45,
    "ellipse_5_4": 70,
    "ellipse_4_3": 85,
    "independent_uji8": 100,
}

TRAIN_SPEEDS_MPS = {
    "fast": 0.5000000000,
    "medium": 0.2500000000,
    "slow": 1.0 / 6.0,
}
TRAIN_SPEED_SCALARS = {
    "fast": 2.0 / 3.0,
    "medium": 1.0 / 3.0,
    "slow": 0.0,
}
VALID_REFERENCE_STEPS = tuple(range(50, 141, 10))
VALID_SPEEDS_MPS = tuple(0.25 / (steps * DT_S) for steps in VALID_REFERENCE_STEPS)
VALID_SPEED_SCALARS = tuple(1.0 - steps / 150.0 for steps in VALID_REFERENCE_STEPS)

COMPOSITION_DIGITS = (0, 4, 6, 9, 8)

CURVE_HEIGHT_UNITS = 0.48
UJI8_WIDTH_TO_HEIGHT = 0.5981519742883377


@dataclass(frozen=True)
class Segment:
    name: str
    points_units: np.ndarray
    shared_id: str | None = None

    def points_m(
        self, scale_m_per_unit: float = GLOBAL_SCALE_M_PER_UNIT
    ) -> np.ndarray:
        return np.asarray(self.points_units, dtype=np.float64) * scale_m_per_unit


def timing_key(segment: Segment) -> str:
    if segment.shared_id in {"curve_A", "curve_B"}:
        return str(segment.shared_id)
    if segment.name.startswith("ellipse_5_4"):
        return "ellipse_5_4"
    if segment.name == "ellipse_4_3":
        return "ellipse_4_3"
    if segment.name == "independent_uji8":
        return "independent_uji8"
    return "line"


def segment_intervals(segment: Segment, selected_reference_steps: int) -> int:
    if selected_reference_steps not in {50, 100}:
        raise ValueError("selected_reference_steps must be 50 or 100")
    scaled = (
        FAST_BASE_INTERVALS[timing_key(segment)]
        * selected_reference_steps
        / 50.0
    )
    return int(math.floor(scaled + 0.5))


def _bezier(control_points: Iterable[Iterable[float]], n: int = 2001) -> np.ndarray:
    control = np.asarray(list(control_points), dtype=np.float64)
    if control.ndim != 2 or control.shape[1] != 2:
        raise ValueError("control_points must have shape [K, 2]")
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    degree = len(control) - 1
    points = np.zeros((n, 2), dtype=np.float64)
    for index, point in enumerate(control):
        weight = comb(degree, index) * (1.0 - t) ** (degree - index) * t**index
        points += weight[:, None] * point
    return points


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    matrix = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )
    return np.asarray(points, dtype=np.float64) @ matrix.T


def _translate(points: np.ndarray, offset: Iterable[float]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) + np.asarray(offset, dtype=np.float64)


def _ellipse(
    rx: float,
    ry: float,
    start_degrees: float,
    end_degrees: float,
    n: int = 3001,
) -> np.ndarray:
    theta = np.deg2rad(np.linspace(start_degrees, end_degrees, n))
    return np.column_stack((rx * np.cos(theta), ry * np.sin(theta)))


def _polyline(*points: Iterable[float]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def _place_start_at_origin(segments: list[Segment]) -> list[Segment]:
    if not segments:
        raise ValueError("digit requires at least one segment")
    origin = np.asarray(segments[0].points_units[0], dtype=np.float64)
    return [
        Segment(
            name=segment.name,
            points_units=np.asarray(segment.points_units, dtype=np.float64) - origin,
            shared_id=segment.shared_id,
        )
        for segment in segments
    ]


def canonical_curve_a() -> np.ndarray:
    """Axis-symmetric curve A, traversed from upper endpoint to lower endpoint."""
    d = 0.38
    h = CURVE_HEIGHT_UNITS
    return _bezier(
        [
            [0.0, +h / 2.0],
            [4.0 * d / 3.0, +h / 2.0],
            [4.0 * d / 3.0, -h / 2.0],
            [0.0, -h / 2.0],
        ]
    )


def canonical_curve_b() -> np.ndarray:
    """Axis-symmetric, fuller curve B, traversed upper endpoint to lower endpoint."""
    d = 0.40
    h = CURVE_HEIGHT_UNITS
    return _bezier(
        [
            [0.0, +h / 2.0],
            [0.95 * d, +h / 2.0],
            [1.22 * d, +0.16 * h],
            [1.22 * d, -0.16 * h],
            [0.95 * d, -h / 2.0],
            [0.0, -h / 2.0],
        ]
    )


def independent_digit8() -> np.ndarray:
    """Independent exact-symmetry figure eight, top start and top end."""
    total_height = 0.96
    b = total_height / 2.0
    a = UJI8_WIDTH_TO_HEIGHT * b
    t = np.linspace(0.0, 2.0 * math.pi, 4001, dtype=np.float64)
    path = np.column_stack(
        (
            -a * np.sin(2.0 * t),
            b * (np.cos(t) - 1.0),
        )
    )
    path[0] = np.array([0.0, 0.0])
    path[-1] = np.array([0.0, 0.0])
    return path


def build_digit_segments_units() -> dict[int, list[Segment]]:
    a = canonical_curve_a()
    b = canonical_curve_b()
    digits: dict[int, list[Segment]] = {}

    # 0: unchanged 4:3 vertical ellipse, top start, counter-clockwise.
    digits[0] = _place_start_at_origin(
        [Segment("ellipse_4_3", _ellipse(0.36, 0.48, 90.0, 450.0), "ellipse_4_3")]
    )

    # 1: unchanged vertical down.
    digits[1] = _place_start_at_origin(
        [Segment("vertical_1", _polyline([0.0, 0.48], [0.0, -0.48]))]
    )

    # 2: A rotated 55 degrees, tangent 55-degree diagonal, longer bottom bar.
    a2 = _rotate(a, 55.0)
    a2 = a2 - a2[0]
    diagonal_direction = np.array(
        [math.cos(math.radians(235.0)), math.sin(math.radians(235.0))],
        dtype=np.float64,
    )
    diagonal_end = a2[-1] + 0.64 * diagonal_direction
    bottom_end = diagonal_end + np.array([0.64, 0.0])
    digits[2] = _place_start_at_origin(
        [
            Segment("curve_A_2", a2, "curve_A"),
            Segment("diagonal_55_2", _polyline(a2[-1], diagonal_end)),
            Segment("horizontal_2", _polyline(diagonal_end, bottom_end)),
        ]
    )

    # 3: upper A followed by lower B.
    a3 = a - a[0]
    b3 = b - b[0] + a3[-1]
    digits[3] = _place_start_at_origin(
        [
            Segment("curve_A_3", a3, "curve_A"),
            Segment("curve_B_3", b3, "curve_B"),
        ]
    )

    # 4: left horizontal 0.72, 55-degree up-right diagonal, unchanged vertical.
    p0 = np.array([0.0, 0.0])
    p1 = np.array([-0.72, 0.0])
    p2 = p1 + np.array([0.78 / math.tan(math.radians(55.0)), 0.78])
    p3 = p2 + np.array([0.0, -1.00])
    digits[4] = _place_start_at_origin(
        [
            Segment("horizontal_4", _polyline(p0, p1)),
            Segment("diagonal_55_4", _polyline(p1, p2)),
            Segment("vertical_4", _polyline(p2, p3)),
        ]
    )

    # 5: top horizontal left, vertical down, lower B.
    p0 = np.array([0.0, 0.0])
    p1 = np.array([-0.42, 0.0])
    p2 = p1 + np.array([0.0, -0.48])
    b5 = b - b[0] + p2
    digits[5] = _place_start_at_origin(
        [
            Segment("horizontal_5", _polyline(p0, p1)),
            Segment("vertical_5", _polyline(p1, p2)),
            Segment("curve_B_5", b5, "curve_B"),
        ]
    )

    # 6: 63.435-degree line tangent to a 5:4 ellipse, then full CCW loop.
    rx_69 = 0.36 * (5.0 / 6.0)
    ry_69 = 0.45 * (5.0 / 6.0)
    phi6 = math.atan2(1.0 / rx_69, -2.0 / ry_69)
    contact = np.array([rx_69 * math.cos(phi6), ry_69 * math.sin(phi6)])
    start = contact + 0.76 * np.array(
        [math.cos(math.radians(63.435)), math.sin(math.radians(63.435))]
    )
    digits[6] = _place_start_at_origin(
        [
            Segment("diagonal_63_435_6", _polyline(start, contact)),
            Segment(
                "ellipse_5_4_6",
                _ellipse(rx_69, ry_69, math.degrees(phi6), math.degrees(phi6) + 360.0),
                "ellipse_5_4",
            ),
        ]
    )

    # 7: unchanged horizontal right then 63.435-degree down-left diagonal.
    digits[7] = _place_start_at_origin(
        [
            Segment("horizontal_7", _polyline([0.0, 0.0], [0.64, 0.0])),
            Segment("diagonal_63_435_7", _polyline([0.64, 0.0], [0.16, -0.96])),
        ]
    )

    # 8: independent path, no shared fragment.
    digits[8] = _place_start_at_origin(
        [Segment("independent_uji8", independent_digit8())]
    )

    # 9: same 5:4 ellipse dimensions, right start, clockwise, then shorter vertical tail.
    rx_69 = 0.36 * (5.0 / 6.0)
    ry_69 = 0.45 * (5.0 / 6.0)
    loop9 = _ellipse(rx_69, ry_69, 0.0, -360.0)
    tail9 = _polyline(loop9[-1], loop9[-1] + np.array([0.0, -0.90]))
    digits[9] = _place_start_at_origin(
        [
            Segment("ellipse_5_4_9", loop9, "ellipse_5_4"),
            Segment("vertical_9", tail9),
        ]
    )

    return digits


def arc_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_linear_arclength(points: np.ndarray, n_intervals: int) -> np.ndarray:
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    points = np.asarray(points, dtype=np.float64)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("segment must have positive finite length")
    target = np.linspace(0.0, total, n_intervals + 1)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    return np.column_stack((x, y))


def sample_digit(
    digit: int,
    speed_mps: float | None,
    dt_s: float = DT_S,
    *,
    scale_m_per_unit: float = GLOBAL_SCALE_M_PER_UNIT,
    timing_mode: str = PHYSICAL_SPEED_ARCLENGTH,
    selected_reference_steps: int | None = None,
) -> dict[str, object]:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if scale_m_per_unit <= 0.0:
        raise ValueError("scale_m_per_unit must be positive")
    if timing_mode == PHYSICAL_SPEED_ARCLENGTH:
        if speed_mps is None or speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive for physical-speed timing")
        if selected_reference_steps is not None:
            raise ValueError(
                "selected_reference_steps is not used for physical-speed timing"
            )
    elif timing_mode == FIXED_SEGMENT_TIMING:
        if selected_reference_steps not in {50, 100}:
            raise ValueError(
                "fixed-segment timing requires selected_reference_steps 50 or 100"
            )
    else:
        raise ValueError(f"unsupported timing_mode: {timing_mode}")
    digits = build_digit_segments_units()
    if digit not in digits:
        raise ValueError(f"unknown digit: {digit}")

    sampled_segments: list[np.ndarray] = []
    segment_records: list[dict[str, object]] = []
    boundaries = [0]
    for segment in digits[digit]:
        dense_m = segment.points_m(scale_m_per_unit)
        length_m = arc_length(dense_m)
        if timing_mode == PHYSICAL_SPEED_ARCLENGTH:
            intervals = int(math.ceil(length_m / (float(speed_mps) * dt_s)))
            intervals = max(intervals, 1)
        else:
            intervals = segment_intervals(segment, int(selected_reference_steps))
        sampled = resample_linear_arclength(dense_m, intervals)
        actual_mean_speed_m_s = length_m / (intervals * dt_s)
        sampled_segments.append(sampled)
        segment_records.append(
            {
                "name": segment.name,
                "shared_id": segment.shared_id,
                "timing_key": timing_key(segment),
                "arc_length_m": length_m,
                "intervals": intervals,
                "samples": intervals + 1,
                "actual_mean_speed_m_s": actual_mean_speed_m_s,
            }
        )
        boundaries.append(boundaries[-1] + intervals)

    path_parts = [sampled_segments[0]]
    for segment in sampled_segments[1:]:
        path_parts.append(segment[1:])
    path = np.vstack(path_parts)

    return {
        "digit": digit,
        "speed_mps": speed_mps,
        "timing_mode": timing_mode,
        "selected_reference_steps": selected_reference_steps,
        "scale_m_per_unit": scale_m_per_unit,
        "dt_s": dt_s,
        "path_m": path,
        "movement_intervals": len(path) - 1,
        "movement_samples": len(path),
        "duration_s": (len(path) - 1) * dt_s,
        "segment_boundaries": boundaries,
        "segments": segment_records,
    }


def _bbox(points: np.ndarray) -> dict[str, float]:
    return {
        "xmin": float(points[:, 0].min()),
        "xmax": float(points[:, 0].max()),
        "ymin": float(points[:, 1].min()),
        "ymax": float(points[:, 1].max()),
        "width": float(points[:, 0].max() - points[:, 0].min()),
        "height": float(points[:, 1].max() - points[:, 1].min()),
    }


def _sha256(points: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float64).tobytes()).hexdigest()


def _symmetry_error_y(points: np.ndarray) -> float:
    reflected = points[::-1].copy()
    reflected[:, 1] *= -1.0
    return float(np.max(np.linalg.norm(points - reflected, axis=1)))


def _rigid_shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Procrustes-like rigid alignment error, no reflection and no scaling."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = a0 @ rotation
    return float(np.max(np.linalg.norm(aligned - b0, axis=1)))


def self_test() -> None:
    digits = build_digit_segments_units()
    assert set(digits) == set(range(10))
    assert math.isclose(
        GLOBAL_SCALE_M_PER_UNIT,
        0.06410256410256411,
        rel_tol=0.0,
        abs_tol=1e-16,
    )

    for digit, segments in digits.items():
        assert np.allclose(segments[0].points_units[0], 0.0, atol=1e-12)
        for segment in segments:
            assert np.isfinite(segment.points_units).all()
            assert arc_length(segment.points_units) > 0.0
        for left, right in zip(segments[:-1], segments[1:]):
            assert np.allclose(left.points_units[-1], right.points_units[0], atol=1e-12)

    a = canonical_curve_a()
    b = canonical_curve_b()
    assert _symmetry_error_y(a) < 1e-10
    assert _symmetry_error_y(b) < 1e-10

    # Shared A and B are exact rigid copies in their uses.
    a2 = resample_linear_arclength(digits[2][0].points_units, 200)
    a3 = resample_linear_arclength(digits[3][0].points_units, 200)
    b3 = resample_linear_arclength(digits[3][1].points_units, 200)
    b5 = resample_linear_arclength(digits[5][2].points_units, 200)
    assert _rigid_shape_distance(a2, a3) < 1e-8
    assert _rigid_shape_distance(b3, b5) < 1e-8

    # Angles.
    angle2 = math.degrees(math.atan2(
        *(digits[2][1].points_units[-1] - digits[2][1].points_units[0])[::-1]
    )) % 180.0
    angle4 = math.degrees(math.atan2(
        *(digits[4][1].points_units[-1] - digits[4][1].points_units[0])[::-1]
    )) % 180.0
    angle6 = math.degrees(math.atan2(
        *(digits[6][0].points_units[-1] - digits[6][0].points_units[0])[::-1]
    )) % 180.0
    angle7 = math.degrees(math.atan2(
        *(digits[7][1].points_units[-1] - digits[7][1].points_units[0])[::-1]
    )) % 180.0
    assert abs(angle2 - 55.0) < 1e-3
    assert abs(angle4 - 55.0) < 1e-3
    assert abs(angle6 - 63.435) < 1e-3
    assert abs(angle7 - 63.43494882292201) < 1e-9

    # Ellipse ratios.
    bbox0 = _bbox(digits[0][0].points_units)
    assert abs((bbox0["height"] / bbox0["width"]) - (4.0 / 3.0)) < 1e-5
    bbox6 = _bbox(digits[6][1].points_units)
    bbox9 = _bbox(digits[9][0].points_units)
    assert abs((bbox6["height"] / bbox6["width"]) - 1.25) < 1e-5
    assert abs((bbox9["height"] / bbox9["width"]) - 1.25) < 1e-5

    # Digit 8 exact left-right and top-bottom symmetry.
    p8 = digits[8][0].points_units
    reflected_lr = p8[::-1].copy()
    reflected_lr[:, 0] *= -1.0
    assert float(np.max(np.linalg.norm(p8 - reflected_lr, axis=1))) < 1e-10
    unique8 = p8[:-1]
    reflected_tb = unique8.copy()
    reflected_tb[:, 1] = -0.96 - reflected_tb[:, 1]
    shifted8 = np.roll(unique8, -(len(unique8) // 2), axis=0)
    assert float(np.max(np.linalg.norm(shifted8 - reflected_tb, axis=1))) < 1e-10

    # Sampling.
    for speed in TRAIN_SPEEDS_MPS.values():
        sampled = {digit: sample_digit(digit, speed) for digit in range(10)}
        for digit in range(10):
            result = sampled[digit]
            path = result["path_m"]
            assert np.isfinite(path).all()
            assert np.allclose(path[0], 0.0, atol=1e-12)
            assert result["movement_intervals"] >= 1
        assert sampled[2]["segments"][0]["intervals"] == sampled[3]["segments"][0]["intervals"]
        assert sampled[3]["segments"][1]["intervals"] == sampled[5]["segments"][2]["intervals"]

    print("All self-tests passed.")


def build_audit() -> dict[str, object]:
    digits = build_digit_segments_units()
    geometry = {}
    for digit, segments in digits.items():
        dense = np.vstack(
            [
                segment.points_m() if index == 0 else segment.points_m()[1:]
                for index, segment in enumerate(segments)
            ]
        )
        geometry[str(digit)] = {
            "total_arc_length_m": sum(arc_length(segment.points_m()) for segment in segments),
            "bbox_m": _bbox(dense),
            "path_sha256": _sha256(dense),
            "segments": [
                {
                    "name": segment.name,
                    "shared_id": segment.shared_id,
                    "arc_length_m": arc_length(segment.points_m()),
                }
                for segment in segments
            ],
        }

    timing = {
        "training": {
            speed_name: {
                str(digit): {
                    "movement_intervals": sample_digit(digit, speed)["movement_intervals"],
                    "duration_s": sample_digit(digit, speed)["duration_s"],
                }
                for digit in range(10)
            }
            for speed_name, speed in TRAIN_SPEEDS_MPS.items()
        },
        "validation": {
            str(reference_steps): {
                str(digit): {
                    "movement_intervals": sample_digit(digit, speed)["movement_intervals"],
                    "duration_s": sample_digit(digit, speed)["duration_s"],
                }
                for digit in range(10)
            }
            for reference_steps, speed in zip(VALID_REFERENCE_STEPS, VALID_SPEEDS_MPS)
        },
    }

    composition = {
        "digits": list(COMPOSITION_DIGITS),
        "arc_lengths_m": {
            str(digit): geometry[str(digit)]["total_arc_length_m"]
            for digit in COMPOSITION_DIGITS
        },
        "validation_index_9_movement_intervals": {
            str(digit): sample_digit(digit, VALID_SPEEDS_MPS[9])["movement_intervals"]
            for digit in COMPOSITION_DIGITS
        },
    }

    return {
        "global_scale_m_per_unit": GLOBAL_SCALE_M_PER_UNIT,
        "digit0_height_m": DIGIT0_HEIGHT_M,
        "dt_s": DT_S,
        "training_speeds_mps": TRAIN_SPEEDS_MPS,
        "training_speed_scalars": TRAIN_SPEED_SCALARS,
        "validation_reference_steps": list(VALID_REFERENCE_STEPS),
        "validation_speeds_mps": list(VALID_SPEEDS_MPS),
        "validation_speed_scalars": list(VALID_SPEED_SCALARS),
        "geometry": geometry,
        "timing": timing,
        "composition_default": composition,
    }


def render_plots(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    digits = build_digit_segments_units()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for digit, ax in enumerate(axes.flat):
        for segment in digits[digit]:
            points = segment.points_m()
            ax.plot(points[:, 0], points[:, 1], linewidth=2.0)
        ax.scatter([0.0], [0.0], marker="*", s=55)
        ax.set_title(str(digit))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "digits_0_to_9.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(canonical_curve_a()[:, 0], canonical_curve_a()[:, 1], label="A")
    ax.plot(canonical_curve_b()[:, 0], canonical_curve_b()[:, 1], label="B")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "curves_A_B.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("digit_geometry_audit"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    with (args.output_dir / "digit_geometry_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    with (args.output_dir / "digit_arc_lengths.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["digit", "total_arc_length_m"])
        for digit in range(10):
            writer.writerow([digit, audit["geometry"][str(digit)]["total_arc_length_m"]])

    if not args.no_plots:
        render_plots(args.output_dir)

    print(f"Wrote audit to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
