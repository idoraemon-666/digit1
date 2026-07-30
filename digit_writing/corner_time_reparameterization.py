"""Pure local time reparameterization for Protocol3 corner easing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


CORNER_EASE_TIMING = "fixed_segment_timing_corner_ease_v1"
TURN_THRESHOLD_DEG = 60.0
BASE_WINDOW_INTERVALS = 10
EXTRA_INTERVALS_PER_SIDE = 5
RESAMPLED_WINDOW_INTERVALS = (
    BASE_WINDOW_INTERVALS + EXTRA_INTERVALS_PER_SIDE
)


@dataclass(frozen=True)
class SharpBoundary:
    boundary_index: int
    previous_segment: str
    next_segment: str
    turn_angle_deg: float
    qualifies: bool


def ease_to_stop(z):
    """Monotone quintic with unit initial slope and zero terminal velocity."""

    value = np.asarray(z, dtype=np.float64)
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("ease parameter must lie in [0, 1]")
    return value + 4.0 * value**3 - 7.0 * value**4 + 3.0 * value**5


def ease_from_stop(z):
    """Time reversal of :func:`ease_to_stop`."""

    value = np.asarray(z, dtype=np.float64)
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("ease parameter must lie in [0, 1]")
    return 1.0 - ease_to_stop(1.0 - value)


def _nonzero_unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def _incoming_unit(points: np.ndarray) -> np.ndarray:
    for index in range(len(points) - 1, 0, -1):
        unit = _nonzero_unit(points[index] - points[index - 1])
        if unit is not None:
            return unit
    raise ValueError("segment has no nonzero incoming tangent")


def _outgoing_unit(points: np.ndarray) -> np.ndarray:
    for index in range(len(points) - 1):
        unit = _nonzero_unit(points[index + 1] - points[index])
        if unit is not None:
            return unit
    raise ValueError("segment has no nonzero outgoing tangent")


def detect_sharp_boundaries(
    sampled_segments: Sequence[np.ndarray],
    segment_names: Sequence[str],
) -> tuple[SharpBoundary, ...]:
    """Classify internal boundaries from the ordered baseline samples."""

    if len(sampled_segments) != len(segment_names):
        raise ValueError("sampled segments and names must have equal length")
    boundaries = []
    for index, (previous, following) in enumerate(
        zip(sampled_segments[:-1], sampled_segments[1:])
    ):
        previous = np.asarray(previous, dtype=np.float64)
        following = np.asarray(following, dtype=np.float64)
        incoming = _incoming_unit(previous)
        outgoing = _outgoing_unit(following)
        cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        angle = float(math.degrees(math.acos(cosine)))
        boundaries.append(
            SharpBoundary(
                boundary_index=index,
                previous_segment=str(segment_names[index]),
                next_segment=str(segment_names[index + 1]),
                turn_angle_deg=angle,
                qualifies=angle >= TURN_THRESHOLD_DEG,
            )
        )
    return tuple(boundaries)


def _sample_arclength_fractions(
    points: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points must have shape [N, 2] with N >= 2")
    if fractions.ndim != 1 or len(fractions) < 2:
        raise ValueError("fractions must be a one-dimensional sample grid")
    if fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("fractions must include exact endpoints 0 and 1")
    if np.any(np.diff(fractions) <= 0.0):
        raise ValueError("fractions must be strictly increasing")

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("segment must have positive finite arc length")
    targets = fractions * total
    return np.column_stack(
        (
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        )
    )


def _corner_ease_fractions(
    base_intervals: int,
    *,
    ease_start: bool,
    ease_end: bool,
) -> np.ndarray:
    if isinstance(base_intervals, bool) or not isinstance(base_intervals, int):
        raise TypeError("base_intervals must be an integer")
    if base_intervals < 1:
        raise ValueError("base_intervals must be positive")
    if ease_start and ease_end:
        if base_intervals < 2 * BASE_WINDOW_INTERVALS + 2:
            raise ValueError("two-sided corner easing requires at least 22 intervals")
    elif (ease_start or ease_end) and base_intervals < BASE_WINDOW_INTERVALS + 1:
        raise ValueError("one-sided corner easing requires at least 11 intervals")

    start_ratio = BASE_WINDOW_INTERVALS / base_intervals
    end_ratio = 1.0 - start_ratio
    parts: list[np.ndarray] = []

    if ease_start:
        z = np.linspace(0.0, 1.0, RESAMPLED_WINDOW_INTERVALS + 1)
        parts.append(start_ratio * ease_from_stop(z))
    else:
        middle_end = end_ratio if ease_end else 1.0
        middle_intervals = base_intervals - (
            BASE_WINDOW_INTERVALS if ease_end else 0
        )
        parts.append(np.linspace(0.0, middle_end, middle_intervals + 1))

    if ease_start:
        middle_end = end_ratio if ease_end else 1.0
        middle_intervals = base_intervals - BASE_WINDOW_INTERVALS - (
            BASE_WINDOW_INTERVALS if ease_end else 0
        )
        middle = np.linspace(start_ratio, middle_end, middle_intervals + 1)
        parts.append(middle[1:])

    if ease_end:
        z = np.linspace(0.0, 1.0, RESAMPLED_WINDOW_INTERVALS + 1)
        ending = end_ratio + start_ratio * ease_to_stop(z)
        parts.append(ending[1:])

    fractions = np.concatenate(parts)
    expected_intervals = base_intervals + EXTRA_INTERVALS_PER_SIDE * (
        int(ease_start) + int(ease_end)
    )
    if len(fractions) != expected_intervals + 1:
        raise RuntimeError("corner-ease interval accounting is inconsistent")
    fractions[0] = 0.0
    fractions[-1] = 1.0
    if np.any(np.diff(fractions) <= 0.0):
        raise RuntimeError("corner-ease mapping is not strictly monotone")
    return fractions


def resample_segment_with_corner_easing(
    points: np.ndarray,
    base_intervals: int,
    *,
    ease_start: bool,
    ease_end: bool,
) -> np.ndarray:
    """Resample one unchanged spatial path with the frozen local timing rule."""

    fractions = _corner_ease_fractions(
        base_intervals,
        ease_start=ease_start,
        ease_end=ease_end,
    )
    return _sample_arclength_fractions(points, fractions)
