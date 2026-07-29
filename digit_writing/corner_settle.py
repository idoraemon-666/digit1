"""Isolated corner-settle timing overlay for the Protocol3 diagnostic gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from digit_writing.geometry import DigitTrajectory, GeometryConfig, build_digit_trajectory


CORNER_DIGITS = (2, 3, 4, 5, 7)
CORNER_TURN_THRESHOLD_DEG = 60.0
CORNER_SETTLE_INTERVALS = 10
CORNER_SETTLE_PHYSICAL_MS = 100


@dataclass(frozen=True)
class CornerBoundary:
    boundary_index: int
    previous_segment: str
    next_segment: str
    turn_angle_deg: float
    turn_qualifies: bool
    settle_applied: bool
    settle_intervals: int
    base_connection_index: int
    connection_index: int
    previous_segment_start_index: int
    settle_start_index: int
    settle_end_index: int
    post_start_index: int
    next_segment_end_index: int
    corner_point_x_m: float
    corner_point_y_m: float


@dataclass(frozen=True)
class CornerSettleTrajectory:
    base: DigitTrajectory
    points: np.ndarray
    movement_intervals: int
    movement_duration_s: float
    corners: tuple[CornerBoundary, ...]


def _nonzero_unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def _incoming_unit(points: np.ndarray, start: int, end: int) -> np.ndarray:
    for index in range(end, start, -1):
        unit = _nonzero_unit(points[index] - points[index - 1])
        if unit is not None:
            return unit
    raise ValueError("previous segment has no nonzero sampled tangent")


def _outgoing_unit(points: np.ndarray, start: int, end: int) -> np.ndarray:
    for index in range(start, end):
        unit = _nonzero_unit(points[index + 1] - points[index])
        if unit is not None:
            return unit
    raise ValueError("next segment has no nonzero sampled tangent")


def _turn_angle_deg(
    points: np.ndarray,
    previous_start: int,
    connection: int,
    next_end: int,
) -> float:
    incoming = _incoming_unit(points, previous_start, connection)
    outgoing = _outgoing_unit(points, connection, next_end)
    cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def build_corner_settle_trajectory(
    digit: int,
    config: GeometryConfig,
    reference_steps: int,
    *,
    settle_intervals: int,
    spatial_angle_rad: float = 0.0,
    anchor: Sequence[float] = (0.0, 0.0),
) -> CornerSettleTrajectory:
    """Insert only repeated target points after qualifying internal boundaries."""

    if digit not in CORNER_DIGITS:
        raise ValueError("corner-settle diagnostic supports only digits 2, 3, 4, 5, and 7")
    if isinstance(settle_intervals, bool) or not isinstance(settle_intervals, int):
        raise TypeError("settle_intervals must be an integer")
    if settle_intervals not in {0, CORNER_SETTLE_INTERVALS}:
        raise ValueError("settle_intervals must be 0 or the frozen 10 intervals")

    base = build_digit_trajectory(
        digit,
        config,
        reference_steps,
        spatial_angle_rad=spatial_angle_rad,
        anchor=anchor,
    )
    base_points = np.asarray(base.points, dtype=np.float64)
    path_parts: list[np.ndarray] = []
    corners: list[CornerBoundary] = []
    cursor = 0
    inserted_before = 0

    for boundary_index, (previous, following) in enumerate(
        zip(base.boundaries[:-1], base.boundaries[1:])
    ):
        if previous.end_index != following.start_index:
            raise RuntimeError("adjacent primitive boundaries do not share one point")
        connection = previous.end_index
        turn_angle = _turn_angle_deg(
            base_points,
            previous.start_index,
            connection,
            following.end_index,
        )
        qualifies = turn_angle >= CORNER_TURN_THRESHOLD_DEG
        applied_intervals = settle_intervals if qualifies else 0
        new_connection = connection + inserted_before
        previous_start = previous.start_index + inserted_before
        if boundary_index and corners[-1].settle_applied:
            previous_start += 1

        path_parts.append(base_points[cursor : connection + 1])
        if applied_intervals:
            path_parts.append(
                np.repeat(base_points[connection : connection + 1], applied_intervals, axis=0)
            )
        inserted_after = inserted_before + applied_intervals
        settle_start = new_connection + 1
        settle_end = settle_start + applied_intervals
        corner_point = base_points[connection]
        corners.append(
            CornerBoundary(
                boundary_index=boundary_index,
                previous_segment=previous.name,
                next_segment=following.name,
                turn_angle_deg=turn_angle,
                turn_qualifies=qualifies,
                settle_applied=bool(applied_intervals),
                settle_intervals=applied_intervals,
                base_connection_index=connection,
                connection_index=new_connection,
                previous_segment_start_index=previous_start,
                settle_start_index=settle_start,
                settle_end_index=settle_end,
                post_start_index=settle_end,
                next_segment_end_index=following.end_index + inserted_after,
                corner_point_x_m=float(corner_point[0]),
                corner_point_y_m=float(corner_point[1]),
            )
        )
        cursor = connection + 1
        inserted_before = inserted_after

    path_parts.append(base_points[cursor:])
    points = np.vstack(path_parts)
    expected_added = sum(corner.settle_intervals for corner in corners)
    if len(points) != len(base_points) + expected_added:
        raise RuntimeError("corner-settle insertion count differs from its manifest")
    points.setflags(write=False)
    movement_intervals = len(points) - 1
    return CornerSettleTrajectory(
        base=base,
        points=points,
        movement_intervals=movement_intervals,
        movement_duration_s=movement_intervals * config.dt_seconds,
        corners=tuple(corners),
    )


def corner_manifest_rows(
    digit: int,
    condition: str,
    trajectory: CornerSettleTrajectory,
) -> list[dict[str, object]]:
    rows = []
    for corner in trajectory.corners:
        row = asdict(corner)
        row.update(
            {
                "digit": digit,
                "condition": condition,
                "segment_type": (
                    "corner_settle" if corner.settle_applied else "baseline_boundary"
                ),
                "segment_name": (
                    f"corner_settle_{corner.boundary_index}"
                    if corner.settle_applied
                    else ""
                ),
            }
        )
        rows.append(row)
    return rows


def save_corner_audit_artifacts(
    output,
    movement_actual,
    movement_target,
    corners: tuple[CornerBoundary, ...],
    dt_seconds: float,
) -> dict[str, object]:
    """Compute uniquely defined local metrics and render diagnostic plots."""

    from pathlib import Path

    output = Path(output)
    actual = movement_actual.detach().cpu().numpy()
    target = movement_target.detach().cpu().numpy()
    corner_metrics = []

    for corner in corners:
        pre_start = max(
            corner.previous_segment_start_index,
            corner.connection_index - 9,
        )
        post_end = min(
            corner.next_segment_end_index + 1,
            corner.post_start_index + 10,
        )
        window_indices = np.arange(pre_start, post_end, dtype=np.int64)
        local_error = np.linalg.norm(
            actual[:, window_indices] - target[:, window_indices], axis=-1
        )
        corner_target = target[:, corner.connection_index]
        corner_distance = np.linalg.norm(
            actual[:, window_indices] - corner_target[:, None, :], axis=-1
        )

        progress_start = (
            corner.settle_end_index - 1
            if corner.settle_intervals
            else corner.connection_index
        )
        progress_end = min(
            corner.next_segment_end_index,
            corner.post_start_index + 9,
        )
        actual_progress = np.linalg.norm(
            np.diff(actual[:, progress_start : progress_end + 1], axis=1),
            axis=-1,
        ).sum(axis=1)
        target_progress = np.linalg.norm(
            np.diff(target[:, progress_start : progress_end + 1], axis=1),
            axis=-1,
        ).sum(axis=1)
        if np.any(target_progress <= 0.0):
            raise RuntimeError("corner post-progress target denominator is not positive")
        corner_metrics.append(
            {
                "boundary_index": corner.boundary_index,
                "previous_segment": corner.previous_segment,
                "next_segment": corner.next_segment,
                "turn_angle_deg": corner.turn_angle_deg,
                "settle_applied": corner.settle_applied,
                "settle_intervals": corner.settle_intervals,
                "window_start_index": int(pre_start),
                "window_end_index_exclusive": int(post_end),
                "post_progress_start_index": int(progress_start),
                "post_progress_end_index": int(progress_end),
                "corner_local_mean_error_m": float(local_error.mean()),
                "corner_local_max_error_m": float(local_error.max()),
                "corner_miss_distance_m": float(corner_distance.min(axis=1).mean()),
                "post_corner_progress_ratio": float(
                    (actual_progress / target_progress).mean()
                ),
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual_mean = actual.mean(axis=0)
    target_mean = target.mean(axis=0)
    figure, axes = plt.subplots(1, len(corners), figsize=(5 * len(corners), 4))
    axes = np.atleast_1d(axes)
    for axis, corner, metrics in zip(axes, corners, corner_metrics):
        start = metrics["window_start_index"]
        end = metrics["window_end_index_exclusive"]
        axis.plot(target_mean[start:end, 0], target_mean[start:end, 1], "k.--", label="target")
        axis.plot(actual_mean[start:end, 0], actual_mean[start:end, 1], ".-", label="actual")
        point = target_mean[corner.connection_index]
        axis.scatter((point[0],), (point[1],), marker="x", s=80, label="corner")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"corner {corner.boundary_index}")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "corner_local_overlay.png", dpi=180)
    plt.close(figure)

    actual_speed = np.linalg.norm(np.diff(actual_mean, axis=0), axis=-1) / dt_seconds
    target_speed = np.linalg.norm(np.diff(target_mean, axis=0), axis=-1) / dt_seconds
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(target_speed, "k--", label="target speed")
    axis.plot(actual_speed, label="actual speed")
    for corner in corners:
        if corner.settle_applied:
            axis.axvspan(
                corner.settle_start_index - 1,
                corner.settle_end_index - 1,
                color="tab:orange",
                alpha=0.2,
            )
    axis.set_xlabel("movement interval")
    axis.set_ylabel("speed (m/s)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "fingertip_speed.png", dpi=180)
    plt.close(figure)

    return {
        "corner_metric_definition": {
            "pre_samples": "last 10 non-settle target samples ending at the connection",
            "post_samples": "first 10 non-settle target samples after the connection/settle",
            "corner_miss_distance_m": "mean across batch of the minimum actual-to-corner Euclidean distance inside the local window",
            "post_corner_progress_ratio": "mean across batch of actual path length divided by target path length over the first 10 post-corner samples",
        },
        "corners": corner_metrics,
        "corner_local_overlay": "corner_local_overlay.png",
        "fingertip_speed_plot": "fingertip_speed.png",
    }
