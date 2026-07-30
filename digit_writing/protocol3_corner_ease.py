"""Protocol3 ten-digit single-condition corner-ease overfit experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from digit_writing.corner_time_reparameterization import (
    BASE_WINDOW_INTERVALS,
    CORNER_EASE_TIMING,
    EXTRA_INTERVALS_PER_SIDE,
    RESAMPLED_WINDOW_INTERVALS,
    TURN_THRESHOLD_DEG,
)
from digit_writing.digit_geometry_final import build_digit_segments_units
from digit_writing.final_protocol_audit import _json_safe
from digit_writing.geometry import (
    _identified_json_hash,
    _identified_sample_hash,
    build_digit_trajectory,
    load_geometry_config,
)
from digit_writing.geometry_audit import build_workspace_audit
from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    protocol_config_sha256,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)


EXPECTED_INTERVALS = (170, 60, 210, 180, 200, 230, 200, 130, 200, 200)
EXPECTED_QUALIFYING_BOUNDARIES = {
    0: (),
    1: (),
    2: (1,),
    3: (0,),
    4: (0, 1),
    5: (0, 1),
    6: (),
    7: (0,),
    8: (),
    9: (),
}
NON_CORNER_DIGITS = (0, 1, 6, 8, 9)
SHARP_DIGITS = (2, 3, 4, 5, 7)
CASE_LABELS = tuple(f"digit{digit}_corner_ease" for digit in range(10))


def select_validation_history(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen stable/pass/fail checkpoint rule to metric rows."""

    passing = []
    stable = []
    stable_intervals = []
    current = []
    for row in rows:
        passed = bool(
            float(row["normalized_mean_error"]) <= 0.08
            and float(row["normalized_endpoint_error"]) <= 0.05
            and 0.85 <= float(row["path_length_ratio"]) <= 1.15
        )
        if passed:
            current.append(row)
            passing.append(row)
            if len(current) == 3:
                stable.extend(current)
                stable_intervals.append(
                    {
                        "start_update": int(current[0]["completed_updates"]),
                        "end_update": int(current[-1]["completed_updates"]),
                    }
                )
            elif len(current) > 3:
                stable.append(row)
                stable_intervals[-1]["end_update"] = int(
                    row["completed_updates"]
                )
        else:
            current = []
    if stable:
        status = "STABLE_PASS"
        candidates = stable
    elif passing:
        status = "PASS_UNSTABLE"
        candidates = passing
    else:
        status = "FAIL"
        candidates = list(rows)
    if not candidates:
        raise ValueError("validation history must not be empty")
    best = min(
        candidates,
        key=lambda row: (
            float(row["normalized_mean_error"]),
            int(row["completed_updates"]),
        ),
    )
    best_metrics = {
        "completed_updates": int(best["completed_updates"]),
        "normalized_mean_error": float(best["normalized_mean_error"]),
        "normalized_endpoint_error": float(best["normalized_endpoint_error"]),
        "path_length_ratio": float(best["path_length_ratio"]),
    }
    return {
        "status": status,
        "best_update": best_metrics["completed_updates"],
        "best_metrics": best_metrics,
        "stable_intervals": stable_intervals,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("configured path escapes the repository") from error
    return path


def _case_by_label(config: Mapping[str, Any], label: str) -> dict[str, Any]:
    matches = [case for case in config["cases"] if case["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown corner-ease case: {label}")
    return dict(matches[0])


def _case_output(run_root: Path, case: Mapping[str, Any]) -> Path:
    output = (run_root / str(case["output_subdirectory"])).resolve()
    try:
        output.relative_to(run_root.resolve())
    except ValueError as error:
        raise ValueError("case output escapes the experiment root") from error
    return output


def _validate_config(
    repository_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from train import _validate_protocol3_gate2_config

    root = Path(repository_root).resolve()
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("corner easing requires Protocol3")
    if config.get("run_kind") != "protocol3_corner_ease_overfit":
        raise ValueError("corner-ease run_kind is invalid")
    if config.get("variant") != "ten_digit_corner_ease_v1_overfit6000":
        raise ValueError("corner-ease variant is not frozen")
    if config.get("device") != "cpu":
        raise ValueError("corner-ease overfit is CPU-only")
    if config.get("scale_multiplier") != 2.5:
        raise ValueError("corner easing is frozen to scale2p50")
    if config.get("selected_reference_steps") != 100:
        raise ValueError("corner easing is frozen to ref100")
    if config.get("timing_mode") != CORNER_EASE_TIMING:
        raise ValueError("corner-ease timing identity differs")
    if config.get("condition_schedule") != (
        "protocol3_corner_ease_single_condition"
    ):
        raise ValueError("corner-ease condition schedule differs")

    source = _read_json(
        _resolve_repository_path(root, str(config["source_gate2_config"]))
    )
    _validate_protocol3_gate2_config(source)
    if source.get("selected_reference_steps") != 100:
        raise ValueError("source Gate2 config is not medium/ref100")
    for key in (
        "seed",
        "validation_seed",
        "device",
        "scale_multiplier",
        "selected_reference_steps",
        "model",
        "optimizer",
        "position_loss",
        "regularization",
    ):
        if config.get(key) != source.get(key):
            raise ValueError(f"corner-ease config differs from source Gate2: {key}")
    if config.get("training") != {
        "batch_size": 8,
        "max_updates": 6000,
        "validation_interval": 100,
    }:
        raise ValueError("corner-ease training boundary differs")
    expected_cases = tuple(
        (
            f"digit{digit}_corner_ease",
            digit,
            0,
            50,
            f"digit{digit}/seed42",
        )
        for digit in range(10)
    )
    actual_cases = tuple(
        (
            case.get("label"),
            case.get("digit"),
            case.get("direction_index"),
            case.get("delay_steps"),
            case.get("output_subdirectory"),
        )
        for case in config.get("cases", ())
    )
    if actual_cases != expected_cases:
        raise ValueError("corner-ease cases differ from the frozen ten digits")
    if config.get("corner_ease") != {
        "turn_angle_deg": TURN_THRESHOLD_DEG,
        "base_window_intervals": BASE_WINDOW_INTERVALS,
        "extra_intervals_per_side": EXTRA_INTERVALS_PER_SIDE,
        "resampled_window_intervals": RESAMPLED_WINDOW_INTERVALS,
        "incoming_speed_ratio_max": 0.1,
        "outgoing_speed_ratio_max": 0.1,
        "automatic_extension": False,
        "automatic_second_seed": False,
        "automatic_full10": False,
    }:
        raise ValueError("corner-ease intervention differs from the frozen rule")
    if config.get("checkpoint_selection") != {
        "stable_consecutive_evaluations": 3,
        "primary_metric": "normalized_mean_error",
        "tie_break": "earlier_update",
    }:
        raise ValueError("corner-ease checkpoint rule differs")
    geometry_identity = config.get("geometry_identity", {})
    if (
        geometry_identity.get("version") != "source-derived-temporal-v1"
        or geometry_identity.get("legacy_canonical_template_sha256_redefined")
        is not False
    ):
        raise ValueError("three-layer geometry identity is not frozen")
    output = config.get("output", {})
    if output.get("best_checkpoint") != "best_checkpoint.pt":
        raise ValueError("corner-ease best checkpoint name differs")
    if output.get("final_checkpoint") != "final_checkpoint.pt":
        raise ValueError("corner-ease final checkpoint name differs")
    if output.get("directory") != (
        "runs/digit_writing_original_protocol3/scale2p50/ref100/"
        "corner_ease_v1/single_digit_overfit"
    ):
        raise ValueError("corner-ease output namespace differs")

    geometry = load_geometry_config(
        _resolve_repository_path(root, str(config["geometry_config"]))
    )
    if geometry.timing_mode != CORNER_EASE_TIMING:
        raise ValueError("corner-ease geometry config timing differs")
    baseline = _read_json(
        _resolve_repository_path(root, str(config["baseline_reference"]))
    )
    if (
        baseline.get("protocol") != config["protocol"]
        or baseline.get("source_archive_sha256")
        != "172a9510ca5e3e9875695c98006e89bb860cf118fa940123b24c14aaea762ecc"
        or tuple(row.get("digit") for row in baseline.get("cases", ()))
        != SHARP_DIGITS
    ):
        raise ValueError("medium baseline reference identity differs")
    if baseline.get("identity") != {
        "scale_multiplier": 2.5,
        "selected_reference_steps": 100,
        "direction_index": 0,
        "delay_steps": 50,
        "seed": 42,
        "batch_size": 8,
        "max_updates": 6000,
        "learning_rate": 0.001,
    }:
        raise ValueError("medium baseline comparison conditions differ")
    if baseline.get("selection_rule") != {
        "mean_max": 0.08,
        "endpoint_max": 0.05,
        "path_ratio_min": 0.85,
        "path_ratio_max": 1.15,
        "stable_consecutive_evaluations": 3,
        "primary_metric": "normalized_mean_error",
        "tie_break": "earlier_update",
    }:
        raise ValueError("medium baseline checkpoint rule differs")
    return source, baseline


def _kinematic_metrics(points: np.ndarray, dt_seconds: float) -> dict[str, float]:
    velocity = np.diff(points, axis=0) / dt_seconds
    acceleration = np.diff(velocity, axis=0) / dt_seconds
    jerk = np.diff(acceleration, axis=0) / dt_seconds

    def values(vectors: np.ndarray) -> np.ndarray:
        return np.linalg.norm(vectors, axis=1) if len(vectors) else np.asarray([])

    speed = values(velocity)
    acceleration_norm = values(acceleration)
    jerk_norm = values(jerk)

    def percentile(array: np.ndarray, value: float) -> float:
        return float(np.percentile(array, value)) if array.size else 0.0

    return {
        "p95_speed_m_s": percentile(speed, 95.0),
        "peak_speed_m_s": float(speed.max()) if speed.size else 0.0,
        "p95_acceleration_m_s2": percentile(acceleration_norm, 95.0),
        "peak_acceleration_m_s2": (
            float(acceleration_norm.max()) if acceleration_norm.size else 0.0
        ),
        "p95_jerk_m_s3": percentile(jerk_norm, 95.0),
        "peak_jerk_m_s3": float(jerk_norm.max()) if jerk_norm.size else 0.0,
    }


def _maximum_distance_to_polyline(
    samples: np.ndarray,
    polyline: np.ndarray,
) -> float:
    starts = np.asarray(polyline[:-1], dtype=np.float64)
    vectors = np.asarray(polyline[1:] - polyline[:-1], dtype=np.float64)
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared_lengths > 0.0
    starts = starts[valid]
    vectors = vectors[valid]
    squared_lengths = squared_lengths[valid]
    if not len(starts):
        raise ValueError("derived path contains no nonzero polyline edge")
    maximum = 0.0
    for point in np.asarray(samples, dtype=np.float64):
        offsets = point - starts
        projection = np.clip(
            np.einsum("ij,ij->i", offsets, vectors) / squared_lengths,
            0.0,
            1.0,
        )
        closest = starts + projection[:, None] * vectors
        distance = float(np.linalg.norm(closest - point, axis=1).min())
        maximum = max(maximum, distance)
    return maximum


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def _local_corner_points(trajectory, boundary_index: int, window: int) -> np.ndarray:
    previous = trajectory.boundaries[boundary_index]
    following = trajectory.boundaries[boundary_index + 1]
    incoming = trajectory.points[previous.end_index - window : previous.end_index + 1]
    outgoing = trajectory.points[
        following.start_index + 1 : following.start_index + window + 1
    ]
    if len(incoming) != window + 1 or len(outgoing) != window:
        raise RuntimeError("corner-local kinematic window is incomplete")
    return np.vstack((incoming, outgoing))


def prepare_experiment(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    output_directory: str | Path,
    evidence_directory: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    source, baseline_reference = _validate_config(root, config)
    output = Path(output_directory).resolve()
    expected_output = _resolve_repository_path(root, config["output"]["directory"])
    if output != expected_output:
        raise ValueError("output directory differs from the checked-in config")
    if output.exists():
        raise FileExistsError(f"corner-ease output already exists: {output}")
    output.mkdir(parents=True)
    evidence = Path(evidence_directory).resolve()
    evidence.mkdir(parents=True, exist_ok=True)

    corner_config = load_geometry_config(
        _resolve_repository_path(root, config["geometry_config"])
    )
    baseline_config = load_geometry_config(
        _resolve_repository_path(root, source["geometry_config"])
    )
    segments_units = build_digit_segments_units()
    manifest_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    kinematic_rows: list[dict[str, Any]] = []
    non_corner_arrays_unchanged = True
    geometry_hashes_unchanged = True
    temporal_hashes_changed_only_where_expected = True
    maximum_path_support_distance_m = 0.0

    for digit in range(10):
        baseline = build_digit_trajectory(digit, baseline_config, 100)
        candidate = build_digit_trajectory(digit, corner_config, 100)
        if candidate.movement_intervals != EXPECTED_INTERVALS[digit]:
            raise RuntimeError(f"digit {digit} corner-ease intervals differ")
        if len(candidate.points) != candidate.movement_intervals + 1:
            raise RuntimeError("movement samples do not equal intervals plus one")
        if not np.isfinite(candidate.points).all():
            raise RuntimeError("corner-ease target contains NaN or Inf")
        step_lengths = np.linalg.norm(np.diff(candidate.points, axis=0), axis=1)
        if np.any(step_lengths <= 0.0):
            raise RuntimeError("corner-ease target contains a pause or reversal sample")
        if max(abs(a - b) for a, b in zip(_bbox(candidate.points), _bbox(baseline.points))) > 1e-10:
            raise RuntimeError("corner easing changed a target bounding box")
        if not np.array_equal(candidate.points[[0, -1]], baseline.points[[0, -1]]):
            raise RuntimeError("corner easing changed a digit endpoint")

        if digit in NON_CORNER_DIGITS:
            unchanged = np.array_equal(candidate.points, baseline.points)
            non_corner_arrays_unchanged &= unchanged
            if not unchanged:
                raise RuntimeError(f"non-corner digit {digit} target changed")

        for base_boundary, candidate_boundary, source_segment in zip(
            baseline.boundaries,
            candidate.boundaries,
            segments_units[digit],
        ):
            if base_boundary.name != candidate_boundary.name:
                raise RuntimeError("segment identity changed under corner easing")
            source_same = (
                base_boundary.source_geometry_sha256
                == candidate_boundary.source_geometry_sha256
            )
            derived_same = (
                base_boundary.derived_path_geometry_sha256
                == candidate_boundary.derived_path_geometry_sha256
            )
            geometry_hashes_unchanged &= source_same and derived_same
            if not source_same or not derived_same:
                raise RuntimeError("corner easing changed source or derived geometry")
            base_endpoint = baseline.points[base_boundary.end_index]
            candidate_endpoint = candidate.points[candidate_boundary.end_index]
            if not np.array_equal(base_endpoint, candidate_endpoint):
                raise RuntimeError("corner easing changed a segment connection point")
            path_distance = _maximum_distance_to_polyline(
                candidate.points[
                    candidate_boundary.start_index : candidate_boundary.end_index + 1
                ],
                source_segment.points_m(corner_config.global_scale_m_per_unit),
            )
            maximum_path_support_distance_m = max(
                maximum_path_support_distance_m, path_distance
            )
            if path_distance > 1e-10:
                raise RuntimeError("corner-ease sample left the source spatial path")

        temporal_equal = tuple(
            left.temporal_sampling_sha256 == right.temporal_sampling_sha256
            for left, right in zip(baseline.boundaries, candidate.boundaries)
        )
        retimed_segments = {
            segment_index
            for boundary_index in EXPECTED_QUALIFYING_BOUNDARIES[digit]
            for segment_index in (boundary_index, boundary_index + 1)
        }
        expected_temporal_equal = tuple(
            index not in retimed_segments for index in range(len(temporal_equal))
        )
        temporal_hashes_changed_only_where_expected &= (
            temporal_equal == expected_temporal_equal
        )
        if not temporal_hashes_changed_only_where_expected:
            raise RuntimeError("temporal sampling hashes changed outside their scope")

        actual_qualifying = tuple(
            int(row["boundary_index"])
            for row in candidate.corner_ease_boundaries
            if row["qualifies"]
        )
        if actual_qualifying != EXPECTED_QUALIFYING_BOUNDARIES[digit]:
            raise RuntimeError(f"digit {digit} sharp-boundary manifest differs")

        base_kinematics = _kinematic_metrics(
            baseline.points, baseline_config.dt_seconds
        )
        candidate_kinematics = _kinematic_metrics(
            candidate.points, corner_config.dt_seconds
        )
        timing_rows.append(
            {
                "digit": digit,
                "base_movement_intervals": baseline.movement_intervals,
                "sharp_corner_count": len(actual_qualifying),
                "corner_ease_movement_intervals": candidate.movement_intervals,
                "base_duration_s": baseline.movement_duration_s,
                "corner_ease_duration_s": candidate.movement_duration_s,
                "source_geometry_unchanged": True,
                "derived_path_geometry_unchanged": True,
                "temporal_sampling_changed": not all(temporal_equal),
                "baseline_source_geometry_sha256": _identified_json_hash(
                    [
                        boundary.source_geometry_sha256
                        for boundary in baseline.boundaries
                    ]
                ),
                "source_geometry_sha256": _identified_json_hash(
                    [
                        boundary.source_geometry_sha256
                        for boundary in candidate.boundaries
                    ]
                ),
                "baseline_derived_path_geometry_sha256": _identified_json_hash(
                    [
                        boundary.derived_path_geometry_sha256
                        for boundary in baseline.boundaries
                    ]
                ),
                "derived_path_geometry_sha256": _identified_json_hash(
                    [
                        boundary.derived_path_geometry_sha256
                        for boundary in candidate.boundaries
                    ]
                ),
                "baseline_temporal_sampling_sha256": _identified_sample_hash(
                    baseline.points / baseline_config.global_scale_m_per_unit
                ),
                "temporal_sampling_sha256": _identified_sample_hash(
                    candidate.points / corner_config.global_scale_m_per_unit
                ),
                "base_p95_acceleration_m_s2": base_kinematics[
                    "p95_acceleration_m_s2"
                ],
                "corner_ease_p95_acceleration_m_s2": candidate_kinematics[
                    "p95_acceleration_m_s2"
                ],
                "base_p95_jerk_m_s3": base_kinematics["p95_jerk_m_s3"],
                "corner_ease_p95_jerk_m_s3": candidate_kinematics[
                    "p95_jerk_m_s3"
                ],
            }
        )

        for corner in candidate.corner_ease_boundaries:
            boundary_index = int(corner["boundary_index"])
            previous_base = baseline.boundaries[boundary_index]
            following_base = baseline.boundaries[boundary_index + 1]
            previous = candidate.boundaries[boundary_index]
            following = candidate.boundaries[boundary_index + 1]
            connection = int(corner["corner_sample_index"])
            base_steps = np.concatenate(
                (
                    np.linalg.norm(
                        np.diff(
                            baseline.points[
                                previous_base.end_index - BASE_WINDOW_INTERVALS :
                                previous_base.end_index + 1
                            ],
                            axis=0,
                        ),
                        axis=1,
                    ),
                    np.linalg.norm(
                        np.diff(
                            baseline.points[
                                following_base.start_index :
                                following_base.start_index + BASE_WINDOW_INTERVALS + 1
                            ],
                            axis=0,
                        ),
                        axis=1,
                    ),
                )
            )
            regular_step = float(np.median(base_steps))
            incoming_step = float(
                np.linalg.norm(candidate.points[connection] - candidate.points[connection - 1])
            )
            outgoing_step = float(
                np.linalg.norm(candidate.points[connection + 1] - candidate.points[connection])
            )
            incoming_ratio = incoming_step / regular_step
            outgoing_ratio = outgoing_step / regular_step
            qualifies = bool(corner["qualifies"])
            row = {
                "digit": digit,
                "boundary_index": boundary_index,
                "previous_segment": corner["previous_segment"],
                "next_segment": corner["next_segment"],
                "turn_angle_deg": corner["turn_angle_deg"],
                "qualifies": qualifies,
                "base_window_intervals": BASE_WINDOW_INTERVALS,
                "extra_intervals_before": (
                    EXTRA_INTERVALS_PER_SIDE if qualifies else 0
                ),
                "extra_intervals_after": (
                    EXTRA_INTERVALS_PER_SIDE if qualifies else 0
                ),
                "corner_sample_index": connection,
                "incoming_step_m": incoming_step,
                "outgoing_step_m": outgoing_step,
                "regular_step_median_m": regular_step,
                "incoming_speed_ratio": incoming_ratio,
                "outgoing_speed_ratio": outgoing_ratio,
                "previous_source_geometry_sha256": previous.source_geometry_sha256,
                "previous_derived_path_geometry_sha256": previous.derived_path_geometry_sha256,
                "previous_temporal_sampling_sha256": previous.temporal_sampling_sha256,
                "next_source_geometry_sha256": following.source_geometry_sha256,
                "next_derived_path_geometry_sha256": following.derived_path_geometry_sha256,
                "next_temporal_sampling_sha256": following.temporal_sampling_sha256,
            }
            manifest_rows.append(row)
            if not qualifies:
                continue
            if incoming_ratio > 0.10 or outgoing_ratio > 0.10:
                raise RuntimeError("corner speed ratio exceeds the frozen maximum")
            if incoming_step == 0.0 or outgoing_step == 0.0:
                raise RuntimeError("corner easing introduced a physical pause")
            base_local = _local_corner_points(
                baseline, boundary_index, BASE_WINDOW_INTERVALS
            )
            candidate_local = _local_corner_points(
                candidate, boundary_index, RESAMPLED_WINDOW_INTERVALS
            )
            base_local_kinematics = _kinematic_metrics(
                base_local, baseline_config.dt_seconds
            )
            candidate_local_kinematics = _kinematic_metrics(
                candidate_local, corner_config.dt_seconds
            )
            acceleration_improved = bool(
                candidate_local_kinematics["p95_acceleration_m_s2"]
                <= base_local_kinematics["p95_acceleration_m_s2"]
            )
            jerk_improved = bool(
                candidate_local_kinematics["p95_jerk_m_s3"]
                <= base_local_kinematics["p95_jerk_m_s3"]
            )
            if not acceleration_improved or not jerk_improved:
                raise RuntimeError("corner easing failed the local kinematic gate")
            kinematic_rows.append(
                {
                    "digit": digit,
                    "boundary_index": boundary_index,
                    "previous_segment": corner["previous_segment"],
                    "next_segment": corner["next_segment"],
                    "turn_angle_deg": corner["turn_angle_deg"],
                    "base_local_p95_speed_m_s": base_local_kinematics[
                        "p95_speed_m_s"
                    ],
                    "corner_ease_local_p95_speed_m_s": candidate_local_kinematics[
                        "p95_speed_m_s"
                    ],
                    "base_local_p95_acceleration_m_s2": base_local_kinematics[
                        "p95_acceleration_m_s2"
                    ],
                    "corner_ease_local_p95_acceleration_m_s2": candidate_local_kinematics[
                        "p95_acceleration_m_s2"
                    ],
                    "base_local_p95_jerk_m_s3": base_local_kinematics[
                        "p95_jerk_m_s3"
                    ],
                    "corner_ease_local_p95_jerk_m_s3": candidate_local_kinematics[
                        "p95_jerk_m_s3"
                    ],
                    "base_local_peak_acceleration_m_s2": base_local_kinematics[
                        "peak_acceleration_m_s2"
                    ],
                    "corner_ease_local_peak_acceleration_m_s2": candidate_local_kinematics[
                        "peak_acceleration_m_s2"
                    ],
                    "base_local_peak_jerk_m_s3": base_local_kinematics[
                        "peak_jerk_m_s3"
                    ],
                    "corner_ease_local_peak_jerk_m_s3": candidate_local_kinematics[
                        "peak_jerk_m_s3"
                    ],
                    "p95_acceleration_not_increased": acceleration_improved,
                    "p95_jerk_not_increased": jerk_improved,
                }
            )

    workspace_audit = build_workspace_audit(corner_config)
    if not workspace_audit["passed"]:
        raise RuntimeError("corner-ease workspace audit failed")
    if maximum_path_support_distance_m > 1e-10:
        raise RuntimeError("corner-ease path support tolerance failed")
    _write_csv(output / "corner_ease_manifest.csv", manifest_rows)
    _write_csv(output / "target_timing_summary.csv", timing_rows)
    _write_csv(output / "target_kinematics_comparison.csv", kinematic_rows)
    _write_json(evidence / "resolved_config.json", config)
    _write_json(evidence / "source_gate2_config.json", source)
    _write_json(evidence / "baseline_reference.json", baseline_reference)
    _write_json(evidence / "workspace_audit.json", workspace_audit)
    summary = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_corner_ease_prepare",
        "git_identity": current_git_identity(root),
        "protocol_config_sha256": protocol_config_sha256(config),
        "source_gate2_config_sha256": protocol_config_sha256(source),
        "baseline_reference_sha256": protocol_config_sha256(baseline_reference),
        "movement_intervals": list(EXPECTED_INTERVALS),
        "qualifying_boundaries": {
            str(key): list(value)
            for key, value in EXPECTED_QUALIFYING_BOUNDARIES.items()
        },
        "non_corner_arrays_unchanged": non_corner_arrays_unchanged,
        "source_and_derived_geometry_hashes_unchanged": geometry_hashes_unchanged,
        "temporal_hashes_changed_only_where_expected": temporal_hashes_changed_only_where_expected,
        "maximum_path_support_distance_m": maximum_path_support_distance_m,
        "workspace_passed": workspace_audit["passed"],
        "training_updates": 0,
        "passed": True,
    }
    _write_json(evidence / "preflight_summary.json", summary)
    return summary


def _hp_for_case(config: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    from train import PROTOCOL3_DELAYS, _base_hp_from_config

    hp = _base_hp_from_config(config)
    hp.update(
        {
            "variant": case["label"],
            "condition_schedule": "protocol3_corner_ease_single_condition",
            "gate2_direction_index": int(case["direction_index"]),
            "gate2_delay_index": PROTOCOL3_DELAYS.index(case["delay_steps"]),
            "stop_after_updates": 6000,
            "artifact_profile": "protocol3_corner_ease_minimal_v1",
            "env_kwargs": {"geometry_config_path": config["geometry_config"]},
        }
    )
    return hp


def _save_best_overlay(
    path: Path,
    actual,
    target,
    *,
    digit: int,
    status: str,
    best_update: int,
    metrics: Mapping[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual_mean = actual.detach().mean(dim=0).cpu().numpy()
    target_mean = target.detach().mean(dim=0).cpu().numpy()
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot(target_mean[:, 0], target_mean[:, 1], "k--", label="target")
    axis.plot(actual_mean[:, 0], actual_mean[:, 1], label="actual")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    axis.set_title(
        f"digit {digit} | {status} | update {best_update}\n"
        f"mean={metrics['normalized_mean_error']:.6f}  "
        f"endpoint={metrics['normalized_endpoint_error']:.6f}  "
        f"path={metrics['path_length_ratio']:.6f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _audit_best_checkpoint(
    policy,
    hp: Mapping[str, Any],
    case: Mapping[str, Any],
    workspace_audit: Mapping[str, Any],
    output: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    import motornet as mn
    import torch

    from digit_writing.final_protocol_audit import _all_finite
    from digit_writing.protocol3_gate2 import (
        _dynamic_safety_metrics,
        _movement_metrics,
    )
    from train import DIGIT_ENV_CLASSES

    before_hash = state_dict_sha256(policy.state_dict())
    policy.eval()
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    environment = DIGIT_ENV_CLASSES[int(case["digit"])](
        effector=effector,
        **hp["env_kwargs"],
    )
    batch_size = int(hp["batch_size"])
    observation, info = environment.reset(
        testing=False,
        seed=int(hp["validation_seed"]),
        options={
            "batch_size": batch_size,
            "reach_conds": np.full(
                batch_size, int(case["direction_index"]), dtype=np.int64
            ),
            "speed_cond": 0,
            "delay_cond": int(hp["gate2_delay_index"]),
            "deterministic": True,
        },
    )
    x = torch.zeros((batch_size, hp["hid_size"]))
    h = torch.zeros_like(x)
    positions = []
    targets = []
    actions = []
    muscles = [info["states"]["muscle"][:, 0].unsqueeze(1)]
    joint_positions = []
    finite = _all_finite(observation) and _all_finite(info)
    terminated = False
    timestep = 0
    while not terminated:
        with torch.no_grad():
            x, h, action = policy(observation, x, h, noise=False)
            observation, _, terminated, info = environment.step(timestep, action)
        positions.append(info["states"]["fingertip"][:, None, :])
        targets.append(info["goal"][:, None, :])
        actions.append(action[:, None, :])
        muscles.append(info["states"]["muscle"][:, 0].unsqueeze(1))
        joint_positions.append(
            info["states"]["joint"][:, : environment.effector.dof].unsqueeze(1)
        )
        finite &= _all_finite(observation) and _all_finite(info)
        timestep += 1

    positions = torch.cat(positions, dim=1)
    targets = torch.cat(targets, dim=1)
    actions = torch.cat(actions, dim=1)
    muscles = torch.cat(muscles, dim=1)
    joint_positions = torch.cat(joint_positions, dim=1)
    movement_actual, movement_target, movement_metrics = _movement_metrics(
        positions, targets, environment.epoch_bounds
    )
    target_span = movement_target.amax(dim=1) - movement_target.amin(dim=1)
    movement_metrics["target_bbox_diagonal_m"] = float(
        torch.linalg.vector_norm(target_span, dim=-1).mean()
    )
    dynamic_safety = _dynamic_safety_metrics(
        environment, positions, joint_positions
    )
    boundary_metrics = {
        "muscle_activation_max": float(muscles.max()),
        "muscle_activation_fraction_ge_0_99": float(
            (muscles >= 0.99).float().mean()
        ),
        "muscle_excitation_fraction_le_0_01": float(
            (actions <= 0.01).float().mean()
        ),
        "muscle_excitation_fraction_ge_0_99": float(
            (actions >= 0.99).float().mean()
        ),
    }
    workspace_minima = {
        "target_min_inner_radius_margin_m": workspace_audit[
            "radial_reach_margin_m"
        ]["minimum_inner"],
        "target_min_outer_radius_margin_m": workspace_audit[
            "radial_reach_margin_m"
        ]["minimum_outer"],
        "target_min_joint_margin_rad": workspace_audit[
            "joint_angle_margin_rad"
        ]["overall_minimum"],
    }
    finite &= all(
        math.isfinite(value)
        for group in (movement_metrics, dynamic_safety, boundary_metrics)
        for value in group.values()
    )
    safety_passed = bool(
        workspace_audit["passed"]
        and min(
            workspace_minima["target_min_inner_radius_margin_m"],
            workspace_minima["target_min_outer_radius_margin_m"],
            dynamic_safety["min_inner_radius_margin_m"],
            dynamic_safety["min_outer_radius_margin_m"],
        )
        >= 0.02
        and min(
            workspace_minima["target_min_joint_margin_rad"],
            dynamic_safety["min_joint_margin_rad"],
        )
        >= 0.10
    )
    after_hash = state_dict_sha256(policy.state_dict())
    selected_metrics = selection["best_metrics"]
    for name in (
        "normalized_mean_error",
        "normalized_endpoint_error",
        "path_length_ratio",
    ):
        if not math.isclose(
            float(movement_metrics[name]),
            float(selected_metrics[name]),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise RuntimeError("best-checkpoint audit differs from selection metrics")
    _save_best_overlay(
        output / "best_movement_overlay.png",
        movement_actual,
        movement_target,
        digit=int(case["digit"]),
        status=str(selection["status"]),
        best_update=int(selection["best_update"]),
        metrics=movement_metrics,
    )
    return {
        "optimizer_steps_during_audit": 0,
        "all_values_finite": finite,
        "state_sha256_before": before_hash,
        "state_sha256_after": after_hash,
        "state_unchanged": before_hash == after_hash,
        "movement_metrics": movement_metrics,
        "workspace_minima": workspace_minima,
        "dynamic_safety": dynamic_safety,
        "boundary_metrics": boundary_metrics,
        "safety_passed": safety_passed,
        "engineering_passed": bool(finite and safety_passed and before_hash == after_hash),
    }


def run_case(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    run_root: str | Path,
    evidence_directory: str | Path,
    case_label: str,
) -> dict[str, Any]:
    from train import (
        _digit_env_dict,
        load_digit_policy_checkpoint,
        train_subsets_base_model,
    )

    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    _validate_config(root, config)
    expected_root = _resolve_repository_path(root, config["output"]["directory"])
    actual_root = Path(run_root).resolve()
    if actual_root != expected_root:
        raise ValueError("run root differs from the checked-in config")
    case = _case_by_label(config, case_label)
    output = _case_output(actual_root, case)
    if not output.is_dir():
        raise FileNotFoundError("server runner must create the case log directory")
    unexpected = {path.name for path in output.iterdir()} - {"run.log"}
    if unexpected:
        raise FileExistsError(f"corner-ease case output is not empty: {sorted(unexpected)}")
    resolved = dict(config)
    resolved["active_case"] = case
    _write_json(output / "resolved_config.json", resolved)
    evidence = Path(evidence_directory).resolve()
    workspace_audit = _read_json(evidence / "workspace_audit.json")
    hp = _hp_for_case(config, case)
    training_summary = train_subsets_base_model(
        str(output),
        config["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((int(case["digit"]),)),
    )
    if int(training_summary["updates"]) != 6000:
        raise RuntimeError("corner-ease case did not stop at 6000 updates")
    selection = training_summary.get("corner_ease_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("corner-ease training did not produce a selection")

    best_checkpoint = output / config["output"]["best_checkpoint"]
    best_policy, best_payload = load_digit_policy_checkpoint(
        best_checkpoint,
        expected_variant=case["label"],
    )
    best_state = validate_protocol3_resume_checkpoint(best_payload, config)
    if int(best_state["completed_updates"]) != int(selection["best_update"]):
        raise RuntimeError("best checkpoint update differs from its selection")
    best_audit = _audit_best_checkpoint(
        best_policy,
        hp,
        case,
        workspace_audit,
        output,
        selection,
    )

    final_checkpoint = output / config["output"]["final_checkpoint"]
    _, final_payload = load_digit_policy_checkpoint(
        final_checkpoint,
        expected_variant=case["label"],
    )
    final_state = validate_protocol3_resume_checkpoint(final_payload, config)
    checkpoint_complete = bool(
        int(final_state["completed_updates"]) == 6000
        and final_payload["git_identity"] == hp["git_identity"]
    )
    engineering_passed = bool(
        checkpoint_complete and best_audit["engineering_passed"]
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_corner_ease_case",
        "case": case,
        "git_identity": hp["git_identity"],
        "completed_updates": 6000,
        "status": selection["status"],
        "best_update": selection["best_update"],
        "best_metrics": selection["best_metrics"],
        "stable_intervals": selection["stable_intervals"],
        "checkpoint_complete": checkpoint_complete,
        "best_checkpoint_audit": best_audit,
        "engineering_passed": engineering_passed,
        "automatic_extension_started": False,
        "automatic_second_seed_started": False,
        "formal_full10_started": False,
    }
    _write_json(output / "run_summary.json", result)
    expected_files = {
        "resolved_config.json",
        "metrics.jsonl",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "best_movement_overlay.png",
        "run_summary.json",
        "run.log",
    }
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"corner-ease case artifacts differ; actual={sorted(actual_files)}"
        )
    return result


def _save_summary_panels(
    run_root: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    for axis, summary in zip(axes.flat, summaries):
        case = summary["case"]
        image_path = _case_output(run_root, case) / "best_movement_overlay.png"
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
        axis.set_title(
            f"digit {case['digit']} | {summary['status']} | "
            f"u={summary['best_update']}"
        )
    figure.tight_layout()
    figure.savefig(run_root / "ten_digit_best_overlays.png", dpi=180)
    plt.close(figure)

    stable = [summary for summary in summaries if summary["status"] == "STABLE_PASS"]
    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = list(axes.flat)
    for axis in axes:
        axis.axis("off")
    if stable:
        for axis, summary in zip(axes, stable):
            image_path = (
                _case_output(run_root, summary["case"])
                / "best_movement_overlay.png"
            )
            axis.imshow(plt.imread(image_path))
            axis.set_title(
                f"digit {summary['case']['digit']} | STABLE_PASS"
            )
    else:
        axes[0].text(
            0.5,
            0.5,
            "No STABLE_PASS digits",
            ha="center",
            va="center",
            fontsize=18,
        )
    figure.tight_layout()
    figure.savefig(run_root / "displayable_digits_overlays.png", dpi=180)
    plt.close(figure)


def summarize_experiment(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    _, baseline_reference = _validate_config(root, config)
    expected_root = _resolve_repository_path(root, config["output"]["directory"])
    output = Path(run_root).resolve()
    if output != expected_root:
        raise ValueError("summary run root differs from the checked-in config")
    summaries = [
        _read_json(_case_output(output, case) / "run_summary.json")
        for case in config["cases"]
    ]
    if [row["case"]["label"] for row in summaries] != list(CASE_LABELS):
        raise RuntimeError("corner-ease case summaries are incomplete or reordered")
    if any(int(row["completed_updates"]) != 6000 for row in summaries):
        raise RuntimeError("one or more corner-ease cases did not finish at 6000")
    if any(not row["engineering_passed"] for row in summaries):
        raise RuntimeError("one or more corner-ease cases failed engineering or safety")

    baseline_by_digit = {
        int(row["digit"]): row for row in baseline_reference["cases"]
    }
    metric_rows = []
    comparisons = []
    for summary in summaries:
        digit = int(summary["case"]["digit"])
        metrics = summary["best_checkpoint_audit"]["movement_metrics"]
        row = {
            "digit": digit,
            "status": summary["status"],
            "best_update": summary["best_update"],
            "normalized_mean_error": metrics["normalized_mean_error"],
            "normalized_endpoint_error": metrics["normalized_endpoint_error"],
            "target_path_length_m": metrics["target_path_length_m"],
            "actual_path_length_m": metrics["actual_path_length_m"],
            "path_length_ratio": metrics["path_length_ratio"],
            "target_bbox_diagonal_m": metrics["target_bbox_diagonal_m"],
            "engineering_passed": summary["engineering_passed"],
            "safety_passed": summary["best_checkpoint_audit"]["safety_passed"],
        }
        metric_rows.append(row)
        if digit in baseline_by_digit:
            baseline = baseline_by_digit[digit]
            comparisons.append(
                {
                    "digit": digit,
                    "baseline_status": baseline["status"],
                    "corner_ease_status": summary["status"],
                    "baseline_best_update": baseline["best_update"],
                    "corner_ease_best_update": summary["best_update"],
                    "baseline_normalized_mean_error": baseline[
                        "best_normalized_mean_error"
                    ],
                    "corner_ease_normalized_mean_error": metrics[
                        "normalized_mean_error"
                    ],
                    "mean_error_change": (
                        metrics["normalized_mean_error"]
                        - baseline["best_normalized_mean_error"]
                    ),
                    "baseline_normalized_endpoint_error": baseline[
                        "best_normalized_endpoint_error"
                    ],
                    "corner_ease_normalized_endpoint_error": metrics[
                        "normalized_endpoint_error"
                    ],
                    "endpoint_error_change": (
                        metrics["normalized_endpoint_error"]
                        - baseline["best_normalized_endpoint_error"]
                    ),
                    "baseline_path_length_ratio": baseline[
                        "best_path_length_ratio"
                    ],
                    "corner_ease_path_length_ratio": metrics[
                        "path_length_ratio"
                    ],
                    "absolute_path_ratio_error_change": (
                        abs(metrics["path_length_ratio"] - 1.0)
                        - abs(baseline["best_path_length_ratio"] - 1.0)
                    ),
                    "baseline_final_update": baseline["final_update"],
                    "baseline_final_normalized_mean_error": baseline[
                        "final_normalized_mean_error"
                    ],
                    "baseline_final_normalized_endpoint_error": baseline[
                        "final_normalized_endpoint_error"
                    ],
                    "baseline_final_path_length_ratio": baseline[
                        "final_path_length_ratio"
                    ],
                }
            )
    _write_csv(output / "ten_digit_metrics.csv", metric_rows)
    _save_summary_panels(output, summaries)

    stable_digits = [
        int(row["case"]["digit"])
        for row in summaries
        if row["status"] == "STABLE_PASS"
    ]
    candidate_supported = len(stable_digits) == 10
    lines = [
        "# Protocol3 ten-digit corner-ease overfit report",
        "",
        f"Repository HEAD: `{summaries[0]['git_identity']['repository_head']}`  ",
        f"mRNNTorch HEAD: `{summaries[0]['git_identity']['mrnntorch_recorded_head']}`  ",
        "Experiment: scale2p50 / ref100 / direction0 / delay50 / seed42 / 6000 updates",
        "",
        "## Ten-digit result",
        "",
        "| digit | status | best update | mean | endpoint | path ratio |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['digit']} | {row['status']} | {row['best_update']} | "
            f"{row['normalized_mean_error']:.6f} | "
            f"{row['normalized_endpoint_error']:.6f} | "
            f"{row['path_length_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "STABLE_PASS digits: "
            + (", ".join(str(value) for value in stable_digits) if stable_digits else "none"),
            "",
            "Only STABLE_PASS digits are included in `displayable_digits_overlays.png`.",
            "",
            "## Matched medium baseline comparison",
            "",
            "Baseline rows were reselected read-only with the same three-consecutive-pass rule.",
            "Negative mean/endpoint deltas and negative absolute path-ratio-error deltas are improvements.",
            "",
            "| digit | baseline | corner ease | mean delta | endpoint delta | abs path-error delta |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['digit']} | {row['baseline_status']} | "
            f"{row['corner_ease_status']} | {row['mean_error_change']:.6f} | "
            f"{row['endpoint_error_change']:.6f} | "
            f"{row['absolute_path_ratio_error_change']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Baseline final@6000 is retained as a secondary, non-selected reference:",
            "",
            "| digit | final mean | final endpoint | final path ratio |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['digit']} | "
            f"{row['baseline_final_normalized_mean_error']:.6f} | "
            f"{row['baseline_final_normalized_endpoint_error']:.6f} | "
            f"{row['baseline_final_path_length_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Cutting, omitted segments, compression and overshoot require human review of "
            "`ten_digit_best_overlays.png`; scalar improvements are not used as a visual claim.",
            "",
            "## Safety and scope",
            "",
            "All ten best-checkpoint audits were finite, state-preserving and passed the frozen workspace/joint margins.",
            "No automatic extension, second seed, fallback or formal full10 was started.",
            "",
            "Full10 candidate conclusion: "
            + (
                "all ten digits reached STABLE_PASS; eligible only for manual candidate review."
                if candidate_supported
                else "not supported yet because fewer than ten digits reached STABLE_PASS."
            ),
            "",
        ]
    )
    with (output / "TEN_DIGIT_CORNER_EASE_OVERFIT_REPORT.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(lines))
    return {
        "completed_cases": 10,
        "stable_pass_digits": stable_digits,
        "engineering_passed": True,
        "full10_candidate_supported": candidate_supported,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repository-root", required=True)
    prepare.add_argument("--protocol-config", required=True)
    prepare.add_argument("--output-directory", required=True)
    prepare.add_argument("--evidence-directory", required=True)
    run = subparsers.add_parser("run-case")
    run.add_argument("--repository-root", required=True)
    run.add_argument("--protocol-config", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--evidence-directory", required=True)
    run.add_argument("--case-label", required=True, choices=CASE_LABELS)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--repository-root", required=True)
    summarize.add_argument("--protocol-config", required=True)
    summarize.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        result = prepare_experiment(
            args.repository_root,
            args.protocol_config,
            args.output_directory,
            args.evidence_directory,
        )
        print(f"CORNER_EASE_PREFLIGHT_PASS={int(result['passed'])}")
        print("TRAINING_UPDATES=0")
    elif args.command == "run-case":
        result = run_case(
            args.repository_root,
            args.protocol_config,
            args.run_root,
            args.evidence_directory,
            args.case_label,
        )
        print(f"CASE_STATUS={result['status']}")
        print(f"COMPLETED_UPDATES={result['completed_updates']}")
        print(f"ENGINEERING_PASSED={int(result['engineering_passed'])}")
    else:
        result = summarize_experiment(
            args.repository_root,
            args.protocol_config,
            args.run_root,
        )
        print("COMPLETED_CASES=10")
        print(
            "STABLE_PASS_DIGITS="
            + ",".join(str(value) for value in result["stable_pass_digits"])
        )
        print(f"ENGINEERING_PASSED={int(result['engineering_passed'])}")
        print(
            "FULL10_CANDIDATE_SUPPORTED="
            f"{int(result['full10_candidate_supported'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
