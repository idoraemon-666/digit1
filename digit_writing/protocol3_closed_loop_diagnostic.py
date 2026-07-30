"""Zero-step joint closed-loop diagnostic for Protocol3 digit 0 and digit 8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from digit_writing.phase_normalized_loss import PHASE_WEIGHTS, position_l1_metrics
from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)
from losses import l1_muscle_act, l1_rate, l1_weight, simple_dynamics


SOURCE_HEAD = "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925"
CONTINUATION_HEAD = "76349d51222eed62bdb26d78d251b7d57760f2e8"
MRNN_HEAD = "ac0c4f589eae37bbde63968912925de99232e306"
CHECKPOINT_LABELS = (
    "d0_source_best4800",
    "d0_source_final6000",
    "d0_lr1e3_final8000",
    "d0_lr3e4_final8000",
    "d8_source_best5800",
    "d8_source_final6000",
    "d9_positive_control_best4800",
)
EXPECTED_CHECKPOINTS = (
    (0, "source", "digit0/seed42/best_checkpoint.pt", "629e353d161281265e40d221cf5bcd30d2906db05b459f2a7be2b62efe81cff7", "digit0/seed42/run_summary.json", "0df65f909782fab0fcd8a1fbcda90511720a451ce9112a18aebbd63070efc65a", "digit0_corner_ease", 4800, SOURCE_HEAD),
    (0, "source", "digit0/seed42/final_checkpoint.pt", "fd4c42c7a5ee546ccdc65615df2648379fdc1b36f36dbe0d918de5ed476ae908", "digit0/seed42/run_summary.json", "0df65f909782fab0fcd8a1fbcda90511720a451ce9112a18aebbd63070efc65a", "digit0_corner_ease", 6000, SOURCE_HEAD),
    (0, "continuation", "digit0/lr1e3/final_checkpoint.pt", "f075309b342c1ecd88c2fa65131ab03e029144dce24712fdf8f2fd505a6bdffc", "digit0/lr1e3/run_summary.json", "136dfb8e8958f85d42b6bc5d3f0174de3c1806d1fa0d262d773487cd7c54008b", "digit0_corner_ease", 8000, CONTINUATION_HEAD),
    (0, "continuation", "digit0/lr3e4/final_checkpoint.pt", "3ec9099d25519add5bad3077b3f10c77c45f43ca4dc2052e0b0a427068823808", "digit0/lr3e4/run_summary.json", "6bfde34728b35757363860dc6762bd15c1536b92b6306a387a1626705145bf50", "digit0_corner_ease", 8000, CONTINUATION_HEAD),
    (8, "source", "digit8/seed42/best_checkpoint.pt", "f70e0ba8e8585414f499f8ae2cc54e9061923983ea444ca62117e5faa5822a39", "digit8/seed42/run_summary.json", "15df42d78db01d52a802f7dcf10c6a8661c342d04c6ade5e30e32fe61817a5b1", "digit8_corner_ease", 5800, SOURCE_HEAD),
    (8, "source", "digit8/seed42/final_checkpoint.pt", "2812c1c336c14d8cb58bb02638592499d0bc018fb6628287d73e10a5dfa445ce", "digit8/seed42/run_summary.json", "15df42d78db01d52a802f7dcf10c6a8661c342d04c6ade5e30e32fe61817a5b1", "digit8_corner_ease", 6000, SOURCE_HEAD),
    (9, "source", "digit9/seed42/best_checkpoint.pt", "4ad0674a167077a6a2935e6143bec9f723e6d9e07335e4a31edae5f2ed384bb9", "digit9/seed42/run_summary.json", "8957fee9d878aa2d8099f953aa782b4102db64b726aa4e2a244cfed467602cfb", "digit9_corner_ease", 4800, SOURCE_HEAD),
)
SOURCE_IDENTITY = {
    "protocol_config": "configurations/digit_writing_original_protocol3_ten_digit_corner_ease_overfit_v3.json",
    "run_directory": "runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit",
    "evidence_directory": "runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/single_digit_overfit_server_evidence",
    "repository_head": SOURCE_HEAD,
}
CONTINUATION_IDENTITY = {
    "protocol_config": "configurations/digit_writing_original_protocol3_corner_ease_lr_continuation_6000_to8000.json",
    "run_directory": "runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/lr_continuation_from6000_to8000",
    "evidence_directory": "runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/lr_continuation_from6000_to8000_server_evidence",
    "repository_head": CONTINUATION_HEAD,
}
AUDIT_IDENTITY = {
    "testing": False,
    "batch_size": 8,
    "direction_index": 0,
    "speed_condition": 0,
    "delay_steps": 50,
    "delay_index": 1,
    "validation_seed": 1042,
    "network_noise": False,
    "deterministic_observations": True,
    "phase_alignment_band_steps": 25,
    "phase_alignment_step_increments": [0, 1, 2],
    "closure_tolerance_bbox_fraction": 0.05,
    "late_movement_fraction": 0.8,
    "crossing_window_steps": 25,
    "crossing_outgoing_steps": 10,
    "gradient_epsilon": 1e-12,
    "optimizer_steps": 0,
}
DECISION_IDENTITY = {
    "automatic_root_cause_selection": False,
    "automatic_stage_b_start": False,
    "automatic_training": False,
    "automatic_geometry_change": False,
    "automatic_loss_change": False,
    "automatic_second_seed": False,
    "formal_full10_start": False,
    "qualitative_review_required": True,
}
OUTPUT_DIRECTORY = (
    "runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/"
    "closed_loop_digit0_digit8_diagnostic"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_ready(value),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV rows are empty: {path}")
    fieldnames = tuple(rows[0])
    if any(tuple(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("configured diagnostic path escapes the repository") from error
    return path


def _checkpoint_expected_row(
    label: str, identity: Sequence[Any]
) -> dict[str, Any]:
    (
        digit,
        origin,
        checkpoint,
        checkpoint_hash,
        summary,
        summary_hash,
        variant,
        update,
        head,
    ) = identity
    return {
        "label": label,
        "digit": digit,
        "origin": origin,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "run_summary": summary,
        "run_summary_sha256": summary_hash,
        "expected_variant": variant,
        "expected_update": update,
        "expected_repository_head": head,
    }


def load_diagnostic_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("closed-loop diagnostic requires Protocol3")
    if config.get("run_kind") != "protocol3_digit0_digit8_closed_loop_diagnostic":
        raise ValueError("closed-loop diagnostic run kind changed")
    if config.get("variant") != "digit0_digit8_joint_zero_step_v1":
        raise ValueError("closed-loop diagnostic variant changed")
    if config.get("source") != SOURCE_IDENTITY:
        raise ValueError("closed-loop diagnostic source identity changed")
    if config.get("continuation") != CONTINUATION_IDENTITY:
        raise ValueError("closed-loop diagnostic continuation identity changed")
    if config.get("mrnntorch_head") != MRNN_HEAD:
        raise ValueError("closed-loop diagnostic mRNNTorch identity changed")
    expected = [
        _checkpoint_expected_row(label, identity)
        for label, identity in zip(CHECKPOINT_LABELS, EXPECTED_CHECKPOINTS)
    ]
    if config.get("checkpoints") != expected:
        raise ValueError("closed-loop diagnostic checkpoint matrix changed")
    if config.get("audit") != AUDIT_IDENTITY:
        raise ValueError("closed-loop diagnostic audit conditions changed")
    if config.get("decision") != DECISION_IDENTITY:
        raise ValueError("closed-loop diagnostic decision boundary changed")
    if config.get("output") != {"directory": OUTPUT_DIRECTORY}:
        raise ValueError("closed-loop diagnostic output identity changed")
    return config


def _load_checked_in_diagnostic_config(
    repository_root: str | Path, path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_path = Path(path).resolve()
    expected_path = root / "configurations" / (
        "digit_writing_original_protocol3_digit0_digit8_"
        "closed_loop_diagnostic.json"
    )
    if config_path != expected_path.resolve():
        raise ValueError("diagnostic config is not the checked-in config")
    return load_diagnostic_config(config_path)


def _numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("numeric summary requires non-empty finite values")
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _target_diagonal(target: np.ndarray) -> float:
    span = np.ptp(np.asarray(target, dtype=np.float64), axis=0)
    diagonal = float(np.linalg.norm(span))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("target bounding-box diagonal must be positive")
    return diagonal


def constrained_monotonic_alignment(
    actual: np.ndarray,
    target: np.ndarray,
    *,
    band_steps: int = 25,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 2:
        raise ValueError("alignment requires matching [time, 2] trajectories")
    count = actual.shape[0]
    if count < 2 or band_steps < 0:
        raise ValueError("alignment trajectory or band is invalid")
    diagonal = _target_diagonal(target)
    distances = np.linalg.norm(actual[:, None, :] - target[None, :, :], axis=-1)
    states: dict[int, tuple[float, int, tuple[int, ...]]] = {
        0: (float(distances[0, 0] / diagonal), 0, (0,))
    }
    for time_index in range(1, count):
        next_states: dict[int, tuple[float, int, tuple[int, ...]]] = {}
        lower = max(0, time_index - band_steps)
        upper = min(count - 1, time_index + band_steps)
        for target_index in range(lower, upper + 1):
            candidates = []
            for increment in (0, 1, 2):
                previous_index = target_index - increment
                previous = states.get(previous_index)
                if previous is None:
                    continue
                candidates.append(
                    (
                        previous[0] + float(distances[time_index, target_index] / diagonal),
                        previous[1] + abs(time_index - target_index),
                        previous[2] + (target_index,),
                    )
                )
            if candidates:
                next_states[target_index] = min(candidates)
        if not next_states:
            raise RuntimeError("phase alignment has no reachable state")
        states = next_states
    best = min(states.values())
    mapping = np.asarray(best[2], dtype=np.int64)
    if mapping.shape != (count,) or mapping[0] != 0:
        raise RuntimeError("phase alignment mapping is incomplete")
    increments = np.diff(mapping)
    if not np.isin(increments, (0, 1, 2)).all():
        raise RuntimeError("phase alignment increment escaped the frozen set")
    if np.max(np.abs(mapping - np.arange(count))) > band_steps:
        raise RuntimeError("phase alignment escaped the frozen band")
    time_error = np.linalg.norm(actual - target, axis=1) / diagonal
    phase_error = np.linalg.norm(actual - target[mapping], axis=1) / diagonal
    lag = np.arange(count, dtype=np.int64) - mapping
    time_mean = float(time_error.mean())
    phase_mean = float(phase_error.mean())
    reduction = time_mean - phase_mean
    return {
        "mapping": mapping,
        "time_aligned_normalized_mean_error": time_mean,
        "phase_aligned_normalized_mean_error": phase_mean,
        "phase_alignment_error_reduction": reduction,
        "phase_alignment_relative_reduction": (
            None if time_mean <= 1e-12 else reduction / time_mean
        ),
        "phase_lag_steps_mean": float(lag.mean()),
        "phase_lag_steps_p95": float(np.percentile(lag, 95.0)),
        "phase_lag_steps_min": int(lag.min()),
        "phase_lag_steps_max": int(lag.max()),
        "phase_lag_steps_final": int(lag[-1]),
        "target_phase_completion_at_movement_end": float(mapping[-1] / (count - 1)),
        "normalized_total_cost": float(best[0]),
        "tie_break_cumulative_absolute_lag": int(best[1]),
    }


def closure_metrics(
    actual_movement: np.ndarray,
    hold_actual: np.ndarray,
    target_endpoint: np.ndarray,
    *,
    target_diagonal: float,
    movement_start_episode_index: int,
    late_fraction: float = 0.8,
    tolerance_fraction: float = 0.05,
) -> dict[str, Any]:
    movement = np.asarray(actual_movement, dtype=np.float64)
    hold = np.asarray(hold_actual, dtype=np.float64)
    endpoint = np.asarray(target_endpoint, dtype=np.float64)
    if (
        movement.ndim != 2
        or hold.ndim != 2
        or movement.shape[1:] != (2,)
        or hold.shape[1:] != (2,)
        or not 0.0 < late_fraction < 1.0
        or target_diagonal <= 0.0
    ):
        raise ValueError("closure diagnostic inputs are invalid")
    tolerance = tolerance_fraction * target_diagonal
    late_start = int(math.floor(late_fraction * len(movement)))
    late_distances = np.linalg.norm(movement[late_start:] - endpoint, axis=1)
    hold_distances = np.linalg.norm(hold - endpoint, axis=1)
    entered_movement = bool(np.any(late_distances <= tolerance))
    entered_hold = bool(np.any(hold_distances <= tolerance))
    reached_during = entered_movement
    reached_only_hold = bool(not entered_movement and entered_hold)
    never_reached = bool(not entered_movement and not entered_hold)
    if sum((reached_during, reached_only_hold, never_reached)) != 1:
        raise RuntimeError("closure arrival categories are not exclusive")
    first_phase = None
    first_episode_index = None
    movement_hits = np.flatnonzero(late_distances <= tolerance)
    hold_hits = np.flatnonzero(hold_distances <= tolerance)
    if movement_hits.size:
        first_phase = "movement"
        first_episode_index = (
            movement_start_episode_index + late_start + int(movement_hits[0])
        )
    elif hold_hits.size:
        first_phase = "hold"
        first_episode_index = (
            movement_start_episode_index + len(movement) + int(hold_hits[0])
        )
    movement_endpoint_distance = float(
        np.linalg.norm(movement[-1] - endpoint)
    )
    instability = bool(
        entered_movement
        and (
            movement_endpoint_distance > tolerance
            or np.any(hold_distances > tolerance)
        )
    )
    labels = []
    if reached_only_hold:
        labels.append("late_closure")
    if never_reached:
        labels.append("persistent_nonclosure")
    if instability:
        labels.append("closure_hold_instability")
    hold_steps = np.diff(hold, axis=0)
    movement_endpoint = movement[-1]
    hold_from_endpoint = np.linalg.norm(hold - movement_endpoint, axis=1)
    return {
        "closure_tolerance_m": float(tolerance),
        "late_movement_local_start_index": late_start,
        "movement_endpoint_distance_m": movement_endpoint_distance,
        "minimum_late_movement_closure_distance_m": float(late_distances.min()),
        "minimum_hold_closure_distance_m": float(hold_distances.min()),
        "first_closure_entry_episode_index": first_episode_index,
        "first_closure_entry_phase": first_phase,
        "hold_path_length_m": float(np.linalg.norm(hold_steps, axis=1).sum()),
        "hold_final_displacement_from_movement_endpoint_m": float(
            hold_from_endpoint[-1]
        ),
        "hold_max_displacement_from_movement_endpoint_m": float(
            hold_from_endpoint.max()
        ),
        "closure_reached_during_movement": reached_during,
        "closure_reached_only_during_hold": reached_only_hold,
        "closure_never_reached": never_reached,
        "closure_hold_instability": instability,
        "diagnostic_labels": labels,
    }


def _padded_derivatives(points: np.ndarray, dt_seconds: float) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or dt_seconds <= 0.0:
        raise ValueError("padded derivatives require [time, 2] points and positive dt")
    count = len(points)
    velocity = np.zeros_like(points)
    acceleration = np.zeros_like(points)
    jerk = np.zeros_like(points)
    if count >= 2:
        raw_velocity = np.diff(points, axis=0) / dt_seconds
        velocity[1:] = raw_velocity
    else:
        raw_velocity = np.empty((0, 2), dtype=np.float64)
    if count >= 3:
        raw_acceleration = np.diff(raw_velocity, axis=0) / dt_seconds
        acceleration[2:] = raw_acceleration
    else:
        raw_acceleration = np.empty((0, 2), dtype=np.float64)
    if count >= 4:
        raw_jerk = np.diff(raw_acceleration, axis=0) / dt_seconds
        jerk[3:] = raw_jerk
    return {
        "speed": np.linalg.norm(velocity, axis=1),
        "acceleration": np.linalg.norm(acceleration, axis=1),
        "jerk": np.linalg.norm(jerk, axis=1),
    }


def _kinematic_metrics(points: np.ndarray, dt_seconds: float) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    velocity = np.diff(points, axis=0) / dt_seconds
    acceleration = np.diff(velocity, axis=0) / dt_seconds
    jerk = np.diff(acceleration, axis=0) / dt_seconds

    def metrics(vectors: np.ndarray, name: str) -> dict[str, float]:
        norms = np.linalg.norm(vectors, axis=1)
        return {
            f"p95_{name}": float(np.percentile(norms, 95.0)) if norms.size else 0.0,
            f"peak_{name}": float(norms.max()) if norms.size else 0.0,
        }

    return {
        **metrics(velocity, "speed_m_s"),
        **metrics(acceleration, "acceleration_m_s2"),
        **metrics(jerk, "jerk_m_s3"),
    }


def _masked_kinematic_metrics(
    points: np.ndarray,
    *,
    velocity_mask: np.ndarray,
    acceleration_mask: np.ndarray,
    jerk_mask: np.ndarray,
    dt_seconds: float,
) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    velocity = np.diff(points, axis=0) / dt_seconds
    acceleration = np.diff(velocity, axis=0) / dt_seconds
    jerk = np.diff(acceleration, axis=0) / dt_seconds
    masks = {
        "speed_m_s": np.asarray(velocity_mask, dtype=bool),
        "acceleration_m_s2": np.asarray(acceleration_mask, dtype=bool),
        "jerk_m_s3": np.asarray(jerk_mask, dtype=bool),
    }
    vectors = {
        "speed_m_s": velocity,
        "acceleration_m_s2": acceleration,
        "jerk_m_s3": jerk,
    }
    result = {}
    for name, values in vectors.items():
        mask = masks[name]
        if mask.shape != (len(values),) or not mask.any():
            raise ValueError(f"regional kinematic mask is invalid: {name}")
        norms = np.linalg.norm(values[mask], axis=1)
        result[f"p95_{name}"] = float(np.percentile(norms, 95.0))
        result[f"peak_{name}"] = float(norms.max())
        result[f"sample_count_{name}"] = int(mask.sum())
    return result


def regional_kinematic_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    *,
    digit: int,
    dt_seconds: float,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    result: dict[str, Any] = {
        "actual_global": _kinematic_metrics(actual, dt_seconds),
        "target_global": _kinematic_metrics(target, dt_seconds),
    }
    if digit == 0:
        center = (target.min(axis=0) + target.max(axis=0)) / 2.0
        velocity_regions = _angle_regions(
            (target[:-1] + target[1:]) / 2.0, center
        )
        acceleration_regions = _angle_regions(target[2:], center)
        jerk_regions = _angle_regions(target[3:], center)
        regions = {}
        for region in ("upper", "right", "lower", "left"):
            masks = {
                "velocity_mask": velocity_regions == region,
                "acceleration_mask": acceleration_regions == region,
                "jerk_mask": jerk_regions == region,
            }
            regions[region] = {
                "actual": _masked_kinematic_metrics(
                    actual, **masks, dt_seconds=dt_seconds
                ),
                "target": _masked_kinematic_metrics(
                    target, **masks, dt_seconds=dt_seconds
                ),
            }
        result["regions"] = regions
    elif digit == 8:
        validate_digit8_landmarks(target)
        segments = {}
        for index, (start, end) in enumerate(
            ((0, 50), (50, 100), (100, 150), (150, 200)), start=1
        ):
            segments[f"segment_{index}"] = {
                "actual": _kinematic_metrics(
                    actual[start : end + 1], dt_seconds
                ),
                "target": _kinematic_metrics(
                    target[start : end + 1], dt_seconds
                ),
            }
        result["segments"] = segments
    return result


def _angle_regions(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    offsets = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    regions = np.empty(len(points), dtype="<U5")
    regions[(angles >= -math.pi / 4) & (angles < math.pi / 4)] = "right"
    regions[(angles >= math.pi / 4) & (angles < 3 * math.pi / 4)] = "upper"
    regions[(angles >= -3 * math.pi / 4) & (angles < -math.pi / 4)] = "lower"
    left = (angles >= 3 * math.pi / 4) | (angles < -3 * math.pi / 4)
    regions[left] = "left"
    if not np.isin(regions, ("right", "upper", "lower", "left")).all():
        raise RuntimeError("digit0 quadrant assignment is incomplete")
    return regions


def digit0_region_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    mapping: np.ndarray,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mapping = np.asarray(mapping, dtype=np.int64)
    if actual.shape != target.shape or mapping.shape != (len(target),):
        raise ValueError("digit0 region inputs differ")
    center = (target.min(axis=0) + target.max(axis=0)) / 2.0
    sample_regions = _angle_regions(target, center)
    midpoint_regions = _angle_regions((target[:-1] + target[1:]) / 2.0, center)
    diagonal = _target_diagonal(target)
    rows = {}
    for region in ("upper", "right", "lower", "left"):
        sample_mask = sample_regions == region
        step_mask = midpoint_regions == region
        if not sample_mask.any() or not step_mask.any():
            raise RuntimeError(f"digit0 region is empty: {region}")
        time_error = np.linalg.norm(actual[sample_mask] - target[sample_mask], axis=1)
        phase_error = np.linalg.norm(
            actual[sample_mask] - target[mapping[sample_mask]], axis=1
        )
        actual_path = float(
            np.linalg.norm(np.diff(actual, axis=0)[step_mask], axis=1).sum()
        )
        target_path = float(
            np.linalg.norm(np.diff(target, axis=0)[step_mask], axis=1).sum()
        )
        target_radius = np.linalg.norm(target[sample_mask] - center, axis=1)
        actual_radius = np.linalg.norm(actual[sample_mask] - center, axis=1)
        rows[region] = {
            "sample_count": int(sample_mask.sum()),
            "interval_count": int(step_mask.sum()),
            "time_aligned_error_m": float(time_error.mean()),
            "time_aligned_normalized_error": float(time_error.mean() / diagonal),
            "phase_aligned_error_m": float(phase_error.mean()),
            "phase_aligned_normalized_error": float(phase_error.mean() / diagonal),
            "actual_to_target_path_ratio": actual_path / target_path,
            "radial_extent_ratio": float(actual_radius.max() / target_radius.max()),
            "maximum_inward_deviation_m": float(
                np.maximum(target_radius - actual_radius, 0.0).max()
            ),
            "mean_phase_lag_steps": float(
                (np.flatnonzero(sample_mask) - mapping[sample_mask]).mean()
            ),
        }
    if sum(row["sample_count"] for row in rows.values()) != len(target):
        raise RuntimeError("digit0 region samples overlap")
    if sum(row["interval_count"] for row in rows.values()) != len(target) - 1:
        raise RuntimeError("digit0 region intervals overlap")
    actual_span = np.ptp(actual, axis=0)
    target_span = np.ptp(target, axis=0)
    return {
        "center_m": center.tolist(),
        "regions": rows,
        "x_bbox_ratio": float(actual_span[0] / target_span[0]),
        "y_bbox_ratio": float(actual_span[1] / target_span[1]),
    }


def validate_digit8_landmarks(target: np.ndarray, *, atol: float = 2e-6) -> None:
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (201, 2):
        raise ValueError("digit8 target must contain 201 movement samples")
    center = (target.min(axis=0) + target.max(axis=0)) / 2.0
    if not (
        np.allclose(target[0], target[200], rtol=0.0, atol=atol)
        and np.allclose(target[50], center, rtol=0.0, atol=atol)
        and np.allclose(target[150], center, rtol=0.0, atol=atol)
        and target[0, 1] > center[1]
        and target[100, 1] < center[1]
    ):
        raise ValueError("digit8 landmark indices differ from the Gerono target")


def _cosine(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < epsilon:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def digit8_segment_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    mapping: np.ndarray,
    *,
    crossing_window_steps: int = 25,
    outgoing_steps: int = 10,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mapping = np.asarray(mapping, dtype=np.int64)
    validate_digit8_landmarks(target)
    if actual.shape != target.shape or mapping.shape != (201,):
        raise ValueError("digit8 segment inputs differ")
    diagonal = _target_diagonal(target)
    segments = {}
    point_ranges = ((0, 50), (50, 100), (100, 150), (150, 201))
    step_ranges = ((0, 50), (50, 100), (100, 150), (150, 200))
    for index, ((point_start, point_end), (step_start, step_end)) in enumerate(
        zip(point_ranges, step_ranges), start=1
    ):
        point_slice = slice(point_start, point_end)
        actual_steps = np.diff(actual, axis=0)[step_start:step_end]
        target_steps = np.diff(target, axis=0)[step_start:step_end]
        time_error = np.linalg.norm(
            actual[point_slice] - target[point_slice], axis=1
        )
        indices = np.arange(point_start, point_end)
        phase_error = np.linalg.norm(
            actual[point_slice] - target[mapping[point_slice]], axis=1
        )
        actual_span = np.ptp(actual[point_slice], axis=0)
        target_span = np.ptp(target[point_slice], axis=0)
        segments[f"segment_{index}"] = {
            "point_start": point_start,
            "point_end_exclusive": point_end,
            "interval_start": step_start,
            "interval_end_exclusive": step_end,
            "point_count": point_end - point_start,
            "interval_count": step_end - step_start,
            "actual_to_target_path_ratio": float(
                np.linalg.norm(actual_steps, axis=1).sum()
                / np.linalg.norm(target_steps, axis=1).sum()
            ),
            "actual_to_target_width_extent_ratio": (
                None if target_span[0] <= 1e-12 else float(actual_span[0] / target_span[0])
            ),
            "actual_to_target_height_extent_ratio": (
                None if target_span[1] <= 1e-12 else float(actual_span[1] / target_span[1])
            ),
            "time_aligned_error_m": float(time_error.mean()),
            "time_aligned_normalized_error": float(time_error.mean() / diagonal),
            "phase_aligned_error_m": float(phase_error.mean()),
            "phase_aligned_normalized_error": float(phase_error.mean() / diagonal),
            "entry_phase_lag_steps": int(indices[0] - mapping[indices[0]]),
            "exit_phase_lag_steps": int(indices[-1] - mapping[indices[-1]]),
        }
    if sum(row["point_count"] for row in segments.values()) != 201:
        raise RuntimeError("digit8 segment samples overlap")
    if sum(row["interval_count"] for row in segments.values()) != 200:
        raise RuntimeError("digit8 segment intervals overlap")
    crossings = {}
    for crossing in (50, 150):
        lower = crossing - crossing_window_steps
        upper = crossing + crossing_window_steps
        local = np.linalg.norm(actual[lower : upper + 1] - target[crossing], axis=1)
        actual_time = lower + int(np.argmin(local))
        outgoing_end = actual_time + outgoing_steps
        target_outgoing_end = crossing + outgoing_steps
        if outgoing_end >= len(actual) or target_outgoing_end >= len(target):
            raise RuntimeError("digit8 outgoing crossing window exceeds movement")
        actual_displacement = actual[outgoing_end] - actual[actual_time]
        target_displacement = target[target_outgoing_end] - target[crossing]
        target_norm = float(np.linalg.norm(target_displacement))
        if target_norm <= 0.0:
            raise RuntimeError("digit8 target branch displacement is zero")
        crossings[f"crossing_{1 if crossing == 50 else 2}"] = {
            "target_index": crossing,
            "actual_crossing_time": actual_time,
            "crossing_position_error_m": float(local.min()),
            "crossing_arrival_lag_steps": actual_time - crossing,
            "actual_velocity_vs_target_tangent_cosine": _cosine(
                actual[actual_time + 1] - actual[actual_time],
                target[crossing + 1] - target[crossing],
            ),
            "outgoing_10_step_progress_ratio": float(
                np.linalg.norm(actual_displacement) / target_norm
            ),
            "wrong_branch_departure": bool(
                np.dot(actual_displacement, target_displacement) < 0.0
            ),
        }
    return {"segments": segments, "crossings": crossings}


def _gradient_vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def _gradient_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float,
) -> dict[str, Any]:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    numerator = float(torch.dot(left, right))
    denominator = left_norm * right_norm
    undefined = left_norm < epsilon or right_norm < epsilon
    return {
        "dot_numerator": numerator,
        "norm_product_denominator": denominator,
        "cosine": None if undefined else numerator / denominator,
        "cosine_undefined": undefined,
        "small_denominator": undefined,
    }


def gradient_decomposition(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    required = (
        "stable_position",
        "delay_position",
        "movement_position",
        "hold_position",
        "total_position",
        "weighted_l1_rate",
        "weighted_l1_weight",
        "weighted_l1_muscle_act",
        "weighted_simple_dynamics",
        "total_regularization",
        "total_objective",
    )
    if tuple(losses) != required:
        raise ValueError("gradient decomposition losses are incomplete or reordered")
    parameter_tuple = tuple(parameters)
    gradients = {
        name: _gradient_vector(loss, parameter_tuple)
        for name, loss in losses.items()
    }
    components = {}
    for name in required:
        gradient = gradients[name]
        components[name] = {
            "loss": float(losses[name].detach()),
            "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        }
    comparisons = {}
    right_names = (
        "weighted_l1_rate",
        "weighted_l1_weight",
        "weighted_l1_muscle_act",
        "weighted_simple_dynamics",
        "total_regularization",
        "total_objective",
    )
    for left_name in ("movement_position", "hold_position", "total_position"):
        comparisons[left_name] = {
            right_name: _gradient_pair(
                gradients[left_name], gradients[right_name], epsilon
            )
            for right_name in right_names
        }
    component_sum = sum(
        (gradients[name] for name in required[:4]),
        torch.zeros_like(gradients["total_position"]),
    ) + sum(
        (gradients[name] for name in required[5:9]),
        torch.zeros_like(gradients["total_position"]),
    )
    residual = component_sum - gradients["total_objective"]
    total_objective_norm = components["total_objective"]["gradient_norm"]
    component_sum_tolerance = 1e-5 * max(1.0, total_objective_norm)
    component_sum_residual_norm = float(torch.linalg.vector_norm(residual))
    return {
        "gradient_epsilon": epsilon,
        "components": components,
        "comparisons": comparisons,
        "component_sum_residual_norm": component_sum_residual_norm,
        "component_sum_tolerance": component_sum_tolerance,
        "component_sum_consistent": (
            component_sum_residual_norm <= component_sum_tolerance
        ),
        "gradients_are_pre_clip": True,
        "optimizer_steps_during_audit": 0,
    }


def _record_for_label(
    config: Mapping[str, Any], label: str
) -> dict[str, Any]:
    matches = [row for row in config["checkpoints"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown closed-loop checkpoint label: {label}")
    return dict(matches[0])


def _artifact_root(
    record: Mapping[str, Any],
    source_run_root: str | Path,
    continuation_run_root: str | Path,
) -> Path:
    return Path(
        source_run_root
        if record["origin"] == "source"
        else continuation_run_root
    ).resolve()


def _artifact_paths(
    record: Mapping[str, Any],
    source_run_root: str | Path,
    continuation_run_root: str | Path,
) -> tuple[Path, Path]:
    root = _artifact_root(record, source_run_root, continuation_run_root)
    checkpoint = (root / record["checkpoint"]).resolve()
    summary = (root / record["run_summary"]).resolve()
    for path in (checkpoint, summary):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("diagnostic artifact path escapes its run root") from error
    return checkpoint, summary


def review_artifact(
    *,
    config: Mapping[str, Any],
    label: str,
    source_run_root: str | Path,
    continuation_run_root: str | Path,
) -> dict[str, Any]:
    record = _record_for_label(config, label)
    checkpoint_path, summary_path = _artifact_paths(
        record, source_run_root, continuation_run_root
    )
    expected_hashes = {
        checkpoint_path: record["checkpoint_sha256"],
        summary_path: record["run_summary_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"closed-loop artifact hash differs: {path}")
    summary = _read_json(summary_path)
    digit = int(record["digit"])
    if (
        int(summary.get("case", {}).get("digit", -1)) != digit
        or summary.get("engineering_passed") is not True
        or summary.get("checkpoint_complete") is not True
    ):
        raise ValueError("closed-loop source summary is not engineering-complete")
    identity = summary.get("git_identity", {})
    if identity.get("repository_head") != record["expected_repository_head"]:
        raise ValueError("closed-loop source summary repository HEAD differs")
    if (
        identity.get("mrnntorch_recorded_head") != MRNN_HEAD
        or identity.get("mrnntorch_worktree_head") != MRNN_HEAD
    ):
        raise ValueError("closed-loop source summary mRNNTorch identity differs")
    if record["origin"] == "source":
        if (
            summary.get("run_kind") != "protocol3_corner_ease_case"
            or int(summary.get("completed_updates", -1)) != 6000
        ):
            raise ValueError("closed-loop source run summary identity differs")
        if int(record["expected_update"]) < 6000 and int(
            summary.get("best_update", -1)
        ) != int(record["expected_update"]):
            raise ValueError("closed-loop source best update differs")
    else:
        if (
            summary.get("run_kind")
            != "protocol3_corner_ease_lr_continuation_arm"
            or int(summary.get("completed_updates", -1)) != 8000
            or summary.get("source_repository_head") != SOURCE_HEAD
            or summary.get("automatic_winner_selection_started") is not False
            or summary.get("automatic_further_continuation_started") is not False
            or summary.get("automatic_second_seed_started") is not False
            or summary.get("digit8_started") is not False
            or summary.get("formal_full10_started") is not False
        ):
            raise ValueError("closed-loop continuation summary identity differs")
    return {
        "label": label,
        "digit": digit,
        "origin": record["origin"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_hashes[checkpoint_path],
        "run_summary": str(summary_path),
        "run_summary_sha256": expected_hashes[summary_path],
        "expected_variant": record["expected_variant"],
        "expected_update": record["expected_update"],
        "expected_repository_head": record["expected_repository_head"],
        "source_summary_status": summary.get("status"),
        "source_summary_engineering_passed": True,
    }


def prepare_diagnostic(
    *,
    repository_root: str | Path,
    diagnostic_config_path: str | Path,
    source_run_root: str | Path,
    source_evidence_root: str | Path,
    continuation_run_root: str | Path,
    continuation_evidence_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _load_checked_in_diagnostic_config(root, diagnostic_config_path)
    source_root = Path(source_run_root).resolve()
    source_evidence = Path(source_evidence_root).resolve()
    continuation_root = Path(continuation_run_root).resolve()
    continuation_evidence = Path(continuation_evidence_root).resolve()
    expected_paths = {
        source_root: _resolve_inside(root, config["source"]["run_directory"]),
        source_evidence: _resolve_inside(root, config["source"]["evidence_directory"]),
        continuation_root: _resolve_inside(
            root, config["continuation"]["run_directory"]
        ),
        continuation_evidence: _resolve_inside(
            root, config["continuation"]["evidence_directory"]
        ),
    }
    for actual, expected in expected_paths.items():
        if actual != expected:
            raise ValueError(f"closed-loop input root changed: {actual}")
    output = Path(run_root).resolve()
    if output != _resolve_inside(root, config["output"]["directory"]):
        raise ValueError("closed-loop diagnostic run root changed")
    if output.exists():
        raise FileExistsError(f"closed-loop diagnostic output exists: {output}")

    source_preflight = _read_json(source_evidence / "preflight_summary.json")
    workspace_audit = _read_json(source_evidence / "workspace_audit.json")
    if (
        source_preflight.get("passed") is not True
        or source_preflight.get("training_updates") != 0
        or source_preflight.get("git_identity", {}).get("repository_head")
        != SOURCE_HEAD
        or source_preflight.get("git_identity", {}).get("mrnntorch_recorded_head")
        != MRNN_HEAD
        or workspace_audit.get("passed") is not True
    ):
        raise ValueError("closed-loop source preflight or workspace audit differs")
    if (source_evidence / "repository_head.txt").read_text(
        encoding="utf-8"
    ).strip() != SOURCE_HEAD:
        raise ValueError("closed-loop source evidence HEAD differs")
    if (continuation_evidence / "repository_head.txt").read_text(
        encoding="utf-8"
    ).strip() != CONTINUATION_HEAD:
        raise ValueError("closed-loop continuation evidence HEAD differs")
    if (continuation_evidence / "source_repository_head.txt").read_text(
        encoding="utf-8"
    ).strip() != SOURCE_HEAD:
        raise ValueError("closed-loop continuation source HEAD differs")
    for evidence in (source_evidence, continuation_evidence):
        if (evidence / "submodule_head.txt").read_text(
            encoding="utf-8"
        ).strip() != MRNN_HEAD:
            raise ValueError("closed-loop evidence submodule HEAD differs")

    reviews = [
        review_artifact(
            config=config,
            label=label,
            source_run_root=source_root,
            continuation_run_root=continuation_root,
        )
        for label in CHECKPOINT_LABELS
    ]
    current_identity = current_git_identity(root)
    if (
        current_identity["mrnntorch_recorded_head"] != MRNN_HEAD
        or current_identity["mrnntorch_worktree_head"] != MRNN_HEAD
    ):
        raise ValueError("closed-loop implementation submodule identity differs")

    output.mkdir(parents=True)
    (output / "per_checkpoint").mkdir()
    _write_json(output / "resolved_config.json", config)
    _write_json(output / "checkpoint_identity.json", {"checkpoints": reviews})
    _write_json(output / "workspace_audit.json", workspace_audit)
    result = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_closed_loop_diagnostic_prepare",
        "implementation_git_identity": current_identity,
        "source_repository_head": SOURCE_HEAD,
        "continuation_repository_head": CONTINUATION_HEAD,
        "mrnntorch_head": MRNN_HEAD,
        "reviewed_checkpoints": len(reviews),
        "optimizer_steps": 0,
        "automatic_stage_b_start": False,
        "automatic_training": False,
        "passed": True,
    }
    _write_json(output / "prepare_summary.json", result)
    return result


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty records")
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records):
        raise ValueError("aggregate records have inconsistent fields")
    result: dict[str, Any] = {}
    for key in keys:
        values = [record[key] for record in records]
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            true_count = sum(bool(value) for value in values)
            result[key] = {
                "true_count": true_count,
                "all": true_count == len(values),
                "any": true_count > 0,
            }
        elif all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            result[key] = _numeric_summary([float(value) for value in values])
        elif all(isinstance(value, Mapping) for value in values):
            result[key] = _aggregate_records(values)
        else:
            result[key] = {
                "values": _json_ready(values),
                "undefined_count": sum(value is None for value in values),
            }
    return result


def _vector_pair_records(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> list[dict[str, Any]]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("paired vectors must have matching [batch, feature] shapes")
    records = []
    for left_row, right_row in zip(left, right):
        left_norm = float(np.linalg.norm(left_row))
        right_norm = float(np.linalg.norm(right_row))
        denominator = 0.5 * (left_norm + right_norm)
        records.append(
            {
                "cosine": _cosine(left_row, right_row, epsilon),
                "l2_distance": float(np.linalg.norm(left_row - right_row)),
                "normalized_l2_distance": (
                    None
                    if denominator < epsilon
                    else float(np.linalg.norm(left_row - right_row) / denominator)
                ),
            }
        )
    return records


def _gather_batch_time(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    values = np.asarray(values)
    indices = np.asarray(indices, dtype=np.int64)
    if values.ndim != 3 or indices.shape != (values.shape[0],):
        raise ValueError("batch-time gather shapes differ")
    if np.any(indices < 0) or np.any(indices >= values.shape[1]):
        raise IndexError("batch-time gather index is out of range")
    return values[np.arange(values.shape[0]), indices]


def _nearest_landmark_time(
    actual: np.ndarray,
    target_point: np.ndarray,
    target_index: int,
    band: int,
) -> int:
    lower = max(0, target_index - band)
    upper = min(len(actual) - 1, target_index + band)
    distances = np.linalg.norm(actual[lower : upper + 1] - target_point, axis=1)
    return lower + int(np.argmin(distances))


def _optional_numeric_summary(values: Sequence[float | None]) -> dict[str, Any]:
    defined = [float(value) for value in values if value is not None]
    return {
        "defined": None if not defined else _numeric_summary(defined),
        "undefined_count": len(values) - len(defined),
        "sample_count": len(values),
    }


def _vector_pair_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, Any]:
    records = _vector_pair_records(left, right, epsilon=epsilon)
    return {
        "cosine": _optional_numeric_summary([row["cosine"] for row in records]),
        "l2_distance": _numeric_summary(
            [row["l2_distance"] for row in records]
        ),
        "normalized_l2_distance": _optional_numeric_summary(
            [row["normalized_l2_distance"] for row in records]
        ),
    }


def _tangent_at(points: np.ndarray, index: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not 0 <= index < len(points):
        raise IndexError("trajectory tangent index is out of range")
    if index == 0:
        return points[1] - points[0]
    if index == len(points) - 1:
        return points[-1] - points[-2]
    return points[index + 1] - points[index - 1]


def _hidden_pair_summary(
    arrays: Mapping[str, np.ndarray],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    *,
    target_left_indices: Sequence[int] | None = None,
    target_right_indices: Sequence[int] | None = None,
    epsilon: float,
) -> dict[str, Any]:
    left_indices = np.asarray(left_indices, dtype=np.int64)
    right_indices = np.asarray(right_indices, dtype=np.int64)
    batch_size = arrays["positions"].shape[0]
    if left_indices.shape != (batch_size,) or right_indices.shape != (batch_size,):
        raise ValueError("hidden pair indices differ from batch size")
    target_left_indices = np.asarray(
        left_indices if target_left_indices is None else target_left_indices,
        dtype=np.int64,
    )
    target_right_indices = np.asarray(
        right_indices if target_right_indices is None else target_right_indices,
        dtype=np.int64,
    )
    if (
        target_left_indices.shape != (batch_size,)
        or target_right_indices.shape != (batch_size,)
    ):
        raise ValueError("hidden target landmark indices differ from batch size")
    fields = {
        "x": "x_states",
        "h": "h_states",
        "action": "actions",
        "pre_action_observation": "observations",
        "joint_position": "joint_positions",
        "joint_velocity": "joint_velocities",
    }
    result = {
        name: _vector_pair_summary(
            _gather_batch_time(arrays[array_name], left_indices),
            _gather_batch_time(arrays[array_name], right_indices),
            epsilon=epsilon,
        )
        for name, array_name in fields.items()
    }
    target_left = []
    target_right = []
    actual_left = []
    actual_right = []
    for batch_index in range(batch_size):
        left = int(left_indices[batch_index])
        right = int(right_indices[batch_index])
        target_left.append(
            _tangent_at(
                arrays["targets"][batch_index],
                int(target_left_indices[batch_index]),
            )
        )
        target_right.append(
            _tangent_at(
                arrays["targets"][batch_index],
                int(target_right_indices[batch_index]),
            )
        )
        actual_left.append(_tangent_at(arrays["positions"][batch_index], left))
        actual_right.append(_tangent_at(arrays["positions"][batch_index], right))
    result["target_tangent"] = _vector_pair_summary(
        np.asarray(target_left), np.asarray(target_right), epsilon=epsilon
    )
    result["actual_tangent"] = _vector_pair_summary(
        np.asarray(actual_left), np.asarray(actual_right), epsilon=epsilon
    )
    result["left_episode_indices"] = left_indices.tolist()
    result["right_episode_indices"] = right_indices.tolist()
    result["target_left_episode_indices"] = target_left_indices.tolist()
    result["target_right_episode_indices"] = target_right_indices.tolist()
    return result


def hidden_phase_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    digit: int,
    movement_start: int,
    movement_end: int,
    phase_band: int,
    digit9_ellipse_boundary: int | None,
    epsilon: float,
) -> dict[str, Any]:
    positions = arrays["positions"]
    targets = arrays["targets"]
    batch_size = positions.shape[0]
    movement_count = movement_end - movement_start
    fixed_start = np.full(batch_size, movement_start, dtype=np.int64)
    fixed_end = np.full(batch_size, movement_end - 1, dtype=np.int64)
    first_hold = np.full(batch_size, movement_end, dtype=np.int64)

    def internal(target_index: int) -> np.ndarray:
        indices = []
        for batch_index in range(batch_size):
            actual_movement = positions[
                batch_index, movement_start:movement_end
            ]
            target_movement = targets[
                batch_index, movement_start:movement_end
            ]
            local = _nearest_landmark_time(
                actual_movement,
                target_movement[target_index],
                target_index,
                phase_band,
            )
            indices.append(movement_start + local)
        return np.asarray(indices, dtype=np.int64)

    if digit == 0:
        half_target = np.full(
            batch_size, movement_start + movement_count // 2, dtype=np.int64
        )
        half = internal(movement_count // 2)
        pairs = {
            "movement_start_vs_half_loop": (
                fixed_start, half, fixed_start, half_target
            ),
            "half_loop_vs_movement_end": (
                half, fixed_end, half_target, fixed_end
            ),
            "movement_start_vs_movement_end": (
                fixed_start, fixed_end, fixed_start, fixed_end
            ),
            "movement_end_vs_first_hold": (
                fixed_end, first_hold, fixed_end, first_hold
            ),
        }
    elif digit == 8:
        crossing_1 = internal(50)
        crossing_2 = internal(150)
        crossing_1_target = np.full(batch_size, movement_start + 50, dtype=np.int64)
        crossing_2_target = np.full(batch_size, movement_start + 150, dtype=np.int64)
        pairs = {
            "crossing_1_vs_crossing_2": (
                crossing_1, crossing_2, crossing_1_target, crossing_2_target
            ),
            "top_start_vs_top_end": (
                fixed_start, fixed_end, fixed_start, fixed_end
            ),
            "movement_end_vs_first_hold": (
                fixed_end, first_hold, fixed_end, first_hold
            ),
        }
    elif digit == 9:
        if digit9_ellipse_boundary is None:
            raise ValueError("digit9 ellipse boundary is required")
        boundary = internal(digit9_ellipse_boundary)
        boundary_target = np.full(
            batch_size,
            movement_start + digit9_ellipse_boundary,
            dtype=np.int64,
        )
        pairs = {
            "ellipse_start_vs_ellipse_boundary": (
                fixed_start, boundary, fixed_start, boundary_target
            ),
            "ellipse_boundary_vs_tail_end": (
                boundary, fixed_end, boundary_target, fixed_end
            ),
        }
    else:
        raise ValueError("hidden phase diagnostic supports only digits 0, 8, and 9")
    return {
        label: _hidden_pair_summary(
            arrays,
            left,
            right,
            target_left_indices=target_left,
            target_right_indices=target_right,
            epsilon=epsilon,
        )
        for label, (left, right, target_left, target_right) in pairs.items()
    }


def _behavior_metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diagonal = _target_diagonal(target)
    error = np.linalg.norm(actual - target, axis=1)
    actual_length = float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum())
    target_length = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    return {
        "mean_euclidean_error_m": float(error.mean()),
        "normalized_mean_error": float(error.mean() / diagonal),
        "endpoint_error_m": float(error[-1]),
        "normalized_endpoint_error": float(error[-1] / diagonal),
        "actual_path_length_m": actual_length,
        "target_path_length_m": target_length,
        "path_length_ratio": actual_length / target_length,
        "target_bbox_diagonal_m": diagonal,
    }


def _digit9_loop_metrics(
    actual: np.ndarray,
    target: np.ndarray,
    boundary_index: int,
    *,
    band: int,
) -> dict[str, Any]:
    actual_time = _nearest_landmark_time(
        actual, target[boundary_index], boundary_index, band
    )
    return {
        "ellipse_boundary_target_index": boundary_index,
        "actual_ellipse_closure_time": actual_time,
        "ellipse_closure_arrival_lag_steps": actual_time - boundary_index,
        "ellipse_closure_position_error_m": float(
            np.linalg.norm(actual[actual_time] - target[boundary_index])
        ),
        "digit0_digit8_closure_label_applicable": False,
    }


def _phase_name(index: int, bounds: Mapping[str, Sequence[int]]) -> str:
    for phase in ("stable", "delay", "movement", "hold"):
        start, end = bounds[phase]
        if int(start) <= index < int(end):
            return phase
    raise IndexError("episode index is outside epoch bounds")


def _save_worker_plots(
    output: Path,
    actual: np.ndarray,
    target: np.ndarray,
    closure_distance: np.ndarray,
    closure_tolerance: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual_mean = actual.mean(axis=0)
    target_mean = target.mean(axis=0)
    color = np.linspace(0.0, 1.0, len(actual_mean))
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(target_mean[:, 0], target_mean[:, 1], "k--", label="target")
    scatter = axis.scatter(
        actual_mean[:, 0], actual_mean[:, 1], c=color, cmap="viridis", s=13
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("movement trajectory colored by time")
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="normalized movement time")
    figure.tight_layout()
    figure.savefig(output / "time_colored_overlay.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for batch_index, values in enumerate(closure_distance):
        axis.plot(values, alpha=0.35, label="sample" if batch_index == 0 else None)
    if closure_tolerance is not None:
        axis.axhline(closure_tolerance, color="red", linestyle="--", label="tolerance")
    axis.set_xlabel("episode index")
    axis.set_ylabel("distance to diagnostic closure point (m)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "closure_distance_vs_time.png", dpi=180)
    plt.close(figure)


def _worker_directory(run_root: Path, label: str) -> Path:
    output = (run_root / "per_checkpoint" / label).resolve()
    try:
        output.relative_to(run_root)
    except ValueError as error:
        raise ValueError("closed-loop worker output escapes the run root") from error
    if not output.is_dir():
        raise FileNotFoundError("server runner must create the worker directory")
    unexpected = {path.name for path in output.iterdir()} - {"run.log"}
    if unexpected:
        raise FileExistsError(f"closed-loop worker output is not empty: {unexpected}")
    return output


def run_checkpoint_audit(
    *,
    repository_root: str | Path,
    diagnostic_config_path: str | Path,
    source_run_root: str | Path,
    continuation_run_root: str | Path,
    run_root: str | Path,
    checkpoint_label: str,
    manual_diagnostic_authorized: bool,
) -> dict[str, Any]:
    if not manual_diagnostic_authorized:
        raise PermissionError("closed-loop checkpoint audit requires manual approval")
    import motornet as mn

    from digit_writing.final_protocol_audit import _all_finite
    from digit_writing.geometry import build_digit_trajectory
    from digit_writing.protocol3_gate2 import (
        _dynamic_safety_metrics,
        _movement_metrics,
    )
    from train import DIGIT_ENV_CLASSES, load_digit_policy_checkpoint

    root = Path(repository_root).resolve()
    config = _load_checked_in_diagnostic_config(root, diagnostic_config_path)
    output_root = Path(run_root).resolve()
    if output_root != _resolve_inside(root, config["output"]["directory"]):
        raise ValueError("closed-loop worker run root changed")
    if Path(source_run_root).resolve() != _resolve_inside(
        root, config["source"]["run_directory"]
    ):
        raise ValueError("closed-loop worker source root changed")
    if Path(continuation_run_root).resolve() != _resolve_inside(
        root, config["continuation"]["run_directory"]
    ):
        raise ValueError("closed-loop worker continuation root changed")
    if not (output_root / "prepare_summary.json").is_file():
        raise ValueError("closed-loop diagnostic was not prepared")
    output = _worker_directory(output_root, checkpoint_label)
    record = _record_for_label(config, checkpoint_label)
    review = review_artifact(
        config=config,
        label=checkpoint_label,
        source_run_root=source_run_root,
        continuation_run_root=continuation_run_root,
    )
    checkpoint_path = Path(review["checkpoint"])
    policy, checkpoint = load_digit_policy_checkpoint(
        checkpoint_path, expected_variant=record["expected_variant"]
    )
    checkpoint_protocol_config = checkpoint.get("protocol_config")
    expected_protocol_config = _read_json(
        _resolve_inside(root, config["source"]["protocol_config"])
    )
    if checkpoint_protocol_config != expected_protocol_config:
        raise ValueError("closed-loop checkpoint protocol config differs")
    restore_state = validate_protocol3_resume_checkpoint(
        checkpoint, checkpoint_protocol_config
    )
    checkpoint_identity = checkpoint.get("git_identity", {})
    hp = checkpoint.get("hp", {})
    audit = config["audit"]
    if (
        int(restore_state["completed_updates"]) != int(record["expected_update"])
        or checkpoint_identity.get("repository_head")
        != record["expected_repository_head"]
        or checkpoint_identity.get("mrnntorch_recorded_head") != MRNN_HEAD
        or checkpoint_identity.get("mrnntorch_worktree_head") != MRNN_HEAD
        or hp.get("git_identity") != checkpoint_identity
        or int(hp.get("batch_size", -1)) != audit["batch_size"]
        or int(hp.get("validation_seed", -1)) != audit["validation_seed"]
        or int(hp.get("gate2_direction_index", -1)) != audit["direction_index"]
        or int(hp.get("gate2_delay_index", -1)) != audit["delay_index"]
    ):
        raise ValueError("closed-loop checkpoint metadata differs")
    current_identity = current_git_identity(root)
    for key in ("branch", "mrnntorch_recorded_head", "mrnntorch_worktree_head"):
        if checkpoint_identity.get(key) != current_identity[key]:
            raise ValueError(f"closed-loop checkpoint {key} differs from implementation")

    before_hash = state_dict_sha256(policy.state_dict())
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("closed-loop policy has pre-existing parameter gradients")
    policy.eval()
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    environment = DIGIT_ENV_CLASSES[int(record["digit"])](
        effector=effector,
        **hp.get("env_kwargs", {}),
    )
    batch_size = int(audit["batch_size"])
    observation, info = environment.reset(
        testing=False,
        seed=int(audit["validation_seed"]),
        options={
            "batch_size": batch_size,
            "reach_conds": np.full(
                batch_size, int(audit["direction_index"]), dtype=np.int64
            ),
            "speed_cond": int(audit["speed_condition"]),
            "delay_cond": int(audit["delay_index"]),
            "deterministic": True,
        },
    )
    if int(environment.delay_time) != int(audit["delay_steps"]):
        raise RuntimeError("closed-loop environment delay differs")
    x = torch.zeros((batch_size, int(hp["hid_size"])))
    h = torch.zeros_like(x)
    observations = []
    positions = []
    targets = []
    actions = []
    x_states = []
    h_states = []
    aligned_muscles = []
    loss_muscles = [info["states"]["muscle"][:, 0].unsqueeze(1)]
    loss_hidden = [h.unsqueeze(1)]
    joint_positions = []
    finite = _all_finite(observation) and _all_finite(info)
    timestep = 0
    terminated = False
    while not terminated:
        observations.append(observation[:, None, :])
        x, h, action = policy(observation, x, h, noise=False)
        observation, _, terminated, info = environment.step(timestep, action)
        positions.append(info["states"]["fingertip"][:, None, :])
        targets.append(info["goal"][:, None, :])
        actions.append(action[:, None, :])
        x_states.append(x[:, None, :])
        h_states.append(h[:, None, :])
        muscle = info["states"]["muscle"][:, 0].unsqueeze(1)
        aligned_muscles.append(muscle)
        loss_muscles.append(muscle)
        loss_hidden.append(h.unsqueeze(1))
        joint_positions.append(
            info["states"]["joint"][:, : environment.effector.dof].unsqueeze(1)
        )
        finite &= _all_finite(observation) and _all_finite(info)
        timestep += 1

    tensors = {
        "observations": torch.cat(observations, dim=1),
        "positions": torch.cat(positions, dim=1),
        "targets": torch.cat(targets, dim=1),
        "actions": torch.cat(actions, dim=1),
        "x_states": torch.cat(x_states, dim=1),
        "h_states": torch.cat(h_states, dim=1),
        "aligned_muscles": torch.cat(aligned_muscles, dim=1),
        "loss_muscles": torch.cat(loss_muscles, dim=1),
        "loss_hidden": torch.cat(loss_hidden, dim=1),
        "joint_positions": torch.cat(joint_positions, dim=1),
    }
    episode_steps = tensors["positions"].shape[1]
    if any(
        tensors[name].shape[1] != episode_steps
        for name in (
            "observations",
            "targets",
            "actions",
            "x_states",
            "h_states",
            "aligned_muscles",
            "joint_positions",
        )
    ):
        raise RuntimeError("closed-loop aligned rollout arrays differ in time")
    if (
        tensors["loss_hidden"].shape[1] != episode_steps + 1
        or tensors["loss_muscles"].shape[1] != episode_steps + 1
    ):
        raise RuntimeError("closed-loop loss arrays do not retain the initial state")

    phase_metrics = position_l1_metrics(
        tensors["positions"], tensors["targets"], environment.epoch_bounds
    )
    position_losses = {
        f"{phase}_position": PHASE_WEIGHTS[phase]
        * phase_metrics[f"{phase}_mean_l1"]
        for phase in ("stable", "delay", "movement", "hold")
    }
    regularization_losses = {
        "weighted_l1_rate": l1_rate(tensors["loss_hidden"], hp["l1_rate"]),
        "weighted_l1_weight": l1_weight(policy, hp["l1_weight"]),
        "weighted_l1_muscle_act": l1_muscle_act(
            tensors["loss_muscles"], hp["l1_muscle_act"]
        ),
        "weighted_simple_dynamics": simple_dynamics(
            tensors["loss_hidden"],
            policy.mrnn,
            weight=hp["simple_dynamics_weight"],
        ),
    }
    total_regularization = sum(regularization_losses.values())
    losses = {
        **position_losses,
        "total_position": phase_metrics["phase_normalized_position_l1"],
        **regularization_losses,
        "total_regularization": total_regularization,
        "total_objective": phase_metrics["phase_normalized_position_l1"]
        + total_regularization,
    }
    gradients = gradient_decomposition(
        losses,
        tuple(parameter for parameter in policy.parameters() if parameter.requires_grad),
        epsilon=float(audit["gradient_epsilon"]),
    )
    after_hash = state_dict_sha256(policy.state_dict())
    parameter_grads_absent = all(
        parameter.grad is None for parameter in policy.parameters()
    )

    movement_start, movement_end = environment.epoch_bounds["movement"]
    hold_start, hold_end = environment.epoch_bounds["hold"]
    movement_actual, movement_target, aggregate_movement_metrics = _movement_metrics(
        tensors["positions"], tensors["targets"], environment.epoch_bounds
    )
    dynamic_safety = _dynamic_safety_metrics(
        environment, tensors["positions"], tensors["joint_positions"]
    )
    workspace_audit = _read_json(output_root / "workspace_audit.json")
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
    boundary_metrics = {
        "muscle_activation_max": float(tensors["loss_muscles"].max()),
        "muscle_activation_fraction_ge_0_99": float(
            (tensors["loss_muscles"] >= 0.99).float().mean()
        ),
        "muscle_excitation_fraction_le_0_01": float(
            (tensors["actions"] <= 0.01).float().mean()
        ),
        "muscle_excitation_fraction_ge_0_99": float(
            (tensors["actions"] >= 0.99).float().mean()
        ),
    }

    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in tensors.items()
    }
    dt_seconds = float(environment.geometry_config.dt_seconds)
    joint_velocities = np.zeros_like(arrays["joint_positions"])
    joint_velocities[:, 1:] = (
        np.diff(arrays["joint_positions"], axis=1) / dt_seconds
    )
    arrays["joint_velocities"] = joint_velocities
    movement_arrays = {
        "actual": arrays["positions"][:, movement_start:movement_end],
        "target": arrays["targets"][:, movement_start:movement_end],
        "hold": arrays["positions"][:, hold_start:hold_end],
    }
    alignments = []
    mappings = []
    behavior = []
    closure = []
    shape = []
    kinematics = []
    digit = int(record["digit"])
    digit9_boundary = None
    if digit == 9:
        formal = build_digit_trajectory(
            9, environment.geometry_config, environment.reference_steps
        )
        digit9_boundary = int(formal.boundaries[0].end_index)
    for batch_index in range(batch_size):
        actual = movement_arrays["actual"][batch_index]
        target = movement_arrays["target"][batch_index]
        alignment = constrained_monotonic_alignment(
            actual,
            target,
            band_steps=int(audit["phase_alignment_band_steps"]),
        )
        mapping = alignment.pop("mapping")
        alignments.append(alignment)
        mappings.append(mapping)
        behavior.append(_behavior_metrics(actual, target))
        if digit in (0, 8):
            closure.append(
                closure_metrics(
                    actual,
                    movement_arrays["hold"][batch_index],
                    target[-1],
                    target_diagonal=_target_diagonal(target),
                    movement_start_episode_index=int(movement_start),
                    late_fraction=float(audit["late_movement_fraction"]),
                    tolerance_fraction=float(
                        audit["closure_tolerance_bbox_fraction"]
                    ),
                )
            )
        if digit == 0:
            shape.append(digit0_region_metrics(actual, target, mapping))
        elif digit == 8:
            shape.append(
                digit8_segment_metrics(
                    actual,
                    target,
                    mapping,
                    crossing_window_steps=int(audit["crossing_window_steps"]),
                    outgoing_steps=int(audit["crossing_outgoing_steps"]),
                )
            )
        else:
            shape.append(
                _digit9_loop_metrics(
                    actual,
                    target,
                    int(digit9_boundary),
                    band=int(audit["phase_alignment_band_steps"]),
                )
            )
        kinematics.append(
            regional_kinematic_metrics(
                actual,
                target,
                digit=digit,
                dt_seconds=dt_seconds,
            )
        )
    mappings_array = np.stack(mappings)
    hidden = hidden_phase_metrics(
        arrays,
        digit=digit,
        movement_start=int(movement_start),
        movement_end=int(movement_end),
        phase_band=int(audit["phase_alignment_band_steps"]),
        digit9_ellipse_boundary=digit9_boundary,
        epsilon=float(audit["gradient_epsilon"]),
    )

    actual_derivatives = [
        _padded_derivatives(row, dt_seconds) for row in arrays["positions"]
    ]
    target_derivatives = [
        _padded_derivatives(row, dt_seconds) for row in arrays["targets"]
    ]
    actual_speed = np.stack([row["speed"] for row in actual_derivatives])
    actual_acceleration = np.stack(
        [row["acceleration"] for row in actual_derivatives]
    )
    actual_jerk = np.stack([row["jerk"] for row in actual_derivatives])
    target_speed = np.stack([row["speed"] for row in target_derivatives])
    target_acceleration = np.stack(
        [row["acceleration"] for row in target_derivatives]
    )
    target_jerk = np.stack([row["jerk"] for row in target_derivatives])

    def cumulative(points: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                np.zeros((points.shape[0], 1), dtype=np.float64),
                np.cumsum(np.linalg.norm(np.diff(points, axis=1), axis=2), axis=1),
            ),
            axis=1,
        )

    actual_cumulative = cumulative(arrays["positions"])
    target_cumulative = cumulative(arrays["targets"])
    if digit in (0, 8):
        reference_points = movement_arrays["target"][:, -1]
        closure_tolerance = float(closure[0]["closure_tolerance_m"])
    else:
        reference_points = movement_arrays["target"][:, int(digit9_boundary)]
        closure_tolerance = None
    closure_distance = np.linalg.norm(
        arrays["positions"] - reference_points[:, None, :], axis=2
    )
    np.savez_compressed(
        output / "raw_rollout.npz",
        observations=arrays["observations"],
        positions=arrays["positions"],
        targets=arrays["targets"],
        actions=arrays["actions"],
        muscle_activations=arrays["aligned_muscles"],
        joint_positions=arrays["joint_positions"],
        joint_velocities=joint_velocities,
        x_states=arrays["x_states"],
        h_states=arrays["h_states"],
        loss_hidden=arrays["loss_hidden"],
        loss_muscles=arrays["loss_muscles"],
        actual_speed_m_s=actual_speed,
        target_speed_m_s=target_speed,
        actual_acceleration_m_s2=actual_acceleration,
        target_acceleration_m_s2=target_acceleration,
        actual_jerk_m_s3=actual_jerk,
        target_jerk_m_s3=target_jerk,
        actual_cumulative_path_m=actual_cumulative,
        target_cumulative_path_m=target_cumulative,
        distance_to_diagnostic_closure_point_m=closure_distance,
        phase_mappings=mappings_array,
    )

    rows = []
    for index in range(episode_steps):
        phase = _phase_name(index, environment.epoch_bounds)
        movement_index = index - movement_start if phase == "movement" else ""
        row: dict[str, Any] = {
            "episode_index": index,
            "phase": phase,
            "movement_index": movement_index,
            "actual_x_m": float(arrays["positions"][:, index, 0].mean()),
            "actual_y_m": float(arrays["positions"][:, index, 1].mean()),
            "target_x_m": float(arrays["targets"][:, index, 0].mean()),
            "target_y_m": float(arrays["targets"][:, index, 1].mean()),
            "actual_speed_m_s": float(actual_speed[:, index].mean()),
            "target_speed_m_s": float(target_speed[:, index].mean()),
            "actual_acceleration_m_s2": float(actual_acceleration[:, index].mean()),
            "target_acceleration_m_s2": float(target_acceleration[:, index].mean()),
            "actual_jerk_m_s3": float(actual_jerk[:, index].mean()),
            "target_jerk_m_s3": float(target_jerk[:, index].mean()),
            "actual_cumulative_path_m": float(actual_cumulative[:, index].mean()),
            "target_cumulative_path_m": float(target_cumulative[:, index].mean()),
            "distance_to_diagnostic_closure_point_m": float(
                closure_distance[:, index].mean()
            ),
        }
        for action_index in range(arrays["actions"].shape[2]):
            row[f"muscle_excitation_{action_index}"] = float(
                arrays["actions"][:, index, action_index].mean()
            )
            row[f"muscle_activation_{action_index}"] = float(
                arrays["aligned_muscles"][:, index, action_index].mean()
            )
        for joint_index in range(arrays["joint_positions"].shape[2]):
            row[f"joint_position_{joint_index}"] = float(
                arrays["joint_positions"][:, index, joint_index].mean()
            )
            row[f"joint_velocity_{joint_index}"] = float(
                joint_velocities[:, index, joint_index].mean()
            )
        rows.append(row)
    _write_csv(output / "timestep_mean.csv", rows)
    _save_worker_plots(
        output,
        movement_arrays["actual"],
        movement_arrays["target"],
        closure_distance,
        closure_tolerance,
    )

    finite &= all(
        np.isfinite(value).all()
        for value in (
            arrays["observations"],
            arrays["positions"],
            arrays["targets"],
            arrays["actions"],
            arrays["x_states"],
            arrays["h_states"],
            arrays["aligned_muscles"],
            arrays["joint_positions"],
            joint_velocities,
            mappings_array,
        )
    )
    behavior_passed = bool(
        aggregate_movement_metrics["normalized_mean_error"] <= 0.08
        and aggregate_movement_metrics["normalized_endpoint_error"] <= 0.05
        and 0.85 <= aggregate_movement_metrics["path_length_ratio"] <= 1.15
    )
    engineering_passed = bool(
        finite
        and safety_passed
        and before_hash == after_hash
        and parameter_grads_absent
        and gradients["optimizer_steps_during_audit"] == 0
        and gradients["component_sum_consistent"]
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_closed_loop_checkpoint_audit",
        "checkpoint": review,
        "implementation_git_identity": current_identity,
        "checkpoint_git_identity": checkpoint_identity,
        "completed_updates": int(restore_state["completed_updates"]),
        "optimizer_steps_during_audit": 0,
        "testing": False,
        "network_noise": False,
        "deterministic_observations": True,
        "batch_size": batch_size,
        "episode_steps": episode_steps,
        "aligned_time_steps": episode_steps,
        "loss_hidden_time_steps": int(tensors["loss_hidden"].shape[1]),
        "loss_muscle_time_steps": int(tensors["loss_muscles"].shape[1]),
        "epoch_bounds": environment.epoch_bounds,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "model_state_unchanged": before_hash == after_hash,
        "parameter_gradients_absent_after_audit": parameter_grads_absent,
        "all_values_finite": bool(finite),
        "behavior_metrics": {
            "aggregate_gate2_semantics": aggregate_movement_metrics,
            "per_batch": behavior,
            "summary": _aggregate_records(behavior),
            "behavior_passed": behavior_passed,
        },
        "temporal_alignment": {
            "per_batch": alignments,
            "summary": _aggregate_records(alignments),
        },
        "closure": (
            {
                "applicable": True,
                "per_batch": closure,
                "summary": _aggregate_records(closure),
            }
            if digit in (0, 8)
            else {
                "applicable": False,
                "positive_control": shape,
                "summary": _aggregate_records(shape),
            }
        ),
        "shape_regions": {
            "per_batch": shape,
            "summary": _aggregate_records(shape),
        },
        "hidden_phase": hidden,
        "gradient_decomposition": gradients,
        "kinematics": {
            "per_batch": kinematics,
            "summary": _aggregate_records(kinematics),
        },
        "workspace_minima": workspace_minima,
        "dynamic_safety": dynamic_safety,
        "boundary_metrics": boundary_metrics,
        "safety_passed": safety_passed,
        "engineering_passed": engineering_passed,
        "automatic_root_cause_selection": False,
        "automatic_stage_b_start": False,
        "automatic_training": False,
        "qualitative_review_required": True,
    }
    _write_json(output / "metrics.json", result)
    _write_json(
        output / "resolved_config.json",
        {
            "diagnostic_config": config,
            "active_checkpoint": record,
            "artifact_review": review,
        },
    )
    expected_files = {
        "run.log",
        "resolved_config.json",
        "raw_rollout.npz",
        "timestep_mean.csv",
        "metrics.json",
        "time_colored_overlay.png",
        "closure_distance_vs_time.png",
    }
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"closed-loop worker artifacts differ: {sorted(actual_files)}"
        )
    return result


def _summary_mean(value: Mapping[str, Any]) -> float:
    if set(value) != {"mean", "min", "max"}:
        raise ValueError("expected a numeric mean/min/max summary")
    return float(value["mean"])


def _flatten_summary_metrics(
    value: Mapping[str, Any], prefix: str = ""
) -> list[tuple[str, Mapping[str, Any]]]:
    rows = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping) and set(item) == {"mean", "min", "max"}:
            rows.append((name, item))
        elif isinstance(item, Mapping):
            rows.extend(_flatten_summary_metrics(item, name))
    return rows


def _save_combined_plots(output: Path, results: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for digit, filename in (
        (0, "digit0_time_colored_overlay.png"),
        (8, "digit8_time_colored_overlay.png"),
    ):
        selected = [row for row in results if int(row["checkpoint"]["digit"]) == digit]
        figure, axes = plt.subplots(
            1, len(selected), figsize=(5 * len(selected), 4), squeeze=False
        )
        for axis, result in zip(axes[0], selected):
            label = result["checkpoint"]["label"]
            with np.load(
                output / "per_checkpoint" / label / "raw_rollout.npz"
            ) as raw:
                movement_start, movement_end = result["epoch_bounds"]["movement"]
                actual = raw["positions"][:, movement_start:movement_end].mean(axis=0)
                target = raw["targets"][:, movement_start:movement_end].mean(axis=0)
            color = np.linspace(0.0, 1.0, len(actual))
            axis.plot(target[:, 0], target[:, 1], "k--", linewidth=1.0)
            axis.scatter(actual[:, 0], actual[:, 1], c=color, cmap="viridis", s=9)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(label)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)

    selected = [row for row in results if int(row["checkpoint"]["digit"]) in (0, 8)]
    figure, axis = plt.subplots(figsize=(10, 5))
    for result in selected:
        label = result["checkpoint"]["label"]
        with np.load(
            output / "per_checkpoint" / label / "raw_rollout.npz"
        ) as raw:
            values = raw["distance_to_diagnostic_closure_point_m"].mean(axis=0)
        axis.plot(values, label=label)
    axis.set_xlabel("episode index")
    axis.set_ylabel("mean distance to closed endpoint (m)")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "closure_distance_vs_time.png", dpi=180)
    plt.close(figure)


def summarize_diagnostic(
    *,
    repository_root: str | Path,
    diagnostic_config_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _load_checked_in_diagnostic_config(root, diagnostic_config_path)
    output = Path(run_root).resolve()
    if output != _resolve_inside(root, config["output"]["directory"]):
        raise ValueError("closed-loop summary root changed")
    results = [
        _read_json(output / "per_checkpoint" / label / "metrics.json")
        for label in CHECKPOINT_LABELS
    ]
    if [row.get("checkpoint", {}).get("label") for row in results] != list(
        CHECKPOINT_LABELS
    ):
        raise RuntimeError("closed-loop checkpoint results are incomplete or reordered")
    current_identity = current_git_identity(root)
    if any(
        row.get("implementation_git_identity") != current_identity for row in results
    ):
        raise RuntimeError("closed-loop checkpoint implementation Git identity differs")
    if any(row.get("engineering_passed") is not True for row in results):
        raise RuntimeError("one or more closed-loop checkpoint audits failed engineering")
    if any(
        row.get("optimizer_steps_during_audit") != 0
        or row.get("model_state_unchanged") is not True
        or row.get("parameter_gradients_absent_after_audit") is not True
        for row in results
    ):
        raise RuntimeError("closed-loop audit violated zero-step read-only semantics")

    comparison_rows = []
    temporal_rows = []
    closure_rows = []
    shape_rows = []
    hidden_summary = {}
    gradient_summary = {}
    for result in results:
        checkpoint = result["checkpoint"]
        label = checkpoint["label"]
        digit = int(checkpoint["digit"])
        behavior = result["behavior_metrics"]["summary"]
        temporal = result["temporal_alignment"]["summary"]
        closure_info = result["closure"]
        diagnostic_labels = []
        if closure_info["applicable"]:
            for row in closure_info["per_batch"]:
                diagnostic_labels.extend(row["diagnostic_labels"])
        comparison_rows.append(
            {
                "label": label,
                "digit": digit,
                "origin": checkpoint["origin"],
                "completed_updates": result["completed_updates"],
                "normalized_mean_error": _summary_mean(
                    behavior["normalized_mean_error"]
                ),
                "normalized_endpoint_error": _summary_mean(
                    behavior["normalized_endpoint_error"]
                ),
                "path_length_ratio": _summary_mean(behavior["path_length_ratio"]),
                "time_aligned_normalized_mean_error": _summary_mean(
                    temporal["time_aligned_normalized_mean_error"]
                ),
                "phase_aligned_normalized_mean_error": _summary_mean(
                    temporal["phase_aligned_normalized_mean_error"]
                ),
                "phase_lag_steps_final": _summary_mean(
                    temporal["phase_lag_steps_final"]
                ),
                "closure_labels": ";".join(sorted(set(diagnostic_labels))),
                "behavior_passed": result["behavior_metrics"]["behavior_passed"],
                "engineering_passed": result["engineering_passed"],
            }
        )
        temporal_rows.append(
            {
                "label": label,
                "digit": digit,
                "time_aligned_normalized_mean_error": _summary_mean(
                    temporal["time_aligned_normalized_mean_error"]
                ),
                "phase_aligned_normalized_mean_error": _summary_mean(
                    temporal["phase_aligned_normalized_mean_error"]
                ),
                "phase_alignment_error_reduction": _summary_mean(
                    temporal["phase_alignment_error_reduction"]
                ),
                "phase_lag_steps_mean": _summary_mean(
                    temporal["phase_lag_steps_mean"]
                ),
                "phase_lag_steps_p95": _summary_mean(
                    temporal["phase_lag_steps_p95"]
                ),
                "phase_lag_steps_final": _summary_mean(
                    temporal["phase_lag_steps_final"]
                ),
                "target_phase_completion_at_movement_end": _summary_mean(
                    temporal["target_phase_completion_at_movement_end"]
                ),
            }
        )
        if closure_info["applicable"]:
            closure_summary = closure_info["summary"]
            closure_rows.append(
                {
                    "label": label,
                    "digit": digit,
                    "applicable": True,
                    "movement_endpoint_distance_m": _summary_mean(
                        closure_summary["movement_endpoint_distance_m"]
                    ),
                    "minimum_late_movement_closure_distance_m": _summary_mean(
                        closure_summary[
                            "minimum_late_movement_closure_distance_m"
                        ]
                    ),
                    "minimum_hold_closure_distance_m": _summary_mean(
                        closure_summary["minimum_hold_closure_distance_m"]
                    ),
                    "hold_path_length_m": _summary_mean(
                        closure_summary["hold_path_length_m"]
                    ),
                    "reached_during_movement_count": closure_summary[
                        "closure_reached_during_movement"
                    ]["true_count"],
                    "reached_only_during_hold_count": closure_summary[
                        "closure_reached_only_during_hold"
                    ]["true_count"],
                    "never_reached_count": closure_summary[
                        "closure_never_reached"
                    ]["true_count"],
                    "hold_instability_count": closure_summary[
                        "closure_hold_instability"
                    ]["true_count"],
                }
            )
        else:
            positive = closure_info["summary"]
            closure_rows.append(
                {
                    "label": label,
                    "digit": digit,
                    "applicable": False,
                    "movement_endpoint_distance_m": "",
                    "minimum_late_movement_closure_distance_m": "",
                    "minimum_hold_closure_distance_m": "",
                    "hold_path_length_m": "",
                    "reached_during_movement_count": "",
                    "reached_only_during_hold_count": "",
                    "never_reached_count": "",
                    "hold_instability_count": "",
                }
            )
            if "ellipse_closure_position_error_m" not in positive:
                raise RuntimeError("digit9 positive-control closure metrics are missing")
        for metric_name, values in _flatten_summary_metrics(
            result["shape_regions"]["summary"]
        ):
            shape_rows.append(
                {
                    "label": label,
                    "digit": digit,
                    "metric": metric_name,
                    "mean": values["mean"],
                    "min": values["min"],
                    "max": values["max"],
                }
            )
        hidden_summary[label] = result["hidden_phase"]
        gradient_summary[label] = result["gradient_decomposition"]

    _write_csv(output / "checkpoint_comparison.csv", comparison_rows)
    _write_csv(output / "temporal_alignment.csv", temporal_rows)
    _write_csv(output / "closure_summary.csv", closure_rows)
    _write_csv(output / "shape_region_summary.csv", shape_rows)
    _write_json(output / "hidden_phase_summary.json", hidden_summary)
    _write_json(output / "gradient_decomposition.json", gradient_summary)
    _save_combined_plots(output, results)

    lines = [
        "# Protocol3 digit0 / digit8 closed-loop diagnostic report",
        "",
        "Seven frozen checkpoints were audited with zero optimizer steps.",
        "The program reports evidence and does not select a root cause.",
        "Stage B training was not started.",
        "",
        "| checkpoint | digit | update | mean | endpoint | path | phase-aligned | final lag | closure labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {label} | {digit} | {completed_updates} | {normalized_mean_error:.6f} "
            "| {normalized_endpoint_error:.6f} | {path_length_ratio:.6f} "
            "| {phase_aligned_normalized_mean_error:.6f} | {phase_lag_steps_final:.2f} "
            "| {closure_labels} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Manual review is required for temporal/phase, spatial compression, hidden-state, gradient, and dynamics evidence.",
            "No geometry, timing, loss, network, seed, or training decision was changed.",
            "",
        ]
    )
    with (output / "CLOSED_LOOP_DIAGNOSTIC_REPORT.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(lines))
    summary = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_digit0_digit8_closed_loop_diagnostic_summary",
        "completed_checkpoints": len(results),
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "optimizer_steps_during_audit": 0,
        "all_model_states_unchanged": True,
        "all_parameter_gradients_absent": True,
        "engineering_passed": True,
        "automatic_root_cause_selection": False,
        "automatic_stage_b_start": False,
        "automatic_training": False,
        "automatic_geometry_change": False,
        "automatic_loss_change": False,
        "automatic_second_seed": False,
        "formal_full10_started": False,
        "qualitative_review_required": True,
        "passed": True,
    }
    _write_json(output / "closed_loop_diagnostic_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    worker = subparsers.add_parser("run-checkpoint")
    summarize = subparsers.add_parser("summarize")
    for subparser in (prepare, worker, summarize):
        subparser.add_argument("--repository-root", required=True)
        subparser.add_argument("--diagnostic-config", required=True)
        subparser.add_argument("--run-root", required=True)
    for subparser in (prepare, worker):
        subparser.add_argument("--source-run-root", required=True)
        subparser.add_argument("--continuation-run-root", required=True)
    prepare.add_argument("--source-evidence-root", required=True)
    prepare.add_argument("--continuation-evidence-root", required=True)
    worker.add_argument("--checkpoint-label", choices=CHECKPOINT_LABELS, required=True)
    worker.add_argument("--manual-diagnostic-authorized", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "repository_root": arguments.repository_root,
        "diagnostic_config_path": arguments.diagnostic_config,
        "run_root": arguments.run_root,
    }
    if arguments.command == "prepare":
        prepare_diagnostic(
            **common,
            source_run_root=arguments.source_run_root,
            source_evidence_root=arguments.source_evidence_root,
            continuation_run_root=arguments.continuation_run_root,
            continuation_evidence_root=arguments.continuation_evidence_root,
        )
    elif arguments.command == "run-checkpoint":
        run_checkpoint_audit(
            **common,
            source_run_root=arguments.source_run_root,
            continuation_run_root=arguments.continuation_run_root,
            checkpoint_label=arguments.checkpoint_label,
            manual_diagnostic_authorized=arguments.manual_diagnostic_authorized,
        )
    else:
        summarize_diagnostic(**common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
