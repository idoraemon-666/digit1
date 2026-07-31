"""Protocol3 digit0/digit8 movement-interval and fixed-LR scratch sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from digit_writing.geometry import build_digit_trajectory, load_geometry_config
from digit_writing.geometry_audit import build_workspace_audit
from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    protocol_config_sha256,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)
from digit_writing.protocol3_corner_ease import (
    _audit_best_checkpoint,
    select_validation_history,
)


RUN_KIND = "protocol3_digit0_digit8_movement_interval_lr_scratch"
VARIANT = "digit0_digit8_twelve_arm_interval_lr_scratch6000"
DIGITS = (0, 8)
SOURCE_INTERVALS = {0: 170, 8: 200}
MOVEMENT_INTERVALS_BY_DIGIT = {
    0: (170, 200, 220),
    8: (200, 220, 240),
}
LEARNING_RATE_ARMS = (("lr1e3", 0.001), ("lr3e4", 0.0003))
MAX_UPDATES = 6000
VALIDATION_INTERVAL = 100
EXPECTED_ARM_COUNT = 12


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_repository_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("configured path escapes the repository") from error
    return path


def _expected_arms() -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            f"digit{digit}_time{movement_intervals}_{lr_label}",
            digit,
            movement_intervals,
            lr_label,
            learning_rate,
            f"digit{digit}/time{movement_intervals}/{lr_label}",
        )
        for lr_label, learning_rate in LEARNING_RATE_ARMS
        for digit in DIGITS
        for movement_intervals in MOVEMENT_INTERVALS_BY_DIGIT[digit]
    )


def load_experiment_config(
    repository_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(config_path)
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("movement-interval sweep requires Protocol3")
    if config.get("run_kind") != RUN_KIND or config.get("variant") != VARIANT:
        raise ValueError("movement-interval sweep identity differs")
    if config.get("device") != "cpu":
        raise ValueError("movement-interval sweep is CPU-only")
    if (
        config.get("scale_multiplier") != 2.5
        or config.get("source_timing_mode")
        != "fixed_segment_timing_corner_ease_v3"
        or config.get("selected_reference_steps") != 100
        or config.get("condition_schedule")
        != "protocol3_corner_ease_single_condition"
    ):
        raise ValueError("movement-interval source timing identity differs")
    if tuple(config.get("digits", ())) != DIGITS:
        raise ValueError("movement-interval digits must be exactly 0 and 8")
    if config.get("source_movement_intervals") != {"0": 170, "8": 200}:
        raise ValueError("source movement intervals differ")
    if config.get("movement_interval_arms") != {
        "0": [170, 200, 220],
        "8": [200, 220, 240],
    }:
        raise ValueError("movement interval arms differ")
    if tuple(
        (arm.get("label"), float(arm.get("learning_rate")))
        for arm in config.get("learning_rate_arms", ())
    ) != LEARNING_RATE_ARMS:
        raise ValueError("learning-rate arms differ")
    actual_arms = tuple(
        (
            arm.get("label"),
            arm.get("digit"),
            arm.get("movement_intervals"),
            arm.get("learning_rate_label"),
            float(arm.get("learning_rate")),
            arm.get("output_subdirectory"),
        )
        for arm in config.get("arms", ())
    )
    if actual_arms != _expected_arms():
        raise ValueError("twelve-arm matrix differs")
    if config.get("optimizer") != {
        "name": "Adam",
        "grad_clip_norm": 1.0,
        "fixed_learning_rate_no_schedule": True,
    }:
        raise ValueError("optimizer identity differs")
    if config.get("training") != {
        "batch_size": 8,
        "max_updates": MAX_UPDATES,
        "validation_interval": VALIDATION_INTERVAL,
        "from_scratch": True,
        "early_stopping": False,
        "synchronized_parallel_arms": EXPECTED_ARM_COUNT,
    }:
        raise ValueError("scratch training boundary differs")
    if config.get("timing_intervention") != {
        "method": "single_primitive_linear_arclength_resample",
        "dt_seconds": 0.01,
        "stable_steps": 25,
        "delay_steps": 50,
        "hold_steps": 25,
        "spatial_geometry_unchanged": True,
        "speed_cue_unchanged": True,
        "digit0_time170_is_original_control": True,
        "digit8_time200_is_original_control": True,
    }:
        raise ValueError("timing intervention differs")
    if config.get("checkpoint_selection") != {
        "stable_consecutive_evaluations": 3,
        "primary_metric": "normalized_mean_error",
        "tie_break": "earlier_update",
    }:
        raise ValueError("checkpoint selection differs")
    if config.get("decision") != {
        "automatic_interval_selection": False,
        "automatic_learning_rate_selection": False,
        "automatic_continuation": False,
        "automatic_second_seed": False,
        "automatic_loss_change": False,
        "automatic_geometry_change": False,
        "formal_full10_start": False,
        "qualitative_overlay_review_required": True,
    }:
        raise ValueError("decision boundary differs")
    if config.get("output") != {
        "directory": (
            "runs/digit_writing_original_protocol3/scale2p50/ref100/"
            "corner_ease_v3/digit0_digit8_movement_interval_lr_scratch6000"
        ),
        "initial_checkpoint": "initial_checkpoint.pt",
        "best_checkpoint": "best_checkpoint.pt",
        "final_checkpoint": "final_checkpoint.pt",
    }:
        raise ValueError("output identity differs")

    source = _read_json(
        _resolve_repository_path(root, str(config["source_protocol_config"]))
    )
    if (
        source.get("protocol") != config["protocol"]
        or source.get("run_kind") != "protocol3_corner_ease_overfit"
        or source.get("variant") != "ten_digit_corner_ease_v3_overfit6000"
        or source.get("geometry_config") != config["geometry_config"]
    ):
        raise ValueError("source Protocol3 config identity differs")
    for key in (
        "seed",
        "validation_seed",
        "device",
        "scale_multiplier",
        "selected_reference_steps",
        "model",
        "position_loss",
        "regularization",
    ):
        if config.get(key) != source.get(key):
            raise ValueError(f"scratch config differs from source: {key}")
    if source.get("optimizer") != {
        "name": "Adam",
        "learning_rate": 0.001,
        "grad_clip_norm": 1.0,
    }:
        raise ValueError("source optimizer identity differs")
    if source.get("training") != {
        "batch_size": 8,
        "max_updates": MAX_UPDATES,
        "validation_interval": VALIDATION_INTERVAL,
    }:
        raise ValueError("source training identity differs")
    geometry = load_geometry_config(
        _resolve_repository_path(root, str(config["geometry_config"]))
    )
    if (
        geometry.protocol != "digit_writing_original_protocol3"
        or geometry.timing_mode != config["source_timing_mode"]
        or geometry.selected_reference_steps != 100
        or not math.isclose(
            geometry.dt_seconds,
            config["timing_intervention"]["dt_seconds"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("geometry config differs from the frozen timing source")
    return config


def _arm_by_label(
    config: Mapping[str, Any], arm_label: str
) -> dict[str, Any]:
    matches = [arm for arm in config["arms"] if arm["label"] == arm_label]
    if len(matches) != 1:
        raise ValueError(f"unknown movement-interval arm: {arm_label}")
    return dict(matches[0])


def _arm_output(
    run_root: Path, arm: Mapping[str, Any]
) -> Path:
    output = (run_root / str(arm["output_subdirectory"])).resolve()
    try:
        output.relative_to(run_root.resolve())
    except ValueError as error:
        raise ValueError("arm output escapes the experiment root") from error
    return output


def _resolved_arm_config(
    master: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> dict[str, Any]:
    output_directory = (
        f"{master['output']['directory']}/{arm['output_subdirectory']}"
    )
    return {
        "protocol": master["protocol"],
        "run_kind": master["run_kind"],
        "variant": arm["label"],
        "seed": master["seed"],
        "validation_seed": master["validation_seed"],
        "device": master["device"],
        "geometry_config": master["geometry_config"],
        "scale_multiplier": master["scale_multiplier"],
        "timing_mode": master["source_timing_mode"],
        "selected_reference_steps": master["selected_reference_steps"],
        "condition_schedule": master["condition_schedule"],
        "case": {
            "label": arm["label"],
            "digit": arm["digit"],
            "direction_index": 0,
            "delay_steps": 50,
            "movement_intervals": arm["movement_intervals"],
        },
        "model": master["model"],
        "optimizer": {
            "name": "Adam",
            "learning_rate": arm["learning_rate"],
            "grad_clip_norm": master["optimizer"]["grad_clip_norm"],
            "fixed_learning_rate_no_schedule": True,
        },
        "training": {
            "batch_size": master["training"]["batch_size"],
            "max_updates": master["training"]["max_updates"],
            "validation_interval": master["training"]["validation_interval"],
            "from_scratch": True,
            "early_stopping": False,
        },
        "position_loss": master["position_loss"],
        "regularization": master["regularization"],
        "timing_intervention": master["timing_intervention"],
        "checkpoint_selection": master["checkpoint_selection"],
        "decision": master["decision"],
        "master_protocol_config_sha256": protocol_config_sha256(master),
        "output": {
            "directory": output_directory,
            "initial_checkpoint": master["output"]["initial_checkpoint"],
            "best_checkpoint": master["output"]["best_checkpoint"],
            "final_checkpoint": master["output"]["final_checkpoint"],
        },
    }


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _target_kinematics(points: np.ndarray, dt_seconds: float) -> dict[str, float]:
    velocity = np.diff(points, axis=0) / dt_seconds
    acceleration = np.diff(velocity, axis=0) / dt_seconds
    jerk = np.diff(acceleration, axis=0) / dt_seconds
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    return {
        "mean_speed_m_s": float(speed.mean()),
        "p95_speed_m_s": _percentile(speed, 95.0),
        "p95_acceleration_m_s2": _percentile(acceleration_norm, 95.0),
        "p95_jerk_m_s3": _percentile(jerk_norm, 95.0),
    }


def prepare_experiment(
    repository_root: str | Path,
    experiment_config_path: str | Path,
    run_root: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_experiment_config(root, experiment_config_path)
    expected_root = _resolve_repository_path(root, config["output"]["directory"])
    output = Path(run_root).resolve()
    if output != expected_root:
        raise ValueError("run root differs from the checked-in config")
    if output.exists():
        raise FileExistsError(f"scratch sweep output already exists: {output}")
    output.mkdir(parents=True)
    evidence = Path(evidence_root).resolve()
    evidence.mkdir(parents=True, exist_ok=True)

    geometry = load_geometry_config(
        _resolve_repository_path(root, config["geometry_config"])
    )
    workspace_audit = build_workspace_audit(geometry)
    if not workspace_audit["passed"]:
        raise RuntimeError("source workspace audit failed")

    timing_rows: list[dict[str, Any]] = []
    all_spatial_identities_unchanged = True
    for digit in DIGITS:
        source = build_digit_trajectory(digit, geometry, 100)
        if source.movement_intervals != SOURCE_INTERVALS[digit]:
            raise RuntimeError(f"digit {digit} source intervals differ")
        if len(source.boundaries) != 1:
            raise RuntimeError("timing sweep requires a one-primitive digit")
        for movement_intervals in MOVEMENT_INTERVALS_BY_DIGIT[digit]:
            candidate = build_digit_trajectory(
                digit,
                geometry,
                100,
                movement_intervals_override=movement_intervals,
            )
            if candidate.movement_intervals != movement_intervals:
                raise RuntimeError("candidate movement intervals differ")
            if len(candidate.points) != movement_intervals + 1:
                raise RuntimeError("candidate sample accounting differs")
            if not np.isfinite(candidate.points).all():
                raise RuntimeError("candidate target contains NaN or Inf")
            source_boundary = source.boundaries[0]
            candidate_boundary = candidate.boundaries[0]
            spatial_identity_unchanged = bool(
                source_boundary.source_geometry_sha256
                == candidate_boundary.source_geometry_sha256
                and source_boundary.derived_path_geometry_sha256
                == candidate_boundary.derived_path_geometry_sha256
                and math.isclose(
                    source.arc_length_m,
                    candidate.arc_length_m,
                    rel_tol=0.0,
                    abs_tol=1e-14,
                )
                and np.array_equal(
                    source.points[[0, -1]],
                    candidate.points[[0, -1]],
                )
            )
            if not spatial_identity_unchanged:
                raise RuntimeError("timing intervention changed spatial identity")
            temporal_identity_changed = bool(
                source_boundary.temporal_sampling_sha256
                != candidate_boundary.temporal_sampling_sha256
            )
            if temporal_identity_changed != (
                movement_intervals != SOURCE_INTERVALS[digit]
            ):
                raise RuntimeError("temporal identity change differs from design")
            all_spatial_identities_unchanged &= spatial_identity_unchanged
            timing_rows.append(
                {
                    "digit": digit,
                    "source_movement_intervals": SOURCE_INTERVALS[digit],
                    "movement_intervals": movement_intervals,
                    "is_original_control": (
                        movement_intervals == SOURCE_INTERVALS[digit]
                    ),
                    "movement_duration_s": candidate.movement_duration_s,
                    "arc_length_m": candidate.arc_length_m,
                    "spatial_identity_unchanged": spatial_identity_unchanged,
                    "temporal_identity_changed": temporal_identity_changed,
                    **_target_kinematics(
                        candidate.points,
                        geometry.dt_seconds,
                    ),
                }
            )

    _write_csv(output / "target_timing_kinematics.csv", timing_rows)
    _write_csv(output / "arm_manifest.csv", config["arms"])
    _write_json(evidence / "resolved_experiment_config.json", config)
    _write_json(evidence / "workspace_audit.json", workspace_audit)
    preflight = {
        "protocol": config["protocol"],
        "run_kind": f"{RUN_KIND}_prepare",
        "git_identity": current_git_identity(root),
        "protocol_config_sha256": protocol_config_sha256(config),
        "digits": list(DIGITS),
        "source_movement_intervals": {
            str(key): value for key, value in SOURCE_INTERVALS.items()
        },
        "movement_interval_arms": {
            str(digit): list(MOVEMENT_INTERVALS_BY_DIGIT[digit])
            for digit in DIGITS
        },
        "learning_rate_arms": [
            {"label": label, "learning_rate": learning_rate}
            for label, learning_rate in LEARNING_RATE_ARMS
        ],
        "expected_arms": EXPECTED_ARM_COUNT,
        "all_spatial_identities_unchanged": all_spatial_identities_unchanged,
        "workspace_passed": True,
        "training_updates": 0,
        "passed": True,
    }
    _write_json(evidence / "preflight_summary.json", preflight)
    return preflight


def _hp_for_arm(
    root: Path,
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    from train import PROTOCOL3_DELAYS, _base_hp_from_config

    hp = _base_hp_from_config(resolved)
    case = resolved["case"]
    hp.update(
        {
            "variant": resolved["variant"],
            "condition_schedule": "protocol3_corner_ease_single_condition",
            "gate2_direction_index": int(case["direction_index"]),
            "gate2_delay_index": PROTOCOL3_DELAYS.index(case["delay_steps"]),
            "stop_after_updates": MAX_UPDATES,
            "artifact_profile": "protocol3_corner_ease_minimal_v1",
            "env_kwargs": {
                "geometry_config_path": str(
                    _resolve_repository_path(root, resolved["geometry_config"])
                ),
                "movement_intervals_override": int(
                    case["movement_intervals"]
                ),
            },
        }
    )
    return hp


def _optimizer_learning_rates(payload: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(group["lr"])
        for group in payload["optimizer_state_dict"]["param_groups"]
    )


def run_arm(
    repository_root: str | Path,
    experiment_config_path: str | Path,
    run_root: str | Path,
    evidence_root: str | Path,
    arm_label: str,
) -> dict[str, Any]:
    from train import (
        _digit_env_dict,
        load_digit_policy_checkpoint,
        train_subsets_base_model,
    )

    root = Path(repository_root).resolve()
    config = load_experiment_config(root, experiment_config_path)
    expected_root = _resolve_repository_path(root, config["output"]["directory"])
    output_root = Path(run_root).resolve()
    if output_root != expected_root:
        raise ValueError("run root differs from the checked-in config")
    arm = _arm_by_label(config, arm_label)
    output = _arm_output(output_root, arm)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in output.iterdir()
        if path.name != "run.log"
    }
    if unexpected:
        raise FileExistsError(f"arm output is not empty: {sorted(unexpected)}")

    resolved = _resolved_arm_config(config, arm)
    _write_json(output / "resolved_config.json", resolved)
    hp = _hp_for_arm(root, resolved)
    training_summary = train_subsets_base_model(
        str(output),
        resolved["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((int(arm["digit"]),)),
    )
    if int(training_summary["updates"]) != MAX_UPDATES:
        raise RuntimeError("scratch arm did not finish 6000 updates")
    selection = training_summary.get("corner_ease_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("scratch arm did not produce checkpoint selection")

    initial_path = output / resolved["output"]["initial_checkpoint"]
    initial_policy, initial_payload = load_digit_policy_checkpoint(
        initial_path,
        expected_variant=arm["label"],
    )
    initial_state = validate_protocol3_resume_checkpoint(
        initial_payload,
        resolved,
    )
    if int(initial_state["completed_updates"]) != 0:
        raise RuntimeError("initial checkpoint is not update zero")
    initial_state_sha256 = state_dict_sha256(initial_policy.state_dict())

    best_path = output / resolved["output"]["best_checkpoint"]
    best_policy, best_payload = load_digit_policy_checkpoint(
        best_path,
        expected_variant=arm["label"],
    )
    best_state = validate_protocol3_resume_checkpoint(best_payload, resolved)
    if int(best_state["completed_updates"]) != int(selection["best_update"]):
        raise RuntimeError("best checkpoint update differs from selection")
    workspace_audit = _read_json(Path(evidence_root) / "workspace_audit.json")
    case = resolved["case"]
    best_audit = _audit_best_checkpoint(
        best_policy,
        hp,
        case,
        workspace_audit,
        output,
        selection,
    )

    final_path = output / resolved["output"]["final_checkpoint"]
    final_policy, final_payload = load_digit_policy_checkpoint(
        final_path,
        expected_variant=arm["label"],
    )
    final_state = validate_protocol3_resume_checkpoint(final_payload, resolved)
    if int(final_state["completed_updates"]) != MAX_UPDATES:
        raise RuntimeError("final checkpoint is not update 6000")
    history = list(final_state["validation_history"])
    if (
        len(history) != MAX_UPDATES // VALIDATION_INTERVAL + 1
        or int(history[0]["completed_updates"]) != 0
        or int(history[-1]["completed_updates"]) != MAX_UPDATES
    ):
        raise RuntimeError("validation history boundary differs")
    independently_selected = select_validation_history(history)
    for key in ("status", "best_update", "best_metrics", "stable_intervals"):
        if independently_selected[key] != selection[key]:
            raise RuntimeError("checkpoint selection differs from history")

    expected_learning_rate = float(arm["learning_rate"])
    initial_group_lrs = _optimizer_learning_rates(initial_payload)
    best_group_lrs = _optimizer_learning_rates(best_payload)
    final_group_lrs = _optimizer_learning_rates(final_payload)
    fixed_learning_rate_verified = bool(
        float(initial_payload["hp"]["lr"]) == expected_learning_rate
        and float(best_payload["hp"]["lr"]) == expected_learning_rate
        and float(final_payload["hp"]["lr"]) == expected_learning_rate
        and all(
            value == expected_learning_rate
            for values in (
                initial_group_lrs,
                best_group_lrs,
                final_group_lrs,
            )
            for value in values
        )
    )
    checkpoint_complete = bool(
        final_payload["git_identity"] == hp["git_identity"]
        and fixed_learning_rate_verified
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": f"{RUN_KIND}_arm",
        "arm": arm,
        "git_identity": hp["git_identity"],
        "from_scratch": True,
        "completed_updates": MAX_UPDATES,
        "status": selection["status"],
        "best_update": selection["best_update"],
        "best_metrics": selection["best_metrics"],
        "final_metrics": {
            name: history[-1][name]
            for name in (
                "normalized_mean_error",
                "normalized_endpoint_error",
                "path_length_ratio",
                "phase_normalized_position_l1",
            )
        },
        "stable_intervals": selection["stable_intervals"],
        "initial_state_sha256": initial_state_sha256,
        "initial_optimizer_param_group_learning_rates": list(
            initial_group_lrs
        ),
        "best_optimizer_param_group_learning_rates": list(best_group_lrs),
        "final_optimizer_param_group_learning_rates": list(final_group_lrs),
        "fixed_learning_rate_no_schedule_verified": (
            fixed_learning_rate_verified
        ),
        "checkpoint_complete": checkpoint_complete,
        "best_checkpoint_audit": best_audit,
        "engineering_passed": bool(
            checkpoint_complete and best_audit["engineering_passed"]
        ),
        "automatic_interval_selection_started": False,
        "automatic_learning_rate_selection_started": False,
        "automatic_continuation_started": False,
        "automatic_second_seed_started": False,
        "automatic_loss_change_started": False,
        "automatic_geometry_change_started": False,
        "formal_full10_started": False,
    }
    _write_json(output / "run_summary.json", result)
    expected_files = {
        "resolved_config.json",
        "metrics.jsonl",
        "initial_checkpoint.pt",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "best_movement_overlay.png",
        "run_summary.json",
        "run.log",
    }
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"scratch arm artifacts differ; actual={sorted(actual_files)}"
        )
    return result


def _save_summary_panel(
    run_root: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 3, figsize=(15, 19))
    for axis, summary in zip(axes.flat, summaries):
        arm = summary["arm"]
        image_path = (
            _arm_output(run_root, arm) / "best_movement_overlay.png"
        )
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
        axis.set_title(
            f"d{arm['digit']} t{arm['movement_intervals']} "
            f"{arm['learning_rate_label']} | {summary['status']} | "
            f"u={summary['best_update']}"
        )
    figure.tight_layout()
    figure.savefig(run_root / "twelve_arm_best_overlays.png", dpi=180)
    plt.close(figure)


def summarize_experiment(
    repository_root: str | Path,
    experiment_config_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_experiment_config(root, experiment_config_path)
    expected_root = _resolve_repository_path(root, config["output"]["directory"])
    output = Path(run_root).resolve()
    if output != expected_root:
        raise ValueError("summary run root differs from the checked-in config")
    summaries = [
        _read_json(_arm_output(output, arm) / "run_summary.json")
        for arm in config["arms"]
    ]
    if [row["arm"]["label"] for row in summaries] != [
        arm["label"] for arm in config["arms"]
    ]:
        raise RuntimeError("scratch arm summaries are incomplete or reordered")
    if any(int(row["completed_updates"]) != MAX_UPDATES for row in summaries):
        raise RuntimeError("one or more scratch arms did not finish at 6000")
    if any(not row["engineering_passed"] for row in summaries):
        raise RuntimeError("one or more scratch arms failed engineering or safety")
    if any(
        not row["fixed_learning_rate_no_schedule_verified"]
        for row in summaries
    ):
        raise RuntimeError("one or more scratch arms changed learning rate")
    initial_hashes = {row["initial_state_sha256"] for row in summaries}
    if len(initial_hashes) != 1:
        raise RuntimeError("twelve arms did not share identical initialization")

    metric_rows = []
    for summary in summaries:
        arm = summary["arm"]
        best = summary["best_metrics"]
        final = summary["final_metrics"]
        metric_rows.append(
            {
                "digit": arm["digit"],
                "movement_intervals": arm["movement_intervals"],
                "is_original_control": (
                    int(arm["movement_intervals"])
                    == SOURCE_INTERVALS[int(arm["digit"])]
                ),
                "learning_rate_label": arm["learning_rate_label"],
                "learning_rate": arm["learning_rate"],
                "status": summary["status"],
                "best_update": summary["best_update"],
                "best_normalized_mean_error": best[
                    "normalized_mean_error"
                ],
                "best_normalized_endpoint_error": best[
                    "normalized_endpoint_error"
                ],
                "best_path_length_ratio": best["path_length_ratio"],
                "final_normalized_mean_error": final[
                    "normalized_mean_error"
                ],
                "final_normalized_endpoint_error": final[
                    "normalized_endpoint_error"
                ],
                "final_path_length_ratio": final["path_length_ratio"],
                "engineering_passed": summary["engineering_passed"],
            }
        )
    _write_csv(output / "twelve_arm_metrics.csv", metric_rows)
    _save_summary_panel(output, summaries)

    lines = [
        "# Protocol3 digit0/digit8 movement-interval and LR scratch sweep",
        "",
        "All twelve arms trained from identical initialization for 6000 updates.",
        "Each arm used one fixed learning rate for the complete run.",
        "",
        "Digit 0 source timing is 170 intervals; time170 is the original control.",
        "Digit 8 source timing is 200 intervals; time200 is the original control.",
        "",
        "| digit | intervals | original control | LR | status | best update | best mean | best endpoint | best path | final mean | final endpoint | final path |",
        "|---:|---:|:---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['digit']} | {row['movement_intervals']} | "
            f"{'yes' if row['is_original_control'] else 'no'} | "
            f"{row['learning_rate']:.4g} | {row['status']} | "
            f"{row['best_update']} | "
            f"{row['best_normalized_mean_error']:.6f} | "
            f"{row['best_normalized_endpoint_error']:.6f} | "
            f"{row['best_path_length_ratio']:.6f} | "
            f"{row['final_normalized_mean_error']:.6f} | "
            f"{row['final_normalized_endpoint_error']:.6f} | "
            f"{row['final_path_length_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "No interval or learning-rate winner was selected automatically.",
            "No continuation, second seed, loss change, geometry change, or formal full10 run was started.",
            "",
        ]
    )
    (output / "REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": RUN_KIND,
        "git_identity": summaries[0]["git_identity"],
        "digits": list(DIGITS),
        "source_movement_intervals": {
            str(key): value for key, value in SOURCE_INTERVALS.items()
        },
        "movement_interval_arms": {
            str(digit): list(MOVEMENT_INTERVALS_BY_DIGIT[digit])
            for digit in DIGITS
        },
        "learning_rate_arms": [
            {"label": label, "learning_rate": learning_rate}
            for label, learning_rate in LEARNING_RATE_ARMS
        ],
        "completed_arms": EXPECTED_ARM_COUNT,
        "completed_updates_per_arm": MAX_UPDATES,
        "from_scratch": True,
        "shared_initial_state_sha256": next(iter(initial_hashes)),
        "fixed_learning_rate_no_schedule_verified": True,
        "engineering_passed": True,
        "automatic_interval_selection": False,
        "automatic_learning_rate_selection": False,
        "automatic_continuation": False,
        "automatic_second_seed": False,
        "automatic_loss_change": False,
        "automatic_geometry_change": False,
        "formal_full10_started": False,
        "arms": summaries,
    }
    _write_json(output / "twelve_arm_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repository-root", required=True)
    prepare.add_argument("--experiment-config", required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--evidence-root", required=True)

    run = subparsers.add_parser("run-arm")
    run.add_argument("--repository-root", required=True)
    run.add_argument("--experiment-config", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--evidence-root", required=True)
    run.add_argument("--arm-label", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--repository-root", required=True)
    summarize.add_argument("--experiment-config", required=True)
    summarize.add_argument("--run-root", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_experiment(
            args.repository_root,
            args.experiment_config,
            args.run_root,
            args.evidence_root,
        )
    elif args.command == "run-arm":
        result = run_arm(
            args.repository_root,
            args.experiment_config,
            args.run_root,
            args.evidence_root,
            args.arm_label,
        )
    else:
        result = summarize_experiment(
            args.repository_root,
            args.experiment_config,
            args.run_root,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
