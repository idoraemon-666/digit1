"""Equal-movement-point learning-rate ablation for Protocol3 digit 8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from digit_writing.protocol3_checkpoint import (
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)
from digit_writing.protocol3_gate2_continuation import (
    inspect_gate2_continuation,
)


ARM_LABELS = (
    "digit8_medium_lr1e3",
    "digit8_medium_lr3e4",
    "digit8_medium_lr1e4",
)
ARM_LEARNING_RATES = (0.001, 0.0003, 0.0001)
SOURCE_CASE = {
    "label": "o3_digit8",
    "digit": 8,
    "direction_index": 0,
    "delay_steps": 50,
}
SOURCE_UPDATES = 6000
TARGET_UPDATES = 7000
VALIDATION_INTERVAL = 100


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def movement_point_error_summary(path: str | Path) -> dict[str, Any]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["movement_index"])
            dx = float(row["actual_x_m"]) - float(row["target_x_m"])
            dy = float(row["actual_y_m"]) - float(row["target_y_m"])
            rows.append((index, math.hypot(dx, dy)))
    if [index for index, _ in rows] != list(range(201)):
        raise ValueError("digit8 movement overlay must contain indices 0 through 200")
    errors = [error for _, error in rows]
    maximum_index, maximum = max(rows, key=lambda row: row[1])
    return {
        "movement_samples": len(errors),
        "all_movement_points_equal_weight_in_training": True,
        "mean_euclidean_error_m": _mean(errors),
        "median_euclidean_error_m": _percentile(errors, 0.5),
        "p95_euclidean_error_m": _percentile(errors, 0.95),
        "maximum_euclidean_error_m": maximum,
        "maximum_error_index": maximum_index,
        "start_error_m": errors[0],
        "endpoint_error_m": errors[-1],
        "first_10_mean_error_m": _mean(errors[:10]),
        "middle_181_mean_error_m": _mean(errors[10:191]),
        "last_10_mean_error_m": _mean(errors[191:]),
    }


def load_ablation_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("digit8 learning-rate ablation requires Protocol3")
    if config.get("run_kind") != "protocol3_digit8_equal_point_lr_ablation":
        raise ValueError("digit8 learning-rate ablation run kind is invalid")
    if config.get("variant") != "digit8_medium_equal_point_lr_from6000_to7000":
        raise ValueError("digit8 learning-rate ablation variant is invalid")
    source = config["source"]
    if source["case_label"] != SOURCE_CASE["label"]:
        raise ValueError("digit8 learning-rate ablation source case is invalid")
    if int(source["completed_updates"]) != SOURCE_UPDATES:
        raise ValueError("digit8 learning-rate ablation source update is invalid")
    for name in (
        "repository_head",
        "checkpoint_sha256",
        "summary_sha256",
        "model_state_sha256",
        "overlay_csv_sha256",
        "overlay_png_sha256",
    ):
        if len(source[name]) != (40 if name == "repository_head" else 64):
            raise ValueError(f"digit8 learning-rate ablation {name} is invalid")
    frozen = config["frozen_case"]
    expected_frozen = {
        "digit": 8,
        "direction_index": 0,
        "delay_steps": 50,
        "scale_multiplier": 2.5,
        "selected_reference_steps": 100,
        "movement_intervals": 200,
        "movement_samples": 201,
        "episode_steps": 301,
        "trajectory_start": "top",
        "initial_direction": "left_down",
        "closed_target": True,
    }
    if frozen != expected_frozen:
        raise ValueError("digit8 trajectory, scale, direction, or timing changed")
    objective = config["objective"]
    expected_objective = {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
        "movement_points_equal_weight": True,
        "start_point_extra_weight": 0.0,
        "endpoint_extra_weight": 0.0,
        "closure_extra_weight": 0.0,
    }
    if objective != expected_objective:
        raise ValueError("digit8 equal-point objective changed")
    optimizer = config["optimizer"]
    if optimizer != {
        "name": "Adam",
        "source_learning_rate": 0.001,
        "restore_optimizer_state": True,
        "override_only_param_group_learning_rate": True,
    }:
        raise ValueError("digit8 learning-rate ablation optimizer identity changed")
    arms = config["arms"]
    if tuple(arm["label"] for arm in arms) != ARM_LABELS:
        raise ValueError("digit8 learning-rate ablation labels changed")
    if tuple(float(arm["learning_rate"]) for arm in arms) != ARM_LEARNING_RATES:
        raise ValueError("digit8 learning-rate ablation rates changed")
    if config["training"] != {
        "target_completed_updates": TARGET_UPDATES,
        "additional_updates": TARGET_UPDATES - SOURCE_UPDATES,
        "validation_interval": VALIDATION_INTERVAL,
        "batch_size": 8,
        "fixed_target_no_early_stop": True,
    }:
        raise ValueError("digit8 learning-rate ablation schedule changed")
    if config["decision"] != {
        "automatic_winner_selection": False,
        "automatic_further_continuation": False,
        "automatic_reference_fallback": False,
        "formal_full10_start": False,
        "qualitative_overlay_review_required": True,
    }:
        raise ValueError("digit8 learning-rate ablation decision boundary changed")
    expected_output = (
        "runs/digit_writing_original_protocol3/scale2p50/ref100/"
        "digit8_equal_point_lr_from6000_to7000"
    )
    if config["output"] != {"directory": expected_output}:
        raise ValueError("digit8 learning-rate ablation output identity changed")
    return config


def _source_config(
    repository_root: str | Path, ablation_config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    root = Path(repository_root).resolve()
    path = (root / ablation_config["source"]["protocol_config"]).resolve()
    if root not in path.parents:
        raise ValueError("source Protocol3 config is outside the repository")
    config = _read_json(path)
    cases = [case for case in config["cases"] if case["label"] == "o3_digit8"]
    if len(cases) != 1 or cases[0] != SOURCE_CASE:
        raise ValueError("source Protocol3 config digit8 identity changed")
    if config["geometry_config"] != (
        "configurations/"
        "digit_writing_original_protocol3_geometry_scale2p50_ref100.json"
    ):
        raise ValueError("source Protocol3 geometry config changed")
    if config["scale_multiplier"] != 2.5:
        raise ValueError("source Protocol3 scale changed")
    if config["selected_reference_steps"] != 100:
        raise ValueError("source Protocol3 reference changed")
    if config["training"] != {
        "batch_size": 8,
        "max_updates": 3000,
        "validation_interval": 100,
    }:
        raise ValueError("source Protocol3 training identity changed")
    expected_position_loss = {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }
    if config["position_loss"] != expected_position_loss:
        raise ValueError("source Protocol3 equal-point loss changed")
    return path, config


def inspect_ablation_source(
    *,
    repository_root: str | Path,
    ablation_config_path: str | Path,
    source_summary_path: str | Path,
    source_checkpoint_path: str | Path,
    source_overlay_csv_path: str | Path,
    source_overlay_png_path: str | Path,
) -> dict[str, Any]:
    config = load_ablation_config(ablation_config_path)
    source = config["source"]
    source_summary_path = Path(source_summary_path).resolve()
    source_checkpoint_path = Path(source_checkpoint_path).resolve()
    source_overlay_csv_path = Path(source_overlay_csv_path).resolve()
    source_overlay_png_path = Path(source_overlay_png_path).resolve()
    expected_hashes = {
        source_summary_path: source["summary_sha256"],
        source_checkpoint_path: source["checkpoint_sha256"],
        source_overlay_csv_path: source["overlay_csv_sha256"],
        source_overlay_png_path: source["overlay_png_sha256"],
    }
    for path, expected in expected_hashes.items():
        if _file_sha256(path) != expected:
            raise ValueError(f"source file hash does not match: {path.name}")
    protocol_config_path, protocol_config = _source_config(
        repository_root, config
    )
    review = inspect_gate2_continuation(
        repository_root=repository_root,
        protocol_config_path=protocol_config_path,
        source_summary_path=source_summary_path,
        resume_checkpoint_path=source_checkpoint_path,
        case_label=source["case_label"],
        expected_source_repository_head=source["repository_head"],
    )
    if int(review["source_completed_updates"]) != SOURCE_UPDATES:
        raise ValueError("source review did not resolve update 6000")
    checkpoint = torch.load(
        source_checkpoint_path, map_location="cpu", weights_only=False
    )
    restore = validate_protocol3_resume_checkpoint(checkpoint, protocol_config)
    model_hash = state_dict_sha256(checkpoint["agent_state_dict"])
    if model_hash != source["model_state_sha256"]:
        raise ValueError("source model state hash does not match")
    source_learning_rate = float(config["optimizer"]["source_learning_rate"])
    param_group_lrs = tuple(
        float(group["lr"])
        for group in checkpoint["optimizer_state_dict"]["param_groups"]
    )
    if not param_group_lrs or any(
        value != source_learning_rate for value in param_group_lrs
    ):
        raise ValueError("source Adam param-group learning rate does not match")
    history = list(restore["validation_history"])
    if [int(row["completed_updates"]) for row in history] != list(
        range(0, SOURCE_UPDATES + 1, VALIDATION_INTERVAL)
    ):
        raise ValueError("source validation history is incomplete")
    return {
        "protocol": "digit_writing_original_protocol3",
        "run_kind": "protocol3_digit8_equal_point_lr_ablation_source_review",
        "case": SOURCE_CASE,
        "source_completed_updates": SOURCE_UPDATES,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_summary_sha256": source["summary_sha256"],
        "source_model_state_sha256": model_hash,
        "source_repository_head": source["repository_head"],
        "current_repository_head": review["current_repository_head"],
        "source_optimizer_param_group_learning_rates": list(param_group_lrs),
        "source_validation_points": len(history),
        "source_validation_history_sha256": _canonical_json_sha256(history),
        "source_gate2_review": review,
        "source_movement_point_errors": movement_point_error_summary(
            source_overlay_csv_path
        ),
        "movement_points_equal_weight_in_training": True,
        "extra_start_endpoint_or_closure_weight": False,
        "eligible_for_three_arm_lr_ablation": True,
        "formal_gate2_continuation": False,
    }


def prepare_ablation(
    *,
    repository_root: str | Path,
    ablation_config_path: str | Path,
    source_summary_path: str | Path,
    source_checkpoint_path: str | Path,
    source_overlay_csv_path: str | Path,
    source_overlay_png_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_ablation_config(ablation_config_path)
    output = Path(run_root).resolve()
    expected = (root / config["output"]["directory"]).resolve()
    if output != expected:
        raise ValueError("digit8 learning-rate ablation run root is invalid")
    if output.exists():
        raise FileExistsError(f"ablation run root already exists: {output}")
    review = inspect_ablation_source(
        repository_root=root,
        ablation_config_path=ablation_config_path,
        source_summary_path=source_summary_path,
        source_checkpoint_path=source_checkpoint_path,
        source_overlay_csv_path=source_overlay_csv_path,
        source_overlay_png_path=source_overlay_png_path,
    )
    from digit_writing.geometry import load_geometry_config
    from digit_writing.geometry_audit import build_workspace_audit

    _, source_config = _source_config(root, config)
    workspace_audit = build_workspace_audit(
        load_geometry_config(source_config["geometry_config"])
    )
    output.mkdir(parents=True)
    _write_json(output / "ablation_config.json", config)
    _write_json(output / "source_review.json", review)
    _write_json(output / "workspace_audit.json", workspace_audit)
    source_artifacts = output / "source_o3_digit8_at6000"
    source_artifacts.mkdir()
    shutil.copy2(source_summary_path, source_artifacts / "gate2_continuation_summary.json")
    shutil.copy2(source_overlay_csv_path, source_artifacts / "movement_overlay.csv")
    shutil.copy2(source_overlay_png_path, source_artifacts / "movement_overlay.png")
    _write_json(
        source_artifacts / "movement_point_error_summary.json",
        movement_point_error_summary(source_overlay_csv_path),
    )
    return review


def _arm_from_config(
    config: Mapping[str, Any], arm_label: str
) -> dict[str, Any]:
    matches = [arm for arm in config["arms"] if arm["label"] == arm_label]
    if len(matches) != 1:
        raise ValueError(f"unknown digit8 learning-rate arm: {arm_label}")
    return dict(matches[0])


def _recent_mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    if len(rows) != 3:
        raise ValueError("recent comparison requires exactly three validation rows")
    return _mean([float(row[name]) for row in rows])


def run_ablation_arm(
    *,
    repository_root: str | Path,
    ablation_config_path: str | Path,
    source_summary_path: str | Path,
    source_checkpoint_path: str | Path,
    source_overlay_csv_path: str | Path,
    source_overlay_png_path: str | Path,
    run_root: str | Path,
    arm_label: str,
    manual_ablation_authorized: bool,
) -> dict[str, Any]:
    if not manual_ablation_authorized:
        raise PermissionError("digit8 learning-rate ablation requires manual approval")
    root = Path(repository_root).resolve()
    config = load_ablation_config(ablation_config_path)
    arm = _arm_from_config(config, arm_label)
    output_root = Path(run_root).resolve()
    if output_root != (root / config["output"]["directory"]).resolve():
        raise ValueError("digit8 learning-rate ablation run root is invalid")
    if not (output_root / "source_review.json").is_file():
        raise ValueError("digit8 learning-rate ablation was not prepared")
    output = output_root / arm_label
    if output.exists():
        raise FileExistsError(f"digit8 learning-rate arm already exists: {output}")
    review = inspect_ablation_source(
        repository_root=root,
        ablation_config_path=ablation_config_path,
        source_summary_path=source_summary_path,
        source_checkpoint_path=source_checkpoint_path,
        source_overlay_csv_path=source_overlay_csv_path,
        source_overlay_png_path=source_overlay_png_path,
    )
    _, protocol_config = _source_config(root, config)
    from digit_writing.protocol3_gate2 import audit_gate2_case
    from train import (
        DIGIT_ENV_CLASSES,
        PROTOCOL3_DELAYS,
        _base_hp_from_config,
        _digit_env_dict,
        _validate_protocol3_gate2_config,
        load_digit_policy_checkpoint,
        train_subsets_base_model,
    )

    _validate_protocol3_gate2_config(protocol_config)
    workspace_audit = _read_json(output_root / "workspace_audit.json")
    hp = _base_hp_from_config(protocol_config)
    hp.update(
        {
            "variant": SOURCE_CASE["label"],
            "condition_schedule": "protocol3_gate2_single_condition",
            "gate2_direction_index": SOURCE_CASE["direction_index"],
            "gate2_delay_index": PROTOCOL3_DELAYS.index(
                SOURCE_CASE["delay_steps"]
            ),
            "stop_after_updates": TARGET_UPDATES,
            "resume_source_repository_head": config["source"]["repository_head"],
            "resume_source_checkpoint_sha256": config["source"][
                "checkpoint_sha256"
            ],
            "experimental_run_kind": config["run_kind"],
            "experimental_lr_ablation_arm": arm_label,
            "experimental_source_learning_rate": config["optimizer"][
                "source_learning_rate"
            ],
            "experimental_resume_learning_rate": arm["learning_rate"],
            "experimental_disable_gate2_early_stopping": True,
        }
    )
    training = train_subsets_base_model(
        str(output),
        protocol_config["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((8,)),
        resume_checkpoint=str(Path(source_checkpoint_path).resolve()),
        target_completed_updates=TARGET_UPDATES,
        manual_resume_authorized=True,
        expected_resume_repository_head=config["source"]["repository_head"],
    )
    final_checkpoint = output / protocol_config["output"][
        "final_checkpoint"
    ]
    policy, checkpoint = load_digit_policy_checkpoint(
        final_checkpoint, expected_variant=SOURCE_CASE["label"]
    )
    restore = validate_protocol3_resume_checkpoint(checkpoint, protocol_config)
    if int(restore["completed_updates"]) != TARGET_UPDATES:
        raise ValueError("digit8 learning-rate arm did not reach update 7000")
    if checkpoint["hp"].get("experimental_lr_ablation_arm") != arm_label:
        raise ValueError("digit8 learning-rate checkpoint arm identity is missing")
    checkpoint_identity_complete = bool(
        checkpoint["git_identity"] == hp["git_identity"]
        and checkpoint["hp"].get("git_identity") == hp["git_identity"]
    )
    final_group_lrs = tuple(
        float(group["lr"])
        for group in checkpoint["optimizer_state_dict"]["param_groups"]
    )
    if not final_group_lrs or any(
        value != float(arm["learning_rate"]) for value in final_group_lrs
    ):
        raise ValueError("digit8 learning-rate arm effective Adam rate is wrong")
    audit = audit_gate2_case(
        policy,
        hp,
        DIGIT_ENV_CLASSES[8],
        SOURCE_CASE,
        output,
        workspace_audit,
    )
    history = list(training["validation_history"])
    source_history = [
        row for row in history if int(row["completed_updates"]) <= SOURCE_UPDATES
    ]
    new_history = [
        row for row in history if int(row["completed_updates"]) > SOURCE_UPDATES
    ]
    if len(source_history) != 61 or [
        int(row["completed_updates"]) for row in new_history
    ] != list(range(6100, 7001, 100)):
        raise ValueError("digit8 learning-rate arm validation history is incomplete")
    result = {
        "protocol": "digit_writing_original_protocol3",
        "run_kind": config["run_kind"],
        "variant": config["variant"],
        "arm": arm,
        "case": SOURCE_CASE,
        "source_completed_updates": SOURCE_UPDATES,
        "target_completed_updates": TARGET_UPDATES,
        "completed_updates": int(training["updates"]),
        "source_checkpoint_sha256": review["source_checkpoint_sha256"],
        "source_model_state_sha256": review["source_model_state_sha256"],
        "final_checkpoint_sha256": _file_sha256(final_checkpoint),
        "checkpoint_identity_complete": checkpoint_identity_complete,
        "effective_optimizer_param_group_learning_rates": list(final_group_lrs),
        "source_validation_history_unchanged": _canonical_json_sha256(
            source_history
        )
        == review["source_validation_history_sha256"],
        "new_validation_history": new_history,
        "recent_3_validation": new_history[-3:],
        "training": training,
        "candidate_audit": audit,
        "movement_point_errors": movement_point_error_summary(
            output / "movement_overlay.csv"
        ),
        "movement_points_equal_weight_in_training": True,
        "extra_start_endpoint_or_closure_weight": False,
        "fixed_target_no_early_stop": True,
        "engineering_passed": bool(
            audit["engineering_passed"]
            and checkpoint_identity_complete
            and int(training["updates"]) == TARGET_UPDATES
            and _canonical_json_sha256(source_history)
            == review["source_validation_history_sha256"]
        ),
        "behavior_passed_at_final_audit": bool(audit["behavior_passed"]),
        "formal_gate2_continuation": False,
        "automatic_winner_selection": False,
        "automatic_further_continuation": False,
        "automatic_reference_fallback": False,
        "formal_full10_started": False,
        "qualitative_overlay_review_required": True,
        "passed": False,
    }
    _write_json(output / "lr_ablation_arm_summary.json", result)
    return result


def summarize_ablation(
    *,
    repository_root: str | Path,
    ablation_config_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_ablation_config(ablation_config_path)
    output = Path(run_root).resolve()
    if output != (root / config["output"]["directory"]).resolve():
        raise ValueError("digit8 learning-rate ablation run root is invalid")
    source_review = _read_json(output / "source_review.json")
    results = [
        _read_json(output / label / "lr_ablation_arm_summary.json")
        for label in ARM_LABELS
    ]
    if [result["arm"]["label"] for result in results] != list(ARM_LABELS):
        raise ValueError("digit8 learning-rate arm summaries are out of order")
    if any(
        result["source_checkpoint_sha256"]
        != source_review["source_checkpoint_sha256"]
        for result in results
    ):
        raise ValueError("digit8 learning-rate arms did not share one source")
    if any(int(result["completed_updates"]) != TARGET_UPDATES for result in results):
        raise ValueError("digit8 learning-rate arm did not complete update 7000")
    engineering_passed = all(result["engineering_passed"] for result in results)
    comparisons = []
    control = results[0]
    control_recent = control["recent_3_validation"]
    source_gate2_metrics = source_review["source_gate2_review"]["metrics"]
    source_point_metrics = source_review["source_movement_point_errors"]
    for result in results:
        final_metrics = result["candidate_audit"]["movement_metrics"]
        point_metrics = result["movement_point_errors"]
        recent = result["recent_3_validation"]
        comparisons.append(
            {
                "label": result["arm"]["label"],
                "learning_rate": result["arm"]["learning_rate"],
                "normalized_mean_error": final_metrics[
                    "normalized_mean_error"
                ],
                "normalized_endpoint_error": final_metrics[
                    "normalized_endpoint_error"
                ],
                "path_length_ratio": final_metrics["path_length_ratio"],
                "p95_euclidean_error_m": point_metrics[
                    "p95_euclidean_error_m"
                ],
                "maximum_euclidean_error_m": point_metrics[
                    "maximum_euclidean_error_m"
                ],
                "recent_3_mean_normalized_mean_error": _recent_mean(
                    recent, "normalized_mean_error"
                ),
                "recent_3_mean_normalized_endpoint_error": _recent_mean(
                    recent, "normalized_endpoint_error"
                ),
                "recent_3_mean_path_length_ratio": _recent_mean(
                    recent, "path_length_ratio"
                ),
                "recent_3_mean_error_relative_to_source": (
                    _recent_mean(recent, "normalized_mean_error")
                    / float(
                        source_gate2_metrics["normalized_mean_error"]["last"]
                    )
                    - 1.0
                ),
                "recent_3_mean_error_relative_to_control": (
                    _recent_mean(recent, "normalized_mean_error")
                    / _recent_mean(control_recent, "normalized_mean_error")
                    - 1.0
                ),
                "p95_error_relative_to_source": (
                    point_metrics["p95_euclidean_error_m"]
                    / source_point_metrics["p95_euclidean_error_m"]
                    - 1.0
                ),
                "maximum_error_relative_to_source": (
                    point_metrics["maximum_euclidean_error_m"]
                    / source_point_metrics["maximum_euclidean_error_m"]
                    - 1.0
                ),
                "mean_error_minus_control": final_metrics[
                    "normalized_mean_error"
                ]
                - control["candidate_audit"]["movement_metrics"][
                    "normalized_mean_error"
                ],
                "automatic_selection": False,
            }
        )
    summary = {
        "protocol": "digit_writing_original_protocol3",
        "run_kind": config["run_kind"],
        "variant": config["variant"],
        "source_review": source_review,
        "arms": results,
        "comparisons": comparisons,
        "completed_updates": TARGET_UPDATES,
        "movement_points_equal_weight_in_training": True,
        "extra_start_endpoint_or_closure_weight": False,
        "engineering_passed": engineering_passed,
        "classification": (
            "manual_equal_point_review_required"
            if engineering_passed
            else "engineering_or_safety_failure"
        ),
        "automatic_winner_selection": False,
        "automatic_further_continuation": False,
        "automatic_reference_fallback": False,
        "formal_full10_started": False,
        "passed": False,
    }
    _write_json(output / "digit8_equal_point_lr_ablation_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "prepare", "run-arm", "summarize"))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--ablation-config", required=True)
    parser.add_argument("--source-summary")
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--source-overlay-csv")
    parser.add_argument("--source-overlay-png")
    parser.add_argument("--run-root")
    parser.add_argument("--arm-label")
    parser.add_argument("--manual-ablation-authorized", action="store_true")
    return parser


def _required(arguments: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(arguments, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")


def main() -> None:
    arguments = _parser().parse_args()
    source_names = (
        "source_summary",
        "source_checkpoint",
        "source_overlay_csv",
        "source_overlay_png",
    )
    common = {
        "repository_root": arguments.repository_root,
        "ablation_config_path": arguments.ablation_config,
    }
    if arguments.mode == "inspect":
        _required(arguments, *source_names)
        result = inspect_ablation_source(
            **common,
            source_summary_path=arguments.source_summary,
            source_checkpoint_path=arguments.source_checkpoint,
            source_overlay_csv_path=arguments.source_overlay_csv,
            source_overlay_png_path=arguments.source_overlay_png,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if arguments.mode == "prepare":
        _required(arguments, *source_names, "run_root")
        prepare_ablation(
            **common,
            source_summary_path=arguments.source_summary,
            source_checkpoint_path=arguments.source_checkpoint,
            source_overlay_csv_path=arguments.source_overlay_csv,
            source_overlay_png_path=arguments.source_overlay_png,
            run_root=arguments.run_root,
        )
        print("DIGIT8_LR_ABLATION_PREPARED=1")
        return
    if arguments.mode == "run-arm":
        _required(arguments, *source_names, "run_root", "arm_label")
        result = run_ablation_arm(
            **common,
            source_summary_path=arguments.source_summary,
            source_checkpoint_path=arguments.source_checkpoint,
            source_overlay_csv_path=arguments.source_overlay_csv,
            source_overlay_png_path=arguments.source_overlay_png,
            run_root=arguments.run_root,
            arm_label=arguments.arm_label,
            manual_ablation_authorized=arguments.manual_ablation_authorized,
        )
        print(f"DIGIT8_LR_ARM={result['arm']['label']}")
        print(f"COMPLETED_UPDATES={result['completed_updates']}")
        print(f"ENGINEERING_PASSED={int(result['engineering_passed'])}")
        return
    _required(arguments, "run_root")
    result = summarize_ablation(
        **common,
        run_root=arguments.run_root,
    )
    print(f"DIGIT8_LR_ABLATION_CLASSIFICATION={result['classification']}")
    print(f"COMPLETED_UPDATES={result['completed_updates']}")
    print(f"ENGINEERING_PASSED={int(result['engineering_passed'])}")
    print("AUTOMATIC_FURTHER_CONTINUATION_STARTED=0")
    print("AUTOMATIC_REFERENCE_FALLBACK_STARTED=0")
    print("FORMAL_FULL10_STARTED=0")


if __name__ == "__main__":
    main()
