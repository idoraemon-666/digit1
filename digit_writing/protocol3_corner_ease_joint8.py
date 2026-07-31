"""Frozen eight-digit shared-RNN experiment for Protocol3 corner-ease v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


JOINT8_SCHEDULE = "protocol3_corner_ease_joint8_balanced_blocks_v1"
JOINT8_DIGITS = (1, 2, 3, 4, 5, 6, 7, 9)
JOINT8_EXCLUDED_DIGITS = (0, 8)
JOINT8_DIRECTION_INDEX = 0
JOINT8_DELAY_STEPS = 50
JOINT8_BATCH_SIZE = 8
JOINT8_UPDATES_PER_DIGIT = 8_000
JOINT8_TOTAL_UPDATES = len(JOINT8_DIGITS) * JOINT8_UPDATES_PER_DIGIT
JOINT8_VALIDATION_INTERVAL = 800
JOINT8_LR_SWITCH_UPDATE = len(JOINT8_DIGITS) * 6_000
JOINT8_INITIAL_LR = 0.001
JOINT8_FINAL_LR = 0.0003
JOINT8_EXPECTED_INTERVALS = {
    1: 60,
    2: 210,
    3: 180,
    4: 200,
    5: 230,
    6: 200,
    7: 130,
    9: 200,
}
PASS_THRESHOLDS = {
    "mean_max": 0.08,
    "endpoint_max": 0.05,
    "path_ratio_min": 0.85,
    "path_ratio_max": 1.15,
}


def _write_json(path: str | Path, value: Any) -> None:
    from digit_writing.final_protocol_audit import _json_safe

    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    candidate = (repository_root / value).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("configured path escapes the repository") from error
    return candidate


def joint8_block_order(block_index: int, seed: int) -> tuple[int, ...]:
    """Return a version-independent seeded permutation for one eight-step block."""
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise TypeError("block_index must be an integer")
    if block_index < 0:
        raise ValueError("block_index must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    def rank(digit: int) -> bytes:
        identity = f"protocol3-joint8-v1:{seed}:{block_index}:{digit}"
        return hashlib.sha256(identity.encode("ascii")).digest()

    return tuple(sorted(JOINT8_DIGITS, key=rank))


def joint8_digit_for_update(completed_updates: int, seed: int) -> int:
    """Select the digit for the next optimizer step (zero-based cursor)."""
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise TypeError("completed_updates must be an integer")
    if not 0 <= completed_updates < JOINT8_TOTAL_UPDATES:
        raise ValueError("completed_updates is outside the frozen joint8 run")
    block_index, position = divmod(completed_updates, len(JOINT8_DIGITS))
    return joint8_block_order(block_index, seed)[position]


def joint8_schedule_sha256(seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(bytes(joint8_digit_for_update(update, seed) for update in range(JOINT8_TOTAL_UPDATES)))
    return digest.hexdigest()


def joint8_learning_rate(completed_updates: int) -> float:
    """Learning rate for the next optimizer step at a zero-based cursor."""
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise TypeError("completed_updates must be an integer")
    if not 0 <= completed_updates < JOINT8_TOTAL_UPDATES:
        raise ValueError("completed_updates is outside the frozen joint8 run")
    return (
        JOINT8_INITIAL_LR
        if completed_updates < JOINT8_LR_SWITCH_UPDATE
        else JOINT8_FINAL_LR
    )


def validate_joint8_condition_counts(
    counts: Mapping[str, Any], completed_updates: int
) -> None:
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise TypeError("completed_updates must be an integer")
    if not 0 <= completed_updates <= JOINT8_TOTAL_UPDATES:
        raise ValueError("completed_updates is outside the frozen joint8 run")
    if completed_updates % len(JOINT8_DIGITS) != 0:
        raise ValueError("joint8 checkpoint must end on an eight-update block boundary")
    expected_per_digit = completed_updates // len(JOINT8_DIGITS)
    expected_digits = [
        expected_per_digit if digit in JOINT8_DIGITS else 0
        for digit in range(10)
    ]
    if counts.get("digit_update_counts") != expected_digits:
        raise ValueError("joint8 digit counts differ from exact block balance")
    expected_delays = {"25": 0, "50": completed_updates, "75": 0}
    if counts.get("delay_update_counts") != expected_delays:
        raise ValueError("joint8 delay counts differ from fixed delay 50")
    joint = counts.get("digit_delay_update_counts")
    expected_joint = [
        [0, expected_digits[digit], 0]
        for digit in range(10)
    ]
    if joint != expected_joint:
        raise ValueError("joint8 digit-delay counts differ from the frozen design")


def digit_metrics_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics["normalized_mean_error"]) <= PASS_THRESHOLDS["mean_max"]
        and float(metrics["normalized_endpoint_error"])
        <= PASS_THRESHOLDS["endpoint_max"]
        and PASS_THRESHOLDS["path_ratio_min"]
        <= float(metrics["path_length_ratio"])
        <= PASS_THRESHOLDS["path_ratio_max"]
    )


def digit_relative_threshold_violation(metrics: Mapping[str, Any]) -> float:
    mean_error = float(metrics["normalized_mean_error"])
    endpoint_error = float(metrics["normalized_endpoint_error"])
    path_ratio = float(metrics["path_length_ratio"])
    path_low = PASS_THRESHOLDS["path_ratio_min"]
    path_high = PASS_THRESHOLDS["path_ratio_max"]
    return max(
        0.0,
        mean_error / PASS_THRESHOLDS["mean_max"] - 1.0,
        endpoint_error / PASS_THRESHOLDS["endpoint_max"] - 1.0,
        (path_low - path_ratio) / path_low,
        (path_ratio - path_high) / path_high,
    )


def joint8_selection_record(row: Mapping[str, Any]) -> dict[str, Any]:
    per_digit = row.get("per_digit")
    if not isinstance(per_digit, Mapping) or set(per_digit) != {
        str(digit) for digit in JOINT8_DIGITS
    }:
        raise ValueError("joint8 validation row requires all eight per-digit metrics")
    pass_count = sum(
        digit_metrics_pass(per_digit[str(digit)]) for digit in JOINT8_DIGITS
    )
    return {
        "completed_updates": int(row["completed_updates"]),
        "passing_digit_count": int(pass_count),
        "worst_relative_threshold_violation": max(
            digit_relative_threshold_violation(per_digit[str(digit)])
            for digit in JOINT8_DIGITS
        ),
        "macro_normalized_mean_error": float(row["normalized_mean_error"]),
        "all_digits_passed": bool(pass_count == len(JOINT8_DIGITS)),
    }


def joint8_selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -float(record["passing_digit_count"]),
        float(record["worst_relative_threshold_violation"]),
        float(record["macro_normalized_mean_error"]),
        float(record["completed_updates"]),
    )


def select_joint8_validation_history(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not history:
        raise ValueError("joint8 selection requires validation history")
    records = [joint8_selection_record(row) for row in history]
    stable_intervals: list[dict[str, int]] = []
    stable_records: list[dict[str, Any]] = []
    passing_records: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for record in records:
        if record["all_digits_passed"]:
            passing_records.append(record)
            current.append(record)
            if len(current) == 3:
                stable_intervals.append(
                    {
                        "start_update": int(current[0]["completed_updates"]),
                        "end_update": int(current[-1]["completed_updates"]),
                    }
                )
                stable_records.extend(current)
            elif len(current) > 3:
                stable_intervals[-1]["end_update"] = int(
                    record["completed_updates"]
                )
                stable_records.append(record)
        else:
            current = []

    if stable_records:
        status = "STABLE_PASS"
        eligible = stable_records
    elif passing_records:
        status = "PASS_UNSTABLE"
        eligible = passing_records
    else:
        status = "FAIL"
        eligible = records
    selected = min(eligible, key=joint8_selection_key)
    return {
        "status": status,
        "best_update": int(selected["completed_updates"]),
        "best_record": dict(selected),
        "stable_intervals": stable_intervals,
        "selection_rule": {
            "stable_consecutive_evaluations": 3,
            "ranking": [
                "maximize_passing_digit_count",
                "minimize_worst_relative_threshold_violation",
                "minimize_macro_normalized_mean_error",
                "earlier_update",
            ],
        },
    }


def validate_config(
    repository_root: str | Path, config: Mapping[str, Any]
) -> None:
    from digit_writing.corner_time_reparameterization import (
        BASE_WINDOW_INTERVALS,
        EXTRA_INTERVALS_PER_SIDE,
        RESAMPLED_WINDOW_INTERVALS,
        TURN_THRESHOLD_DEG,
        V3_STEP_WEIGHT_DENOMINATOR,
        V3_STEP_WEIGHT_NUMERATORS,
    )
    from digit_writing.geometry import CORNER_EASE_TIMING, build_digit_trajectory, load_geometry_config

    root = Path(repository_root).resolve()
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("joint8 requires Protocol3")
    if config.get("run_kind") != "protocol3_corner_ease_joint8":
        raise ValueError("joint8 run_kind differs")
    if config.get("variant") != "corner_ease_v3_joint8_excluding_digit0_digit8_seed42":
        raise ValueError("joint8 variant differs")
    if config.get("seed") != 42 or config.get("validation_seed") != 1042:
        raise ValueError("joint8 seeds differ")
    if config.get("device") != "cpu":
        raise ValueError("joint8 is CPU-only")
    if config.get("scale_multiplier") != 2.5:
        raise ValueError("joint8 requires scale2p50")
    if config.get("timing_mode") != CORNER_EASE_TIMING:
        raise ValueError("joint8 requires corner-ease v3 timing")
    if config.get("selected_reference_steps") != 100:
        raise ValueError("joint8 requires ref100")
    if tuple(config.get("train_digits", ())) != JOINT8_DIGITS:
        raise ValueError("joint8 included digits differ")
    if tuple(config.get("excluded_digits", ())) != JOINT8_EXCLUDED_DIGITS:
        raise ValueError("joint8 excluded digits differ")
    if config.get("direction_index") != JOINT8_DIRECTION_INDEX:
        raise ValueError("joint8 direction must remain zero")
    if config.get("delay_steps") != JOINT8_DELAY_STEPS:
        raise ValueError("joint8 delay must remain 50")
    if config.get("condition_schedule") != JOINT8_SCHEDULE:
        raise ValueError("joint8 condition schedule differs")
    if config.get("digit_order_within_block") != "sha256_ranked_permutation_v1":
        raise ValueError("joint8 block permutation differs")
    if config.get("digit_movement_intervals") != {
        str(digit): intervals
        for digit, intervals in JOINT8_EXPECTED_INTERVALS.items()
    }:
        raise ValueError("joint8 movement intervals differ")

    expected_model = {
        "network": "rnn",
        "input_size": 28,
        "hidden_size": 256,
        "output_size": 6,
        "activation": "softplus",
        "recurrent_noise_std": 0.1,
        "input_noise_std": 0.01,
        "constrained": False,
        "rnn_dt_ms": 10,
        "rnn_tau_ms": 20,
        "batch_first": True,
    }
    if config.get("model") != expected_model:
        raise ValueError("joint8 model differs")
    if config.get("training") != {
        "batch_size": JOINT8_BATCH_SIZE,
        "updates_per_digit": JOINT8_UPDATES_PER_DIGIT,
        "max_updates": JOINT8_TOTAL_UPDATES,
        "validation_interval": JOINT8_VALIDATION_INTERVAL,
        "samples_per_digit": JOINT8_UPDATES_PER_DIGIT * JOINT8_BATCH_SIZE,
        "total_training_samples": JOINT8_TOTAL_UPDATES * JOINT8_BATCH_SIZE,
    }:
        raise ValueError("joint8 training boundary differs")
    if config.get("optimizer") != {
        "name": "Adam",
        "learning_rate": JOINT8_INITIAL_LR,
        "grad_clip_norm": 1.0,
        "schedule": [
            {
                "per_digit_exposure_start": 1,
                "per_digit_exposure_end": 6000,
                "global_update_start": 1,
                "global_update_end": JOINT8_LR_SWITCH_UPDATE,
                "learning_rate": JOINT8_INITIAL_LR,
            },
            {
                "per_digit_exposure_start": 6001,
                "per_digit_exposure_end": JOINT8_UPDATES_PER_DIGIT,
                "global_update_start": JOINT8_LR_SWITCH_UPDATE + 1,
                "global_update_end": JOINT8_TOTAL_UPDATES,
                "learning_rate": JOINT8_FINAL_LR,
            },
        ],
        "preserve_adam_state_at_switch": True,
    }:
        raise ValueError("joint8 optimizer schedule differs")
    if config.get("position_loss") != {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }:
        raise ValueError("joint8 position loss differs")
    if config.get("regularization") != {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }:
        raise ValueError("joint8 regularization differs")
    if config.get("checkpoint_selection") != {
        **PASS_THRESHOLDS,
        "stable_consecutive_evaluations": 3,
        "ranking": [
            "maximize_passing_digit_count",
            "minimize_worst_relative_threshold_violation",
            "minimize_macro_normalized_mean_error",
            "earlier_update",
        ],
    }:
        raise ValueError("joint8 checkpoint selection differs")
    if config.get("corner_ease") != {
        "mapping_version": "fixed_minimax_rational_step_table_v3",
        "turn_angle_deg": TURN_THRESHOLD_DEG,
        "base_window_intervals": BASE_WINDOW_INTERVALS,
        "extra_intervals_per_side": EXTRA_INTERVALS_PER_SIDE,
        "resampled_window_intervals": RESAMPLED_WINDOW_INTERVALS,
        "step_weight_denominator": V3_STEP_WEIGHT_DENOMINATOR,
        "step_weight_numerators": list(V3_STEP_WEIGHT_NUMERATORS),
        "incoming_speed_ratio_max": 0.1,
        "outgoing_speed_ratio_max": 0.1,
        "automatic_extension": False,
        "automatic_second_seed": False,
        "automatic_full10": False,
    }:
        raise ValueError("joint8 corner-ease identity differs")
    geometry_identity = config.get("geometry_identity", {})
    if (
        geometry_identity.get("version") != "source-derived-temporal-v1"
        or geometry_identity.get("legacy_canonical_template_sha256_redefined") is not False
    ):
        raise ValueError("joint8 geometry identity differs")
    if config.get("output") != {
        "directory": (
            "runs/digit_writing_original_protocol3/scale2p50/ref100/"
            "corner_ease_v3/joint8_excluding_digit0_digit8/seed42"
        ),
        "initial_checkpoint": "initial_checkpoint.pt",
        "best_checkpoint": "best_checkpoint.pt",
        "final_checkpoint": "final_checkpoint.pt",
    }:
        raise ValueError("joint8 output identity differs")

    source = _read_json(
        _resolve_repository_path(root, str(config["source_gate2_config"]))
    )
    if (
        source.get("protocol") != config["protocol"]
        or source.get("run_kind") != "protocol3_gate2"
        or source.get("selected_reference_steps") != 100
        or source.get("scale_multiplier") != 2.5
    ):
        raise ValueError("joint8 source Gate2 identity differs")
    for key in (
        "seed",
        "validation_seed",
        "device",
        "model",
        "position_loss",
        "regularization",
    ):
        if config.get(key) != source.get(key):
            raise ValueError(f"joint8 differs from source Gate2: {key}")
    source_optimizer = source.get("optimizer", {})
    if (
        source_optimizer.get("name") != "Adam"
        or source_optimizer.get("learning_rate") != JOINT8_INITIAL_LR
        or source_optimizer.get("grad_clip_norm") != 1.0
    ):
        raise ValueError("joint8 source optimizer identity differs")

    geometry_path = _resolve_repository_path(root, str(config["geometry_config"]))
    geometry = load_geometry_config(geometry_path)
    for digit, expected_intervals in JOINT8_EXPECTED_INTERVALS.items():
        trajectory = build_digit_trajectory(digit, geometry, 100)
        if trajectory.movement_intervals != expected_intervals:
            raise ValueError(f"digit {digit} movement intervals differ")


def hp_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    from train import PROTOCOL3_DELAYS, _base_hp_from_config

    hp = _base_hp_from_config(config)
    hp.update(
        {
            "condition_schedule": JOINT8_SCHEDULE,
            "gate2_direction_index": JOINT8_DIRECTION_INDEX,
            "gate2_delay_index": PROTOCOL3_DELAYS.index(JOINT8_DELAY_STEPS),
            "joint8_digits": list(JOINT8_DIGITS),
            "stop_after_updates": JOINT8_TOTAL_UPDATES,
            "artifact_profile": "protocol3_joint8_minimal_v1",
            "env_kwargs": {"geometry_config_path": config["geometry_config"]},
        }
    )
    return hp


def prepare_experiment(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    evidence_directory: str | Path,
) -> dict[str, Any]:
    from digit_writing.geometry import load_geometry_config
    from digit_writing.geometry_audit import build_workspace_audit

    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    validate_config(root, config)
    evidence = Path(evidence_directory).resolve()
    if evidence.exists():
        raise FileExistsError(f"joint8 evidence directory already exists: {evidence}")
    evidence.mkdir(parents=True)
    geometry = load_geometry_config(
        _resolve_repository_path(root, str(config["geometry_config"]))
    )
    workspace_audit = build_workspace_audit(geometry)
    if not workspace_audit["passed"]:
        raise RuntimeError("joint8 workspace audit failed")
    summary = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_corner_ease_joint8_prepare",
        "train_digits": list(JOINT8_DIGITS),
        "excluded_digits": list(JOINT8_EXCLUDED_DIGITS),
        "total_optimizer_steps": JOINT8_TOTAL_UPDATES,
        "schedule_sha256": joint8_schedule_sha256(int(config["seed"])),
        "workspace_passed": True,
        "formal_training_started": False,
    }
    _write_json(evidence / "workspace_audit.json", workspace_audit)
    _write_json(evidence / "prepare_summary.json", summary)
    return summary


def run_experiment(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    run_directory: str | Path,
    evidence_directory: str | Path,
) -> dict[str, Any]:
    from digit_writing.protocol3_checkpoint import validate_protocol3_resume_checkpoint
    from digit_writing.protocol3_corner_ease import (
        _audit_best_checkpoint,
        select_validation_history,
    )
    from train import (
        _digit_env_dict,
        load_digit_policy_checkpoint,
        train_subsets_base_model,
    )

    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    validate_config(root, config)
    expected_run = _resolve_repository_path(root, config["output"]["directory"])
    run = Path(run_directory).resolve()
    if run != expected_run:
        raise ValueError("joint8 run directory differs from checked-in config")
    if not run.is_dir():
        raise FileNotFoundError("server runner must create the joint8 log directory")
    unexpected = {path.name for path in run.iterdir()} - {"run.log"}
    if unexpected:
        raise FileExistsError(f"joint8 run directory is not empty: {sorted(unexpected)}")
    evidence = Path(evidence_directory).resolve()
    workspace_audit = _read_json(evidence / "workspace_audit.json")
    if not workspace_audit.get("passed"):
        raise RuntimeError("joint8 prepared workspace audit did not pass")

    _write_json(run / "resolved_config.json", config)
    hp = hp_from_config(config)
    training_summary = train_subsets_base_model(
        str(run),
        config["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict(JOINT8_DIGITS),
    )
    if int(training_summary["updates"]) != JOINT8_TOTAL_UPDATES:
        raise RuntimeError("joint8 training did not reach 64000 optimizer steps")
    selection = training_summary.get("joint8_selection")
    if not isinstance(selection, Mapping):
        raise RuntimeError("joint8 training did not produce common selection")

    best_checkpoint = run / config["output"]["best_checkpoint"]
    best_policy, best_payload = load_digit_policy_checkpoint(
        best_checkpoint,
        expected_variant=config["variant"],
    )
    best_state = validate_protocol3_resume_checkpoint(best_payload, config)
    if int(best_state["completed_updates"]) != int(selection["best_update"]):
        raise RuntimeError("joint8 best checkpoint update differs from selection")
    best_row = next(
        row
        for row in training_summary["validation_history"]
        if int(row["completed_updates"]) == int(selection["best_update"])
    )
    final_row = training_summary["validation_history"][-1]

    audit_root = run / "common_best_digit_audits"
    audit_root.mkdir()
    digit_summaries: dict[str, Any] = {}
    for digit in JOINT8_DIGITS:
        digit_history = [
            {
                "completed_updates": row["completed_updates"],
                **row["per_digit"][str(digit)],
            }
            for row in training_summary["validation_history"]
        ]
        digit_selection = select_validation_history(digit_history)
        common_metrics = best_row["per_digit"][str(digit)]
        digit_output = audit_root / f"digit{digit}"
        digit_output.mkdir()
        common_passed = digit_metrics_pass(common_metrics)
        audit_selection = {
            "status": "PASS_AT_COMMON_CHECKPOINT" if common_passed else "FAIL_AT_COMMON_CHECKPOINT",
            "best_update": int(selection["best_update"]),
            "best_metrics": common_metrics,
        }
        case = {
            "digit": digit,
            "direction_index": JOINT8_DIRECTION_INDEX,
            "delay_steps": JOINT8_DELAY_STEPS,
        }
        audit = _audit_best_checkpoint(
            best_policy,
            hp,
            case,
            workspace_audit,
            digit_output,
            audit_selection,
        )
        digit_summaries[str(digit)] = {
            "history_status": digit_selection["status"],
            "history_best_update": digit_selection["best_update"],
            "history_best_metrics": digit_selection["best_metrics"],
            "common_checkpoint_passed": common_passed,
            "common_checkpoint_metrics": common_metrics,
            "final_checkpoint_metrics": final_row["per_digit"][str(digit)],
            "common_checkpoint_audit": audit,
        }

    final_checkpoint = run / config["output"]["final_checkpoint"]
    _, final_payload = load_digit_policy_checkpoint(
        final_checkpoint,
        expected_variant=config["variant"],
    )
    final_state = validate_protocol3_resume_checkpoint(final_payload, config)
    validate_joint8_condition_counts(
        final_state["condition_counts"], JOINT8_TOTAL_UPDATES
    )
    checkpoint_complete = bool(
        int(final_state["completed_updates"]) == JOINT8_TOTAL_UPDATES
        and final_payload["git_identity"] == hp["git_identity"]
        and len(final_payload["optimizer_state_dict"]["state"]) > 0
    )
    audits_passed = all(
        row["common_checkpoint_audit"]["engineering_passed"]
        for row in digit_summaries.values()
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": config["run_kind"],
        "variant": config["variant"],
        "git_identity": hp["git_identity"],
        "train_digits": list(JOINT8_DIGITS),
        "excluded_digits": list(JOINT8_EXCLUDED_DIGITS),
        "completed_optimizer_steps": JOINT8_TOTAL_UPDATES,
        "updates_per_digit": JOINT8_UPDATES_PER_DIGIT,
        "samples_per_digit": JOINT8_UPDATES_PER_DIGIT * JOINT8_BATCH_SIZE,
        "selection": selection,
        "digit_summaries": digit_summaries,
        "checkpoint_complete": checkpoint_complete,
        "all_common_checkpoint_audits_passed": audits_passed,
        "engineering_passed": bool(checkpoint_complete and audits_passed),
        "automatic_extension_started": False,
        "automatic_second_seed_started": False,
        "formal_full10_started": False,
    }
    _write_json(run / "joint8_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repository-root", required=True)
    prepare.add_argument("--protocol-config", required=True)
    prepare.add_argument("--evidence-directory", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repository-root", required=True)
    run.add_argument("--protocol-config", required=True)
    run.add_argument("--run-directory", required=True)
    run.add_argument("--evidence-directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_experiment(
            args.repository_root,
            args.protocol_config,
            args.evidence_directory,
        )
    else:
        result = run_experiment(
            args.repository_root,
            args.protocol_config,
            args.run_directory,
            args.evidence_directory,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
