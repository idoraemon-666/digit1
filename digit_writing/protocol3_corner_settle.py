"""Isolated, manually reviewed Protocol3 corner-settle experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from digit_writing.corner_settle import (
    CORNER_SETTLE_INTERVALS,
    CORNER_SETTLE_PHYSICAL_MS,
    CORNER_TURN_THRESHOLD_DEG,
    build_corner_settle_trajectory,
    corner_manifest_rows,
)
from digit_writing.geometry import load_geometry_config
from digit_writing.geometry_audit import build_workspace_audit
from digit_writing.final_protocol_audit import _json_safe
from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    protocol_config_sha256,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)


CASE_SPECS = (
    ("digit4_baseline", 4, "baseline", 0, "digit4/baseline"),
    ("digit4_settle100ms", 4, "settle100ms", 10, "digit4/settle100ms"),
    ("digit7_baseline", 7, "baseline", 0, "digit7/baseline"),
    ("digit7_settle100ms", 7, "settle100ms", 10, "digit7/settle100ms"),
)
REVIEW_INTERVAL_UPDATES = 6000


def validate_next_review_target(source_updates: int, target_updates: int) -> int:
    if isinstance(source_updates, bool) or isinstance(target_updates, bool):
        raise TypeError("review update counts must be integers")
    source = int(source_updates)
    target = int(target_updates)
    if source < REVIEW_INTERVAL_UPDATES or source % REVIEW_INTERVAL_UPDATES != 0:
        raise ValueError("source must be a completed 6000-update review boundary")
    if target != source + REVIEW_INTERVAL_UPDATES:
        raise ValueError("each continuation must add exactly 6000 updates")
    return target


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


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = (repository_root / value).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("configured path escapes the repository") from error
    return path


def _validate_config(
    repository_root: str | Path,
    protocol_config: Mapping[str, Any],
) -> dict[str, Any]:
    from train import _validate_protocol3_gate2_config

    root = Path(repository_root).resolve()
    if protocol_config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("corner-settle gate requires Protocol3")
    if protocol_config.get("run_kind") != "protocol3_corner_settle_gate":
        raise ValueError("corner-settle run_kind is invalid")
    if protocol_config.get("variant") != "corner_settle_fast_review6000":
        raise ValueError("only the frozen fast 6000-update variant is allowed")
    if protocol_config.get("device") != "cpu":
        raise ValueError("corner-settle gate is CPU-only")
    if protocol_config.get("selected_reference_steps") != 50:
        raise ValueError("the initial corner-settle gate is fast/ref50 only")
    if protocol_config.get("scale_multiplier") != 2.5:
        raise ValueError("the initial corner-settle gate uses scale2p50")
    if protocol_config.get("timing_mode") != "fixed_segment_timing":
        raise ValueError("corner-settle gate requires fixed segment timing")
    if protocol_config.get("condition_schedule") != "protocol3_corner_settle_matrix_v1":
        raise ValueError("corner-settle matrix identity differs")

    source_path = _resolve_repository_path(
        root, str(protocol_config["source_gate2_config"])
    )
    source = _read_json(source_path)
    _validate_protocol3_gate2_config(source)
    for key in (
        "seed",
        "validation_seed",
        "device",
        "geometry_config",
        "scale_multiplier",
        "timing_mode",
        "selected_reference_steps",
        "model",
        "optimizer",
        "position_loss",
        "regularization",
    ):
        if protocol_config.get(key) != source.get(key):
            raise ValueError(f"corner-settle config differs from source Gate 2: {key}")
    if protocol_config.get("training") != {
        "batch_size": source["training"]["batch_size"],
        "max_updates": REVIEW_INTERVAL_UPDATES,
        "validation_interval": source["training"]["validation_interval"],
    }:
        raise ValueError("corner-settle training boundary must be exactly 6000 updates")
    if protocol_config.get("corner_settle") != {
        "turn_angle_deg": CORNER_TURN_THRESHOLD_DEG,
        "physical_pause_ms": CORNER_SETTLE_PHYSICAL_MS,
        "settle_intervals": CORNER_SETTLE_INTERVALS,
        "review_interval_updates": REVIEW_INTERVAL_UPDATES,
        "automatic_convergence_decision": False,
        "automatic_continuation": False,
        "automatic_medium_fallback": False,
    }:
        raise ValueError("corner-settle intervention or manual gates differ")

    actual_cases = tuple(
        (
            case.get("label"),
            case.get("digit"),
            case.get("condition"),
            case.get("settle_intervals"),
            case.get("output_subdirectory"),
        )
        for case in protocol_config.get("cases", ())
    )
    if actual_cases != CASE_SPECS:
        raise ValueError("corner-settle cases must be the frozen matched 4/7 pairs")
    for case in protocol_config["cases"]:
        if case.get("direction_index") != 0 or case.get("delay_steps") != 50:
            raise ValueError("corner-settle cases require direction 0 and delay 50")

    output = protocol_config.get("output", {})
    if output.get("initial_checkpoint") != "initial_checkpoint.pt":
        raise ValueError("corner-settle initial checkpoint name differs")
    if output.get("best_checkpoint") != "best_checkpoint.pt":
        raise ValueError("corner-settle best checkpoint name differs")
    if output.get("final_checkpoint") != "final_continuation_checkpoint.pt":
        raise ValueError("corner-settle continuation checkpoint name differs")
    directory = str(output.get("directory", ""))
    if (
        not directory.startswith("runs/digit_writing_original_protocol3_corner_settle_gate/")
        or "protocol2" in directory
        or "full10" in directory
    ):
        raise ValueError("corner-settle output namespace is invalid")
    return source


def _case_by_label(config: Mapping[str, Any], label: str) -> dict[str, Any]:
    matches = [case for case in config["cases"] if case["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown corner-settle case: {label}")
    return dict(matches[0])


def _case_output(run_root: Path, case: Mapping[str, Any]) -> Path:
    output = (run_root / str(case["output_subdirectory"])).resolve()
    try:
        output.relative_to(run_root.resolve())
    except ValueError as error:
        raise ValueError("case output escapes the experiment root") from error
    return output


def _hp_for_case(config: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    from train import _base_hp_from_config, PROTOCOL3_DELAYS

    hp = _base_hp_from_config(config)
    hp.update(
        {
            "variant": case["label"],
            "condition_schedule": "protocol3_corner_settle_single_condition",
            "gate2_direction_index": int(case["direction_index"]),
            "gate2_delay_index": PROTOCOL3_DELAYS.index(case["delay_steps"]),
            "stop_after_updates": REVIEW_INTERVAL_UPDATES,
            "env_kwargs": {
                "geometry_config_path": config["geometry_config"],
                "corner_settle_intervals": int(case["settle_intervals"]),
            },
        }
    )
    return hp


def prepare_experiment(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    source = _validate_config(root, config)
    output = Path(output_directory).resolve()
    expected_output = _resolve_repository_path(root, config["output"]["directory"])
    if output != expected_output:
        raise ValueError("initial output directory differs from the checked-in config")
    if output.exists():
        raise FileExistsError(f"corner-settle output already exists: {output}")
    output.mkdir(parents=True)
    _write_json(output / "resolved_config.json", config)
    _write_json(output / "source_gate2_config.json", source)

    geometry_path = _resolve_repository_path(root, config["geometry_config"])
    geometry_config = load_geometry_config(geometry_path)
    workspace_audit = build_workspace_audit(geometry_config)
    _write_json(output / "workspace_audit.json", workspace_audit)

    manifest_rows = []
    for case in config["cases"]:
        trajectory = build_corner_settle_trajectory(
            int(case["digit"]),
            geometry_config,
            int(config["selected_reference_steps"]),
            settle_intervals=int(case["settle_intervals"]),
        )
        manifest_rows.extend(
            corner_manifest_rows(
                int(case["digit"]), str(case["condition"]), trajectory
            )
        )
    with (output / "corner_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    result = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_corner_settle_prepare",
        "git_identity": current_git_identity(root),
        "source_gate2_config": config["source_gate2_config"],
        "source_gate2_config_sha256": protocol_config_sha256(source),
        "selected_reference_steps": 50,
        "settle_intervals": 10,
        "physical_pause_ms": 100,
        "training_updates": 0,
        "case_labels": [case["label"] for case in config["cases"]],
        "parallel_process_count": 4,
        "automatic_continuation_allowed": False,
        "automatic_medium_fallback_allowed": False,
        "formal_full10_started": False,
    }
    _write_json(output / "preflight_summary.json", result)
    return result


def _run_case_training(
    repository_root: Path,
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    output: Path,
    workspace_audit: Mapping[str, Any],
    *,
    resume_checkpoint: str | Path | None = None,
    target_completed_updates: int | None = None,
    expected_source_head: str | None = None,
    source_initial_state_sha256: str | None = None,
    source_best_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    from train import (
        DIGIT_ENV_CLASSES,
        _digit_env_dict,
        load_digit_policy_checkpoint,
        train_subsets_base_model,
    )
    from digit_writing.protocol3_gate2 import audit_gate2_case

    if output.exists():
        raise FileExistsError(f"corner-settle case output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    hp = _hp_for_case(config, case)
    training_summary = train_subsets_base_model(
        str(output),
        config["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((int(case["digit"]),)),
        resume_checkpoint=(None if resume_checkpoint is None else str(resume_checkpoint)),
        manual_resume_authorized=resume_checkpoint is not None,
        target_completed_updates=target_completed_updates,
        expected_resume_repository_head=expected_source_head,
    )
    best_checkpoint = output / config["output"]["best_checkpoint"]
    best_checkpoint_inherited = False
    if not best_checkpoint.is_file():
        if source_best_checkpoint is None:
            raise FileNotFoundError("training did not produce a best checkpoint")
        shutil.copy2(source_best_checkpoint, best_checkpoint)
        best_checkpoint_inherited = True
    if resume_checkpoint is None:
        initial_checkpoint = output / config["output"]["initial_checkpoint"]
        _, initial_payload = load_digit_policy_checkpoint(
            initial_checkpoint,
            expected_variant=case["label"],
        )
        initial_state_sha256 = state_dict_sha256(
            initial_payload["agent_state_dict"]
        )
        initial_state = validate_protocol3_resume_checkpoint(initial_payload, config)
        initial_checkpoint_complete = bool(
            int(initial_state["completed_updates"]) == 0
            and initial_payload["git_identity"] == hp["git_identity"]
        )
    else:
        if not source_initial_state_sha256:
            raise ValueError("continuation source initial-state identity is missing")
        initial_state_sha256 = source_initial_state_sha256
        initial_checkpoint_complete = True
    final_checkpoint = output / config["output"]["final_checkpoint"]
    policy, checkpoint = load_digit_policy_checkpoint(
        final_checkpoint,
        expected_variant=case["label"],
    )
    checkpoint_error = None
    try:
        checkpoint_state = validate_protocol3_resume_checkpoint(checkpoint, config)
        checkpoint_complete = bool(
            checkpoint["git_identity"] == hp["git_identity"]
            and int(checkpoint_state["completed_updates"])
            == int(training_summary["updates"])
        )
    except (KeyError, TypeError, ValueError) as error:
        checkpoint_complete = False
        checkpoint_error = str(error)
    candidate_audit = audit_gate2_case(
        policy,
        hp,
        DIGIT_ENV_CLASSES[int(case["digit"])],
        case,
        output,
        workspace_audit,
        audit_filename="corner_settle_candidate_audit.json",
    )
    best_policy, _ = load_digit_policy_checkpoint(
        best_checkpoint,
        expected_variant=case["label"],
    )
    best_audit = audit_gate2_case(
        best_policy,
        hp,
        DIGIT_ENV_CLASSES[int(case["digit"])],
        case,
        output / "best_checkpoint_audit",
        workspace_audit,
        audit_filename="corner_settle_best_checkpoint_audit.json",
    )
    engineering_passed = bool(
        initial_checkpoint_complete
        and checkpoint_complete
        and candidate_audit["engineering_passed"]
        and best_audit["engineering_passed"]
    )
    classification = (
        "pending_manual_review"
        if engineering_passed
        else "engineering_or_safety_failure"
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_corner_settle_case",
        "case": dict(case),
        "git_identity": hp["git_identity"],
        "completed_updates": int(training_summary["updates"]),
        "review_interval_updates": REVIEW_INTERVAL_UPDATES,
        "training_summary": training_summary,
        "initial_checkpoint_complete": initial_checkpoint_complete,
        "initial_state_sha256": initial_state_sha256,
        "best_checkpoint_inherited": best_checkpoint_inherited,
        "checkpoint_complete": checkpoint_complete,
        "checkpoint_error": checkpoint_error,
        "candidate_audit": candidate_audit,
        "best_checkpoint_audit": best_audit,
        "engineering_passed": engineering_passed,
        "classification": classification,
        "automatic_convergence_decision": False,
        "automatic_continuation_allowed": False,
        "automatic_medium_fallback_allowed": False,
        "formal_full10_started": False,
    }
    _write_json(output / "corner_settle_case_summary.json", result)
    return result


def run_initial_case(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    run_root: str | Path,
    case_label: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    _validate_config(root, config)
    expected_root = _resolve_repository_path(root, config["output"]["directory"])
    actual_root = Path(run_root).resolve()
    if actual_root != expected_root:
        raise ValueError("initial run root differs from the checked-in config")
    workspace_audit = _read_json(actual_root / "workspace_audit.json")
    case = _case_by_label(config, case_label)
    return _run_case_training(
        root,
        config,
        case,
        _case_output(actual_root, case),
        workspace_audit,
    )


def run_continuation_case(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    source_run_root: str | Path,
    output_run_root: str | Path,
    case_label: str,
    target_completed_updates: int,
    expected_source_head: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    _validate_config(root, config)
    case = _case_by_label(config, case_label)
    source_root = Path(source_run_root).resolve()
    output_root = Path(output_run_root).resolve()
    namespace = (
        root
        / "runs"
        / "digit_writing_original_protocol3_corner_settle_gate"
        / "scale2p50"
        / "ref50"
    ).resolve()
    try:
        source_root.relative_to(namespace)
    except ValueError as error:
        raise ValueError("continuation source escapes the corner-settle namespace") from error
    if output_root.parent != namespace / "continuations":
        raise ValueError("continuation output must use the isolated continuation root")
    source_case = _case_output(source_root, case)
    source_summary = _read_json(source_case / "corner_settle_case_summary.json")
    source_updates = int(source_summary["completed_updates"])
    target_completed_updates = validate_next_review_target(
        source_updates, target_completed_updates
    )
    if source_summary.get("case") != case:
        raise ValueError("continuation source case identity differs")
    if not source_summary.get("engineering_passed"):
        raise ValueError("unsafe or incomplete source case must not be continued")
    checkpoint = source_case / config["output"]["final_checkpoint"]
    workspace_audit = _read_json(source_root / "workspace_audit.json")
    result = _run_case_training(
        root,
        config,
        case,
        _case_output(output_root, case),
        workspace_audit,
        resume_checkpoint=checkpoint,
        target_completed_updates=target_completed_updates,
        expected_source_head=expected_source_head,
        source_initial_state_sha256=source_summary.get("initial_state_sha256"),
        source_best_checkpoint=source_case / config["output"]["best_checkpoint"],
    )
    result["source_run_root"] = str(source_root)
    result["source_completed_updates"] = source_updates
    _write_json(
        _case_output(output_root, case) / "corner_settle_case_summary.json",
        result,
    )
    return result


def _pair_comparison(
    baseline: Mapping[str, Any], intervention: Mapping[str, Any]
) -> dict[str, Any]:
    def compare_audits(baseline_audit, intervention_audit):
        baseline_global = baseline_audit["movement_metrics"]
        intervention_global = intervention_audit["movement_metrics"]

        def relative_reduction(name: str) -> float | None:
            denominator = float(baseline_global[name])
            if denominator == 0.0:
                return None
            return (denominator - float(intervention_global[name])) / denominator

        baseline_corners = baseline_audit["corner_audit"]["corners"]
        intervention_corners = intervention_audit["corner_audit"]["corners"]
        local = []
        for left, right in zip(baseline_corners, intervention_corners):
            if left["boundary_index"] != right["boundary_index"]:
                raise ValueError("paired corner identities differ")
            left_mean = float(left["corner_local_mean_error_m"])
            left_miss = float(left["corner_miss_distance_m"])
            local.append(
                {
                    "boundary_index": left["boundary_index"],
                    "baseline": left,
                    "settle100ms": right,
                    "corner_local_mean_error_relative_reduction": (
                        None
                        if left_mean == 0.0
                        else (left_mean - float(right["corner_local_mean_error_m"])) / left_mean
                    ),
                    "corner_miss_distance_relative_reduction": (
                        None
                        if left_miss == 0.0
                        else (left_miss - float(right["corner_miss_distance_m"])) / left_miss
                    ),
                }
            )
        return {
            "baseline_movement_metrics": baseline_global,
            "settle100ms_movement_metrics": intervention_global,
            "normalized_mean_error_relative_reduction": relative_reduction(
                "normalized_mean_error"
            ),
            "normalized_endpoint_error_relative_reduction": relative_reduction(
                "normalized_endpoint_error"
            ),
            "corners": local,
        }

    final_comparison = compare_audits(
        baseline["candidate_audit"], intervention["candidate_audit"]
    )
    best_comparison = compare_audits(
        baseline["best_checkpoint_audit"], intervention["best_checkpoint_audit"]
    )
    return {
        "matched_initialization": (
            baseline["initial_state_sha256"]
            == intervention["initial_state_sha256"]
        ),
        "baseline_initial_state_sha256": baseline["initial_state_sha256"],
        "settle100ms_initial_state_sha256": intervention["initial_state_sha256"],
        "final_checkpoint_comparison": final_comparison,
        "best_checkpoint_comparison": best_comparison,
    }


def _manual_convergence_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    history = summary["training_summary"]["validation_history"]
    completed = int(summary["completed_updates"])
    by_update = {int(row["completed_updates"]): row for row in history}
    requested = (max(0, completed - 2000), max(0, completed - 1000), completed)
    snapshots = {str(update): by_update[update] for update in requested if update in by_update}

    def relative_change(metric: str, start: int) -> float | None:
        if start not in by_update or completed not in by_update:
            return None
        denominator = float(by_update[start][metric])
        if denominator == 0.0:
            return None
        return (float(by_update[completed][metric]) - denominator) / denominator

    return {
        "best_checkpoint_update": summary["training_summary"]["best_checkpoint_update"],
        "best_validation_loss": summary["training_summary"]["best_validation_loss"],
        "last_validation_loss": summary["training_summary"]["last_validation_loss"],
        "snapshots": snapshots,
        "last_1000_relative_change": {
            metric: relative_change(metric, max(0, completed - 1000))
            for metric in (
                "phase_normalized_position_l1",
                "normalized_mean_error",
                "normalized_endpoint_error",
                "path_length_ratio",
            )
        },
        "last_2000_relative_change": {
            metric: relative_change(metric, max(0, completed - 2000))
            for metric in (
                "phase_normalized_position_l1",
                "normalized_mean_error",
                "normalized_endpoint_error",
                "path_length_ratio",
            )
        },
        "automatic_convergence_decision": None,
    }


def _save_review_curve(output: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = summary["training_summary"]["validation_history"]
    updates = [int(row["completed_updates"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        updates,
        [float(row["normalized_mean_error"]) for row in history],
        label="normalized mean",
    )
    axes[0].plot(
        updates,
        [float(row["normalized_endpoint_error"]) for row in history],
        label="normalized endpoint",
    )
    axes[0].set_xlabel("completed updates")
    axes[0].set_ylabel("normalized error")
    axes[0].legend()
    axes[1].plot(
        updates,
        [float(row["phase_normalized_position_l1"]) for row in history],
        label="position loss",
    )
    axes[1].plot(
        updates,
        [float(row["path_length_ratio"]) for row in history],
        label="path-length ratio",
    )
    axes[1].set_xlabel("completed updates")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "training_review_curve.png", dpi=180)
    plt.close(figure)


def summarize_experiment(
    repository_root: str | Path,
    protocol_config_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_json(protocol_config_path)
    _validate_config(root, config)
    output = Path(run_root).resolve()
    summaries = {
        case["label"]: _read_json(
            _case_output(output, case) / "corner_settle_case_summary.json"
        )
        for case in config["cases"]
    }
    for case in config["cases"]:
        _save_review_curve(
            _case_output(output, case), summaries[case["label"]]
        )
    updates = {int(summary["completed_updates"]) for summary in summaries.values()}
    if len(updates) != 1 or next(iter(updates)) % REVIEW_INTERVAL_UPDATES != 0:
        raise ValueError("all paired cases must stop on the same 6000-update boundary")
    paired_comparisons = {
        "digit4": _pair_comparison(
            summaries["digit4_baseline"], summaries["digit4_settle100ms"]
        ),
        "digit7": _pair_comparison(
            summaries["digit7_baseline"], summaries["digit7_settle100ms"]
        ),
    }
    engineering_passed = bool(
        all(bool(summary["engineering_passed"]) for summary in summaries.values())
        and all(pair["matched_initialization"] for pair in paired_comparisons.values())
    )
    with (output / "review_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            "case_label",
            "completed_updates",
            "phase_normalized_position_l1",
            "normalized_mean_error",
            "normalized_endpoint_error",
            "path_length_ratio",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, summary in summaries.items():
            for row in summary["training_summary"]["validation_history"]:
                writer.writerow({"case_label": label, **{key: row[key] for key in fieldnames[1:]}})
    result = {
        "protocol": config["protocol"],
        "run_kind": "protocol3_corner_settle_gate",
        "git_identity": current_git_identity(root),
        "selected_reference_steps": 50,
        "physical_pause_ms": 100,
        "settle_intervals": 10,
        "completed_updates": next(iter(updates)),
        "cases": summaries,
        "paired_comparisons": paired_comparisons,
        "manual_convergence_evidence": {
            label: _manual_convergence_evidence(summary)
            for label, summary in summaries.items()
        },
        "review_metrics_csv": "review_metrics.csv",
        "engineering_passed": engineering_passed,
        "classification": (
            "pending_manual_review"
            if engineering_passed
            else "engineering_or_safety_failure"
        ),
        "automatic_convergence_decision": False,
        "automatic_continuation_started": False,
        "automatic_medium_fallback_started": False,
        "formal_full10_started": False,
    }
    _write_json(output / "corner_settle_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command):
        command.add_argument("--repository-root", required=True)
        command.add_argument("--protocol-config", required=True)

    prepare = subparsers.add_parser("prepare")
    common(prepare)
    prepare.add_argument("--output-directory", required=True)

    run_case = subparsers.add_parser("run-case")
    common(run_case)
    run_case.add_argument("--run-root", required=True)
    run_case.add_argument("--case-label", required=True)

    continuation = subparsers.add_parser("continue-case")
    common(continuation)
    continuation.add_argument("--source-run-root", required=True)
    continuation.add_argument("--output-run-root", required=True)
    continuation.add_argument("--case-label", required=True)
    continuation.add_argument("--target-completed-updates", required=True, type=int)
    continuation.add_argument("--expected-source-head", required=True)
    continuation.add_argument("--manual-resume-authorized", action="store_true")

    summarize = subparsers.add_parser("summarize")
    common(summarize)
    summarize.add_argument("--run-root", required=True)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare_experiment(
            arguments.repository_root,
            arguments.protocol_config,
            arguments.output_directory,
        )
    elif arguments.command == "run-case":
        result = run_initial_case(
            arguments.repository_root,
            arguments.protocol_config,
            arguments.run_root,
            arguments.case_label,
        )
    elif arguments.command == "continue-case":
        if not arguments.manual_resume_authorized:
            raise PermissionError("corner-settle continuation requires manual approval")
        result = run_continuation_case(
            arguments.repository_root,
            arguments.protocol_config,
            arguments.source_run_root,
            arguments.output_run_root,
            arguments.case_label,
            arguments.target_completed_updates,
            arguments.expected_source_head,
        )
    else:
        result = summarize_experiment(
            arguments.repository_root,
            arguments.protocol_config,
            arguments.run_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
