"""Manual, state-complete continuation for one failed Protocol3 Gate 2 case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    validate_protocol3_resume_checkpoint,
)


REVIEW_METRICS = (
    "phase_normalized_position_l1",
    "normalized_mean_error",
    "normalized_endpoint_error",
    "path_length_ratio",
)


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


def _case_from_config(
    config: Mapping[str, Any], case_label: str
) -> dict[str, Any]:
    matches = [case for case in config["cases"] if case["label"] == case_label]
    if len(matches) != 1:
        raise ValueError(f"Gate 2 config does not contain one case named {case_label}")
    return dict(matches[0])


def _source_case_record(
    summary: Mapping[str, Any], case_label: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if summary.get("run_kind") == "protocol3_gate2":
        matches = [
            case for case in summary["cases"] if case["label"] == case_label
        ]
        if len(matches) != 1:
            raise ValueError("source Gate 2 summary case identity does not match")
        return dict(matches[0]), summary["git_identity"]
    if summary.get("run_kind") == "protocol3_gate2_continuation":
        case = summary.get("case", {})
        if case.get("label") != case_label:
            raise ValueError("source continuation summary case identity does not match")
        return {
            **case,
            "updates": summary["completed_updates"],
            "engineering_passed": summary["engineering_passed"],
            "behavior_passed": summary["behavior_passed"],
            "classification": summary["classification"],
        }, summary["git_identity"]
    raise ValueError("source summary is not Gate 2 or a Gate 2 continuation")


def _metric_review(history: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in REVIEW_METRICS:
        values = [float(row[name]) for row in history]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"source validation metric is not finite: {name}")
        recent = values[-10:]
        row = {
            "first": values[0],
            "last": values[-1],
            "last_10": recent,
            "last_10_mean": sum(recent) / len(recent),
        }
        if len(values) >= 20:
            previous = values[-20:-10]
            row["previous_10_mean"] = sum(previous) / len(previous)
            row["last_10_minus_previous_10_mean"] = (
                row["last_10_mean"] - row["previous_10_mean"]
            )
        if name == "path_length_ratio":
            row["closest_to_one"] = min(
                values, key=lambda value: abs(value - 1.0)
            )
            row["last_distance_to_one"] = abs(values[-1] - 1.0)
        else:
            row["minimum"] = min(values)
        result[name] = row
    return result


def review_gate2_continuation(
    protocol_config: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    case_label: str,
    expected_source_repository_head: str,
    current_identity: Mapping[str, str],
) -> dict[str, Any]:
    if protocol_config.get("run_kind") != "protocol3_gate2":
        raise ValueError("continuation requires the original Gate 2 config")
    case = _case_from_config(protocol_config, case_label)
    source_case, source_summary_identity = _source_case_record(
        source_summary, case_label
    )
    restore_state = validate_protocol3_resume_checkpoint(
        checkpoint, protocol_config
    )
    source_identity = checkpoint["git_identity"]
    if source_identity != source_summary_identity:
        raise ValueError("source summary and checkpoint Git identities differ")
    if source_identity["repository_head"] != expected_source_repository_head:
        raise ValueError("source checkpoint repository HEAD does not match")
    if checkpoint.get("hp", {}).get("git_identity") != source_identity:
        raise ValueError("source checkpoint embedded Git identity does not match")
    for key in (
        "branch",
        "mrnntorch_recorded_head",
        "mrnntorch_worktree_head",
    ):
        if source_identity[key] != current_identity[key]:
            raise ValueError(f"source checkpoint {key} does not match current code")
    if checkpoint.get("variant") != case_label:
        raise ValueError("source checkpoint variant does not match the case")
    if any(source_case.get(key) != case[key] for key in case):
        raise ValueError("source summary case does not match the Gate 2 config")
    if source_case.get("classification") != "behavior_failure":
        raise ValueError("only a behavior-failure case can be continued")
    if source_case.get("engineering_passed") is not True:
        raise ValueError("engineering or safety failure must not be continued")
    if source_case.get("behavior_passed") is not False:
        raise ValueError("a behavior-passed case must not be continued")

    completed_updates = int(restore_state["completed_updates"])
    if int(source_case["updates"]) != completed_updates:
        raise ValueError("source summary and checkpoint update counts differ")
    initial_max = int(protocol_config["training"]["max_updates"])
    interval = int(protocol_config["training"]["validation_interval"])
    if completed_updates < initial_max:
        raise ValueError("continuation requires a completed initial Gate 2 case")
    if completed_updates % interval != 0:
        raise ValueError("source checkpoint is not on a validation boundary")
    consecutive_passes = int(restore_state.get("gate2_consecutive_passes", -1))
    if consecutive_passes < 0:
        raise ValueError("source checkpoint lacks Gate 2 consecutive-pass state")
    if consecutive_passes >= 2:
        raise ValueError("a Gate 2 case that already passed must not continue")

    history = list(restore_state["validation_history"])
    expected_updates = list(range(0, completed_updates + 1, interval))
    actual_updates = [int(row["completed_updates"]) for row in history]
    if actual_updates != expected_updates:
        raise ValueError("source Gate 2 validation history is incomplete")
    return {
        "protocol": "digit_writing_original_protocol3",
        "run_kind": "protocol3_gate2_continuation_review",
        "case": case,
        "source_git_identity": dict(source_identity),
        "source_repository_head": source_identity["repository_head"],
        "current_repository_head": current_identity["repository_head"],
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_completed_updates": completed_updates,
        "validation_interval": interval,
        "validation_points": len(history),
        "gate2_consecutive_passes": consecutive_passes,
        "metrics": _metric_review(history),
        "engineering_passed": True,
        "behavior_passed": False,
        "eligible_for_manual_continuation": True,
        "automatic_continuation_allowed": False,
        "manual_target_required": True,
    }


def inspect_gate2_continuation(
    *,
    repository_root: str | Path,
    protocol_config_path: str | Path,
    source_summary_path: str | Path,
    resume_checkpoint_path: str | Path,
    case_label: str,
    expected_source_repository_head: str,
) -> dict[str, Any]:
    checkpoint_path = Path(resume_checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return review_gate2_continuation(
        _read_json(protocol_config_path),
        _read_json(source_summary_path),
        checkpoint,
        checkpoint_sha256=_file_sha256(checkpoint_path),
        case_label=case_label,
        expected_source_repository_head=expected_source_repository_head,
        current_identity=current_git_identity(repository_root),
    )


def validate_gate2_continuation_target(
    review: Mapping[str, Any], target_completed_updates: int
) -> int:
    source_updates = int(review["source_completed_updates"])
    interval = int(review["validation_interval"])
    target = int(target_completed_updates)
    if target <= source_updates:
        raise ValueError("continuation target must exceed the source update count")
    if target % interval != 0:
        raise ValueError("continuation target must end on a validation boundary")
    return target


def run_gate2_continuation(
    *,
    repository_root: str | Path,
    protocol_config_path: str | Path,
    source_summary_path: str | Path,
    resume_checkpoint_path: str | Path,
    case_label: str,
    target_completed_updates: int,
    output_directory: str | Path,
    expected_source_repository_head: str,
    manual_resume_authorized: bool,
) -> dict[str, Any]:
    if not manual_resume_authorized:
        raise PermissionError("Gate 2 continuation requires explicit manual approval")
    repository_root = Path(repository_root).resolve()
    output = Path(output_directory).resolve()

    config = _read_json(protocol_config_path)
    review = inspect_gate2_continuation(
        repository_root=repository_root,
        protocol_config_path=protocol_config_path,
        source_summary_path=source_summary_path,
        resume_checkpoint_path=resume_checkpoint_path,
        case_label=case_label,
        expected_source_repository_head=expected_source_repository_head,
    )
    source_updates = int(review["source_completed_updates"])
    target_completed_updates = validate_gate2_continuation_target(
        review, target_completed_updates
    )
    base_gate2_output = (repository_root / config["output"]["directory"]).resolve()
    expected_output = (
        Path(f"{base_gate2_output}_continuations")
        / case_label
        / f"from{source_updates}_to{target_completed_updates}"
    )
    if output != expected_output:
        raise ValueError("Gate 2 continuation output identity does not match")
    if output.exists():
        raise FileExistsError(f"Gate 2 continuation output already exists: {output}")

    from digit_writing.geometry import load_geometry_config
    from digit_writing.geometry_audit import build_workspace_audit
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

    _validate_protocol3_gate2_config(config)
    case = _case_from_config(config, case_label)
    output.mkdir(parents=True)
    request = {
        **review,
        "source_summary": str(Path(source_summary_path).resolve()),
        "source_checkpoint": str(Path(resume_checkpoint_path).resolve()),
        "target_completed_updates": target_completed_updates,
        "output_directory": str(output),
        "manual_resume_authorized": True,
    }
    _write_json(output / "continuation_request.json", request)
    _write_json(output / "protocol_config.json", config)
    workspace_audit = build_workspace_audit(
        load_geometry_config(config["geometry_config"])
    )
    _write_json(output / "workspace_audit.json", workspace_audit)

    hp = _base_hp_from_config(config)
    hp.update(
        {
            "variant": case_label,
            "condition_schedule": "protocol3_gate2_single_condition",
            "gate2_direction_index": case["direction_index"],
            "gate2_delay_index": PROTOCOL3_DELAYS.index(case["delay_steps"]),
            "stop_after_updates": target_completed_updates,
            "resume_source_repository_head": expected_source_repository_head,
            "resume_source_checkpoint_sha256": review[
                "source_checkpoint_sha256"
            ],
        }
    )
    training_summary = train_subsets_base_model(
        str(output),
        config["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((case["digit"],)),
        resume_checkpoint=str(Path(resume_checkpoint_path).resolve()),
        target_completed_updates=target_completed_updates,
        manual_resume_authorized=True,
        expected_resume_repository_head=expected_source_repository_head,
    )
    final_checkpoint = output / config["output"]["final_checkpoint"]
    policy, checkpoint = load_digit_policy_checkpoint(
        final_checkpoint, expected_variant=case_label
    )
    checkpoint_error = None
    try:
        checkpoint_state = validate_protocol3_resume_checkpoint(checkpoint, config)
        checkpoint_complete = bool(
            checkpoint["git_identity"] == hp["git_identity"]
            and int(checkpoint_state["completed_updates"])
            == training_summary["updates"]
        )
    except (KeyError, TypeError, ValueError) as error:
        checkpoint_complete = False
        checkpoint_error = str(error)
    candidate_audit = audit_gate2_case(
        policy,
        hp,
        DIGIT_ENV_CLASSES[case["digit"]],
        case,
        output,
        workspace_audit,
    )
    engineering_passed = bool(
        checkpoint_complete and candidate_audit["engineering_passed"]
    )
    behavior_passed = bool(
        training_summary["gate2_passed"] and candidate_audit["behavior_passed"]
    )
    if not engineering_passed:
        classification = "engineering_or_safety_failure"
    elif not behavior_passed:
        classification = "behavior_failure"
    else:
        classification = "provisional_pass_pending_qualitative_review"
    result = {
        "protocol": "digit_writing_original_protocol3",
        "run_kind": "protocol3_gate2_continuation",
        "git_identity": hp["git_identity"],
        "source_git_identity": review["source_git_identity"],
        "scale_multiplier": config["scale_multiplier"],
        "selected_reference_steps": config["selected_reference_steps"],
        "case": case,
        "source_completed_updates": source_updates,
        "target_completed_updates": target_completed_updates,
        "completed_updates": training_summary["updates"],
        "source_checkpoint_sha256": review["source_checkpoint_sha256"],
        "final_checkpoint_sha256": _file_sha256(final_checkpoint),
        "training": training_summary,
        "checkpoint_complete": checkpoint_complete,
        "checkpoint_error": checkpoint_error,
        "candidate_audit": candidate_audit,
        "engineering_passed": engineering_passed,
        "behavior_passed": behavior_passed,
        "automatic_metrics_passed": bool(
            engineering_passed and behavior_passed
        ),
        "gradient_metrics_are_diagnostic_only": True,
        "qualitative_overlay_review_required": behavior_passed,
        "classification": classification,
        "automatic_continuation_allowed": False,
        "automatic_medium_fallback_allowed": False,
        "further_continuation_requires_new_manual_approval": True,
        "passed": False,
    }
    _write_json(output / "gate2_continuation_summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "run"))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--source-summary", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--target-completed-updates", type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--manual-resume-authorized", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    common = {
        "repository_root": arguments.repository_root,
        "protocol_config_path": arguments.protocol_config,
        "source_summary_path": arguments.source_summary,
        "resume_checkpoint_path": arguments.resume_checkpoint,
        "case_label": arguments.case_label,
        "expected_source_repository_head": arguments.expected_source_head,
    }
    if arguments.mode == "inspect":
        result = inspect_gate2_continuation(**common)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if (
        arguments.target_completed_updates is None
        or arguments.output_directory is None
    ):
        raise ValueError("run mode requires target updates and output directory")
    result = run_gate2_continuation(
        **common,
        target_completed_updates=arguments.target_completed_updates,
        output_directory=arguments.output_directory,
        manual_resume_authorized=arguments.manual_resume_authorized,
    )
    print(f"GATE2_CONTINUATION_CLASSIFICATION={result['classification']}")
    print(f"GATE2_CONTINUATION_COMPLETED_UPDATES={result['completed_updates']}")
    print("AUTOMATIC_MEDIUM_FALLBACK_STARTED=0")


if __name__ == "__main__":
    main()
