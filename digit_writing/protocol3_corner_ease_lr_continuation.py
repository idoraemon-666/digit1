"""Controlled learning-rate continuation for six Protocol3 corner-ease cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from digit_writing.protocol3_checkpoint import (
    current_git_identity,
    state_dict_sha256,
    validate_protocol3_resume_checkpoint,
)
from digit_writing.protocol3_corner_ease import (
    _audit_best_checkpoint,
    _hp_for_case,
    _validate_config,
    reconstruct_corner_ease_selection_state,
    select_validation_history,
)


SOURCE_UPDATES = 6000
TARGET_UPDATES = 8000
VALIDATION_INTERVAL = 100
SOURCE_REPOSITORY_HEAD = "ee5a1900a33f8a8b7921fd9fc5b09aeed6600925"
CASE_DIGITS = (0, 3, 4, 5, 6, 7)
JOINT_CASE_DIGITS = (0, 8)
ARM_LABELS = ("lr1e3", "lr3e4")
ARM_LEARNING_RATES = (0.001, 0.0003)
JOINT_RUN_KIND = "protocol3_digit0_digit8_matched_lr_continuation"
JOINT_VARIANT = "digit0_digit8_four_arm_from6000_to8000"


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("continuation comparison has no rows")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_label(digit: int) -> str:
    return f"digit{digit}_corner_ease"


def _arm_run_label(digit: int, arm_label: str) -> str:
    return f"digit{digit}_{arm_label}"


def _is_joint_config(config: Mapping[str, Any]) -> bool:
    return config.get("run_kind") == JOINT_RUN_KIND


def _case_digits(config: Mapping[str, Any]) -> tuple[int, ...]:
    return JOINT_CASE_DIGITS if _is_joint_config(config) else CASE_DIGITS


def _output_names(config: Mapping[str, Any]) -> dict[str, str]:
    if _is_joint_config(config):
        return {
            "comparison": "digit0_digit8_lr_comparison.csv",
            "summary": "digit0_digit8_matched_lr_continuation_summary.json",
            "report": "DIGIT0_DIGIT8_MATCHED_LR_CONTINUATION_REPORT.md",
        }
    return {
        "comparison": "six_digit_lr_comparison.csv",
        "summary": "corner_ease_lr_continuation_summary.json",
        "report": "CORNER_EASE_LR_CONTINUATION_REPORT.md",
    }


def load_continuation_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("corner-ease continuation requires Protocol3")
    run_kind = config.get("run_kind")
    if run_kind not in (
        "protocol3_corner_ease_lr_continuation",
        JOINT_RUN_KIND,
    ):
        raise ValueError("corner-ease continuation run kind is invalid")
    joint = _is_joint_config(config)
    expected_variant = (
        JOINT_VARIANT if joint else "six_digit_lr_control_from6000_to8000"
    )
    if config.get("variant") != expected_variant:
        raise ValueError("corner-ease continuation variant is invalid")
    digits = _case_digits(config)
    source = config["source"]
    expected_source_paths = {
        "protocol_config": (
            "configurations/"
            "digit_writing_original_protocol3_ten_digit_corner_ease_overfit_v3.json"
        ),
        "output_directory": (
            "runs/digit_writing_original_protocol3/scale2p50/ref100/"
            "corner_ease_v3/single_digit_overfit"
        ),
        "evidence_directory": (
            "runs/digit_writing_original_protocol3/scale2p50/ref100/"
            "corner_ease_v3/single_digit_overfit_server_evidence"
        ),
    }
    for name, expected in expected_source_paths.items():
        if source.get(name) != expected:
            raise ValueError(f"corner-ease continuation source {name} changed")
    if source.get("repository_head") != SOURCE_REPOSITORY_HEAD:
        raise ValueError("corner-ease continuation source HEAD changed")
    if int(source.get("completed_updates", -1)) != SOURCE_UPDATES:
        raise ValueError("corner-ease continuation source update changed")
    source_cases = source.get("cases", [])
    if tuple(int(row.get("digit", -1)) for row in source_cases) != digits:
        raise ValueError("corner-ease continuation digit scope changed")
    if tuple(row.get("label") for row in source_cases) != tuple(
        _case_label(digit) for digit in digits
    ):
        raise ValueError("corner-ease continuation case labels changed")
    for row in source_cases:
        digit = int(row["digit"])
        expected_status = (
            "FAIL" if joint and digit == 8 else "PASS_UNSTABLE"
        )
        if row.get("source_status") != expected_status:
            raise ValueError("corner-ease continuation source status changed")
        for name in (
            "best_checkpoint_sha256",
            "final_checkpoint_sha256",
            "run_summary_sha256",
        ):
            if len(row.get(name, "")) != 64:
                raise ValueError(f"corner-ease continuation {name} is invalid")
    if config.get("optimizer") != {
        "name": "Adam",
        "source_learning_rate": 0.001,
        "restore_optimizer_state": True,
        "override_only_param_group_learning_rate": True,
    }:
        raise ValueError("corner-ease continuation optimizer identity changed")
    arms = config.get("arms", [])
    if tuple(arm.get("label") for arm in arms) != ARM_LABELS:
        raise ValueError("corner-ease continuation arm labels changed")
    if tuple(float(arm.get("learning_rate")) for arm in arms) != (
        ARM_LEARNING_RATES
    ):
        raise ValueError("corner-ease continuation learning rates changed")
    expected_training = {
        "target_completed_updates": TARGET_UPDATES,
        "additional_updates": TARGET_UPDATES - SOURCE_UPDATES,
        "validation_interval": VALIDATION_INTERVAL,
        "batch_size": 8,
        "fixed_target_no_early_stop": True,
    }
    if joint:
        expected_training["synchronized_parallel_arms"] = 4
    if config.get("training") != expected_training:
        raise ValueError("corner-ease continuation training schedule changed")
    expected_decision = {
        "automatic_winner_selection": False,
        "automatic_further_continuation": False,
        "automatic_second_seed": False,
        "digit8_included": joint,
        "formal_full10_start": False,
        "qualitative_overlay_review_required": True,
    }
    if joint:
        expected_decision.update(
            {
                "automatic_geometry_change": False,
                "automatic_loss_change": False,
                "synchronized_parallel_training": True,
            }
        )
    if config.get("decision") != expected_decision:
        raise ValueError("corner-ease continuation decision boundary changed")
    expected_output = (
        (
            "runs/digit_writing_original_protocol3/scale2p50/ref100/"
            "corner_ease_v3/"
            "digit0_digit8_matched_lr_continuation_from6000_to8000"
        )
        if joint
        else (
            "runs/digit_writing_original_protocol3/scale2p50/ref100/"
            "corner_ease_v3/lr_continuation_from6000_to8000"
        )
    )
    if config.get("output") != {"directory": expected_output}:
        raise ValueError("corner-ease continuation output identity changed")
    return config


def _resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("configured continuation path escapes the repository") from error
    return path


def _source_case_record(
    config: Mapping[str, Any], case_label: str
) -> dict[str, Any]:
    matches = [
        row for row in config["source"]["cases"] if row["label"] == case_label
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown corner-ease continuation case: {case_label}")
    return dict(matches[0])


def _arm_record(config: Mapping[str, Any], arm_label: str) -> dict[str, Any]:
    matches = [row for row in config["arms"] if row["label"] == arm_label]
    if len(matches) != 1:
        raise ValueError(f"unknown corner-ease continuation arm: {arm_label}")
    return dict(matches[0])


def _source_protocol_config(
    root: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_inside(root, config["source"]["protocol_config"])
    protocol_config = _read_json(path)
    _validate_config(root, protocol_config)
    return path, protocol_config


def _source_case_paths(
    source_root: Path, protocol_config: Mapping[str, Any], digit: int
) -> tuple[dict[str, Any], Path, Path, Path]:
    matches = [case for case in protocol_config["cases"] if case["digit"] == digit]
    if len(matches) != 1 or matches[0]["label"] != _case_label(digit):
        raise ValueError("source corner-ease case identity changed")
    case = dict(matches[0])
    case_root = (source_root / case["output_subdirectory"]).resolve()
    try:
        case_root.relative_to(source_root)
    except ValueError as error:
        raise ValueError("source corner-ease case escapes its run root") from error
    return (
        case,
        case_root / protocol_config["output"]["best_checkpoint"],
        case_root / protocol_config["output"]["final_checkpoint"],
        case_root / "run_summary.json",
    )


def review_source_case(
    *,
    repository_root: str | Path,
    continuation_config_path: str | Path,
    source_run_root: str | Path,
    case_label: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_continuation_config(continuation_config_path)
    expected_source_root = _resolve_inside(
        root, config["source"]["output_directory"]
    )
    source_root = Path(source_run_root).resolve()
    if source_root != expected_source_root:
        raise ValueError("corner-ease continuation source run root changed")
    source_record = _source_case_record(config, case_label)
    digit = int(source_record["digit"])
    _, protocol_config = _source_protocol_config(root, config)
    case, best_path, final_path, summary_path = _source_case_paths(
        source_root, protocol_config, digit
    )
    expected_hashes = {
        best_path: source_record["best_checkpoint_sha256"],
        final_path: source_record["final_checkpoint_sha256"],
        summary_path: source_record["run_summary_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"corner-ease source hash differs: {path.name}")
    summary = _read_json(summary_path)
    if (
        summary.get("run_kind") != "protocol3_corner_ease_case"
        or summary.get("case") != case
        or int(summary.get("completed_updates", -1)) != SOURCE_UPDATES
        or summary.get("status") != source_record["source_status"]
        or summary.get("checkpoint_complete") is not True
        or summary.get("engineering_passed") is not True
    ):
        raise ValueError("corner-ease source summary is not eligible")
    source_identity = summary.get("git_identity")
    if source_identity.get("repository_head") != config["source"]["repository_head"]:
        raise ValueError("corner-ease source summary HEAD differs")
    current_identity = current_git_identity(root)
    for key in ("branch", "mrnntorch_recorded_head", "mrnntorch_worktree_head"):
        if source_identity.get(key) != current_identity[key]:
            raise ValueError(f"corner-ease source {key} differs from current code")

    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    final_state = validate_protocol3_resume_checkpoint(
        final_checkpoint, protocol_config
    )
    if (
        final_checkpoint.get("variant") != case_label
        or final_checkpoint.get("git_identity") != source_identity
        or final_checkpoint.get("hp", {}).get("git_identity") != source_identity
        or int(final_state["completed_updates"]) != SOURCE_UPDATES
    ):
        raise ValueError("corner-ease final checkpoint identity differs")
    param_group_lrs = tuple(
        float(group["lr"])
        for group in final_checkpoint["optimizer_state_dict"]["param_groups"]
    )
    if not param_group_lrs or any(value != 0.001 for value in param_group_lrs):
        raise ValueError("corner-ease source Adam learning rate differs")
    history = list(final_state["validation_history"])
    if [int(row["completed_updates"]) for row in history] != list(
        range(0, SOURCE_UPDATES + 1, VALIDATION_INTERVAL)
    ):
        raise ValueError("corner-ease source validation history is incomplete")
    selection = select_validation_history(history)
    for key in ("status", "best_update", "best_metrics", "stable_intervals"):
        if summary.get(key) != selection[key]:
            raise ValueError(f"corner-ease source selection differs: {key}")
    rebuilt = reconstruct_corner_ease_selection_state(history)
    saved = final_state.get("corner_ease_selection")
    if not isinstance(saved, dict):
        raise ValueError("corner-ease source selector state is missing")
    for key in (
        "pass_streak",
        "stable_best",
        "passing_best",
        "mean_best",
        "stable_intervals",
    ):
        if saved.get(key) != rebuilt[key]:
            raise ValueError(f"corner-ease source selector state differs: {key}")
    current_best = rebuilt["current_streak_best"]
    if current_best is not None and int(current_best["completed_updates"]) not in (
        SOURCE_UPDATES,
        int(selection["best_update"]),
    ):
        raise ValueError("corner-ease trailing streak checkpoint is unavailable")

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    best_state = validate_protocol3_resume_checkpoint(best_checkpoint, protocol_config)
    best_model_hash = state_dict_sha256(best_checkpoint["agent_state_dict"])
    if (
        best_checkpoint.get("variant") != case_label
        or best_checkpoint.get("git_identity") != source_identity
        or int(best_state["completed_updates"]) != int(selection["best_update"])
        or best_model_hash
        != summary["best_checkpoint_audit"]["state_sha256_before"]
    ):
        raise ValueError("corner-ease source best checkpoint identity differs")
    return {
        "case": case,
        "source_status": selection["status"],
        "source_best_update": selection["best_update"],
        "source_best_metrics": selection["best_metrics"],
        "source_stable_intervals": selection["stable_intervals"],
        "source_completed_updates": SOURCE_UPDATES,
        "source_repository_head": source_identity["repository_head"],
        "current_repository_head": current_identity["repository_head"],
        "source_best_checkpoint": str(best_path),
        "source_final_checkpoint": str(final_path),
        "source_best_checkpoint_sha256": expected_hashes[best_path],
        "source_final_checkpoint_sha256": expected_hashes[final_path],
        "source_best_model_state_sha256": best_model_hash,
        "source_final_model_state_sha256": state_dict_sha256(
            final_checkpoint["agent_state_dict"]
        ),
        "source_optimizer_param_group_learning_rates": list(param_group_lrs),
        "source_validation_points": len(history),
        "trailing_pass_streak": rebuilt["pass_streak"],
        "eligible_for_controlled_lr_continuation": True,
    }


def prepare_continuation(
    *,
    repository_root: str | Path,
    continuation_config_path: str | Path,
    source_run_root: str | Path,
    source_evidence_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_continuation_config(continuation_config_path)
    digits = _case_digits(config)
    output = Path(run_root).resolve()
    if output != _resolve_inside(root, config["output"]["directory"]):
        raise ValueError("corner-ease continuation run root changed")
    if output.exists():
        raise FileExistsError(f"continuation run root already exists: {output}")
    expected_evidence = _resolve_inside(root, config["source"]["evidence_directory"])
    evidence = Path(source_evidence_root).resolve()
    if evidence != expected_evidence:
        raise ValueError("corner-ease continuation source evidence root changed")
    source_preflight = _read_json(evidence / "preflight_summary.json")
    workspace_audit = _read_json(evidence / "workspace_audit.json")
    if (
        source_preflight.get("passed") is not True
        or source_preflight.get("training_updates") != 0
        or source_preflight.get("git_identity", {}).get("repository_head")
        != config["source"]["repository_head"]
        or workspace_audit.get("passed") is not True
    ):
        raise ValueError("corner-ease source preflight or workspace audit differs")
    reviews = [
        review_source_case(
            repository_root=root,
            continuation_config_path=continuation_config_path,
            source_run_root=source_run_root,
            case_label=_case_label(digit),
        )
        for digit in digits
    ]
    output.mkdir(parents=True)
    _write_json(output / "continuation_config.json", config)
    _write_json(
        output / "source_review.json",
        {
            "protocol": config["protocol"],
            "run_kind": f"{config['run_kind']}_source_review",
            "source_repository_head": config["source"]["repository_head"],
            "current_repository_head": reviews[0]["current_repository_head"],
            "cases": reviews,
            "eligible_cases": len(reviews),
            "digit8_included": 8 in digits,
            "synchronized_parallel_training": _is_joint_config(config),
            "passed": True,
        },
    )
    _write_json(output / "workspace_audit.json", workspace_audit)
    return {"cases": reviews, "passed": True}


def run_continuation_arm(
    *,
    repository_root: str | Path,
    continuation_config_path: str | Path,
    source_run_root: str | Path,
    run_root: str | Path,
    case_label: str,
    arm_label: str,
    manual_continuation_authorized: bool,
) -> dict[str, Any]:
    if not manual_continuation_authorized:
        raise PermissionError("corner-ease LR continuation requires manual approval")
    root = Path(repository_root).resolve()
    config = load_continuation_config(continuation_config_path)
    source_record = _source_case_record(config, case_label)
    arm = _arm_record(config, arm_label)
    digit = int(source_record["digit"])
    output_root = Path(run_root).resolve()
    if output_root != _resolve_inside(root, config["output"]["directory"]):
        raise ValueError("corner-ease continuation run root changed")
    if not (output_root / "source_review.json").is_file():
        raise ValueError("corner-ease continuation was not prepared")
    output = (output_root / f"digit{digit}" / arm_label).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("corner-ease continuation arm escapes its run root") from error
    if not output.is_dir():
        raise FileNotFoundError("server runner must create the arm log directory")
    unexpected = {path.name for path in output.iterdir()} - {"run.log"}
    if unexpected:
        raise FileExistsError(f"continuation arm is not empty: {sorted(unexpected)}")

    review = review_source_case(
        repository_root=root,
        continuation_config_path=continuation_config_path,
        source_run_root=source_run_root,
        case_label=case_label,
    )
    _, protocol_config = _source_protocol_config(root, config)
    case, best_source, final_source, _ = _source_case_paths(
        Path(source_run_root).resolve(), protocol_config, digit
    )
    resolved = {
        "continuation_config": config,
        "active_case": case,
        "active_arm": arm,
        "source_review": review,
    }
    _write_json(output / "resolved_config.json", resolved)

    from train import (
        _digit_env_dict,
        load_digit_policy_checkpoint,
        train_subsets_base_model,
    )

    hp = _hp_for_case(protocol_config, case)
    arm_run_label = _arm_run_label(digit, arm_label)
    hp.update(
        {
            "stop_after_updates": TARGET_UPDATES,
            "resume_source_repository_head": config["source"]["repository_head"],
            "resume_source_checkpoint_sha256": source_record[
                "final_checkpoint_sha256"
            ],
            "experimental_run_kind": config["run_kind"],
            "experimental_lr_ablation_arm": arm_run_label,
            "experimental_source_learning_rate": config["optimizer"][
                "source_learning_rate"
            ],
            "experimental_resume_learning_rate": arm["learning_rate"],
        }
    )
    training = train_subsets_base_model(
        str(output),
        protocol_config["output"]["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((digit,)),
        resume_checkpoint=str(final_source),
        resume_best_checkpoint=str(best_source),
        target_completed_updates=TARGET_UPDATES,
        manual_resume_authorized=True,
        expected_resume_repository_head=config["source"]["repository_head"],
    )
    if int(training["updates"]) != TARGET_UPDATES:
        raise RuntimeError("corner-ease continuation did not stop at update 8000")
    selection = training.get("corner_ease_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("corner-ease continuation selection is missing")

    best_path = output / protocol_config["output"]["best_checkpoint"]
    best_policy, best_checkpoint = load_digit_policy_checkpoint(
        best_path, expected_variant=case_label
    )
    best_state = validate_protocol3_resume_checkpoint(
        best_checkpoint, protocol_config
    )
    if int(best_state["completed_updates"]) != int(selection["best_update"]):
        raise RuntimeError("continuation best checkpoint update differs")
    workspace_audit = _read_json(output_root / "workspace_audit.json")
    best_audit = _audit_best_checkpoint(
        best_policy, hp, case, workspace_audit, output, selection
    )

    final_path = output / protocol_config["output"]["final_checkpoint"]
    _, final_checkpoint = load_digit_policy_checkpoint(
        final_path, expected_variant=case_label
    )
    final_state = validate_protocol3_resume_checkpoint(
        final_checkpoint, protocol_config
    )
    source_checkpoint = torch.load(
        final_source, map_location="cpu", weights_only=False
    )
    source_state = validate_protocol3_resume_checkpoint(
        source_checkpoint, protocol_config
    )
    final_history = list(final_state["validation_history"])
    if final_history[: len(source_state["validation_history"])] != list(
        source_state["validation_history"]
    ):
        raise RuntimeError("corner-ease continuation changed source validation history")
    continuation_updates = [
        int(row["completed_updates"])
        for row in final_history
        if int(row["completed_updates"]) > SOURCE_UPDATES
    ]
    if continuation_updates != list(
        range(SOURCE_UPDATES + VALIDATION_INTERVAL, TARGET_UPDATES + 1, VALIDATION_INTERVAL)
    ):
        raise RuntimeError("corner-ease continuation validation history is incomplete")
    final_group_lrs = tuple(
        float(group["lr"])
        for group in final_checkpoint["optimizer_state_dict"]["param_groups"]
    )
    checkpoint_complete = bool(
        int(final_state["completed_updates"]) == TARGET_UPDATES
        and final_checkpoint["git_identity"] == hp["git_identity"]
        and final_checkpoint.get("hp", {}).get("experimental_lr_ablation_arm")
        == arm_run_label
        and final_group_lrs
        and all(value == float(arm["learning_rate"]) for value in final_group_lrs)
    )
    engineering_passed = bool(
        checkpoint_complete and best_audit["engineering_passed"]
    )
    result = {
        "protocol": config["protocol"],
        "run_kind": f"{config['run_kind']}_arm",
        "case": case,
        "arm": arm,
        "arm_run_label": arm_run_label,
        "git_identity": hp["git_identity"],
        "source_repository_head": config["source"]["repository_head"],
        "source_completed_updates": SOURCE_UPDATES,
        "completed_updates": TARGET_UPDATES,
        "additional_updates": TARGET_UPDATES - SOURCE_UPDATES,
        "source_status": review["source_status"],
        "source_best_update": review["source_best_update"],
        "source_best_metrics": review["source_best_metrics"],
        "status": selection["status"],
        "best_update": selection["best_update"],
        "best_checkpoint_origin": (
            "source" if int(selection["best_update"]) <= SOURCE_UPDATES else "continuation"
        ),
        "best_metrics": selection["best_metrics"],
        "stable_intervals": selection["stable_intervals"],
        "continuation_validation_points": len(continuation_updates),
        "source_history_preserved": True,
        "effective_optimizer_param_group_learning_rates": list(final_group_lrs),
        "checkpoint_complete": checkpoint_complete,
        "best_checkpoint_audit": best_audit,
        "engineering_passed": engineering_passed,
        "automatic_winner_selection_started": False,
        "automatic_further_continuation_started": False,
        "automatic_second_seed_started": False,
        "digit8_started": digit == 8,
        "synchronized_parallel_training": _is_joint_config(config),
        "automatic_geometry_change_started": False,
        "automatic_loss_change_started": False,
        "formal_full10_started": False,
    }
    _write_json(output / "run_summary.json", result)
    expected_files = {
        "run.log",
        "resolved_config.json",
        "metrics.jsonl",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "best_movement_overlay.png",
        "run_summary.json",
    }
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"continuation arm artifacts differ; actual={sorted(actual_files)}"
        )
    return result


def summarize_continuation(
    *,
    repository_root: str | Path,
    continuation_config_path: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_continuation_config(continuation_config_path)
    digits = _case_digits(config)
    output_names = _output_names(config)
    output = Path(run_root).resolve()
    if output != _resolve_inside(root, config["output"]["directory"]):
        raise ValueError("corner-ease continuation summary root changed")
    results = []
    for digit in digits:
        for arm_label in ARM_LABELS:
            results.append(
                _read_json(
                    output
                    / f"digit{digit}"
                    / arm_label
                    / "run_summary.json"
                )
            )
    expected_labels = [
        _arm_run_label(digit, arm)
        for digit in digits
        for arm in ARM_LABELS
    ]
    if [row.get("arm_run_label") for row in results] != expected_labels:
        raise RuntimeError("corner-ease continuation arms are incomplete or reordered")
    if any(int(row.get("completed_updates", -1)) != TARGET_UPDATES for row in results):
        raise RuntimeError("one or more continuation arms did not reach update 8000")
    current_identity = current_git_identity(root)
    if any(row.get("git_identity") != current_identity for row in results):
        raise RuntimeError("continuation arm implementation Git identity differs")
    engineering_passed = all(row.get("engineering_passed") is True for row in results)
    comparison_rows = []
    for digit in digits:
        control, lower = [
            row for row in results if int(row["case"]["digit"]) == digit
        ]
        control_metrics = control["best_checkpoint_audit"]["movement_metrics"]
        lower_metrics = lower["best_checkpoint_audit"]["movement_metrics"]
        comparison_rows.append(
            {
                "digit": digit,
                "source_status": control["source_status"],
                "lr1e3_status": control["status"],
                "lr3e4_status": lower["status"],
                "lr1e3_best_update": control["best_update"],
                "lr3e4_best_update": lower["best_update"],
                "lr3e4_minus_lr1e3_mean_error": (
                    lower_metrics["normalized_mean_error"]
                    - control_metrics["normalized_mean_error"]
                ),
                "lr3e4_minus_lr1e3_endpoint_error": (
                    lower_metrics["normalized_endpoint_error"]
                    - control_metrics["normalized_endpoint_error"]
                ),
                "lr3e4_minus_lr1e3_absolute_path_error": (
                    abs(lower_metrics["path_length_ratio"] - 1.0)
                    - abs(control_metrics["path_length_ratio"] - 1.0)
                ),
                "lr1e3_engineering_passed": control["engineering_passed"],
                "lr3e4_engineering_passed": lower["engineering_passed"],
            }
        )
    _write_csv(output / output_names["comparison"], comparison_rows)
    summary = {
        "protocol": config["protocol"],
        "run_kind": config["run_kind"],
        "variant": config["variant"],
        "source_repository_head": config["source"]["repository_head"],
        "current_repository_head": current_identity["repository_head"],
        "source_completed_updates": SOURCE_UPDATES,
        "completed_updates": TARGET_UPDATES,
        "additional_updates_per_arm": TARGET_UPDATES - SOURCE_UPDATES,
        "digits": list(digits),
        "arms": results,
        "comparisons": comparison_rows,
        "completed_arms": len(results),
        "engineering_passed": engineering_passed,
        "classification": (
            "manual_lr_comparison_review_required"
            if engineering_passed
            else "engineering_or_safety_failure"
        ),
        "automatic_winner_selection": False,
        "automatic_further_continuation": False,
        "automatic_second_seed": False,
        "digit8_included": 8 in digits,
        "synchronized_parallel_training": _is_joint_config(config),
        "automatic_geometry_change": False,
        "automatic_loss_change": False,
        "formal_full10_started": False,
        "passed": False,
    }
    _write_json(output / output_names["summary"], summary)
    title = (
        "# Protocol3 digit0 / digit8 matched LR continuation report"
        if _is_joint_config(config)
        else "# Protocol3 corner-ease LR continuation report"
    )
    scope_line = (
        "Digits 0 and 8 each resumed in two independent synchronized arms "
        "from state-complete final@6000 to 8000."
        if _is_joint_config(config)
        else "Six PASS_UNSTABLE digits resumed from state-complete final@6000 to 8000."
    )
    lines = [
        title,
        "",
        scope_line,
        "The original 1e-3 rate is retained as a control against 3e-4.",
        "No arm is selected automatically.",
        "",
        "| digit | source | 1e-3 | 3e-4 | mean delta | endpoint delta | abs path-error delta |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['digit']} | {row['source_status']} | "
            f"{row['lr1e3_status']} | {row['lr3e4_status']} | "
            f"{row['lr3e4_minus_lr1e3_mean_error']:.6f} | "
            f"{row['lr3e4_minus_lr1e3_endpoint_error']:.6f} | "
            f"{row['lr3e4_minus_lr1e3_absolute_path_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Negative deltas favor 3e-4. Status, metrics and overlays require manual review together.",
            (
                "Second seeds, further continuation, geometry/loss changes and "
                "formal full10 were not started."
                if _is_joint_config(config)
                else (
                    "Digit 8, second seeds, further continuation and formal "
                    "full10 were not started."
                )
            ),
            "",
        ]
    )
    with (output / output_names["report"]).open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("\n".join(lines))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run-arm", "summarize"))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--continuation-config", required=True)
    parser.add_argument("--source-run-root")
    parser.add_argument("--source-evidence-root")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--case-label")
    parser.add_argument("--arm-label")
    parser.add_argument("--manual-continuation-authorized", action="store_true")
    return parser


def _required(arguments: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(arguments, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")


def main() -> None:
    arguments = _parser().parse_args()
    common = {
        "repository_root": arguments.repository_root,
        "continuation_config_path": arguments.continuation_config,
        "run_root": arguments.run_root,
    }
    if arguments.mode == "prepare":
        _required(arguments, "source_run_root", "source_evidence_root")
        result = prepare_continuation(
            **common,
            source_run_root=arguments.source_run_root,
            source_evidence_root=arguments.source_evidence_root,
        )
        print(f"ELIGIBLE_CASES={len(result['cases'])}")
        print("CORNER_EASE_LR_CONTINUATION_PREPARED=1")
        return
    if arguments.mode == "run-arm":
        _required(arguments, "source_run_root", "case_label", "arm_label")
        result = run_continuation_arm(
            **common,
            source_run_root=arguments.source_run_root,
            case_label=arguments.case_label,
            arm_label=arguments.arm_label,
            manual_continuation_authorized=(
                arguments.manual_continuation_authorized
            ),
        )
        print(f"ARM={result['arm_run_label']}")
        print(f"COMPLETED_UPDATES={result['completed_updates']}")
        print(f"STATUS={result['status']}")
        print(f"ENGINEERING_PASSED={int(result['engineering_passed'])}")
        return
    result = summarize_continuation(**common)
    print(f"CLASSIFICATION={result['classification']}")
    print(f"COMPLETED_ARMS={result['completed_arms']}")
    print(f"COMPLETED_UPDATES={result['completed_updates']}")
    print(f"ENGINEERING_PASSED={int(result['engineering_passed'])}")
    print("AUTOMATIC_WINNER_SELECTION_STARTED=0")
    print("AUTOMATIC_FURTHER_CONTINUATION_STARTED=0")
    print("AUTOMATIC_SECOND_SEED_STARTED=0")
    print(f"DIGIT8_STARTED={int(result['digit8_included'])}")
    print(
        "SYNCHRONIZED_PARALLEL_TRAINING="
        f"{int(result['synchronized_parallel_training'])}"
    )
    print("AUTOMATIC_GEOMETRY_CHANGE_STARTED=0")
    print("AUTOMATIC_LOSS_CHANGE_STARTED=0")
    print("FORMAL_FULL10_STARTED=0")


if __name__ == "__main__":
    main()
