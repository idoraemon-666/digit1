"""Frozen shared-fragment neural analysis for the Protocol3 joint8 RNN.

The module is intentionally split into server-runnable stages.  ``validate``
pins repository, training-config, checkpoint, and model-state identities;
``prepare`` discovers the geometry-only shared fragments; ``collect`` performs
read-only rollouts; ``analyze`` runs the six approved mathematical analyses;
and ``report`` renders the audit report and figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_RUN_KIND = "protocol3_joint8_shared_fragment_analysis"
ANALYSIS_VARIANT = "corner_ease_v3_joint8_shared_fragment_analysis_v1"
ANALYSIS_DIGITS = (1, 2, 3, 4, 5, 6, 7, 9)
STAGE_DIRECTORIES = {
    "validate": "00_identity",
    "prepare": "01_prepare",
    "collect": "02_activity",
    "analyze": "03_metrics",
    "figures": "04_figures",
    "report": "05_report",
}


@dataclass(frozen=True, order=True)
class FragmentOccurrence:
    digit: int
    start_interval: int
    end_interval: int

    @property
    def intervals(self) -> int:
        return self.end_interval - self.start_interval


@dataclass(frozen=True)
class FragmentFamily:
    family_id: str
    intervals: int
    occurrences: tuple[FragmentOccurrence, ...]


@dataclass(frozen=True)
class OccurrencePair:
    pair_id: str
    family_id: str
    first: FragmentOccurrence
    second: FragmentOccurrence
    behavior_stratum: str
    pair_type: str = "shared"
    control_for_pair_id: str = ""


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite a different artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        replaced = True
    finally:
        if replaced and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes_once(path, _canonical_json_bytes(value))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    _write_bytes_once(path, buffer.getvalue().encode("utf-8"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )
    _write_bytes_once(path, payload)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_bytes_once(path, buffer.getvalue())


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"configured path escapes repository: {relative}") from error
    return candidate


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def validate_analysis_config(repository_root: Path, config: Mapping[str, Any]) -> None:
    _require_exact_keys(
        config,
        {
            "protocol", "run_kind", "variant", "training_identity", "checkpoint",
            "condition", "fragment_matching", "collection", "principal_angles",
            "cross_projection", "cca", "procrustes", "fixed_points", "dsa",
            "behavior_strata", "output",
        },
        "analysis config",
    )
    nested_keys = {
        "training_identity": {"repository_head", "training_config", "digits"},
        "checkpoint": {"update", "variant", "sha256"},
        "condition": {
            "direction_index", "delay_index", "delay_steps", "reference_steps",
            "speed_condition",
        },
        "fragment_matching": {
            "allow_translation", "allow_rotation", "allow_scaling", "allow_reflection",
            "allow_reversal", "length_tolerance_m", "turn_tolerance_rad",
            "rigid_residual_tolerance_m", "minimum_discovery_intervals",
            "minimum_analysis_intervals", "corner_base_window_intervals",
            "corner_actual_window_intervals", "expected_family_count",
            "expected_primary_pair_count", "expected_families",
        },
        "collection": {
            "deterministic_reset_seed", "deterministic_batch_size",
            "stochastic_rollouts", "stochastic_batch_size", "stochastic_batch_count",
            "stochastic_seed_base", "environment_observation_noise",
            "deterministic_network_noise", "stochastic_network_noise",
        },
        "principal_angles": {"components", "variance_explained_warning_below"},
        "cross_projection": {"components"},
        "cca": {"pca_components", "canonical_components"},
        "procrustes": {"center", "unit_frobenius_norm", "allow_neural_reflection"},
        "fixed_points": {
            "phases", "interpolation_alphas", "dtype", "optimizer", "learning_rate",
            "maximum_steps", "residual_rms_max", "dedup_activation_rms_max",
            "initial_states",
        },
        "dsa": {
            "n_delays", "delay_interval", "steps_ahead", "rank", "ridge",
            "score_method", "optimizer", "learning_rate", "iterations",
            "initializations",
        },
        "behavior_strata": {"pass_digits", "fail_digits", "allowed_labels"},
        "output": {"directory", "artifact_version"},
    }
    for name, keys in nested_keys.items():
        if not isinstance(config[name], Mapping):
            raise ValueError(f"{name} must be a JSON object")
        _require_exact_keys(config[name], keys, name)
    if config["protocol"] != "digit_writing_original_protocol3":
        raise ValueError("analysis protocol differs from Protocol3")
    if config["run_kind"] != ANALYSIS_RUN_KIND or config["variant"] != ANALYSIS_VARIANT:
        raise ValueError("analysis run identity differs from the frozen v1 design")
    training = config["training_identity"]
    if tuple(training["digits"]) != ANALYSIS_DIGITS:
        raise ValueError("analysis digits differ from frozen joint8 digits")
    if training["repository_head"] != "bd39e09b2203296f459b7d723bd4f9bcacd69ae9":
        raise ValueError("training repository HEAD differs from the accepted checkpoint")
    training_config = _resolve_inside(repository_root, training["training_config"])
    if not training_config.is_file():
        raise FileNotFoundError(training_config)
    if config["checkpoint"] != {
        "update": 61600,
        "variant": "corner_ease_v3_joint8_excluding_digit0_digit8_seed42",
        "sha256": "df0e634021bf24eae1018c2d54325258b20457d67e2db0656921257d3342d345",
    }:
        raise ValueError("checkpoint identity differs from the accepted common checkpoint")
    condition = config["condition"]
    if condition != {
        "direction_index": 0,
        "delay_index": 1,
        "delay_steps": 50,
        "reference_steps": 100,
        "speed_condition": 0,
    }:
        raise ValueError("analysis condition is not direction0/delay50/ref100")
    collection = config["collection"]
    if int(collection["stochastic_rollouts"]) != (
        int(collection["stochastic_batch_size"])
        * int(collection["stochastic_batch_count"])
    ):
        raise ValueError("stochastic rollout count differs from batch product")
    if collection["environment_observation_noise"] is not False:
        raise ValueError("environment observation noise must remain disabled")
    if collection["deterministic_network_noise"] is not False:
        raise ValueError("deterministic rollout must disable network noise")
    if collection["stochastic_network_noise"] is not True:
        raise ValueError("stochastic rollouts must enable network noise")
    expected_collection_numbers = {
        "deterministic_reset_seed": 1042,
        "deterministic_batch_size": 1,
        "stochastic_rollouts": 64,
        "stochastic_batch_size": 8,
        "stochastic_batch_count": 8,
        "stochastic_seed_base": 42042000,
    }
    if any(collection[key] != value for key, value in expected_collection_numbers.items()):
        raise ValueError("activity collection counts or seeds differ from frozen design")
    matching = config["fragment_matching"]
    frozen_transform = {
        "allow_translation": True,
        "allow_rotation": True,
        "allow_scaling": False,
        "allow_reflection": False,
        "allow_reversal": False,
    }
    if {key: matching[key] for key in frozen_transform} != frozen_transform:
        raise ValueError("fragment transform rules differ from accepted definitions")
    if int(matching["corner_actual_window_intervals"]) != 15:
        raise ValueError("corner exclusion window must be 15 intervals per side")
    if int(matching["expected_family_count"]) != 3:
        raise ValueError("expected shared-fragment family count must be three")
    if int(matching["expected_primary_pair_count"]) != 4:
        raise ValueError("expected primary pair count must be four")
    matching_numbers = {
        "length_tolerance_m": 1e-10,
        "turn_tolerance_rad": 1e-9,
        "rigid_residual_tolerance_m": 1e-10,
        "minimum_discovery_intervals": 2,
        "minimum_analysis_intervals": 30,
        "corner_base_window_intervals": 10,
        "corner_actual_window_intervals": 15,
    }
    if any(matching[key] != value for key, value in matching_numbers.items()):
        raise ValueError("fragment matching tolerance or boundary differs from frozen design")
    expected_family_rows = [
        {
            "family_id": "curve_a_shared_core",
            "intervals": 70,
            "occurrences": [[2, 0, 70], [3, 0, 70]],
        },
        {
            "family_id": "curve_b_shared_core",
            "intervals": 80,
            "occurrences": [[3, 100, 180], [5, 150, 230]],
        },
        {
            "family_id": "equal_length_straight_core",
            "intervals": 50,
            "occurrences": [[2, 80, 130], [2, 160, 210], [7, 0, 50]],
        },
    ]
    if matching["expected_families"] != expected_family_rows:
        raise ValueError("expected shared-fragment oracle differs from accepted preflight")
    frozen_analysis_parameters = {
        "principal_angles": {"components": 12, "variance_explained_warning_below": 0.95},
        "cross_projection": {"components": 12},
        "cca": {"pca_components": 10, "canonical_components": 10},
        "procrustes": {
            "center": True,
            "unit_frobenius_norm": True,
            "allow_neural_reflection": True,
        },
        "fixed_points": {
            "phases": [0.0, 0.25, 0.5, 0.75, 1.0],
            "interpolation_alphas": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "dtype": "float64",
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "maximum_steps": 10000,
            "residual_rms_max": 1e-6,
            "dedup_activation_rms_max": 1e-4,
            "initial_states": 64,
        },
        "dsa": {
            "n_delays": 5,
            "delay_interval": 1,
            "steps_ahead": 1,
            "rank": 10,
            "ridge": 1e-8,
            "score_method": "normalized_euclidean",
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "iterations": 200,
            "initializations": ["identity", "odd_permutation"],
        },
    }
    for name, expected_value in frozen_analysis_parameters.items():
        if config[name] != expected_value:
            raise ValueError(f"{name} settings differ from frozen design")
    if tuple(config["behavior_strata"]["pass_digits"]) != (1, 2, 9):
        raise ValueError("behavior PASS stratum differs from common checkpoint audit")
    if tuple(config["behavior_strata"]["fail_digits"]) != (3, 4, 5, 6, 7):
        raise ValueError("behavior FAIL stratum differs from common checkpoint audit")
    if config["behavior_strata"]["allowed_labels"] != [
        "PASS-PASS", "PASS-FAIL", "FAIL-FAIL"
    ]:
        raise ValueError("behavior stratum labels differ from frozen design")
    if config["output"]["artifact_version"] != "protocol3_joint8_shared_fragment_analysis_v1":
        raise ValueError("analysis artifact version differs")
    _resolve_inside(repository_root, config["output"]["directory"])


def _stage_manifest(
    stage: str,
    output_root: Path,
    config_path: Path,
    inputs: Sequence[Path],
    outputs: Sequence[Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "stage": stage,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "inputs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in inputs
        ],
        "outputs": [
            {"path": str(path.relative_to(output_root)), "sha256": _sha256_file(path)}
            for path in outputs
        ],
    }
    if extra:
        value.update(extra)
    return value


def _verify_stage_manifest(path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    for row in manifest.get("inputs", []):
        candidate = Path(row["path"])
        if not candidate.is_file() or _sha256_file(candidate) != row["sha256"]:
            raise RuntimeError(f"stage input hash is stale: {candidate}")
    for row in manifest.get("outputs", []):
        candidate = output_root / row["path"]
        if not candidate.is_file() or _sha256_file(candidate) != row["sha256"]:
            raise RuntimeError(f"stage output hash is stale: {candidate}")
    return manifest


def _parameter_flag_audit(policy: Any) -> dict[str, Any]:
    parameters = list(policy.named_parameters())
    if not parameters:
        raise RuntimeError("loaded policy has no named parameters")
    if any(parameter.grad is not None for _, parameter in parameters):
        raise RuntimeError("loaded policy has unexpected parameter gradients")
    return {
        "parameter_tensor_count": len(parameters),
        "trainable_parameter_tensor_count": sum(
            bool(parameter.requires_grad) for _, parameter in parameters
        ),
        "frozen_parameter_tensor_count": sum(
            not parameter.requires_grad for _, parameter in parameters
        ),
        "frozen_parameter_names": [
            name for name, parameter in parameters if not parameter.requires_grad
        ],
        "parameter_gradients_absent": True,
    }


def validate_checkpoint_identity(
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Load the checkpoint only after its byte hash has been verified."""
    config = _read_json(config_path)
    validate_analysis_config(repository_root, config)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_sha256 != config["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA256 differs from frozen analysis config")

    import torch

    from digit_writing.protocol3_checkpoint import (
        current_git_identity,
        protocol_config_sha256,
        state_dict_sha256,
        validate_protocol3_resume_checkpoint,
    )
    from digit_writing.protocol3_corner_ease_joint8 import validate_config as validate_joint8_config
    from train import load_digit_policy_checkpoint

    training_config_path = _resolve_inside(
        repository_root, config["training_identity"]["training_config"]
    )
    training_config = _read_json(training_config_path)
    validate_joint8_config(repository_root, training_config)
    policy, checkpoint = load_digit_policy_checkpoint(
        checkpoint_path, expected_variant=config["checkpoint"]["variant"]
    )
    state = validate_protocol3_resume_checkpoint(checkpoint, training_config)
    if int(state["completed_updates"]) != int(config["checkpoint"]["update"]):
        raise ValueError("checkpoint update differs from frozen update 61600")
    if int(checkpoint.get("update", -1)) != int(config["checkpoint"]["update"]):
        raise ValueError("checkpoint payload update differs from frozen update 61600")
    if checkpoint["protocol_config"] != training_config:
        raise ValueError("checkpoint embeds a different training config")
    if checkpoint["protocol_config_sha256"] != protocol_config_sha256(training_config):
        raise ValueError("embedded training config hash differs")
    checkpoint_git = dict(checkpoint["git_identity"])
    if checkpoint_git["repository_head"] != config["training_identity"]["repository_head"]:
        raise ValueError("checkpoint training HEAD differs from frozen identity")
    if checkpoint.get("hp", {}).get("git_identity") != checkpoint_git:
        raise ValueError("checkpoint hp and payload Git identities differ")
    current_identity = current_git_identity(repository_root)
    for key in ("branch", "mrnntorch_recorded_head", "mrnntorch_worktree_head"):
        if checkpoint_git[key] != current_identity[key]:
            raise ValueError(f"checkpoint {key} differs from analysis implementation")
    if (
        int(policy.mrnn.total_num_inputs) != 28
        or int(policy.mrnn.total_num_units) != 256
        or int(policy.output_dim) != 6
    ):
        raise ValueError("checkpoint model dimensions differ from 28-256-6")
    model_hash = state_dict_sha256(policy.state_dict())
    raw_model_hash = state_dict_sha256(checkpoint["agent_state_dict"])
    if model_hash != raw_model_hash:
        raise RuntimeError("loaded policy state differs from checkpoint model state")
    parameter_flag_audit = _parameter_flag_audit(policy)
    del policy
    if torch.is_grad_enabled() is False:
        raise RuntimeError("global torch gradient state was unexpectedly changed")

    stage = output_root / STAGE_DIRECTORIES["validate"]
    identity_path = stage / "checkpoint_identity.json"
    frozen_config_path = stage / "frozen_analysis_config.json"
    identity = {
        "protocol": config["protocol"],
        "run_kind": config["run_kind"],
        "analysis_variant": config["variant"],
        "analysis_git_identity": current_identity,
        "training_git_identity": checkpoint_git,
        "training_config_path": str(training_config_path),
        "training_config_sha256": _sha256_file(training_config_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_update": int(state["completed_updates"]),
        "checkpoint_variant": checkpoint["variant"],
        "model_state_sha256": model_hash,
        "parameter_flag_audit": parameter_flag_audit,
        "optimizer_used": False,
    }
    _write_json(frozen_config_path, config)
    _write_json(identity_path, identity)
    manifest_path = stage / "manifest.json"
    manifest = _stage_manifest(
        "validate", output_root, config_path,
        [training_config_path, checkpoint_path], [frozen_config_path, identity_path],
    )
    _write_json(manifest_path, manifest)
    return identity


def _signed_turns(points: np.ndarray) -> np.ndarray:
    edges = np.diff(points, axis=0)
    cross = edges[:-1, 0] * edges[1:, 1] - edges[:-1, 1] * edges[1:, 0]
    dot = np.sum(edges[:-1] * edges[1:], axis=1)
    return np.arctan2(cross, dot)


def _rotation_only_residual(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean(axis=0, keepdims=True)
    second_centered = second - second.mean(axis=0, keepdims=True)
    left, _, right_t = np.linalg.svd(first_centered.T @ second_centered)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    fitted = first_centered @ rotation.T
    return float(np.sqrt(np.mean(np.sum((fitted - second_centered) ** 2, axis=1))))


def _fragment_equal(
    first: np.ndarray,
    second: np.ndarray,
    length_tolerance: float,
    turn_tolerance: float,
    rigid_tolerance: float,
) -> bool:
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2:
        return False
    first_lengths = np.linalg.norm(np.diff(first, axis=0), axis=1)
    second_lengths = np.linalg.norm(np.diff(second, axis=0), axis=1)
    if not np.allclose(first_lengths, second_lengths, rtol=0.0, atol=length_tolerance):
        return False
    if not np.allclose(
        _signed_turns(first), _signed_turns(second), rtol=0.0, atol=turn_tolerance
    ):
        return False
    return _rotation_only_residual(first, second) <= rigid_tolerance


def _eligible_interval_mask(trajectory: Any, half_window: int) -> np.ndarray:
    mask = np.ones(trajectory.movement_intervals, dtype=bool)
    for row in trajectory.corner_ease_boundaries:
        if not bool(row["qualifies"]):
            continue
        corner = int(row["corner_sample_index"])
        mask[max(0, corner - half_window) : min(mask.size, corner + half_window)] = False
    return mask


def _expected_families(config: Mapping[str, Any]) -> tuple[FragmentFamily, ...]:
    return tuple(
        FragmentFamily(
            family_id=str(row["family_id"]),
            intervals=int(row["intervals"]),
            occurrences=tuple(
                FragmentOccurrence(int(digit), int(start), int(end))
                for digit, start, end in row["occurrences"]
            ),
        )
        for row in config["fragment_matching"]["expected_families"]
    )


def _candidate_maximal_matches(
    first_digit: int,
    second_digit: int,
    trajectories: Mapping[int, np.ndarray],
    masks: Mapping[int, np.ndarray],
    minimum_intervals: int,
    length_tolerance: float,
    turn_tolerance: float,
    rigid_tolerance: float,
) -> list[tuple[FragmentOccurrence, FragmentOccurrence]]:
    first = trajectories[first_digit]
    second = trajectories[second_digit]
    first_lengths = np.linalg.norm(np.diff(first, axis=0), axis=1)
    second_lengths = np.linalg.norm(np.diff(second, axis=0), axis=1)
    matches: list[tuple[FragmentOccurrence, FragmentOccurrence]] = []
    for first_start in range(first_lengths.size):
        if not masks[first_digit][first_start]:
            continue
        for second_start in range(second_lengths.size):
            if not masks[second_digit][second_start]:
                continue
            if abs(first_lengths[first_start] - second_lengths[second_start]) > length_tolerance:
                continue
            length = 1
            while (
                first_start + length < first_lengths.size
                and second_start + length < second_lengths.size
                and masks[first_digit][first_start + length]
                and masks[second_digit][second_start + length]
                and abs(
                    first_lengths[first_start + length]
                    - second_lengths[second_start + length]
                ) <= length_tolerance
            ):
                if length >= 1:
                    first_turn = _signed_turns(
                        first[first_start + length - 1 : first_start + length + 2]
                    )[0]
                    second_turn = _signed_turns(
                        second[second_start + length - 1 : second_start + length + 2]
                    )[0]
                    if abs(first_turn - second_turn) > turn_tolerance:
                        break
                length += 1
            if length < minimum_intervals:
                continue
            first_points = first[first_start : first_start + length + 1]
            second_points = second[second_start : second_start + length + 1]
            if not _fragment_equal(
                first_points, second_points, length_tolerance, turn_tolerance, rigid_tolerance
            ):
                continue
            matches.append(
                (
                    FragmentOccurrence(first_digit, first_start, first_start + length),
                    FragmentOccurrence(second_digit, second_start, second_start + length),
                )
            )
    matches.sort(key=lambda pair: (-pair[0].intervals, pair[0], pair[1]))
    primary: list[tuple[FragmentOccurrence, FragmentOccurrence]] = []
    for candidate in matches:
        contained = any(
            candidate[0].digit == accepted[0].digit
            and candidate[1].digit == accepted[1].digit
            and accepted[0].start_interval <= candidate[0].start_interval
            and candidate[0].end_interval <= accepted[0].end_interval
            and accepted[1].start_interval <= candidate[1].start_interval
            and candidate[1].end_interval <= accepted[1].end_interval
            for accepted in primary
        )
        if not contained:
            primary.append(candidate)
    return primary


def _behavior_stratum(first_digit: int, second_digit: int, config: Mapping[str, Any]) -> str:
    passing = set(int(value) for value in config["behavior_strata"]["pass_digits"])
    count = int(first_digit in passing) + int(second_digit in passing)
    return ("FAIL-FAIL", "PASS-FAIL", "PASS-PASS")[count]


def _rotation_category(points: np.ndarray, tolerance: float) -> str:
    turns = _signed_turns(points)
    positive = bool(np.any(turns > tolerance))
    negative = bool(np.any(turns < -tolerance))
    if positive and negative:
        return "mixed_rotation"
    if positive:
        return "counter_clockwise"
    if negative:
        return "clockwise"
    return "straight"


def _quantize_half_away_from_zero(values: np.ndarray, tolerance: float) -> list[int]:
    array = np.asarray(values, dtype=np.float64)
    quantized = np.sign(array) * np.floor(np.abs(array) / tolerance + 0.5)
    return quantized.astype(np.int64).tolist()


def _family_pairs(families: Sequence[FragmentFamily], config: Mapping[str, Any]) -> list[OccurrencePair]:
    pairs: list[OccurrencePair] = []
    for family in families:
        for index, first in enumerate(family.occurrences):
            for second in family.occurrences[index + 1 :]:
                if first.digit == second.digit:
                    continue
                pairs.append(
                    OccurrencePair(
                        pair_id=f"{family.family_id}__d{first.digit}_{first.start_interval}__d{second.digit}_{second.start_interval}",
                        family_id=family.family_id,
                        first=first,
                        second=second,
                        behavior_stratum=_behavior_stratum(first.digit, second.digit, config),
                    )
                )
    return pairs


def _find_matched_controls(
    shared_pairs: Sequence[OccurrencePair],
    trajectories: Mapping[int, np.ndarray],
    masks: Mapping[int, np.ndarray],
    config: Mapping[str, Any],
) -> list[OccurrencePair]:
    matching = config["fragment_matching"]
    length_tolerance = float(matching["length_tolerance_m"])
    turn_tolerance = float(matching["turn_tolerance_rad"])
    rigid_tolerance = float(matching["rigid_residual_tolerance_m"])
    controls: list[OccurrencePair] = []
    seen: set[tuple[str, FragmentOccurrence, FragmentOccurrence]] = set()
    for shared in shared_pairs:
        intervals = shared.first.intervals
        reference_points = trajectories[shared.first.digit][
            shared.first.start_interval : shared.first.end_interval + 1
        ]
        reference_lengths = np.linalg.norm(np.diff(reference_points, axis=0), axis=1)
        reference_category = _rotation_category(reference_points, turn_tolerance)
        first_digit = shared.first.digit
        second_digit = shared.second.digit
        for first_start in range(trajectories[first_digit].shape[0] - intervals):
            first_end = first_start + intervals
            if not np.all(masks[first_digit][first_start:first_end]):
                continue
            first_points = trajectories[first_digit][first_start : first_end + 1]
            first_lengths = np.linalg.norm(np.diff(first_points, axis=0), axis=1)
            if not np.allclose(first_lengths, reference_lengths, rtol=0.0, atol=length_tolerance):
                continue
            if _rotation_category(first_points, turn_tolerance) != reference_category:
                continue
            for second_start in range(trajectories[second_digit].shape[0] - intervals):
                second_end = second_start + intervals
                if not np.all(masks[second_digit][second_start:second_end]):
                    continue
                second_points = trajectories[second_digit][second_start : second_end + 1]
                second_lengths = np.linalg.norm(np.diff(second_points, axis=0), axis=1)
                if not np.allclose(
                    second_lengths, reference_lengths, rtol=0.0, atol=length_tolerance
                ):
                    continue
                if _rotation_category(second_points, turn_tolerance) != reference_category:
                    continue
                if _fragment_equal(
                    first_points,
                    second_points,
                    length_tolerance,
                    turn_tolerance,
                    rigid_tolerance,
                ):
                    continue
                first_occurrence = FragmentOccurrence(first_digit, first_start, first_end)
                second_occurrence = FragmentOccurrence(second_digit, second_start, second_end)
                key = (shared.family_id, first_occurrence, second_occurrence)
                if key in seen:
                    continue
                seen.add(key)
                controls.append(
                    OccurrencePair(
                        pair_id=(
                            f"control_for__{shared.pair_id}__"
                            f"d{first_digit}_{first_start}__d{second_digit}_{second_start}"
                        ),
                        family_id=shared.family_id,
                        first=first_occurrence,
                        second=second_occurrence,
                        behavior_stratum=shared.behavior_stratum,
                        pair_type="control",
                        control_for_pair_id=shared.pair_id,
                    )
                )
    return sorted(controls, key=lambda row: row.pair_id)


def prepare_fragments(
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_analysis_config(repository_root, config)
    _verify_stage_manifest(
        output_root / STAGE_DIRECTORIES["validate"] / "manifest.json", output_root
    )
    identity_path = output_root / STAGE_DIRECTORIES["validate"] / "checkpoint_identity.json"
    identity = _read_json(identity_path)
    if identity["checkpoint_sha256"] != _sha256_file(checkpoint_path):
        raise RuntimeError("validated checkpoint identity is stale")

    from digit_writing.geometry import build_digit_trajectory, load_geometry_config

    training_config = _read_json(
        _resolve_inside(repository_root, config["training_identity"]["training_config"])
    )
    geometry_path = _resolve_inside(repository_root, training_config["geometry_config"])
    geometry = load_geometry_config(geometry_path)
    trajectories: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    matching = config["fragment_matching"]
    for digit in ANALYSIS_DIGITS:
        trajectory = build_digit_trajectory(
            digit,
            geometry,
            int(config["condition"]["reference_steps"]),
            spatial_angle_rad=0.0,
            anchor=(0.0, 0.0),
        )
        trajectories[digit] = np.asarray(trajectory.points, dtype=np.float64)
        masks[digit] = _eligible_interval_mask(
            trajectory, int(matching["corner_actual_window_intervals"])
        )

    expected = _expected_families(config)
    expected_pair_keys: set[tuple[FragmentOccurrence, FragmentOccurrence]] = set()
    for family in expected:
        if family.intervals < int(matching["minimum_analysis_intervals"]):
            raise ValueError("expected family is shorter than analysis minimum")
        for occurrence in family.occurrences:
            if occurrence.intervals != family.intervals:
                raise ValueError("expected occurrence interval count differs from family")
            if not np.all(masks[occurrence.digit][occurrence.start_interval : occurrence.end_interval]):
                raise ValueError(f"expected occurrence intersects a corner exclusion: {occurrence}")
        for index, first in enumerate(family.occurrences):
            for second in family.occurrences[index + 1 :]:
                if first.digit == second.digit:
                    continue
                first_points = trajectories[first.digit][first.start_interval : first.end_interval + 1]
                second_points = trajectories[second.digit][second.start_interval : second.end_interval + 1]
                if not _fragment_equal(
                    first_points,
                    second_points,
                    float(matching["length_tolerance_m"]),
                    float(matching["turn_tolerance_rad"]),
                    float(matching["rigid_residual_tolerance_m"]),
                ):
                    raise RuntimeError(f"configured expected shared fragment failed geometry checks: {family.family_id}")
                expected_pair_keys.add((first, second))

    discovered: list[tuple[FragmentOccurrence, FragmentOccurrence]] = []
    for first_index, first_digit in enumerate(ANALYSIS_DIGITS):
        for second_digit in ANALYSIS_DIGITS[first_index + 1 :]:
            discovered.extend(
                _candidate_maximal_matches(
                    first_digit,
                    second_digit,
                    trajectories,
                    masks,
                    int(matching["minimum_discovery_intervals"]),
                    float(matching["length_tolerance_m"]),
                    float(matching["turn_tolerance_rad"]),
                    float(matching["rigid_residual_tolerance_m"]),
                )
            )
    discovered_eligible = {
        pair for pair in discovered if pair[0].intervals >= int(matching["minimum_analysis_intervals"])
    }
    if discovered_eligible != expected_pair_keys:
        missing = sorted(expected_pair_keys - discovered_eligible)
        unexpected = sorted(discovered_eligible - expected_pair_keys)
        raise RuntimeError(
            f"geometry discovery differs from frozen oracle; missing={missing}, unexpected={unexpected}"
        )

    pairs = _family_pairs(expected, config)
    if len(expected) != int(matching["expected_family_count"]):
        raise RuntimeError("family count differs from frozen expectation")
    if len(pairs) != int(matching["expected_primary_pair_count"]):
        raise RuntimeError("primary occurrence-pair count differs from frozen expectation")
    controls = _find_matched_controls(pairs, trajectories, masks, config)
    stage = output_root / STAGE_DIRECTORIES["prepare"]
    family_path = stage / "shared_fragment_families.json"
    pair_path = stage / "primary_pairs.json"
    control_path = stage / "matched_controls.json"
    catalog_path = stage / "shared_fragment_catalog.csv"
    excluded_path = stage / "excluded_candidates.csv"
    geometry_path_out = stage / "geometry_arrays.npz"
    mask_path = stage / "exclusion_masks.npz"
    _write_json(
        family_path,
        {
            "family_count": len(expected),
            "families": [asdict(family) for family in expected],
            "discovery_candidate_count": len(discovered),
            "analysis_eligible_pair_count": len(discovered_eligible),
        },
    )
    _write_json(pair_path, {"pair_count": len(pairs), "pairs": [asdict(pair) for pair in pairs]})
    _write_json(
        control_path,
        {"control_count": len(controls), "controls": [asdict(pair) for pair in controls]},
    )
    catalog_rows: list[dict[str, Any]] = []
    for family in expected:
        reference = family.occurrences[0]
        reference_points = trajectories[reference.digit][
            reference.start_interval : reference.end_interval + 1
        ]
        signature = {
            "version": "protocol3_shared_fragment_signature_v1",
            "T": family.intervals,
            "lengths": _quantize_half_away_from_zero(
                np.linalg.norm(np.diff(reference_points, axis=0), axis=1),
                float(matching["length_tolerance_m"]),
            ),
            "signed_turns": _quantize_half_away_from_zero(
                _signed_turns(reference_points),
                float(matching["turn_tolerance_rad"]),
            ),
            "rotation_category": _rotation_category(
                reference_points, float(matching["turn_tolerance_rad"])
            ),
        }
        signature_sha256 = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for occurrence in family.occurrences:
            catalog_rows.append(
                {
                    "family_id": family.family_id,
                    "digit": occurrence.digit,
                    "start_interval": occurrence.start_interval,
                    "end_interval": occurrence.end_interval,
                    "T": family.intervals,
                    "rotation_category": signature["rotation_category"],
                    "canonical_signature_sha256": signature_sha256,
                    "behavior_label": (
                        "PASS"
                        if occurrence.digit in config["behavior_strata"]["pass_digits"]
                        else "FAIL"
                    ),
                }
            )
    _write_csv(
        catalog_path,
        [
            "family_id", "digit", "start_interval", "end_interval", "T",
            "rotation_category", "canonical_signature_sha256", "behavior_label",
        ],
        catalog_rows,
    )
    excluded_rows = [
        {
            "digit_a": first.digit,
            "start_a": first.start_interval,
            "end_a": first.end_interval,
            "digit_b": second.digit,
            "start_b": second.start_interval,
            "end_b": second.end_interval,
            "T": first.intervals,
            "reason": "below_minimum_analysis_intervals",
        }
        for first, second in discovered
        if first.intervals < int(matching["minimum_analysis_intervals"])
    ]
    _write_csv(
        excluded_path,
        ["digit_a", "start_a", "end_a", "digit_b", "start_b", "end_b", "T", "reason"],
        excluded_rows,
    )
    _write_npz(geometry_path_out, **{f"digit{digit}": trajectories[digit] for digit in ANALYSIS_DIGITS})
    _write_npz(mask_path, **{f"digit{digit}": masks[digit] for digit in ANALYSIS_DIGITS})
    manifest_path = stage / "manifest.json"
    manifest = _stage_manifest(
        "prepare", output_root, config_path,
        [identity_path],
        [
            family_path, pair_path, control_path, catalog_path, excluded_path,
            geometry_path_out, mask_path,
        ],
        {
            "family_count": len(expected),
            "primary_pair_count": len(pairs),
            "matched_control_count": len(controls),
        },
    )
    _write_json(manifest_path, manifest)
    return manifest


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rollout_batch(
    policy: Any,
    hp: Mapping[str, Any],
    environment: Any,
    *,
    batch_size: int,
    seed: int,
    direction_index: int,
    speed_condition: int,
    delay_index: int,
    network_noise: bool,
) -> dict[str, np.ndarray]:
    import torch

    _seed_everything(seed)
    observation, info = environment.reset(
        testing=False,
        seed=seed,
        options={
            "batch_size": batch_size,
            "reach_conds": np.full(batch_size, direction_index, dtype=np.int64),
            "speed_cond": speed_condition,
            "delay_cond": delay_index,
            "deterministic": True,
        },
    )
    x = torch.zeros((batch_size, int(hp["hid_size"])), dtype=torch.float32)
    h = torch.zeros_like(x)
    arrays: dict[str, list[Any]] = {
        "observation": [],
        "x": [],
        "h": [],
        "actual": [],
        "target": [],
    }
    terminated = False
    timestep = 0
    while not terminated:
        arrays["observation"].append(observation[:, None, :])
        with torch.no_grad():
            x, h, action = policy(observation, x, h, noise=network_noise)
        observation, _, terminated, info = environment.step(timestep, action)
        arrays["x"].append(x[:, None, :])
        arrays["h"].append(h[:, None, :])
        arrays["actual"].append(info["states"]["fingertip"][:, None, :])
        arrays["target"].append(info["goal"][:, None, :])
        timestep += 1
    movement_start, movement_end = (
        int(value) for value in environment.epoch_bounds["movement"]
    )
    if int(environment.delay_time) != 50:
        raise RuntimeError("rollout environment delay differs from frozen 50 steps")
    result: dict[str, np.ndarray] = {}
    for name, pieces in arrays.items():
        tensor = torch.cat(pieces, dim=1)[:, movement_start:movement_end]
        value = tensor.detach().cpu().numpy()
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"non-finite rollout array: {name}")
        result[name] = value
    expected_steps = int(environment.traj.shape[1])
    if any(value.shape[1] != expected_steps for value in result.values()):
        raise RuntimeError("movement arrays do not align with environment trajectory")
    if not np.allclose(
        result["target"], result["target"][0:1], rtol=0.0, atol=1e-8
    ):
        raise RuntimeError("fixed-condition rollout targets differ within batch")
    expected_target = environment.traj
    if hasattr(expected_target, "detach"):
        expected_target = expected_target.detach().cpu().numpy()
    expected_target = np.asarray(expected_target)
    if not np.array_equal(result["target"], expected_target):
        raise RuntimeError("movement target differs from environment.traj")
    return result


def collect_activity(
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_analysis_config(repository_root, config)
    identity_path = output_root / STAGE_DIRECTORIES["validate"] / "checkpoint_identity.json"
    identity = _read_json(identity_path)
    if identity["checkpoint_sha256"] != _sha256_file(checkpoint_path):
        raise RuntimeError("validated checkpoint identity is stale")
    prepare_manifest_path = output_root / STAGE_DIRECTORIES["prepare"] / "manifest.json"
    _verify_stage_manifest(prepare_manifest_path, output_root)

    import motornet as mn

    from digit_writing.protocol3_checkpoint import state_dict_sha256
    from train import DIGIT_ENV_CLASSES, load_digit_policy_checkpoint

    policy, checkpoint = load_digit_policy_checkpoint(
        checkpoint_path, expected_variant=config["checkpoint"]["variant"]
    )
    hp = checkpoint["hp"]
    policy.eval()
    before_hash = state_dict_sha256(policy.state_dict())
    if before_hash != identity["model_state_sha256"]:
        raise RuntimeError("collection model state differs from validated identity")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("collection policy has pre-existing gradients")

    collection = config["collection"]
    condition = config["condition"]
    stage = output_root / STAGE_DIRECTORIES["collect"]
    family_catalog = _read_json(
        output_root / STAGE_DIRECTORIES["prepare"] / "shared_fragment_families.json"
    )
    artifact_paths: list[Path] = []
    digit_records: dict[str, Any] = {}
    for digit in ANALYSIS_DIGITS:
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        environment = DIGIT_ENV_CLASSES[digit](
            effector=effector,
            **hp.get("env_kwargs", {}),
        )
        deterministic = _rollout_batch(
            policy,
            hp,
            environment,
            batch_size=int(collection["deterministic_batch_size"]),
            seed=int(collection["deterministic_reset_seed"]),
            direction_index=int(condition["direction_index"]),
            speed_condition=int(condition["speed_condition"]),
            delay_index=int(condition["delay_index"]),
            network_noise=bool(collection["deterministic_network_noise"]),
        )
        stochastic_pieces: dict[str, list[np.ndarray]] = {
            name: [] for name in deterministic
        }
        stochastic_seeds: list[int] = []
        for batch_index in range(int(collection["stochastic_batch_count"])):
            seed = int(collection["stochastic_seed_base"]) + digit * 100 + batch_index
            stochastic_seeds.append(seed)
            rollout = _rollout_batch(
                policy,
                hp,
                environment,
                batch_size=int(collection["stochastic_batch_size"]),
                seed=seed,
                direction_index=int(condition["direction_index"]),
                speed_condition=int(condition["speed_condition"]),
                delay_index=int(condition["delay_index"]),
                network_noise=bool(collection["stochastic_network_noise"]),
            )
            for name, value in rollout.items():
                stochastic_pieces[name].append(value)
        stochastic = {
            name: np.concatenate(pieces, axis=0)
            for name, pieces in stochastic_pieces.items()
        }
        if any(
            value.shape[0] != int(collection["stochastic_rollouts"])
            for value in stochastic.values()
        ):
            raise RuntimeError("stochastic rollout count differs from frozen design")
        if not np.allclose(
            stochastic["target"], deterministic["target"], rtol=0.0, atol=1e-8
        ):
            raise RuntimeError("deterministic and stochastic target trajectories differ")

        artifact_path = stage / f"digit{digit}_activity.npz"
        stored_names = ("observation", "x", "h", "actual")
        payload = {
            "target": deterministic["target"][0],
            **{
                f"deterministic_{name}": deterministic[name][0]
                for name in stored_names
            },
            **{
                f"stochastic_{name}": stochastic[name]
                for name in stored_names
            },
        }
        _write_npz(artifact_path, **payload)
        artifact_paths.append(artifact_path)
        digit_records[str(digit)] = {
            "path": str(artifact_path.relative_to(output_root)),
            "sha256": _sha256_file(artifact_path),
            "movement_steps": int(deterministic["h"].shape[1]),
            "hidden_size": int(deterministic["h"].shape[2]),
            "full_episode_steps": int(environment.epoch_bounds["hold"][1]),
            "movement_bounds": [
                int(value) for value in environment.epoch_bounds["movement"]
            ],
            "fragment_bounds": [
                [int(occurrence["start_interval"]), int(occurrence["end_interval"])]
                for family in family_catalog["families"]
                for occurrence in family["occurrences"]
                if int(occurrence["digit"]) == digit
            ],
            "deterministic_seed": int(collection["deterministic_reset_seed"]),
            "stochastic_seeds": stochastic_seeds,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in payload.items()
            },
            "array_sha256": {
                name: _sha256_array(value) for name, value in payload.items()
            },
        }

    after_hash = state_dict_sha256(policy.state_dict())
    if after_hash != before_hash:
        raise RuntimeError("policy parameters changed during read-only activity collection")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("activity collection unexpectedly created parameter gradients")
    activity_manifest_path = stage / "activity_collection_manifest.json"
    activity_manifest = {
        "digits": digit_records,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "parameter_gradients_absent": True,
        "optimizer_used": False,
        "environment_observation_noise": False,
        "deterministic_network_noise": False,
        "stochastic_network_noise": True,
    }
    _write_json(activity_manifest_path, activity_manifest)
    manifest_path = stage / "manifest.json"
    manifest = _stage_manifest(
        "collect", output_root, config_path,
        [identity_path, prepare_manifest_path, checkpoint_path],
        [*artifact_paths, activity_manifest_path],
        {"model_state_unchanged": True, "digit_count": len(ANALYSIS_DIGITS)},
    )
    _write_json(manifest_path, manifest)
    return manifest


def _centered_svd(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    matrix = np.asarray(activity, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("activity must be a time-by-unit matrix")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    total = float(np.sum(centered * centered))
    if total <= 0.0:
        raise ValueError("activity has zero centered variance")
    left, singular, right_t = np.linalg.svd(centered, full_matrices=False)
    return left, singular, right_t, total


def principal_angle_metrics(
    first: np.ndarray, second: np.ndarray, components: int
) -> dict[str, Any]:
    _, first_singular, first_right_t, first_total = _centered_svd(first)
    _, second_singular, second_right_t, second_total = _centered_svd(second)
    rank = min(
        int(components),
        int(np.sum(first_singular > np.finfo(np.float64).eps * first_singular[0])),
        int(np.sum(second_singular > np.finfo(np.float64).eps * second_singular[0])),
    )
    if rank < 1:
        raise ValueError("principal-angle numerical rank is zero")
    cosines = np.linalg.svd(
        first_right_t[:rank] @ second_right_t[:rank].T,
        compute_uv=False,
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return {
        "components_requested": int(components),
        "components_used": rank,
        "angles_deg": angles.tolist(),
        "median_angle_deg": float(np.median(angles)),
        "maximum_angle_deg": float(np.max(angles)),
        "variance_explained_first": float(np.sum(first_singular[:rank] ** 2) / first_total),
        "variance_explained_second": float(np.sum(second_singular[:rank] ** 2) / second_total),
    }


def cross_projection_metrics(
    first: np.ndarray, second: np.ndarray, components: int
) -> dict[str, Any]:
    _, first_singular, first_right_t, first_total = _centered_svd(first)
    _, second_singular, second_right_t, second_total = _centered_svd(second)
    rank = min(components, first_right_t.shape[0], second_right_t.shape[0])
    if rank < 1:
        raise ValueError("cross-projection numerical rank is zero")
    first_centered = first - np.mean(first, axis=0, keepdims=True)
    second_centered = second - np.mean(second, axis=0, keepdims=True)
    first_in_second = float(
        np.sum((first_centered @ second_right_t[:rank].T) ** 2) / first_total
    )
    second_in_first = float(
        np.sum((second_centered @ first_right_t[:rank].T) ** 2) / second_total
    )
    first_in_first = float(np.sum(first_singular[:rank] ** 2) / first_total)
    second_in_second = float(np.sum(second_singular[:rank] ** 2) / second_total)
    normalized_first = first_in_second / first_in_first
    normalized_second = second_in_first / second_in_second
    return {
        "components_used": int(rank),
        "first_variance_in_second": first_in_second,
        "second_variance_in_first": second_in_first,
        "first_self_variance": first_in_first,
        "second_self_variance": second_in_second,
        "first_normalized_cross_projection": normalized_first,
        "second_normalized_cross_projection": normalized_second,
        "symmetric_normalized_cross_projection": float(
            0.5 * (normalized_first + normalized_second)
        ),
    }


def _inverse_sqrt_symmetric(matrix: np.ndarray, tolerance: float = 1e-12) -> tuple[np.ndarray, int]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = max(tolerance, tolerance * float(np.max(np.abs(eigenvalues))))
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise ValueError("covariance numerical rank is zero")
    inverse_sqrt = (
        eigenvectors[:, keep]
        * (1.0 / np.sqrt(eigenvalues[keep]))[None, :]
    ) @ eigenvectors[:, keep].T
    return inverse_sqrt, int(np.sum(keep))


def cca_metrics(
    first: np.ndarray,
    second: np.ndarray,
    pca_components: int,
    canonical_components: int,
) -> dict[str, Any]:
    first_left, first_singular, _, _ = _centered_svd(first)
    second_left, second_singular, _, _ = _centered_svd(second)
    pca_rank = min(
        int(pca_components),
        first_left.shape[1],
        second_left.shape[1],
        first.shape[0] - 1,
    )
    if pca_rank < 1:
        raise ValueError("CCA PCA rank is zero")
    first_scores = first_left[:, :pca_rank] * first_singular[:pca_rank]
    second_scores = second_left[:, :pca_rank] * second_singular[:pca_rank]
    first_scores -= first_scores.mean(axis=0, keepdims=True)
    second_scores -= second_scores.mean(axis=0, keepdims=True)
    denominator = max(1, first_scores.shape[0] - 1)
    first_cov = first_scores.T @ first_scores / denominator
    second_cov = second_scores.T @ second_scores / denominator
    cross_cov = first_scores.T @ second_scores / denominator
    first_whitener, first_rank = _inverse_sqrt_symmetric(first_cov)
    second_whitener, second_rank = _inverse_sqrt_symmetric(second_cov)
    correlations = np.linalg.svd(
        first_whitener @ cross_cov @ second_whitener,
        compute_uv=False,
    )
    count = min(int(canonical_components), first_rank, second_rank, correlations.size)
    correlations = np.clip(correlations[:count], 0.0, 1.0)
    return {
        "pca_components_used": pca_rank,
        "canonical_components_used": count,
        "canonical_correlations": correlations.tolist(),
        "mean_canonical_correlation": float(np.mean(correlations)),
        "minimum_canonical_correlation": float(np.min(correlations)),
    }


def procrustes_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    if first_value.shape != second_value.shape or first_value.ndim != 2:
        raise ValueError("Procrustes inputs must have equal time-by-unit shape")
    first_centered = first_value - first_value.mean(axis=0, keepdims=True)
    second_centered = second_value - second_value.mean(axis=0, keepdims=True)
    first_norm = float(np.linalg.norm(first_centered))
    second_norm = float(np.linalg.norm(second_centered))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Procrustes input has zero centered norm")
    first_normalized = first_centered / first_norm
    second_normalized = second_centered / second_norm
    left, _, right_t = np.linalg.svd(
        first_normalized.T @ second_normalized, full_matrices=False
    )
    transform = left @ right_t
    residual = first_normalized @ transform - second_normalized
    return {
        "disparity": float(np.sum(residual * residual)),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "orthogonality_error": float(
            np.linalg.norm(transform.T @ transform - np.eye(transform.shape[0]))
        ),
        "transform_determinant": float(np.linalg.det(transform)),
    }


def _load_pairs(output_root: Path) -> list[OccurrencePair]:
    primary = _read_json(output_root / STAGE_DIRECTORIES["prepare"] / "primary_pairs.json")
    controls = _read_json(output_root / STAGE_DIRECTORIES["prepare"] / "matched_controls.json")
    pairs: list[OccurrencePair] = []
    for row in [*primary["pairs"], *controls["controls"]]:
        pairs.append(
            OccurrencePair(
                pair_id=str(row["pair_id"]),
                family_id=str(row["family_id"]),
                first=FragmentOccurrence(**row["first"]),
                second=FragmentOccurrence(**row["second"]),
                behavior_stratum=str(row["behavior_stratum"]),
                pair_type=str(row.get("pair_type", "shared")),
                control_for_pair_id=str(row.get("control_for_pair_id", "")),
            )
        )
    return pairs


def _load_digit_activity(output_root: Path, digit: int) -> dict[str, np.ndarray]:
    path = output_root / STAGE_DIRECTORIES["collect"] / f"digit{digit}_activity.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _occurrence_activity(
    activity: Mapping[str, np.ndarray],
    occurrence: FragmentOccurrence,
    source: str,
    state: str = "h",
) -> np.ndarray:
    if source == "deterministic":
        value = activity[f"deterministic_{state}"]
    elif source == "stochastic_mean":
        value = activity[f"stochastic_{state}"].mean(axis=0)
    elif source == "stochastic_trials":
        value = activity[f"stochastic_{state}"]
    else:
        raise ValueError(f"unsupported activity source: {source}")
    return value[..., occurrence.start_interval : occurrence.end_interval + 1, :]


def _base_metric_row(pair: OccurrencePair, source: str, metric: str) -> dict[str, Any]:
    return {
        "family_id": pair.family_id,
        "pair_id": pair.pair_id,
        "digit_a": pair.first.digit,
        "digit_b": pair.second.digit,
        "occurrence_a": f"[{pair.first.start_interval},{pair.first.end_interval})",
        "occurrence_b": f"[{pair.second.start_interval},{pair.second.end_interval})",
        "T": pair.first.intervals,
        "behavior_stratum": pair.behavior_stratum,
        "pair_type": pair.pair_type,
        "control_for_pair_id": pair.control_for_pair_id,
        "activity_source": source,
        "metric": metric,
    }


def _run_first_four(
    pairs: Sequence[OccurrencePair],
    activities: Mapping[int, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {
        "principal_angles": [],
        "cross_projection": [],
        "cca": [],
        "procrustes": [],
    }
    for pair in pairs:
        for source in ("deterministic", "stochastic_mean"):
            first = _occurrence_activity(activities[pair.first.digit], pair.first, source)
            second = _occurrence_activity(activities[pair.second.digit], pair.second, source)
            functions = {
                "principal_angles": lambda: principal_angle_metrics(
                    first, second, int(config["principal_angles"]["components"])
                ),
                "cross_projection": lambda: cross_projection_metrics(
                    first, second, int(config["cross_projection"]["components"])
                ),
                "cca": lambda: cca_metrics(
                    first,
                    second,
                    int(config["cca"]["pca_components"]),
                    int(config["cca"]["canonical_components"]),
                ),
                "procrustes": lambda: procrustes_metrics(first, second),
            }
            for metric, function in functions.items():
                row = _base_metric_row(pair, source, metric)
                try:
                    row.update(function())
                    row.update({"valid": True, "failure_reason": ""})
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    row.update({"valid": False, "failure_reason": str(error)})
                results[metric].append(row)
    return results


def round_half_up_nonnegative(value: float) -> int:
    if value < 0.0:
        raise ValueError("round-half-up helper accepts only non-negative values")
    return int(math.floor(value + 0.5))


def _mrnn_one_step(policy: Any, x: Any, observation: Any) -> Any:
    h = policy.mrnn.activation(x)
    next_x, _ = policy.mrnn(
        x,
        observation[:, None, :],
        noise=False,
        h0=h,
    )
    return next_x.squeeze(1)


def _deduplicate_fixed_points(
    x_values: np.ndarray,
    h_values: np.ndarray,
    residuals: np.ndarray,
    tolerance: float,
) -> list[dict[str, Any]]:
    order = np.argsort(residuals, kind="stable")
    clusters: list[dict[str, Any]] = []
    for index in order:
        matching = None
        for cluster in clusters:
            representative = h_values[int(cluster["representative_index"])]
            distance = float(np.sqrt(np.mean((h_values[index] - representative) ** 2)))
            if distance <= tolerance:
                matching = cluster
                break
        if matching is None:
            clusters.append(
                {
                    "representative_index": int(index),
                    "member_indices": [int(index)],
                }
            )
        else:
            matching["member_indices"].append(int(index))
    for cluster in clusters:
        index = int(cluster["representative_index"])
        cluster.update(
            {
                "cluster_size": len(cluster["member_indices"]),
                "residual_rms": float(residuals[index]),
                "x": x_values[index].tolist(),
                "h": h_values[index].tolist(),
            }
        )
    return clusters


def solve_fixed_points(
    policy: Any,
    observation: np.ndarray,
    initial_x: np.ndarray,
    settings: Mapping[str, Any],
    progress_label: str | None = None,
) -> dict[str, Any]:
    import torch

    initial = np.asarray(initial_x, dtype=np.float64)
    if initial.ndim != 2:
        raise ValueError("fixed-point initial states must be state-by-unit")
    if initial.shape[0] < 1:
        raise ValueError("fixed-point solver requires at least one initial state")
    observation_value = np.asarray(observation, dtype=np.float64)
    if observation_value.ndim == 1:
        observation_value = np.repeat(observation_value[None, :], initial.shape[0], axis=0)
    if observation_value.shape[0] == 1:
        observation_value = np.repeat(observation_value, initial.shape[0], axis=0)
    if observation_value.shape[0] != initial.shape[0]:
        raise ValueError("fixed-point observation count differs from initial states")

    x = torch.nn.Parameter(torch.as_tensor(initial, dtype=torch.float64).clone())
    obs = torch.as_tensor(observation_value, dtype=torch.float64)
    optimizer = torch.optim.Adam([x], lr=float(settings["learning_rate"]))
    maximum_steps = int(settings["maximum_steps"])
    threshold = float(settings["residual_rms_max"])
    steps_completed = 0
    for step in range(maximum_steps):
        optimizer.zero_grad(set_to_none=True)
        next_x = _mrnn_one_step(policy, x, obs)
        squared = torch.mean((next_x - x) ** 2, dim=1)
        loss = torch.mean(squared)
        if not torch.isfinite(loss):
            raise FloatingPointError("fixed-point optimization produced non-finite loss")
        loss.backward()
        optimizer.step()
        steps_completed = step + 1
        if progress_label is not None and (
            steps_completed == 1 or steps_completed % 1000 == 0
        ):
            current = torch.sqrt(squared.detach())
            print(
                f"FIXED_POINT_PROGRESS label={progress_label} step={steps_completed} "
                f"converged={int(torch.sum(current <= threshold))}/{initial.shape[0]} "
                f"max_residual={float(torch.max(current)):.9g}",
                flush=True,
            )
        if bool(torch.all(torch.sqrt(squared.detach()) <= threshold)):
            break
    with torch.no_grad():
        next_x = _mrnn_one_step(policy, x, obs)
        residuals = torch.sqrt(torch.mean((next_x - x) ** 2, dim=1))
        h = policy.mrnn.activation(x)
    x_numpy = x.detach().cpu().numpy()
    h_numpy = h.detach().cpu().numpy()
    residual_numpy = residuals.detach().cpu().numpy()
    converged = residual_numpy <= threshold
    clusters = _deduplicate_fixed_points(
        x_numpy[converged],
        h_numpy[converged],
        residual_numpy[converged],
        float(settings["dedup_activation_rms_max"]),
    ) if np.any(converged) else []
    return {
        "steps_completed": steps_completed,
        "initial_state_count": int(initial.shape[0]),
        "converged_count": int(np.sum(converged)),
        "residual_rms": residual_numpy.tolist(),
        "clusters": clusters,
    }


def _fixed_point_stability(policy: Any, observation: np.ndarray, x: Sequence[float]) -> dict[str, Any]:
    import torch

    state = torch.as_tensor(x, dtype=torch.float64).clone().requires_grad_(True)
    obs = torch.as_tensor(observation, dtype=torch.float64)[None, :]

    def mapping(value: Any) -> Any:
        return _mrnn_one_step(policy, value[None, :], obs)[0]

    jacobian = torch.autograd.functional.jacobian(mapping, state, vectorize=True)
    eigenvalues = np.linalg.eigvals(jacobian.detach().cpu().numpy())
    magnitudes = np.abs(eigenvalues)
    return {
        "spectral_radius": float(np.max(magnitudes)),
        "stable": bool(np.max(magnitudes) < 1.0),
        "eigenvalue_magnitude_min": float(np.min(magnitudes)),
        "eigenvalue_magnitude_median": float(np.median(magnitudes)),
        "eigenvalue_magnitude_max": float(np.max(magnitudes)),
    }


def _cluster_set_comparison(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not first or not second:
        return {
            "valid": False,
            "failure_reason": "one or both occurrences have no converged fixed point",
            "first_cluster_count": len(first),
            "second_cluster_count": len(second),
        }
    first_h = np.asarray([row["h"] for row in first], dtype=np.float64)
    second_h = np.asarray([row["h"] for row in second], dtype=np.float64)
    distances = np.sqrt(np.mean((first_h[:, None, :] - second_h[None, :, :]) ** 2, axis=2))
    first_to_second = np.min(distances, axis=1)
    second_to_first = np.min(distances, axis=0)
    first_radii = np.asarray([row["stability"]["spectral_radius"] for row in first])
    second_radii = np.asarray([row["stability"]["spectral_radius"] for row in second])
    closest = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return {
        "valid": True,
        "failure_reason": "",
        "first_cluster_count": len(first),
        "second_cluster_count": len(second),
        "nearest_activation_rms": float(np.min(distances)),
        "symmetric_hausdorff_activation_rms": float(
            max(np.max(first_to_second), np.max(second_to_first))
        ),
        "nearest_stability_radius_difference": float(
            abs(first_radii[closest[0]] - second_radii[closest[1]])
        ),
        "nearest_stability_class_match": bool(
            first[closest[0]]["stability"]["stable"]
            == second[closest[1]]["stability"]["stable"]
        ),
    }


def _run_fixed_point_analysis(
    policy: Any,
    pairs: Sequence[OccurrencePair],
    activities: Mapping[int, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import copy
    import torch

    settings = config["fixed_points"]
    isolated = copy.deepcopy(policy).double().eval()
    for parameter in isolated.parameters():
        parameter.requires_grad_(False)
    occurrence_keys = sorted(
        {pair.first for pair in pairs} | {pair.second for pair in pairs}
    )
    solved: dict[tuple[FragmentOccurrence, float], dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    for occurrence in occurrence_keys:
        print(
            "FIXED_POINT_OCCURRENCE_START "
            f"digit={occurrence.digit} start={occurrence.start_interval} "
            f"end={occurrence.end_interval}",
            flush=True,
        )
        activity = activities[occurrence.digit]
        for phase in settings["phases"]:
            local_index = round_half_up_nonnegative(float(phase) * occurrence.intervals)
            movement_index = occurrence.start_interval + local_index
            observation = activity["deterministic_observation"][movement_index]
            initial_x = activity["stochastic_x"][:, movement_index, :]
            solution = solve_fixed_points(
                isolated,
                observation,
                initial_x,
                settings,
                progress_label=(
                    f"digit{occurrence.digit}_{occurrence.start_interval}_"
                    f"{occurrence.end_interval}_phase{phase}"
                ),
            )
            for cluster in solution["clusters"]:
                cluster["stability"] = _fixed_point_stability(
                    isolated, observation, cluster["x"]
                )
            solved[(occurrence, float(phase))] = solution
            raw_rows.append(
                {
                    "record_type": "occurrence_phase",
                    "digit": occurrence.digit,
                    "start_interval": occurrence.start_interval,
                    "end_interval": occurrence.end_interval,
                    "phase": float(phase),
                    "movement_index": movement_index,
                    **solution,
                }
            )

    comparison_rows: list[dict[str, Any]] = []
    for pair in pairs:
        print(f"FIXED_POINT_PAIR_START pair={pair.pair_id}", flush=True)
        for phase in settings["phases"]:
            first_solution = solved[(pair.first, float(phase))]
            second_solution = solved[(pair.second, float(phase))]
            row = _base_metric_row(
                pair, "deterministic_observation_stochastic_initial_states", "fixed_points"
            )
            row["phase"] = float(phase)
            row.update(
                _cluster_set_comparison(
                    first_solution["clusters"], second_solution["clusters"]
                )
            )
            comparison_rows.append(row)

        midpoint_first = pair.first.start_interval + round_half_up_nonnegative(
            0.5 * pair.first.intervals
        )
        midpoint_second = pair.second.start_interval + round_half_up_nonnegative(
            0.5 * pair.second.intervals
        )
        first_activity = activities[pair.first.digit]
        second_activity = activities[pair.second.digit]
        source_midpoint_clusters = solved[(pair.first, 0.5)]["clusters"]
        previous_cluster_x: np.ndarray | None = (
            np.asarray(source_midpoint_clusters[0]["x"], dtype=np.float64)
            if source_midpoint_clusters
            else None
        )
        interpolation_records: list[dict[str, Any]] = []
        for alpha in settings["interpolation_alphas"]:
            alpha_value = float(alpha)
            observation = (
                (1.0 - alpha_value)
                * first_activity["deterministic_observation"][midpoint_first]
                + alpha_value
                * second_activity["deterministic_observation"][midpoint_second]
            )
            raw_initials = (
                (1.0 - alpha_value) * first_activity["stochastic_x"][:, midpoint_first, :]
                + alpha_value * second_activity["stochastic_x"][:, midpoint_second, :]
            )
            initial_x = raw_initials
            if previous_cluster_x is not None:
                initial_x = np.concatenate([previous_cluster_x[None, :], raw_initials], axis=0)
            solution = solve_fixed_points(
                isolated,
                observation,
                initial_x,
                settings,
                progress_label=f"{pair.pair_id}_alpha{alpha_value}",
            )
            for cluster in solution["clusters"]:
                cluster["stability"] = _fixed_point_stability(
                    isolated, observation, cluster["x"]
                )
            chosen_index: int | None = None
            if solution["clusters"]:
                if previous_cluster_x is None:
                    chosen_index = 0
                else:
                    previous_h = np.asarray(
                        isolated.mrnn.activation(
                            torch.as_tensor(previous_cluster_x, dtype=torch.float64)
                        ).detach().cpu().numpy()
                    )
                    chosen_index = int(
                        np.argmin(
                            [
                                np.sqrt(np.mean((np.asarray(cluster["h"]) - previous_h) ** 2))
                                for cluster in solution["clusters"]
                            ]
                        )
                    )
                previous_cluster_x = np.asarray(
                    solution["clusters"][chosen_index]["x"], dtype=np.float64
                )
            interpolation_records.append(
                {
                    "alpha": alpha_value,
                    "chosen_cluster_index": chosen_index,
                    "solution": solution,
                }
            )
        target_midpoint_clusters = solved[(pair.second, 0.5)]["clusters"]
        continuity = bool(source_midpoint_clusters and target_midpoint_clusters) and all(
            row["chosen_cluster_index"] is not None for row in interpolation_records
        )
        chosen_clusters = [
            row["solution"]["clusters"][row["chosen_cluster_index"]]
            for row in interpolation_records
            if row["chosen_cluster_index"] is not None
        ]
        interpolation_summary: dict[str, Any] = {
            "record_type": "pair_midpoint_interpolation",
            "pair_id": pair.pair_id,
            "family_id": pair.family_id,
            "branch_continuous": continuity,
            "records": interpolation_records,
        }
        if continuity and chosen_clusters:
            first_cluster = chosen_clusters[0]
            last_cluster = chosen_clusters[-1]
            interpolation_summary.update(
                {
                    "endpoint_activation_rms": float(
                        np.sqrt(
                            np.mean(
                                (
                                    np.asarray(first_cluster["h"])
                                    - np.asarray(last_cluster["h"])
                                ) ** 2
                            )
                        )
                    ),
                    "endpoint_stability_difference": float(
                        abs(
                            first_cluster["stability"]["spectral_radius"]
                            - last_cluster["stability"]["spectral_radius"]
                        )
                    ),
                    "endpoint_stability_class_match": bool(
                        first_cluster["stability"]["stable"]
                        == last_cluster["stability"]["stable"]
                    ),
                }
            )
        raw_rows.append(interpolation_summary)
        interpolation_row = _base_metric_row(
            pair, "deterministic_observation_stochastic_initial_states", "fixed_points"
        )
        interpolation_row["phase"] = "midpoint_interpolation"
        interpolation_row["branch_continuous"] = continuity
        if continuity and chosen_clusters:
            interpolation_row.update(
                {
                    "valid": True,
                    "failure_reason": "",
                    "endpoint_activation_rms": interpolation_summary[
                        "endpoint_activation_rms"
                    ],
                    "endpoint_stability_difference": interpolation_summary[
                        "endpoint_stability_difference"
                    ],
                    "endpoint_stability_class_match": interpolation_summary[
                        "endpoint_stability_class_match"
                    ],
                }
            )
        else:
            interpolation_row.update(
                {
                    "valid": False,
                    "failure_reason": "fixed-point interpolation branch is discontinuous",
                }
            )
        comparison_rows.append(interpolation_row)
    return comparison_rows, raw_rows


def delay_embed_trials(
    activity: np.ndarray, n_delays: int, delay_interval: int
) -> np.ndarray:
    value = np.asarray(activity, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("DSA activity must be trial-by-time-by-unit")
    embedded_time = value.shape[1] - (n_delays - 1) * delay_interval
    if embedded_time < 2:
        raise ValueError("DSA activity is too short for delay embedding")
    return np.concatenate(
        [
            value[:, delay * delay_interval : delay * delay_interval + embedded_time, :]
            for delay in range(n_delays)
        ],
        axis=2,
    )


def fit_dsa_linear_system(
    activity: np.ndarray,
    *,
    n_delays: int,
    delay_interval: int,
    steps_ahead: int,
    rank: int,
    ridge: float,
) -> dict[str, Any]:
    embedded = delay_embed_trials(activity, n_delays, delay_interval)
    trials, embedded_time, _ = embedded.shape
    flattened = embedded.reshape(trials * embedded_time, -1)
    _, singular, right_coordinates = np.linalg.svd(flattened.T, full_matrices=False)
    available_rank = min(
        int(rank),
        int(np.sum(singular > np.finfo(np.float64).eps * singular[0])),
    )
    if available_rank < 1:
        raise ValueError("DSA delay embedding has zero numerical rank")
    coordinates = right_coordinates[:available_rank].T.reshape(
        trials, embedded_time, available_rank
    )
    if steps_ahead < 1 or steps_ahead >= embedded_time:
        raise ValueError("DSA steps_ahead is outside embedded trial length")
    minus = coordinates[:, :-steps_ahead, :].reshape(-1, available_rank)
    plus = coordinates[:, steps_ahead:, :].reshape(-1, available_rank)
    gram = minus.T @ minus + float(ridge) * np.eye(available_rank)
    cross = minus.T @ plus
    system = np.linalg.solve(gram, cross).T
    return {
        "system": system,
        "singular_values": singular.tolist(),
        "rank_used": available_rank,
        "trial_count": trials,
        "embedded_time": embedded_time,
        "regression_samples": int(minus.shape[0]),
        "minus": minus,
        "plus": plus,
    }


def pavf_distance(
    first_system: np.ndarray,
    second_system: np.ndarray,
    *,
    learning_rate: float,
    iterations: int,
) -> dict[str, Any]:
    import torch

    first = np.asarray(first_system, dtype=np.float64)
    second = np.asarray(second_system, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("PAVF systems must be equal square matrices")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("PAVF system has zero Frobenius norm")
    first_tensor = torch.as_tensor(first / first_norm, dtype=torch.float64)
    second_tensor = torch.as_tensor(second / second_norm, dtype=torch.float64)
    dimension = first.shape[0]
    identity = torch.eye(dimension, dtype=torch.float64)
    odd_permutation = torch.eye(dimension, dtype=torch.float64)
    if dimension == 1:
        odd_permutation[0, 0] = -1.0
    else:
        odd_permutation[[0, 1]] = odd_permutation[[1, 0]]
    records: list[dict[str, Any]] = []
    for initialization in ("identity", "odd_permutation"):
        raw = torch.nn.Parameter(torch.zeros((dimension, dimension), dtype=torch.float64))
        optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
        loss_curve: list[float] = []
        for _ in range(int(iterations)):
            optimizer.zero_grad(set_to_none=True)
            skew = raw - raw.T
            rotation = torch.linalg.solve(identity + skew, identity - skew)
            transform = rotation if initialization == "identity" else rotation @ odd_permutation
            residual = first_tensor - transform @ second_tensor @ transform.T
            objective = torch.sum(residual * residual)
            if not torch.isfinite(objective):
                raise FloatingPointError("PAVF optimization produced non-finite loss")
            objective.backward()
            optimizer.step()
            loss_curve.append(float(torch.sqrt(objective.detach())))
        with torch.no_grad():
            skew = raw - raw.T
            rotation = torch.linalg.solve(identity + skew, identity - skew)
            transform = rotation if initialization == "identity" else rotation @ odd_permutation
            residual = first_tensor - transform @ second_tensor @ transform.T
            final_loss = float(torch.linalg.matrix_norm(residual, ord="fro"))
            orthogonality_error = float(
                torch.linalg.matrix_norm(transform.T @ transform - identity, ord="fro")
            )
            determinant = float(torch.linalg.det(transform))
        records.append(
            {
                "initialization": initialization,
                "loss_curve": loss_curve,
                "final_distance": final_loss,
                "orthogonality_error": orthogonality_error,
                "transform_determinant": determinant,
            }
        )
    best = min(records, key=lambda row: row["final_distance"])
    return {
        "distance": float(best["final_distance"]),
        "selected_initialization": best["initialization"],
        "initializations": records,
    }


def _run_dsa(
    pairs: Sequence[OccurrencePair],
    activities: Mapping[int, Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = config["dsa"]
    occurrence_keys = sorted(
        {pair.first for pair in pairs} | {pair.second for pair in pairs}
    )
    systems: dict[FragmentOccurrence, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []
    for occurrence in occurrence_keys:
        print(
            "DSA_FIT_START "
            f"digit={occurrence.digit} start={occurrence.start_interval} "
            f"end={occurrence.end_interval}",
            flush=True,
        )
        activity = _occurrence_activity(
            activities[occurrence.digit], occurrence, "stochastic_trials"
        )
        fitted = fit_dsa_linear_system(
            activity,
            n_delays=int(settings["n_delays"]),
            delay_interval=int(settings["delay_interval"]),
            steps_ahead=int(settings["steps_ahead"]),
            rank=int(settings["rank"]),
            ridge=float(settings["ridge"]),
        )
        systems[occurrence] = fitted
        audits.append(
            {
                "digit": occurrence.digit,
                "start_interval": occurrence.start_interval,
                "end_interval": occurrence.end_interval,
                "rank_used": fitted["rank_used"],
                "trial_count": fitted["trial_count"],
                "embedded_time": fitted["embedded_time"],
                "regression_samples": fitted["regression_samples"],
                "singular_values": fitted["singular_values"],
                "system": fitted["system"].tolist(),
            }
        )
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        print(f"DSA_PAIR_START pair={pair.pair_id}", flush=True)
        row = _base_metric_row(pair, "stochastic_trials", "dsa")
        first = systems[pair.first]
        second = systems[pair.second]
        if first["rank_used"] != second["rank_used"]:
            row.update(
                {
                    "valid": False,
                    "failure_reason": "paired DSA systems have different numerical ranks",
                }
            )
        else:
            try:
                row.update(
                    pavf_distance(
                        first["system"],
                        second["system"],
                        learning_rate=float(settings["learning_rate"]),
                        iterations=int(settings["iterations"]),
                    )
                )
                row.update({"valid": True, "failure_reason": ""})
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                row.update({"valid": False, "failure_reason": str(error)})
        rows.append(row)
    return rows, audits


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "family_id", "pair_id", "digit_a", "digit_b", "occurrence_a",
        "occurrence_b", "T", "behavior_stratum", "pair_type", "activity_source",
        "control_for_pair_id", "metric", "valid", "failure_reason",
    ]
    keys = set().union(*(set(row) for row in rows)) if rows else set(preferred)
    fieldnames = [name for name in preferred if name in keys] + sorted(keys - set(preferred))
    _write_csv(
        path,
        fieldnames,
        [{name: _csv_value(value) for name, value in row.items()} for row in rows],
    )


def _control_evidence(
    pairs: Sequence[OccurrencePair],
    metric_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    shared_pairs = [pair for pair in pairs if pair.pair_type == "shared"]
    controls = [pair for pair in pairs if pair.pair_type == "control"]
    if not controls:
        return {
            "status": "descriptive_only_no_matched_control",
            "pair_decisions": [],
            "family_decisions": [],
        }
    specifications = {
        "principal_angles": ("median_angle_deg", "lower", "stochastic_mean"),
        "cross_projection": (
            "symmetric_normalized_cross_projection", "upper", "stochastic_mean"
        ),
        "cca": ("mean_canonical_correlation", "upper", "stochastic_mean"),
        "procrustes": ("disparity", "lower", "stochastic_mean"),
        "dsa": ("distance", "lower", "stochastic_trials"),
    }
    pair_decisions: list[dict[str, Any]] = []
    for pair in shared_pairs:
        tool_decisions: dict[str, Any] = {}
        for metric, (field, direction, source) in specifications.items():
            shared_candidates = [
                row for row in metric_rows[metric]
                if row["pair_id"] == pair.pair_id
                and row["activity_source"] == source
                and bool(row.get("valid"))
            ]
            control_values = [
                float(row[field]) for row in metric_rows[metric]
                if row.get("control_for_pair_id") == pair.pair_id
                and row["activity_source"] == source
                and bool(row.get("valid"))
                and field in row
            ]
            if len(shared_candidates) != 1 or not control_values:
                tool_decisions[metric] = {
                    "valid": False,
                    "reason": "missing valid shared metric or matched-control distribution",
                }
                continue
            shared_value = float(shared_candidates[0][field])
            quantile = 0.05 if direction == "lower" else 0.95
            threshold = float(np.quantile(control_values, quantile))
            passed = shared_value <= threshold if direction == "lower" else shared_value >= threshold
            tool_decisions[metric] = {
                "valid": True,
                "field": field,
                "direction": direction,
                "shared_value": shared_value,
                "control_count": len(control_values),
                "control_extreme_5_percent_threshold": threshold,
                "passed": bool(passed),
            }

        shared_fixed = [
            row for row in metric_rows["fixed_points"]
            if row["pair_id"] == pair.pair_id
            and row.get("phase") == "midpoint_interpolation"
            and bool(row.get("valid"))
        ]
        control_fixed = [
            row for row in metric_rows["fixed_points"]
            if row.get("control_for_pair_id") == pair.pair_id
            and row.get("phase") == "midpoint_interpolation"
            and bool(row.get("valid"))
        ]
        if len(shared_fixed) == 1 and control_fixed:
            endpoint_threshold = float(
                np.quantile([row["endpoint_activation_rms"] for row in control_fixed], 0.05)
            )
            stability_threshold = float(
                np.quantile(
                    [row["endpoint_stability_difference"] for row in control_fixed], 0.05
                )
            )
            row = shared_fixed[0]
            tool_decisions["fixed_points"] = {
                "valid": True,
                "endpoint_activation_rms": float(row["endpoint_activation_rms"]),
                "endpoint_control_threshold": endpoint_threshold,
                "endpoint_stability_difference": float(
                    row["endpoint_stability_difference"]
                ),
                "stability_control_threshold": stability_threshold,
                "branch_continuous": bool(row["branch_continuous"]),
                "endpoint_stability_class_match": bool(
                    row["endpoint_stability_class_match"]
                ),
                "passed": bool(
                    row["endpoint_activation_rms"] <= endpoint_threshold
                    and row["endpoint_stability_difference"] <= stability_threshold
                    and row["branch_continuous"]
                    and row["endpoint_stability_class_match"]
                ),
            }
        else:
            tool_decisions["fixed_points"] = {
                "valid": False,
                "reason": "missing valid shared fixed-point result or matched controls",
            }

        def group_pass(first: str, second: str) -> bool:
            return bool(
                tool_decisions[first].get("valid")
                and tool_decisions[first].get("passed")
                and tool_decisions[second].get("valid")
                and tool_decisions[second].get("passed")
            )

        groups = {
            "population_activity_direction": group_pass(
                "principal_angles", "cross_projection"
            ),
            "population_trajectory_geometry": group_pass("cca", "procrustes"),
            "dynamical_law": group_pass("fixed_points", "dsa"),
        }
        pair_decisions.append(
            {
                "pair_id": pair.pair_id,
                "family_id": pair.family_id,
                "tools": tool_decisions,
                "groups": groups,
                "all_three_groups_consistent": all(groups.values()),
            }
        )
    family_decisions = []
    for family_id in sorted({pair.family_id for pair in shared_pairs}):
        decisions = [row for row in pair_decisions if row["family_id"] == family_id]
        family_decisions.append(
            {
                "family_id": family_id,
                "shared_pair_count": len(decisions),
                "all_pairs_have_all_three_groups": bool(
                    decisions and all(row["all_three_groups_consistent"] for row in decisions)
                ),
            }
        )
    return {
        "status": "matched_control_inference_completed",
        "pair_decisions": pair_decisions,
        "family_decisions": family_decisions,
    }


def analyze_activity(
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_analysis_config(repository_root, config)
    identity_path = output_root / STAGE_DIRECTORIES["validate"] / "checkpoint_identity.json"
    identity = _read_json(identity_path)
    if identity["checkpoint_sha256"] != _sha256_file(checkpoint_path):
        raise RuntimeError("validated checkpoint identity is stale")
    prepare_manifest_path = output_root / STAGE_DIRECTORIES["prepare"] / "manifest.json"
    collect_manifest_path = output_root / STAGE_DIRECTORIES["collect"] / "manifest.json"
    _verify_stage_manifest(prepare_manifest_path, output_root)
    _verify_stage_manifest(collect_manifest_path, output_root)
    activity_manifest = _read_json(
        output_root / STAGE_DIRECTORIES["collect"] / "activity_collection_manifest.json"
    )
    if activity_manifest["model_state_sha256_after"] != identity["model_state_sha256"]:
        raise RuntimeError("activity manifest model hash differs from identity")
    pairs = _load_pairs(output_root)
    activities = {
        digit: _load_digit_activity(output_root, digit) for digit in ANALYSIS_DIGITS
    }

    from digit_writing.protocol3_checkpoint import state_dict_sha256
    from train import load_digit_policy_checkpoint

    policy, _ = load_digit_policy_checkpoint(
        checkpoint_path, expected_variant=config["checkpoint"]["variant"]
    )
    policy.eval()
    before_hash = state_dict_sha256(policy.state_dict())
    first_four = _run_first_four(pairs, activities, config)
    fixed_rows, fixed_raw = _run_fixed_point_analysis(policy, pairs, activities, config)
    dsa_rows, dsa_audits = _run_dsa(pairs, activities, config)
    after_hash = state_dict_sha256(policy.state_dict())
    if before_hash != after_hash or after_hash != identity["model_state_sha256"]:
        raise RuntimeError("original model state changed during analysis")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("original analysis policy unexpectedly has parameter gradients")

    stage = output_root / STAGE_DIRECTORIES["analyze"]
    metric_rows = {**first_four, "fixed_points": fixed_rows, "dsa": dsa_rows}
    output_paths: list[Path] = []
    for metric, rows in metric_rows.items():
        path = stage / f"{metric}.csv"
        _write_rows_csv(path, rows)
        output_paths.append(path)
    fixed_raw_path = stage / "fixed_points.jsonl"
    dsa_raw_path = stage / "dsa_systems.jsonl"
    _write_jsonl(fixed_raw_path, fixed_raw)
    _write_jsonl(dsa_raw_path, dsa_audits)
    output_paths.extend((fixed_raw_path, dsa_raw_path))
    evidence_path = stage / "control_evidence.json"
    evidence = _control_evidence(pairs, metric_rows)
    _write_json(evidence_path, evidence)
    output_paths.append(evidence_path)
    behavior_path = stage / "behavior_stratified_summary.json"
    behavior_summary: dict[str, Any] = {
        "strata_kept_separate": True,
        "pass_digits": config["behavior_strata"]["pass_digits"],
        "fail_digits": config["behavior_strata"]["fail_digits"],
        "strata": {},
    }
    for stratum in config["behavior_strata"]["allowed_labels"]:
        stratum_pairs = [pair for pair in pairs if pair.behavior_stratum == stratum]
        behavior_summary["strata"][stratum] = {
            "shared_pair_ids": [pair.pair_id for pair in stratum_pairs if pair.pair_type == "shared"],
            "control_pair_ids": [pair.pair_id for pair in stratum_pairs if pair.pair_type == "control"],
            "metric_rows": {
                metric: sum(row["behavior_stratum"] == stratum for row in rows)
                for metric, rows in metric_rows.items()
            },
            "valid_metric_rows": {
                metric: sum(
                    row["behavior_stratum"] == stratum and bool(row.get("valid"))
                    for row in rows
                )
                for metric, rows in metric_rows.items()
            },
        }
    _write_json(behavior_path, behavior_summary)
    output_paths.append(behavior_path)
    valid_counts = {
        metric: sum(bool(row.get("valid")) for row in rows)
        for metric, rows in metric_rows.items()
    }
    shared_pair_count = sum(pair.pair_type == "shared" for pair in pairs)
    control_count = sum(pair.pair_type == "control" for pair in pairs)
    summary_path = stage / "analysis_summary.json"
    summary = {
        "status": evidence["status"],
        "shared_family_count": 3,
        "shared_pair_count": shared_pair_count,
        "matched_control_count": control_count,
        "metric_row_counts": {metric: len(rows) for metric, rows in metric_rows.items()},
        "valid_metric_row_counts": valid_counts,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "six_tools_only": [
            "principal_angles", "cross_projection", "cca", "procrustes",
            "fixed_points", "dsa",
        ],
        "dsa_uses_stochastic_trials_only": True,
        "no_cross_digit_stochastic_trial_pairing": True,
    }
    _write_json(summary_path, summary)
    output_paths.append(summary_path)
    manifest_path = stage / "manifest.json"
    manifest = _stage_manifest(
        "analyze", output_root, config_path,
        [identity_path, prepare_manifest_path, collect_manifest_path, checkpoint_path],
        output_paths,
        {"model_state_unchanged": True, "status": summary["status"]},
    )
    _write_json(manifest_path, manifest)
    return manifest


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _render_family_tool_figure(
    family_id: str,
    metric: str,
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["family_id"] == family_id and row["valid"] == "True"]
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if not selected:
        axis.text(0.5, 0.5, "No valid result", ha="center", va="center")
        axis.set_axis_off()
    elif metric == "principal_angles":
        for row in selected:
            axis.plot(
                json.loads(row["angles_deg"]),
                marker="o",
                label=f'{row["pair_type"]}:{row["activity_source"]}:{row["pair_id"]}',
            )
        axis.set_xlabel("principal-angle index")
        axis.set_ylabel("angle (degrees; lower is more similar)")
    elif metric == "cca":
        for row in selected:
            axis.plot(
                json.loads(row["canonical_correlations"]),
                marker="o",
                label=f'{row["pair_type"]}:{row["activity_source"]}:{row["pair_id"]}',
            )
        axis.set_xlabel("canonical component")
        axis.set_ylabel("canonical correlation (higher is more similar)")
        axis.set_ylim(0.0, 1.05)
    else:
        field = {
            "cross_projection": "symmetric_normalized_cross_projection",
            "procrustes": "disparity",
            "fixed_points": "nearest_activation_rms",
            "dsa": "distance",
        }[metric]
        usable = [row for row in selected if row.get(field, "") not in ("", None)]
        labels = [
            f'{row["pair_type"]}:{row["pair_id"]}'
            + (f':p{row.get("phase")}' if metric == "fixed_points" else "")
            for row in usable
        ]
        values = [float(row[field]) for row in usable]
        axis.bar(np.arange(len(values)), values)
        axis.set_xticks(np.arange(len(values)), labels, rotation=75, ha="right", fontsize=7)
        ylabel = {
            "cross_projection": "symmetric normalized cross-projection (higher is more similar)",
            "procrustes": "disparity (lower is more similar)",
            "fixed_points": "nearest fixed-point activation RMS (lower is more similar)",
            "dsa": "DSA distance (lower is more similar)",
        }[metric]
        axis.set_ylabel(ylabel)
    axis.set_title(f"{family_id}: {metric}")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=6, loc="best")
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=160,
        metadata={"Software": "Protocol3 shared-fragment analysis"},
    )
    plt.close(figure)
    _write_bytes_once(path, buffer.getvalue())


def render_report(
    repository_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_analysis_config(repository_root, config)
    identity_path = output_root / STAGE_DIRECTORIES["validate"] / "checkpoint_identity.json"
    identity = _read_json(identity_path)
    if identity["checkpoint_sha256"] != _sha256_file(checkpoint_path):
        raise RuntimeError("validated checkpoint identity is stale")
    _verify_stage_manifest(
        output_root / STAGE_DIRECTORIES["validate"] / "manifest.json", output_root
    )
    _verify_stage_manifest(
        output_root / STAGE_DIRECTORIES["prepare"] / "manifest.json", output_root
    )
    _verify_stage_manifest(
        output_root / STAGE_DIRECTORIES["collect"] / "manifest.json", output_root
    )
    analysis_manifest_path = output_root / STAGE_DIRECTORIES["analyze"] / "manifest.json"
    _verify_stage_manifest(analysis_manifest_path, output_root)
    metric_stage = output_root / STAGE_DIRECTORIES["analyze"]
    metrics = (
        "principal_angles", "cross_projection", "cca", "procrustes",
        "fixed_points", "dsa",
    )
    rows = {
        metric: _read_csv_rows(metric_stage / f"{metric}.csv") for metric in metrics
    }
    families_value = _read_json(
        output_root / STAGE_DIRECTORIES["prepare"] / "shared_fragment_families.json"
    )
    family_ids = [row["family_id"] for row in families_value["families"]]
    figure_stage = output_root / STAGE_DIRECTORIES["figures"]
    figure_paths: list[Path] = []
    for family_id in family_ids:
        for metric in metrics:
            path = figure_stage / f"{family_id}__{metric}.png"
            _render_family_tool_figure(family_id, metric, rows[metric], path)
            figure_paths.append(path)

    summary = _read_json(metric_stage / "analysis_summary.json")
    evidence = _read_json(metric_stage / "control_evidence.json")
    behavior = _read_json(metric_stage / "behavior_stratified_summary.json")
    report_lines = [
        "# Protocol3 joint8 shared-fragment neural analysis",
        "",
        "## Frozen identity",
        "",
        f'- Training checkpoint: update `{identity["checkpoint_update"]}`, SHA256 `{identity["checkpoint_sha256"]}`.',
        f'- Training HEAD: `{identity["training_git_identity"]["repository_head"]}`.',
        f'- Analysis implementation HEAD: `{identity["analysis_git_identity"]["repository_head"]}`.',
        f'- Model state SHA256 remained unchanged: `{summary["model_state_sha256_before"] == summary["model_state_sha256_after"]}`.',
        "- Condition: direction index 0, delay 50, reference 100; digits 1,2,3,4,5,6,7,9.",
        "",
        "## Scope and conclusion boundary",
        "",
        f'- Shared families: {summary["shared_family_count"]}; primary cross-digit pairs: {summary["shared_pair_count"]}.',
        f'- Matched controls: {summary["matched_control_count"]}; inference status: `{summary["status"]}`.',
        "- The six tools are principal angles, cross-projection, CCA, Procrustes, fixed points, and DSA only.",
        "- Deterministic and stochastic-mean results remain separate. DSA uses the 64 stochastic trials without trial averaging.",
        "- PASS/FAIL strata remain separate; these results must not be read as evidence that all eight digits have successful behavioral control.",
    ]
    if evidence["status"] == "descriptive_only_no_matched_control":
        report_lines.extend(
            [
                "- No strict matched control exists. All similarity metrics below are descriptive; no control-percentile claim is made.",
            ]
        )
    report_lines.extend(["", "## Behavior strata", ""])
    for stratum, value in behavior["strata"].items():
        report_lines.append(
            f'- {stratum}: {len(value["shared_pair_ids"])} shared pair(s), '
            f'{len(value["control_pair_ids"])} control pair(s).'
        )
    report_lines.extend(["", "## Family figures", ""])
    for family_id in family_ids:
        report_lines.extend([f"### {family_id}", ""])
        for metric in metrics:
            report_lines.append(
                f'![{family_id} {metric}](../{STAGE_DIRECTORIES["figures"]}/{family_id}__{metric}.png)'
            )
            report_lines.append("")
    report_lines.extend(
        [
            "## Raw artifacts",
            "",
            "All scalar rows, full fixed-point solver records, DSA systems, seeds, array shapes, and hashes are retained in the stage directories and manifests.",
            "",
        ]
    )
    report_stage = output_root / STAGE_DIRECTORIES["report"]
    report_path = report_stage / "analysis_report.md"
    _write_bytes_once(report_path, "\n".join(report_lines).encode("utf-8"))
    manifest_path = report_stage / "manifest.json"
    manifest = _stage_manifest(
        "report", output_root, config_path,
        [identity_path, analysis_manifest_path], [*figure_paths, report_path],
        {
            "family_count": len(family_ids),
            "figures_per_family": len(metrics),
            "conclusion_status": summary["status"],
        },
    )
    _write_json(manifest_path, manifest)
    return manifest


def _parse_common_paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repository_root = Path(arguments.repository_root).resolve()
    config_path = Path(arguments.config).resolve()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    if not repository_root.is_dir():
        raise NotADirectoryError(repository_root)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    config = _read_json(config_path)
    validate_analysis_config(repository_root, config)
    configured_output = _resolve_inside(repository_root, config["output"]["directory"])
    output_root = Path(arguments.output_root).resolve()
    if output_root != configured_output:
        raise ValueError("output root differs from frozen analysis config")
    output_root.mkdir(parents=True, exist_ok=True)
    return repository_root, config_path, checkpoint_path, output_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "prepare", "collect", "analyze", "report"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--repository-root", required=True)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--checkpoint", required=True)
        subparser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    repository_root, config_path, checkpoint_path, output_root = _parse_common_paths(arguments)
    function = {
        "validate": validate_checkpoint_identity,
        "prepare": prepare_fragments,
        "collect": collect_activity,
        "analyze": analyze_activity,
        "report": render_report,
    }[arguments.command]
    result = function(repository_root, config_path, checkpoint_path, output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
