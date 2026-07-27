"""Phase-normalized position loss for the final digit protocol."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


PHASES = ("stable", "delay", "movement", "hold")
PHASE_WEIGHTS = {
    "stable": 0.1,
    "delay": 0.1,
    "movement": 0.6,
    "hold": 0.2,
}


def _validated_bounds(
    epoch_bounds: Mapping[str, Sequence[int]], time_steps: int
) -> dict[str, tuple[int, int]]:
    if set(epoch_bounds) != set(PHASES):
        raise ValueError("epoch_bounds must contain stable, delay, movement, and hold")
    bounds = {
        phase: (int(epoch_bounds[phase][0]), int(epoch_bounds[phase][1]))
        for phase in PHASES
    }
    expected_start = 0
    for phase in PHASES:
        start, end = bounds[phase]
        if start != expected_start or end <= start:
            raise ValueError("phase bounds must be contiguous, ordered, and non-empty")
        expected_start = end
    if expected_start != time_steps:
        raise ValueError(
            f"phase bounds cover {expected_start} steps but tensors contain {time_steps}"
        )
    return bounds


def position_l1_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epoch_bounds: Mapping[str, Sequence[int]],
) -> dict[str, torch.Tensor]:
    """Return the fixed phase-normalized objective and diagnostic position metrics."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("position tensors must have shape [batch, time, 2]")
    bounds = _validated_bounds(epoch_bounds, prediction.shape[1])
    error = torch.sum(torch.abs(prediction - target), dim=-1)
    phase_means = {
        phase: torch.mean(error[:, bounds[phase][0] : bounds[phase][1]])
        for phase in PHASES
    }
    phase_normalized = sum(
        PHASE_WEIGHTS[phase] * phase_means[phase] for phase in PHASES
    )

    movement_end = bounds["movement"][1]
    hold_start, hold_end = bounds["hold"]
    final_movement_position = prediction[:, movement_end - 1 : movement_end, :]
    hold_drift = torch.mean(
        torch.sum(
            torch.abs(prediction[:, hold_start:hold_end, :] - final_movement_position),
            dim=-1,
        )
    )
    return {
        "phase_normalized_position_l1": phase_normalized,
        "full_trial_position_l1": torch.mean(error),
        "stable_mean_l1": phase_means["stable"],
        "delay_mean_l1": phase_means["delay"],
        "movement_mean_l1": phase_means["movement"],
        "hold_mean_l1": phase_means["hold"],
        "endpoint_error": torch.mean(error[:, movement_end - 1]),
        "hold_drift": hold_drift,
    }


def phase_normalized_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epoch_bounds: Mapping[str, Sequence[int]],
) -> torch.Tensor:
    return position_l1_metrics(prediction, target, epoch_bounds)[
        "phase_normalized_position_l1"
    ]


def detached_position_metrics(
    metrics: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in metrics.items()
    }
