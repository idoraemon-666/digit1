"""Pure condition-sampling helpers for digit-writing Protocol3."""

from __future__ import annotations

import numpy as np


DELAYS = (25, 50, 75)
ENVIRONMENT_SEED_UPPER_BOUND = 2**32


def balanced_direction_indices(batch_size: int) -> np.ndarray:
    if batch_size % 8 != 0:
        raise ValueError("protocol3 batch_size must be divisible by 8")
    return np.tile(
        np.arange(8, dtype=np.int64),
        batch_size // 8,
    )


def sample_independent_digit_delay(rng, digit_count: int = 10) -> tuple[int, int]:
    if digit_count < 1:
        raise ValueError("digit_count must be positive")
    return rng.randrange(digit_count), rng.randrange(len(DELAYS))


def sample_protocol3_update(
    rng, digit_count: int = 10
) -> tuple[int, int, int]:
    digit, delay_index = sample_independent_digit_delay(rng, digit_count)
    environment_seed = rng.randrange(ENVIRONMENT_SEED_UPPER_BOUND)
    return digit, delay_index, environment_seed


def new_condition_counts() -> dict[str, object]:
    return {
        "digit_update_counts": [0] * 10,
        "delay_update_counts": {str(delay): 0 for delay in DELAYS},
        "digit_delay_update_counts": [[0, 0, 0] for _ in range(10)],
    }


def record_condition(counts: dict[str, object], digit: int, delay_index: int) -> None:
    if not 0 <= digit <= 9:
        raise ValueError("digit must be in [0, 9]")
    if not 0 <= delay_index < len(DELAYS):
        raise ValueError("delay_index must be in [0, 2]")
    counts["digit_update_counts"][digit] += 1
    counts["delay_update_counts"][str(DELAYS[delay_index])] += 1
    counts["digit_delay_update_counts"][digit][delay_index] += 1


def validate_condition_counts(counts: dict[str, object], completed_updates: int) -> None:
    digit_counts = counts.get("digit_update_counts")
    delay_counts = counts.get("delay_update_counts")
    joint_counts = counts.get("digit_delay_update_counts")
    if not isinstance(digit_counts, list) or len(digit_counts) != 10:
        raise ValueError("condition counts require 10 digit counts")
    if (
        not isinstance(joint_counts, list)
        or len(joint_counts) != 10
        or any(
            not isinstance(row, list) or len(row) != len(DELAYS)
            for row in joint_counts
        )
    ):
        raise ValueError("condition counts require a 10 by 3 digit-delay table")
    if not isinstance(delay_counts, dict) or set(delay_counts) != {
        str(delay) for delay in DELAYS
    }:
        raise ValueError("condition counts require all three delay counts")
    totals = (
        sum(digit_counts),
        sum(delay_counts.values()),
        sum(sum(row) for row in joint_counts),
    )
    if totals != (completed_updates, completed_updates, completed_updates):
        raise ValueError("condition-count totals do not match completed updates")
    for digit, row in enumerate(joint_counts):
        if sum(row) != digit_counts[digit]:
            raise ValueError("digit and digit-delay counts disagree")
    for delay_index, delay in enumerate(DELAYS):
        if sum(row[delay_index] for row in joint_counts) != delay_counts[str(delay)]:
            raise ValueError("delay and digit-delay counts disagree")
