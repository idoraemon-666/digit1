"""MotorNet environments for digit_writing_original_protocol2."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch as th
from motornet import environment as env

from digit_writing.geometry import (
    build_digit_trajectory,
    load_geometry_config,
    training_angles,
    validation_angles,
)


DEFAULT_GEOMETRY_CONFIG = (
    Path(__file__).resolve().parent
    / "configurations"
    / "digit_writing_original_protocol2_geometry.json"
)


class DigitWritingEnv(env.Environment):
    """A digit trajectory rotated around the original fixed starting point."""

    FIXED_DIGIT: int | None = None

    def __init__(
        self,
        *args: Any,
        geometry_config_path: str | Path = DEFAULT_GEOMETRY_CONFIG,
        **kwargs: Any,
    ) -> None:
        self.geometry_config_path = Path(geometry_config_path)
        self.geometry_config = load_geometry_config(self.geometry_config_path)
        super().__init__(*args, **kwargs)
        if self.action_frame_stacking != 0:
            raise ValueError("digit environments require action_frame_stacking=0")
        self.obs_noise[: self.skeleton.space_dim] = [0.0] * self.skeleton.space_dim
        self.dt = self.geometry_config.dt_seconds
        self._deterministic_observations = False

    def get_obs(
        self, t: int, action: th.Tensor | np.ndarray | None = None, deterministic: bool = False
    ) -> th.Tensor | np.ndarray:
        """Return the protocol-defined 28-dimensional observation."""

        self.update_obs_buffer(action=action)
        obs = th.cat(
            [
                self.rule_input,
                self.speed_scalar[:, t],
                self.go_cue[:, t],
                self.vis_inp[:, t],
                self.obs_buffer["vision"][0],
                self.obs_buffer["proprioception"][0],
            ],
            dim=-1,
        )
        if obs.shape[-1] != 28:
            raise RuntimeError(f"digit observation must have 28 features, got {obs.shape[-1]}")
        if not deterministic:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs if self.differentiable else self.detach(obs)

    def get_proprioception(self) -> th.Tensor:
        if not self._deterministic_observations:
            return super().get_proprioception()
        muscle_length = self.states["muscle"][:, 1:2, :] / self.muscle.l0_ce
        muscle_velocity = self.states["muscle"][:, 2:3, :] / self.muscle.vmax
        return th.concatenate(
            [muscle_length, muscle_velocity], dim=-1
        ).squeeze(dim=1)

    def get_vision(self) -> th.Tensor:
        if not self._deterministic_observations:
            return super().get_vision()
        return self.states["fingertip"]

    def step(
        self,
        t: int,
        action: th.Tensor | np.ndarray,
        **kwargs: Any,
    ) -> tuple[th.Tensor | np.ndarray, Any, bool, dict[str, Any]]:
        """Advance MotorNet once and expose the target for timestep ``t``."""

        action = action if th.is_tensor(action) else th.tensor(action, dtype=th.float32)
        action = action.to(self.device)
        noisy_action = action
        self.effector.step(noisy_action, **kwargs)

        obs = self.get_obs(
            t,
            action=noisy_action,
            deterministic=self._deterministic_observations,
        )
        reward = None if self.differentiable else np.zeros((action.shape[0], 1))
        terminated = bool(t >= self.max_ep_duration)
        self.hidden_goal = self._target_at(t)

        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": noisy_action,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }
        return obs, reward, terminated, info

    def reset(
        self,
        *,
        testing: bool = False,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Sample one batch condition and build its digit trajectories."""

        self._set_generator(seed=seed)
        options = {} if options is None else options
        batch_size = int(options.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._deterministic_observations = bool(options.get("deterministic", False))

        self.current_digit = self._resolve_digit(options.get("digit"))
        direction_count = 32 if testing else 8
        self.direction_indices = self._resolve_direction_indices(
            options.get("reach_conds"), batch_size, direction_count
        )

        reference_values = (
            self.geometry_config.validation_reference_steps
            if testing
            else self.geometry_config.training_reference_steps
        )
        self.speed_cond = self._resolve_condition_index(
            options.get("speed_cond"), len(reference_values), "speed_cond"
        )
        self.reference_steps = reference_values[self.speed_cond]
        self.delay_time = self._resolve_delay(options)

        joint_state = self._initial_joint_state(batch_size)
        self.initial_pos = joint_state.clone()
        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})
        fingertip = self.joint2cartesian(joint_state).chunk(2, dim=-1)[0]

        angles = validation_angles() if testing else training_angles()
        angle_values = angles[self.direction_indices]
        trajectories = []
        fingertip_numpy = fingertip.detach().cpu().numpy()
        for batch_index, angle in enumerate(angle_values):
            trajectory = build_digit_trajectory(
                self.current_digit,
                self.geometry_config,
                self.reference_steps,
                spatial_angle_rad=float(angle),
                anchor=fingertip_numpy[batch_index],
            )
            trajectories.append(trajectory.points)

        trajectory_points = np.stack(trajectories)
        self.traj = th.as_tensor(
            trajectory_points, dtype=fingertip.dtype, device=self.device
        )
        self.movement_intervals = self.traj.shape[1] - 1
        self.movement_time = self.traj.shape[1]
        self._build_trial_inputs(fingertip, angle_values)

        action = th.zeros(
            (batch_size, self.action_space.shape[0]),
            dtype=fingertip.dtype,
            device=self.device,
        )
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(
            self.obs_buffer["vision"]
        )
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action_output = action if self.differentiable else self.detach(action)
        obs = self.get_obs(0, deterministic=self._deterministic_observations)
        info = {
            "states": self._maybe_detach_states(),
            "action": action_output,
            "noisy action": action_output,
            "goal": self.hidden_goal if self.differentiable else self.detach(self.hidden_goal),
        }
        return obs, info

    def _initial_joint_state(self, batch_size: int) -> th.Tensor:
        position_range = th.as_tensor(
            self.effector.pos_range_bound, dtype=th.float32, device=self.device
        )
        position_upper = th.as_tensor(
            self.effector.pos_upper_bound, dtype=th.float32, device=self.device
        )
        offsets = th.tensor((0.1, 0.5), dtype=th.float32, device=self.device)
        position = position_range * 0.5 + position_upper + offsets
        joint_state = th.cat((position, th.zeros(2, device=self.device)))
        return joint_state.unsqueeze(0).repeat(batch_size, 1)

    def _build_trial_inputs(
        self, fingertip: th.Tensor, angle_values: np.ndarray
    ) -> None:
        config = self.geometry_config
        self.stable_time = config.stable_steps
        self.hold_time = config.hold_steps
        movement_start = self.stable_time + self.delay_time
        movement_end = movement_start + self.traj.shape[1]
        hold_end = movement_end + self.hold_time
        self.epoch_bounds = {
            "stable": (0, self.stable_time),
            "delay": (self.stable_time, movement_start),
            "movement": (movement_start, movement_end),
            "hold": (movement_end, hold_end),
        }
        self.max_ep_duration = hold_end - 1

        batch_size = fingertip.shape[0]
        dtype = fingertip.dtype
        self.rule_input = th.zeros((batch_size, 10), dtype=dtype, device=self.device)
        self.rule_input[:, self.current_digit] = 1.0

        self.speed_scalar = th.zeros(
            (batch_size, hold_end, 1), dtype=dtype, device=self.device
        )
        speed_value = 1.0 - self.reference_steps / 150.0
        self.speed_scalar[:, self.stable_time :, 0] = speed_value

        self.go_cue = th.zeros((batch_size, hold_end, 1), dtype=dtype, device=self.device)
        self.go_cue[:, movement_start:, 0] = 1.0

        angle_tensor = th.as_tensor(angle_values, dtype=dtype, device=self.device)
        unit_directions = th.stack((th.cos(angle_tensor), th.sin(angle_tensor)), dim=-1)
        direction_cues = (
            fingertip
            + self.geometry_config.reach_distance_m * unit_directions
        )
        self.vis_inp = th.zeros((batch_size, hold_end, 2), dtype=dtype, device=self.device)
        self.vis_inp[:, self.stable_time :] = direction_cues[:, None, :]
        self.hidden_goal = self.traj[:, 0, :].clone()

    def _target_at(self, t: int) -> th.Tensor:
        movement_start, movement_end = self.epoch_bounds["movement"]
        if t < movement_start:
            return self.traj[:, 0, :].clone()
        if t < movement_end:
            return self.traj[:, t - movement_start, :].clone()
        return self.traj[:, -1, :].clone()

    def _resolve_digit(self, requested: Any) -> int:
        fixed = self.FIXED_DIGIT
        if requested is None:
            digit = random.randrange(10) if fixed is None else fixed
        else:
            digit = self._as_index(requested, 10, "digit")
            if fixed is not None and digit != fixed:
                raise ValueError(f"{type(self).__name__} is fixed to digit {fixed}")
        return digit

    def _resolve_delay(self, options: dict[str, Any]) -> int:
        delay_cond = options.get("delay_cond")
        custom_delay = options.get("custom_delay")
        if delay_cond is not None:
            index = self._as_index(
                delay_cond, len(self.geometry_config.delay_steps), "delay_cond"
            )
            return self.geometry_config.delay_steps[index]
        if custom_delay is not None:
            if isinstance(custom_delay, bool) or not isinstance(custom_delay, Integral):
                raise TypeError("custom_delay must be an integer")
            if custom_delay < 0:
                raise ValueError("custom_delay must be non-negative")
            return int(custom_delay)
        return random.choice(self.geometry_config.delay_steps)

    def _resolve_condition_index(self, value: Any, count: int, label: str) -> int:
        return random.randrange(count) if value is None else self._as_index(value, count, label)

    @staticmethod
    def _as_index(value: Any, count: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{label} must be an integer")
        index = int(value)
        if not 0 <= index < count:
            raise ValueError(f"{label} must be in [0, {count - 1}]")
        return index

    @classmethod
    def _resolve_direction_indices(
        cls, value: Any, batch_size: int, direction_count: int
    ) -> np.ndarray:
        if value is None:
            return np.random.randint(0, direction_count, size=batch_size, dtype=np.int64)
        if th.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.full(batch_size, array.item())
        if array.shape != (batch_size,):
            raise ValueError("reach_conds must be scalar or have one index per batch sample")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("reach_conds must contain integers")
        indices = array.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= direction_count):
            raise ValueError(f"reach_conds must be in [0, {direction_count - 1}]")
        return indices


class DlyHalfReach(DigitWritingEnv):
    FIXED_DIGIT = 0


class DlyHalfCircleClk(DigitWritingEnv):
    FIXED_DIGIT = 1


class DlyHalfCircleCClk(DigitWritingEnv):
    FIXED_DIGIT = 2


class DlySinusoid(DigitWritingEnv):
    FIXED_DIGIT = 3


class DlySinusoidInv(DigitWritingEnv):
    FIXED_DIGIT = 4


class DlyFullReach(DigitWritingEnv):
    FIXED_DIGIT = 5


class DlyFullCircleClk(DigitWritingEnv):
    FIXED_DIGIT = 6


class DlyFullCircleCClk(DigitWritingEnv):
    FIXED_DIGIT = 7


class DlyFigure8(DigitWritingEnv):
    FIXED_DIGIT = 8


class DlyFigure8Inv(DigitWritingEnv):
    FIXED_DIGIT = 9


class ComposableEnv:
    """Reserved name for the Phase D composition implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ComposableEnv is unavailable until Phase D")
