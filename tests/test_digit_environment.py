from __future__ import annotations

import unittest

import numpy as np

try:
    import motornet as mn
    import torch as th

    from digit_writing.geometry import build_digit_trajectory
    from envs import (
        DlyFigure8,
        DlyFigure8Inv,
        DlyFullCircleCClk,
        DlyFullCircleClk,
        DlyFullReach,
        DlyHalfCircleCClk,
        DlyHalfCircleClk,
        DlyHalfReach,
        DlySinusoid,
        DlySinusoidInv,
    )
except ModuleNotFoundError as exc:
    if exc.name != "motornet":
        raise
    mn = None
    th = None


@unittest.skipUnless(mn is not None, "MotorNet is available only in the server environment")
class DigitEnvironmentTests(unittest.TestCase):
    ENV_CLASSES = () if mn is None else (
        DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid,
        DlySinusoidInv, DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk,
        DlyFigure8, DlyFigure8Inv,
    )

    @staticmethod
    def make_env(environment_class=None):
        environment_class = DlyHalfReach if environment_class is None else environment_class
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        return environment_class(effector=effector, action_frame_stacking=0)

    @staticmethod
    def as_numpy(value):
        return value.detach().cpu().numpy() if th.is_tensor(value) else np.asarray(value)

    def test_original_environment_names_map_to_digits_zero_through_nine(self):
        self.assertEqual(
            [environment_class.FIXED_DIGIT for environment_class in self.ENV_CLASSES],
            list(range(10)),
        )

    def test_observation_contract_and_trial_cues(self):
        env = self.make_env(DlySinusoid)
        directions = np.array((0, 1, 2, 3), dtype=np.int64)
        obs, _ = env.reset(
            testing=False,
            options={
                "batch_size": 4,
                "reach_conds": directions,
                "speed_cond": 1,
                "delay_cond": 1,
                "deterministic": True,
            },
        )

        self.assertEqual(tuple(obs.shape), (4, 28))
        self.assertEqual(env.action_frame_stacking, 0)
        np.testing.assert_allclose(self.as_numpy(obs[:, 0:10]), self.as_numpy(env.rule_input))
        np.testing.assert_allclose(self.as_numpy(obs[:, 10:11]), 0.0)
        np.testing.assert_allclose(self.as_numpy(obs[:, 11:12]), 0.0)
        np.testing.assert_allclose(self.as_numpy(obs[:, 12:14]), 0.0)
        self.assertTrue(th.all(env.rule_input[:, 3] == 1.0))
        self.assertEqual(int(th.count_nonzero(env.rule_input).item()), 4)

        stable_end = env.epoch_bounds["stable"][1]
        movement_start = env.epoch_bounds["movement"][0]
        delay_obs = env.get_obs(stable_end, deterministic=True)
        movement_obs = env.get_obs(movement_start, deterministic=True)
        np.testing.assert_allclose(
            self.as_numpy(delay_obs[:, 0:10]), self.as_numpy(env.rule_input)
        )
        np.testing.assert_allclose(self.as_numpy(delay_obs[:, 10:11]), 1.0 / 3.0, atol=1e-7)
        np.testing.assert_allclose(self.as_numpy(delay_obs[:, 11:12]), 0.0)
        np.testing.assert_allclose(
            self.as_numpy(delay_obs[:, 12:14]), self.as_numpy(env.vis_inp[:, stable_end])
        )
        np.testing.assert_allclose(self.as_numpy(movement_obs[:, 11:12]), 1.0)
        np.testing.assert_allclose(
            self.as_numpy(movement_obs[:, 14:16]),
            self.as_numpy(env.obs_buffer["vision"][0]),
        )
        np.testing.assert_allclose(
            self.as_numpy(movement_obs[:, 16:28]),
            self.as_numpy(env.obs_buffer["proprioception"][0]),
        )

        anchors = env.traj[:, 0, :]
        angles = th.as_tensor(directions, dtype=anchors.dtype) * (2.0 * np.pi / 8.0)
        expected_cues = anchors + env.geometry_config.reach_distance_m * th.stack(
            (th.cos(angles), th.sin(angles)), dim=-1
        )
        np.testing.assert_allclose(
            self.as_numpy(env.vis_inp[:, stable_end]), self.as_numpy(expected_cues), atol=1e-7
        )
        np.testing.assert_allclose(self.as_numpy(env.speed_scalar[:, :stable_end]), 0.0)
        np.testing.assert_allclose(self.as_numpy(env.vis_inp[:, :stable_end]), 0.0)
        np.testing.assert_allclose(
            self.as_numpy(env.go_cue[:, :movement_start]), 0.0
        )
        np.testing.assert_allclose(
            self.as_numpy(env.go_cue[:, movement_start:]), 1.0
        )

    def test_each_batch_sample_keeps_its_own_spatial_direction(self):
        env = self.make_env(DlyFullReach)
        directions = np.arange(8, dtype=np.int64)
        env.reset(
            testing=False,
            options={
                "batch_size": 8,
                "reach_conds": directions,
                "speed_cond": 0,
                "delay_cond": 0,
                "deterministic": True,
            },
        )
        np.testing.assert_array_equal(env.direction_indices, directions)
        self.assertEqual(env.current_digit, 5)
        self.assertEqual(env.speed_cond, 0)
        self.assertEqual(env.delay_time, 25)

        for batch_index, direction_index in enumerate(directions):
            expected = build_digit_trajectory(
                5,
                env.geometry_config,
                50,
                spatial_angle_rad=direction_index * (2.0 * np.pi / 8.0),
                anchor=env.traj[batch_index, 0].cpu().numpy(),
            )
            np.testing.assert_allclose(
                self.as_numpy(env.traj[batch_index]), expected.points, atol=2e-7
            )

    def test_training_validation_and_custom_delay_have_no_off_by_one(self):
        env = self.make_env(DlyHalfCircleCClk)
        cases = (
            (False, {"speed_cond": 0, "delay_cond": 0}, 25),
            (True, {"speed_cond": 9, "delay_cond": 2}, 75),
            (False, {"speed_cond": 2, "custom_delay": 150}, 150),
        )
        for testing, conditions, expected_delay in cases:
            with self.subTest(testing=testing, conditions=conditions):
                options = {
                    "batch_size": 1,
                    "reach_conds": 0,
                    "deterministic": True,
                    **conditions,
                }
                env.reset(testing=testing, options=options)
                stable_end = env.epoch_bounds["stable"][1]
                movement_start, movement_end = env.epoch_bounds["movement"]
                hold_start, hold_end = env.epoch_bounds["hold"]

                self.assertEqual(env.delay_time, expected_delay)
                self.assertEqual(movement_start, stable_end + expected_delay)
                self.assertEqual(movement_end - movement_start, env.movement_intervals)
                self.assertEqual(env.traj.shape[1], env.movement_intervals + 1)
                self.assertEqual(hold_start, movement_end)
                self.assertEqual(hold_end - hold_start, env.geometry_config.hold_steps)
                self.assertEqual(env.speed_scalar.shape[1], hold_end)
                self.assertEqual(env.go_cue.shape[1], hold_end)
                self.assertEqual(env.vis_inp.shape[1], hold_end)
                self.assertEqual(env.max_ep_duration, hold_end - 1)
                np.testing.assert_allclose(
                    self.as_numpy(env._target_at(movement_start)),
                    self.as_numpy(env.traj[:, 0]),
                )
                np.testing.assert_allclose(
                    self.as_numpy(env._target_at(hold_start)),
                    self.as_numpy(env.traj[:, -1]),
                )

    def test_short_cpu_closed_loop_smoke_supervises_both_endpoints(self):
        env = self.make_env(DlyHalfCircleClk)
        obs, _ = env.reset(
            testing=False,
            options={
                "batch_size": 2,
                "reach_conds": np.array((0, 1), dtype=np.int64),
                "speed_cond": 0,
                "delay_cond": 0,
                "deterministic": True,
            },
        )
        movement_start, movement_end = env.epoch_bounds["movement"]
        goals = {}
        timestep = 0
        terminated = False
        while not terminated:
            action = th.zeros((2, env.action_space.shape[0]), dtype=th.float32)
            obs, _, terminated, info = env.step(timestep, action)
            self.assertEqual(tuple(obs.shape), (2, 28))
            self.assertTrue(bool(th.isfinite(obs).all()))
            self.assertTrue(bool(th.isfinite(info["goal"]).all()))
            if timestep in {movement_start, movement_end}:
                goals[timestep] = info["goal"].clone()
            timestep += 1

        self.assertEqual(timestep, env.epoch_bounds["hold"][1])
        np.testing.assert_allclose(
            self.as_numpy(goals[movement_start]), self.as_numpy(env.traj[:, 0])
        )
        np.testing.assert_allclose(
            self.as_numpy(goals[movement_end]), self.as_numpy(env.traj[:, -1])
        )


if __name__ == "__main__":
    unittest.main()
