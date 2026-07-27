from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from losses import (
    l1_muscle_act,
    l1_rate,
    l1_weight,
    position_l1_metrics,
    simple_dynamics,
)
from model import RNNPolicy

try:
    import motornet as mn

    import train as train_module
    import digit_writing.experiments as experiment_module
    from digit_writing.experiments import _full_rule, prepare_composition
except ModuleNotFoundError as exc:
    if exc.name != "motornet":
        raise
    mn = None
    train_module = None
    experiment_module = None
    prepare_composition = None
    _full_rule = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configurations"


def make_policy(hidden_size=8):
    return RNNPolicy(
        28,
        hidden_size,
        6,
        activation_name="softplus",
        noise_level_act=0.0,
        noise_level_inp=0.0,
        constrained=False,
        dt=10,
        t_const=20,
        batch_first=True,
        device="cpu",
    )


class PhaseDConfigurationTests(unittest.TestCase):
    @staticmethod
    def load(name):
        with open(CONFIG_ROOT / name, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_base_models_are_protocol_fixed_and_independent(self):
        full10 = self.load("digit_writing_original_protocol2_full10_dev42.json")
        heldout5 = self.load("digit_writing_original_protocol2_heldout5_dev42.json")
        self.assertEqual(full10["train_digits"], list(range(10)))
        self.assertEqual(heldout5["train_digits"], [0, 1, 2, 3, 4, 6, 7, 8, 9])
        self.assertNotEqual(full10["output"]["directory"], heldout5["output"]["directory"])
        self.assertNotEqual(full10["variant"], heldout5["variant"])

        for config in (full10, heldout5):
            self.assertEqual(config["device"], "cpu")
            self.assertNotIn("source_checkpoint", config)
            self.assertEqual(config["model"]["input_size"], 28)
            self.assertEqual(config["model"]["hidden_size"], 256)
            self.assertEqual(config["model"]["activation"], "softplus")
            self.assertEqual(config["model"]["recurrent_noise_std"], 0.1)
            self.assertEqual(config["model"]["input_noise_std"], 0.01)
            self.assertEqual(config["optimizer"]["learning_rate"], 0.001)
            self.assertEqual(config["optimizer"]["grad_clip_norm"], 1.0)
            self.assertEqual(config["training"]["batch_size"], 32)
            self.assertEqual(config["training"]["max_updates"], 75000)
            self.assertEqual(config["training"]["validation_interval"], 500)
            self.assertEqual(
                config["position_loss"],
                {
                    "type": "phase_normalized_l1",
                    "stable_weight": 0.1,
                    "delay_weight": 0.1,
                    "movement_weight": 0.6,
                    "hold_weight": 0.2,
                },
            )
            self.assertEqual(
                config["regularization"],
                {
                    "l1_rate": 0.001,
                    "l1_weight": 0.001,
                    "l1_muscle_act": 0.01,
                    "simple_dynamics_weight": 0.001,
                },
            )

    def test_composition_configuration_reproduces_original_optimization(self):
        config = self.load("digit_writing_original_protocol2_composition.json")
        self.assertIn("full10", config["source_checkpoint"])
        self.assertEqual(config["digit_group"], [0, 4, 6, 9, 8])
        self.assertTrue(config["leave_one_out"])
        self.assertEqual(config["batch_size"], 8)
        self.assertEqual(config["direction_indices"], list(range(0, 32, 4)))
        self.assertEqual(config["validation_speed_index"], 9)
        self.assertEqual(config["custom_delay"], 150)
        self.assertEqual(config["iterations"], 250)
        self.assertEqual(config["coefficient_learning_rate"], 0.1)
        self.assertFalse(config["network_noise"])
        if experiment_module is not None:
            source = inspect.getsource(experiment_module.run_composition_config)
            self.assertIn('checkpoint.get("protocol_config")', source)

    def test_transfer_configuration_uses_only_heldout5_checkpoint(self):
        config = self.load("digit_writing_original_protocol2_transfer5.json")
        self.assertIn("heldout5", config["source_checkpoint"])
        self.assertEqual(config["target_digit"], 5)
        self.assertNotIn("composition", config["source_checkpoint"])
        self.assertEqual(config["optimizer"]["learning_rate"], 0.001)

    def test_experiment_runners_require_exact_phase_e_authorization(self):
        generic = (
            ROOT / "server" / "run_digit_writing_original_protocol2_experiment.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("DIGIT_PROTOCOL2_AUTHORIZED_RUN", generic)
        self.assertIn("must name this exact project-2 experiment run", generic)
        self.assertIn("source_checkpoint.sha256", generic)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", generic)
        self.assertGreaterEqual(generic.count("status --porcelain"), 4)
        for name in (
            "run_digit_writing_original_protocol2_full10_dev42.sh",
            "run_digit_writing_original_protocol2_heldout5_dev42.sh",
            "run_digit_writing_original_protocol2_composition.sh",
            "run_digit_writing_original_protocol2_transfer5.sh",
        ):
            wrapper = (ROOT / "server" / name).read_text(encoding="utf-8")
            self.assertIn(
                "run_digit_writing_original_protocol2_experiment.sh", wrapper
            )


@unittest.skipUnless(train_module is not None, "MotorNet is server-only")
class PhaseDOriginalCoreReuseTests(unittest.TestCase):
    @staticmethod
    def load(name):
        with open(CONFIG_ROOT / name, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_no_parallel_base_rollout_loss_or_evaluation_helpers_remain(self):
        for name in (
            "rollout_digit_episode",
            "digit_base_loss",
            "evaluate_digit_policy",
        ):
            self.assertFalse(hasattr(train_module, name), name)

    def test_digit_base_entry_delegates_to_original_subset_trainer(self):
        for filename, expected_digits in (
            ("digit_writing_original_protocol2_full10_dev42.json", tuple(range(10))),
            (
                "digit_writing_original_protocol2_heldout5_dev42.json",
                (0, 1, 2, 3, 4, 6, 7, 8, 9),
            ),
        ):
            config = self.load(filename)
            with tempfile.TemporaryDirectory() as directory:
                config["output"] = dict(config["output"])
                config["output"]["directory"] = directory
                marker = {"delegated": True}
                with mock.patch.object(
                    train_module,
                    "train_subsets_base_model",
                    return_value=marker,
                ) as core:
                    self.assertIs(
                        train_module.train_digit_base_model(config), marker
                    )
                args = core.call_args
                self.assertEqual(args.args[0], directory)
                self.assertEqual(
                    args.args[1], config["output"]["best_checkpoint"]
                )
                hp = args.kwargs["hp"]
                self.assertEqual(hp["hid_size"], 256)
                self.assertEqual(hp["epochs"], 75000)
                self.assertEqual(hp["save_iter"], 500)
                self.assertEqual(hp["variant"], config["variant"])
                self.assertEqual(
                    tuple(
                        environment.FIXED_DIGIT
                        for environment in args.kwargs["env_dict"].values()
                    ),
                    expected_digits,
                )

    def test_original_subset_core_retains_five_term_loss_and_isolated_defaults(self):
        source = inspect.getsource(train_module.train_subsets_base_model)
        self.assertIn("DEF_HP.copy()", source)
        for loss_name in (
            "position_l1_metrics",
            "l1_rate",
            "l1_weight",
            "l1_muscle_act",
            "simple_dynamics",
        ):
            self.assertIn(loss_name, source)
        self.assertIn("_run_validation", source)


class PhaseDModelFreezingTests(unittest.TestCase):
    def test_mrnn_forward_uses_the_pinned_submodule_contract(self):
        policy = make_policy()
        observation = torch.zeros((4, 28))
        x = torch.zeros((4, 8))
        h = torch.zeros((4, 8))
        x_next, h_next, action = policy(observation, x, h, noise=False)
        self.assertEqual(tuple(x_next.shape), (4, 8))
        self.assertEqual(tuple(h_next.shape), (4, 8))
        self.assertEqual(tuple(action.shape), (4, 6))

    def test_transfer_reinitializes_only_digit_five_rule_column(self):
        torch.manual_seed(3)
        policy = make_policy()
        before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        generator = torch.Generator().manual_seed(42)
        policy.prepare_rule_column_transfer(5, generator=generator)
        after = policy.state_dict()
        input_name = next(
            name
            for name, parameter in policy.named_parameters()
            if parameter is policy.rule_input_weight()
        )
        for name in before:
            if name == input_name:
                self.assertTrue(torch.equal(before[name][:, :5], after[name][:, :5]))
                self.assertTrue(torch.equal(before[name][:, 6:], after[name][:, 6:]))
                self.assertFalse(torch.equal(before[name][:, 5], after[name][:, 5]))
            else:
                self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertEqual(
            sum(parameter.requires_grad for parameter in policy.parameters()), 1
        )

    def test_transfer_adam_changes_only_digit_five_rule_column(self):
        policy = make_policy()
        parameter = policy.prepare_rule_column_transfer(
            5, generator=torch.Generator().manual_seed(7)
        )
        before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        optimizer = torch.optim.Adam(policy.transfer_trainable_parameters(), lr=0.001)
        observation = torch.zeros((4, 28))
        observation[:, 5] = 1.0
        for _ in range(3):
            _, _, action = policy(
                observation,
                torch.zeros((4, 8)),
                torch.zeros((4, 8)),
                noise=False,
            )
            optimizer.zero_grad()
            action.sum().backward()
            optimizer.step()

        self.assertEqual(int(torch.count_nonzero(parameter.grad[:, :5])), 0)
        self.assertEqual(int(torch.count_nonzero(parameter.grad[:, 6:])), 0)
        self.assertGreater(int(torch.count_nonzero(parameter.grad[:, 5])), 0)
        input_name = next(
            name
            for name, candidate in policy.named_parameters()
            if candidate is policy.rule_input_weight()
        )
        after = policy.state_dict()
        for name in before:
            if name == input_name:
                self.assertTrue(torch.equal(before[name][:, :5], after[name][:, :5]))
                self.assertTrue(torch.equal(before[name][:, 6:], after[name][:, 6:]))
                self.assertFalse(torch.equal(before[name][:, 5], after[name][:, 5]))
            else:
                self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_reinitializing_digit_five_preserves_other_digit_outputs(self):
        torch.manual_seed(11)
        policy = make_policy()
        observations = torch.zeros((9, 28))
        observations[
            torch.arange(9), torch.tensor([0, 1, 2, 3, 4, 6, 7, 8, 9])
        ] = 1.0
        x = torch.zeros((9, 8))
        h = torch.zeros((9, 8))
        before = policy(observations, x, h, noise=False)
        policy.prepare_rule_column_transfer(
            5, generator=torch.Generator().manual_seed(9)
        )
        after = policy(observations, x, h, noise=False)
        for before_value, after_value in zip(before, after):
            self.assertTrue(torch.equal(before_value, after_value))


@unittest.skipUnless(prepare_composition is not None, "MotorNet is server-only")
class PhaseDCompositionTests(unittest.TestCase):
    def test_external_coefficients_are_the_only_composition_parameters(self):
        policy = make_policy()
        before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        coefficients, optimizer = prepare_composition(
            policy, 0, (4, 6, 9, 8), 8, learning_rate=0.1
        )
        rule_input = _full_rule(coefficients, (4, 6, 9, 8))
        self.assertEqual(tuple(rule_input.shape), (8, 10))
        self.assertTrue(torch.equal(rule_input[:, 0], torch.zeros(8)))
        self.assertEqual(int(torch.count_nonzero(rule_input[:, 1:4])), 0)
        self.assertEqual(
            sum(parameter.requires_grad for parameter in policy.parameters()), 0
        )
        optimizer.zero_grad()
        rule_input.square().sum().backward()
        optimizer.step()
        self.assertIsNotNone(coefficients.grad)
        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_cpu_closed_loop_gradient_smoke_uses_original_five_term_loss(self):
        config = PhaseDConfigurationTests.load(
            "digit_writing_original_protocol2_full10_dev42.json"
        )
        torch.manual_seed(21)
        policy = make_policy()
        effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        environment = train_module.DIGIT_ENV_CLASSES[1](
            effector=effector,
            action_frame_stacking=0,
            geometry_config_path=config["geometry_config"],
        )
        x = torch.zeros((2, 8))
        h = torch.zeros_like(x)
        observation, info = environment.reset(options={"batch_size": 2})
        xy = []
        target = []
        muscle = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hidden = [h.unsqueeze(1)]
        terminated = False
        timestep = 0
        while not terminated:
            x, h, action = policy(observation, x, h, noise=False)
            observation, _, terminated, info = environment.step(
                timestep, action=action
            )
            xy.append(info["states"]["fingertip"][:, None, :])
            target.append(info["goal"][:, None, :])
            muscle.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hidden.append(h.unsqueeze(1))
            timestep += 1

        xy = torch.cat(xy, dim=1)
        target = torch.cat(target, dim=1)
        muscle = torch.cat(muscle, dim=1)
        hidden = torch.cat(hidden, dim=1)
        loss = position_l1_metrics(
            xy, target, environment.epoch_bounds
        )["phase_normalized_position_l1"]
        loss = loss + l1_rate(hidden, 0.001)
        loss = loss + l1_weight(policy, 0.001)
        loss = loss + l1_muscle_act(muscle, 0.01)
        loss = loss + simple_dynamics(hidden, policy.mrnn, weight=0.001)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in policy.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
