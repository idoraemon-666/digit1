import os
import sys
import json
import torch
import motornet as mn
import random
import numpy as np
from contextlib import contextmanager
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import RNNPolicy, GRUPolicy
from losses import (
    detached_position_metrics,
    l1_dist,
    l1_muscle_act,
    l1_rate,
    l1_weight,
    position_l1_metrics,
    simple_dynamics,
)
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
from envs import ComposableEnv
from utils import save_hp, create_dir, load_hp
from itertools import product
from digit_writing.protocol3_schedule import (
    DELAYS as PROTOCOL3_DELAYS,
    balanced_direction_indices,
    new_condition_counts,
    record_condition,
    sample_protocol3_update,
)
from digit_writing.protocol3_checkpoint import (
    capture_rng_state,
    current_git_identity,
    protocol_config_sha256,
    restore_rng_state,
    validate_protocol3_resume_checkpoint,
)

DEF_HP = {
    "network": "rnn",
    "inp_size": 28,
    "hid_size": 512,
    "activation_name": "softplus",
    "noise_level_act": 0.1,
    "noise_level_inp": 0.01,
    "constrained": False,
    "dt": 10,
    "t_const": 20,
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 75_000,
    "save_iter": 500,
    "l1_rate": 0.001,
    "l1_weight": 0.001,
    "l1_muscle_act": 0.01,
    "simple_dynamics_weight": 0.001
}

DIGIT_ENV_CLASSES = (
    DlyHalfReach,
    DlyHalfCircleClk,
    DlyHalfCircleCClk,
    DlySinusoid,
    DlySinusoidInv,
    DlyFullReach,
    DlyFullCircleClk,
    DlyFullCircleCClk,
    DlyFigure8,
    DlyFigure8Inv,
)
FULL10_DIGITS = tuple(range(10))
HELDOUT5_DIGITS = tuple(digit for digit in FULL10_DIGITS if digit != 5)


def _build_policy(hp, output_dim, device):
    if hp["network"] == "rnn":
        return RNNPolicy(
            hp["inp_size"],
            hp["hid_size"],
            output_dim,
            activation_name=hp["activation_name"],
            noise_level_act=hp["noise_level_act"],
            noise_level_inp=hp["noise_level_inp"],
            constrained=hp["constrained"],
            dt=hp["dt"],
            t_const=hp["t_const"],
            batch_first=hp.get("batch_first", True),
            device=device,
        )
    if hp["network"] == "gru":
        return GRUPolicy(
            hp["inp_size"], hp["hid_size"], output_dim, batch_first=True
        )
    raise ValueError("Not a valid architecture")


@contextmanager
def _fixed_rng(seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path, value):
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True))
        handle.write("\n")


def _checkpoint_payload(
    policy,
    optimizer,
    hp,
    update,
    validation_loss,
    training_state=None,
):
    payload = {
        "agent_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hp": hp,
        "protocol_config": hp.get("protocol_config"),
        "variant": hp.get("variant"),
        "update": update,
        "validation_loss": validation_loss,
        "rng_state": capture_rng_state(),
    }
    if hp.get("protocol") == "digit_writing_original_protocol3":
        if hp.get("protocol_config") is None or hp.get("git_identity") is None:
            raise ValueError("protocol3 checkpoints require config and Git identity")
        payload["protocol_config_sha256"] = protocol_config_sha256(
            hp["protocol_config"]
        )
        payload["git_identity"] = hp["git_identity"]
    if training_state is not None:
        payload["training_state"] = training_state
    return payload


def _run_validation(
    policy, hp, env_dict, network_noise=True, return_metrics=False
):
    seed = hp.get("validation_seed")
    evaluator = (
        _do_protocol3_eval
        if hp.get("protocol") == "digit_writing_original_protocol3"
        else do_eval
    )
    if seed is None:
        return evaluator(
            policy,
            hp,
            env_dict=env_dict,
            network_noise=network_noise,
            return_metrics=return_metrics,
        )
    with _fixed_rng(seed):
        return evaluator(
            policy,
            hp,
            env_dict=env_dict,
            network_noise=network_noise,
            return_metrics=return_metrics,
        )

def do_eval(
    policy, hp, env_dict=None, network_noise=True, return_metrics=False
):

    if env_dict == None:
        env_dict = {
            "DlyHalfReach": DlyHalfReach, 
            "DlyHalfCircleClk": DlyHalfCircleClk, 
            "DlyHalfCircleCClk": DlyHalfCircleCClk, 
            "DlySinusoid": DlySinusoid, 
            "DlySinusoidInv": DlySinusoidInv,
            "DlyFullReach": DlyFullReach,
            "DlyFullCircleClk": DlyFullCircleClk,
            "DlyFullCircleCClk": DlyFullCircleCClk,
            "DlyFigure8": DlyFigure8,
            "DlyFigure8Inv": DlyFigure8Inv
        }

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    # Currently 10 speed conds during testing, 3 for training
    speed_conds = list(np.arange(0, 10))

    total_metrics = None
    condition_count = 0
    condition_losses = {}
    for env in env_dict:
        condition_loss = 0
        for speed in speed_conds:

            # initialize batch
            # use 32 to get every possible direction
            x = torch.zeros(size=(32, hp["hid_size"]))
            h = torch.zeros(size=(32, hp["hid_size"]))

            cur_env = env_dict[env](
                effector=effector, **hp.get("env_kwargs", {})
            )

            # Get first timestep
            obs, info = cur_env.reset(testing=True, options={"batch_size": 32, "reach_conds": np.arange(0, 32), "speed_cond": speed})
            terminated = False

            xy = []
            tg = []

            timestep = 0
            # simulate whole episode
            while not terminated:  # will run until `max_ep_duration` is reached

                with torch.no_grad():
                    x, h, action = policy(obs, x, h, noise=network_noise)
                    obs, reward, terminated, info = cur_env.step(timestep, action=action)

                xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
                tg.append(info["goal"][:, None, :])  # targets

                timestep += 1

            # concatenate into a (batch_size, n_timesteps, xy) tensor
            xy = torch.cat(xy, axis=1)
            tg = torch.cat(tg, axis=1)

            metrics = detached_position_metrics(
                position_l1_metrics(xy, tg, cur_env.epoch_bounds)
            )
            if total_metrics is None:
                total_metrics = {name: 0.0 for name in metrics}
            for name, value in metrics.items():
                total_metrics[name] += value
            condition_count += 1
            condition_loss += metrics["phase_normalized_position_l1"]
        condition_loss /= len(speed_conds)
        condition_losses[env] = condition_loss
    averaged_metrics = {
        name: value / condition_count for name, value in total_metrics.items()
    }
    total_test_loss = averaged_metrics["phase_normalized_position_l1"]

    print("\n")
    print("Eval Results:")
    for env in condition_losses:
        print(f"Total Loss for Environment {env}| {condition_losses[env]}")
    print(f"Total Testing Loss: {total_test_loss}")
    print(f"Position Metrics: {averaged_metrics}")
    print("\n")

    return averaged_metrics if return_metrics else total_test_loss


def _protocol3_rollout_metrics(
    policy,
    hp,
    environment_class,
    *,
    testing,
    options,
    network_noise,
    deterministic,
):
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    environment = environment_class(
        effector=effector,
        **hp.get("env_kwargs", {}),
    )
    reset_options = dict(options)
    reset_options["deterministic"] = deterministic
    observation, _ = environment.reset(
        testing=testing,
        seed=random.randrange(2**32),
        options=reset_options,
    )
    batch_size = int(reset_options["batch_size"])
    x = torch.zeros((batch_size, hp["hid_size"]))
    h = torch.zeros_like(x)
    positions = []
    targets = []
    timestep = 0
    terminated = False
    while not terminated:
        with torch.no_grad():
            x, h, action = policy(
                observation,
                x,
                h,
                noise=network_noise,
            )
            observation, _, terminated, info = environment.step(
                timestep,
                action=action,
            )
        positions.append(info["states"]["fingertip"][:, None, :])
        targets.append(info["goal"][:, None, :])
        timestep += 1

    actual = torch.cat(positions, dim=1)
    target = torch.cat(targets, dim=1)
    metrics = detached_position_metrics(
        position_l1_metrics(actual, target, environment.epoch_bounds)
    )
    movement_start, movement_end = environment.epoch_bounds["movement"]
    actual_movement = actual[:, movement_start:movement_end]
    target_movement = target[:, movement_start:movement_end]
    euclidean_error = torch.linalg.vector_norm(
        actual_movement - target_movement,
        dim=-1,
    )
    target_span = target_movement.amax(dim=1) - target_movement.amin(dim=1)
    target_diagonal = torch.linalg.vector_norm(target_span, dim=-1)
    if bool(torch.any(target_diagonal <= 0.0)):
        raise RuntimeError("target bounding-box diagonal must be positive")
    mean_error = euclidean_error.mean(dim=1)
    endpoint_error = euclidean_error[:, -1]
    actual_length = torch.linalg.vector_norm(
        torch.diff(actual_movement, dim=1),
        dim=-1,
    ).sum(dim=1)
    target_length = torch.linalg.vector_norm(
        torch.diff(target_movement, dim=1),
        dim=-1,
    ).sum(dim=1)
    if bool(torch.any(target_length <= 0.0)):
        raise RuntimeError("target movement length must be positive")
    metrics.update(
        {
            "mean_euclidean_error_m": float(mean_error.mean()),
            "normalized_mean_error": float(
                (mean_error / target_diagonal).mean()
            ),
            "endpoint_error_m": float(endpoint_error.mean()),
            "normalized_endpoint_error": float(
                (endpoint_error / target_diagonal).mean()
            ),
            "actual_path_length_m": float(actual_length.mean()),
            "target_path_length_m": float(target_length.mean()),
            "path_length_ratio": float((actual_length / target_length).mean()),
        }
    )
    return metrics


def _do_protocol3_eval(
    policy,
    hp,
    env_dict=None,
    network_noise=True,
    return_metrics=False,
):
    if env_dict is None:
        raise ValueError("protocol3 evaluation requires an explicit digit set")
    single_condition = hp.get("condition_schedule") in {
        "protocol3_gate2_single_condition",
        "protocol3_corner_settle_single_condition",
    }
    deterministic = single_condition or bool(hp.get("deterministic_evaluation", False))
    if single_condition:
        cases = (
            (
                next(iter(env_dict.values())),
                False,
                {
                    "batch_size": hp["batch_size"],
                    "reach_conds": np.full(
                        hp["batch_size"],
                        hp["gate2_direction_index"],
                        dtype=np.int64,
                    ),
                    "speed_cond": 0,
                    "delay_cond": hp["gate2_delay_index"],
                },
            ),
        )
    else:
        cases = tuple(
            (
                environment_class,
                True,
                {
                    "batch_size": 32,
                    "reach_conds": np.arange(32, dtype=np.int64),
                    "speed_cond": 0,
                    "delay_cond": delay_index,
                },
            )
            for environment_class in env_dict.values()
            for delay_index in range(3)
        )

    totals = None
    for environment_class, testing, options in cases:
        metrics = _protocol3_rollout_metrics(
            policy,
            hp,
            environment_class,
            testing=testing,
            options=options,
            network_noise=network_noise,
            deterministic=deterministic,
        )
        if totals is None:
            totals = {name: 0.0 for name in metrics}
        for name, value in metrics.items():
            totals[name] += value
    averaged = {
        name: value / len(cases)
        for name, value in totals.items()
    }
    return averaged if return_metrics else averaged["phase_normalized_position_l1"]


def _balanced_protocol3_directions(batch_size):
    return balanced_direction_indices(batch_size)


def _protocol3_training_condition(env_list, hp):
    schedule = hp["condition_schedule"]
    if schedule == "independent_random_digit_delay_v1":
        digit_index, delay_index, environment_seed = sample_protocol3_update(
            random,
            len(env_list),
        )
        environment_class = env_list[digit_index]
        directions = _balanced_protocol3_directions(hp["batch_size"])
    elif schedule in {
        "protocol3_gate2_single_condition",
        "protocol3_corner_settle_single_condition",
    }:
        environment_class = env_list[0]
        delay_index = hp["gate2_delay_index"]
        environment_seed = random.randrange(2**32)
        directions = np.full(
            hp["batch_size"],
            hp["gate2_direction_index"],
            dtype=np.int64,
        )
    else:
        raise ValueError(f"unsupported protocol3 condition schedule: {schedule}")
    return environment_class, delay_index, environment_seed, {
        "batch_size": hp["batch_size"],
        "reach_conds": directions,
        "speed_cond": 0,
        "delay_cond": delay_index,
    }


def _new_protocol3_condition_counts():
    return new_condition_counts()


def _record_protocol3_condition(counts, digit, delay_index):
    record_condition(counts, digit, delay_index)





def do_eval_single_task(policy, hp, task):

    env_dict = {
        "DlyHalfReach": DlyHalfReach, 
        "DlyHalfCircleClk": DlyHalfCircleClk, 
        "DlyHalfCircleCClk": DlyHalfCircleCClk, 
        "DlySinusoid": DlySinusoid, 
        "DlySinusoidInv": DlySinusoidInv,
        "DlyFullReach": DlyFullReach,
        "DlyFullCircleClk": DlyFullCircleClk,
        "DlyFullCircleCClk": DlyFullCircleCClk,
        "DlyFigure8": DlyFigure8,
        "DlyFigure8Inv": DlyFigure8Inv
    }

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    # Currently 10 speed conds during testing, 3 for training
    speed_conds = list(np.arange(0, 10))

    condition_loss = 0
    for speed in speed_conds:

        # initialize batch
        # use 32 to get every possible direction
        x = torch.zeros(size=(32, hp["hid_size"]))
        h = torch.zeros(size=(32, hp["hid_size"]))

        cur_env = env_dict[task](effector=effector)

        # Get first timestep
        obs, info = cur_env.reset(testing=True, options={"batch_size": 32, "reach_conds": np.arange(0, 32), "speed_cond": speed})
        terminated = False

        # initial positions and targets
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]

        timestep = 0
        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached

            with torch.no_grad():
                x, h, action = policy(obs, x, h)
                obs, reward, terminated, info = cur_env.step(timestep, action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets

            timestep += 1

        # concatenate into a (batch_size, n_timesteps, xy) tensor
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)

        # Implement loss function
        loss = l1_dist(xy, tg)  # L1 loss on position
        condition_loss += loss.item()
    condition_loss /= len(speed_conds)

    print("\n")
    print("Eval Results:")
    print(f"Total Loss for Environment {task}| {condition_loss}")
    print("\n")
    
    return condition_loss




def do_eval_compositional_env(policy, hp):

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    # Currently 10 speed conds during testing, 3 for training
    speed_conds = list(np.arange(0, 10))

    forward_motifs = [
        "forward_halfreach",
        "forward_halfcircleclk",
        "forward_halfcirclecclk",
        "forward_sinusoid",
        "forward_sinusoidinv"
    ]

    backward_motifs = [
        "backward_fullreach",
        "backward_fullcircleclk",
        "backward_fullcirclecclk",
        "backward_figure8",
        "backward_figure8inv"
    ]

    combination_idx = list(product(forward_motifs, backward_motifs))
    combination_idx.remove(("forward_halfreach", "backward_fullreach"))
    combination_idx.remove(("forward_halfcircleclk", "backward_fullcircleclk"))
    combination_idx.remove(("forward_halfcirclecclk", "backward_fullcirclecclk"))
    combination_idx.remove(("forward_sinusoid", "backward_figure8"))
    combination_idx.remove(("forward_sinusoidinv", "backward_figure8inv"))

    total_test_loss = 0
    condition_losses = {}
    for combination in combination_idx:
        condition_loss = 0
        for speed in speed_conds:

            # initialize batch
            # use 32 to get every possible direction
            x = torch.zeros(size=(32, hp["hid_size"]))
            h = torch.zeros(size=(32, hp["hid_size"]))

            cur_env = ComposableEnv(effector=effector)

            # Get first timestep
            obs, info = cur_env.reset(
                testing=True, 
                options={
                    "batch_size": 32, 
                    "reach_conds": np.arange(0, 32), 
                    "speed_cond": speed,
                    "forward_key": combination[0], 
                    "backward_key": combination[1], 
                }
            )

            terminated = False

            # initial positions and targets
            xy = [info["states"]["fingertip"][:, None, :]]
            tg = [info["goal"][:, None, :]]

            timestep = 0
            # simulate whole episode
            while not terminated:  # will run until `max_ep_duration` is reached

                with torch.no_grad():
                    x, h, action = policy(obs, x, h)
                    obs, reward, terminated, info = cur_env.step(timestep, action=action)

                xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
                tg.append(info["goal"][:, None, :])  # targets

                timestep += 1

            # concatenate into a (batch_size, n_timesteps, xy) tensor
            xy = torch.cat(xy, axis=1)
            tg = torch.cat(tg, axis=1)

            # Implement loss function
            loss = l1_dist(xy, tg)  # L1 loss on position
            condition_loss += loss.item()
        condition_loss /= len(speed_conds)
        condition_losses[combination] = condition_loss
        total_test_loss += condition_loss
    total_test_loss /= len(combination_idx)

    print("\n")
    print("Eval Results:")
    for combination in condition_losses:
        print(f"Total Loss for Environment {combination}| {condition_losses[combination]}")
    print(f"Total Testing Loss: {total_test_loss}")
    print("\n")
    
    return total_test_loss



def train_2link(model_path, model_file, hp=None):

    # create model path for saving model and hp
    create_dir(model_path)

    def_hp = DEF_HP
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp

    # save hyperparameters
    save_hp(hp, model_path)

    device = torch.device("cpu")
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    if hp["network"] == "rnn":
        policy = RNNPolicy(
            hp["inp_size"],
            hp["hid_size"],
            effector.n_muscles, 
            activation_name=hp["activation_name"],
            noise_level_act=hp["noise_level_act"], 
            noise_level_inp=hp["noise_level_inp"], 
            constrained=hp["constrained"], 
            dt=hp["dt"],
            t_const=hp["t_const"],
            device=device
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")

    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    losses = []
    interval = 100
    best_test_loss = np.inf

    env_list = [
        DlyHalfReach, 
        DlyHalfCircleClk, 
        DlyHalfCircleCClk, 
        DlySinusoid, 
        DlySinusoidInv,
        DlyFullReach,
        DlyFullCircleClk,
        DlyFullCircleCClk,
        DlyFigure8,
        DlyFigure8Inv
    ]

    probs = [1/len(env_list)] * len(env_list)

    for batch in range(hp["epochs"]):

        # initialize batch
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

        rand_env = random.choices(env_list, probs)
        env = rand_env[0](effector=effector)
        epoch_bounds = env.epoch_bounds

        # Get first timestep
        obs, info = env.reset(options={"batch_size": hp["batch_size"]})
        terminated = False

        xy = []
        tg = []
        muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hs = [h.unsqueeze(1)]

        timestep = 0
        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached

            x, h, action = policy(obs, x, h)
            obs, reward, terminated, info = env.step(timestep, action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hs.append(h.unsqueeze(1))

            timestep += 1

        # concatenate into a (batch_size, n_timesteps, xy) tensor
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        muscle_acts = torch.cat(muscle_acts, axis=1)
        hs = torch.cat(hs, axis=1)

        position_metrics = position_l1_metrics(xy, tg, env.epoch_bounds)
        loss = position_metrics["phase_normalized_position_l1"]
        loss += l1_rate(hs, hp["l1_rate"])
        loss += l1_weight(policy, hp["l1_weight"])
        loss += l1_muscle_act(muscle_acts, hp["l1_muscle_act"])
        if hp["activation_name"] != "tanh":
            loss += simple_dynamics(hs, policy.mrnn, weight=hp["simple_dynamics_weight"])
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], sum(losses[-interval:])/interval))
            np.savetxt(os.path.join(model_path, "losses.txt"), losses)

        if (batch % hp["save_iter"] == 0):
            # Get test loss
            test_loss = do_eval(policy, hp)
            # If current test loss is better than previous, save model and update best loss
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'agent_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_path + "/" + model_file)
                print("Model Saved!")
                print(f"Directory: {model_path}/{model_file}")
                print("\n")




def load_prev_training(model_path, model_file):

    hp = load_hp(model_path)
    hp = hp.copy()

    device = torch.device("cpu")
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    if hp["network"] == "rnn":
        policy = RNNPolicy(
            hp["inp_size"],
            hp["hid_size"],
            effector.n_muscles, 
            activation_name=hp["activation_name"],
            noise_level_act=hp["noise_level_act"], 
            noise_level_inp=hp["noise_level_inp"], 
            constrained=hp["constrained"], 
            dt=hp["dt"],
            t_const=hp["t_const"],
            device=device
        )
    elif hp["network"] == "gru":
        policy = GRUPolicy(hp["inp_size"], hp["hid_size"], effector.n_muscles, batch_first=True)
    else:
        raise ValueError("Not a valid architecture")


    checkpoint = torch.load(os.path.join(model_path, model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'])

    # Did not save optimizer (should add later)
    # For now then, just make learning rate really small
    # Later on, include an option to check if optimizer was saved and load it instead
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    losses = []
    interval = 100
    best_test_loss = np.inf

    env_list = [
        DlyHalfReach, 
        DlyHalfCircleClk, 
        DlyHalfCircleCClk, 
        DlySinusoid, 
        DlySinusoidInv,
        DlyFullReach,
        DlyFullCircleClk,
        DlyFullCircleCClk,
        DlyFigure8,
        DlyFigure8Inv
    ]

    probs = [1/len(env_list)] * len(env_list)

    for batch in range(hp["epochs"]):

        # initialize batch
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

        rand_env = random.choices(env_list, probs)
        env = rand_env[0](effector=effector)
        epoch_bounds = env.epoch_bounds

        # Get first timestep
        obs, info = env.reset(options={"batch_size": hp["batch_size"]})
        terminated = False

        # initial positions and targets
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]
        muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hs = [h.unsqueeze(1)]

        timestep = 0
        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached

            x, h, action = policy(obs, x, h)
            obs, reward, terminated, info = env.step(timestep, action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hs.append(h.unsqueeze(1))

            timestep += 1

        # concatenate into a (batch_size, n_timesteps, xy) tensor
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        muscle_acts = torch.cat(muscle_acts, axis=1)
        hs = torch.cat(hs, axis=1)

        # Implement loss function
        loss = l1_dist(xy, tg)  # L1 loss on position
        loss += l1_rate(hs)
        loss += l1_weight(policy, hp["l1_weight"])
        loss += l1_muscle_act(muscle_acts, hp["l1_muscle_act"])
        loss += simple_dynamics(hs, policy.mrnn, weight=hp["simple_dynamics_weight"])
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], sum(losses[-interval:])/interval))

        if (batch % hp["save_iter"] == 0):
            # Get test loss
            test_loss = do_eval(policy, hp)
            # If current test loss is better than previous, save model and update best loss
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                torch.save({
                    'agent_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_path + "/" + model_file)
                print("Model Saved!")
                print(f"Directory: {os.path.join(model_path, model_file)}")
                print("\n")





def _apply_digit8_lr_ablation_optimizer_override(optimizer, checkpoint, hp):
    learning_rate = float(hp["experimental_resume_learning_rate"])
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(
            "experimental resume learning rate must be finite and positive"
        )
    source_learning_rate = float(checkpoint.get("hp", {}).get("lr", -1.0))
    if source_learning_rate != float(hp["experimental_source_learning_rate"]):
        raise ValueError(
            "source checkpoint learning rate differs from the ablation"
        )
    source_group_learning_rates = tuple(
        float(group["lr"]) for group in optimizer.param_groups
    )
    if (
        source_learning_rate <= 0.0
        or not source_group_learning_rates
        or any(
            not np.isclose(
                value,
                source_learning_rate,
                rtol=0.0,
                atol=0.0,
            )
            for value in source_group_learning_rates
        )
    ):
        raise ValueError(
            "source checkpoint optimizer learning rate is inconsistent"
        )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return source_group_learning_rates


def train_subsets_base_model(
    model_path,
    model_file,
    hp=None,
    env_dict=None,
    *,
    resume_checkpoint=None,
    target_completed_updates=None,
    manual_resume_authorized=False,
    expected_resume_repository_head=None,
):

    # create model path for saving model and hp
    create_dir(model_path)

    def_hp = DEF_HP.copy()
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp

    if "seed" in hp:
        random.seed(hp["seed"])
        np.random.seed(hp["seed"])
        torch.manual_seed(hp["seed"])

    # save hyperparameters
    save_hp(hp, model_path)

    device = torch.device("cpu")
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    policy = _build_policy(hp, effector.n_muscles, device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    losses = []
    test_losses = []
    interval = 100
    best_test_loss = np.inf
    last_test_loss = None
    protocol3 = hp.get("protocol") == "digit_writing_original_protocol3"
    condition_counts = _new_protocol3_condition_counts() if protocol3 else None
    gate2_consecutive_passes = 0
    gate2_passed = False
    completed_updates = 0
    best_checkpoint_update = None
    validation_history = []
    deterministic_sentinel_history = []

    if env_dict == None:
        env_dict = {
            "DlyHalfReach": DlyHalfReach, 
            "DlyHalfCircleClk": DlyHalfCircleClk, 
            "DlySinusoid": DlySinusoid, 
            "DlySinusoidInv": DlySinusoidInv, 
            "DlyFullReach": DlyFullReach,
            "DlyFullCircleClk": DlyFullCircleClk,
            "DlyFigure8": DlyFigure8,
            "DlyFigure8Inv": DlyFigure8Inv
        }
    
    # Build an environment list
    env_list = []
    for env in env_dict.values():
        env_list.append(env)

    probs = [1/len(env_list)] * len(env_list)

    schedule = hp.get("condition_schedule")
    gate2 = schedule == "protocol3_gate2_single_condition"
    single_condition = schedule in {
        "protocol3_gate2_single_condition",
        "protocol3_corner_settle_single_condition",
    }
    experimental_resume_learning_rate = hp.get(
        "experimental_resume_learning_rate"
    )
    experimental_disable_gate2_early_stopping = bool(
        hp.get("experimental_disable_gate2_early_stopping", False)
    )
    experimental_lr_arms = {
        "digit8_medium_lr1e3": 0.001,
        "digit8_medium_lr3e4": 0.0003,
        "digit8_medium_lr1e4": 0.0001,
    }
    if (
        experimental_resume_learning_rate is not None
        or experimental_disable_gate2_early_stopping
    ) and not (
        protocol3
        and gate2
        and resume_checkpoint is not None
        and manual_resume_authorized
        and expected_resume_repository_head is not None
    ):
        raise ValueError(
            "experimental Gate 2 resume controls require an authorized "
            "single-condition continuation"
        )
    if (
        experimental_resume_learning_rate is not None
        or experimental_disable_gate2_early_stopping
    ):
        experimental_arm = hp.get("experimental_lr_ablation_arm")
        if (
            experimental_resume_learning_rate is None
            or hp.get("experimental_run_kind")
            != "protocol3_digit8_equal_point_lr_ablation"
            or experimental_arm not in experimental_lr_arms
            or float(experimental_resume_learning_rate)
            != experimental_lr_arms[experimental_arm]
            or float(hp.get("experimental_source_learning_rate", -1.0))
            != 0.001
            or not experimental_disable_gate2_early_stopping
            or hp.get("variant") != "o3_digit8"
            or hp.get("gate2_direction_index") != 0
            or hp.get("gate2_delay_index") != PROTOCOL3_DELAYS.index(50)
            or env_dict is None
            or len(env_dict) != 1
            or next(iter(env_dict.values())).FIXED_DIGIT != 8
        ):
            raise ValueError("digit8 learning-rate ablation identity is invalid")
    if resume_checkpoint is not None:
        if not protocol3:
            raise ValueError("only protocol3 supports continuation resume")
        if not manual_resume_authorized:
            raise PermissionError("protocol3 resume requires explicit manual approval")
        checkpoint = torch.load(
            resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        restore_state = validate_protocol3_resume_checkpoint(
            checkpoint,
            hp["protocol_config"],
        )
        if single_condition:
            source_identity = checkpoint["git_identity"]
            if expected_resume_repository_head is None:
                raise ValueError("single-condition continuation requires its expected source HEAD")
            if (
                source_identity["repository_head"]
                != expected_resume_repository_head
            ):
                raise ValueError("Gate 2 continuation source HEAD does not match")
            for key in (
                "branch",
                "mrnntorch_recorded_head",
                "mrnntorch_worktree_head",
            ):
                if source_identity[key] != hp["git_identity"][key]:
                    raise ValueError(
                        f"single-condition continuation source {key} does not match"
                    )
            if checkpoint.get("variant") != hp.get("variant"):
                raise ValueError("Gate 2 continuation case identity does not match")
            source_hp = checkpoint.get("hp", {})
            if source_hp.get("git_identity") != source_identity:
                raise ValueError(
                    "Gate 2 continuation checkpoint Git identities differ"
                )
            for key in (
                "condition_schedule",
                "gate2_direction_index",
                "gate2_delay_index",
                "batch_size",
                "save_iter",
                "env_kwargs",
            ):
                if source_hp.get(key) != hp.get(key):
                    raise ValueError(
                        f"single-condition continuation source {key} does not match"
                    )
            if "gate2_consecutive_passes" not in restore_state:
                raise ValueError("single-condition continuation state is incomplete")
            source_updates = int(restore_state["completed_updates"])
            if source_updates < int(hp["epochs"]):
                raise ValueError("continuation requires a completed initial run")
            if source_updates % int(hp["save_iter"]) != 0:
                raise ValueError("continuation source must end on a validation boundary")
            if gate2 and int(restore_state["gate2_consecutive_passes"]) >= 2:
                raise ValueError("a passed Gate 2 case must not be continued")
        else:
            if expected_resume_repository_head is not None:
                raise ValueError(
                    "full10 continuation does not accept a different source HEAD"
                )
            if checkpoint["git_identity"] != hp["git_identity"]:
                raise ValueError("continuation checkpoint Git identity does not match")
            if int(restore_state["completed_updates"]) != 5_000:
                raise ValueError("protocol3 continuation must start at update 5000")
        policy.load_state_dict(checkpoint["agent_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if experimental_resume_learning_rate is not None:
            _apply_digit8_lr_ablation_optimizer_override(
                optimizer, checkpoint, hp
            )
        completed_updates = int(restore_state["completed_updates"])
        if (
            experimental_resume_learning_rate is not None
            and completed_updates != 6000
        ):
            raise ValueError("digit8 learning-rate ablation must start at update 6000")
        condition_counts = restore_state["condition_counts"]
        validation_history = list(restore_state["validation_history"])
        deterministic_sentinel_history = list(
            restore_state.get("deterministic_sentinel_history", [])
        )
        test_losses = [
            float(row["phase_normalized_position_l1"])
            for row in validation_history
        ]
        best_test_loss = float(restore_state["best_validation_loss"])
        best_checkpoint_update = int(restore_state["best_checkpoint_update"])
        last_test_loss = float(restore_state["last_validation_loss"])
        gate2_consecutive_passes = int(
            restore_state.get("gate2_consecutive_passes", 0)
        )
        restore_rng_state(checkpoint["rng_state"])

    stop_after_updates = int(
        target_completed_updates
        if target_completed_updates is not None
        else hp.get("stop_after_updates", hp["epochs"])
    )
    if (
        experimental_resume_learning_rate is not None
        and stop_after_updates != 7000
    ):
        raise ValueError("digit8 learning-rate ablation must stop at update 7000")
    if stop_after_updates > int(hp["epochs"]):
        single_condition_extension = bool(
            protocol3
            and single_condition
            and resume_checkpoint is not None
            and manual_resume_authorized
            and expected_resume_repository_head is not None
        )
        if not single_condition_extension:
            raise ValueError("target completed updates exceed the frozen maximum")
    if stop_after_updates <= completed_updates:
        raise ValueError("target completed updates must exceed the checkpoint state")
    if single_condition and stop_after_updates % int(hp["save_iter"]) != 0:
        raise ValueError("single-condition target must end on a validation boundary")

    def protocol3_training_state():
        return {
            "completed_updates": completed_updates,
            "next_update": completed_updates,
            "condition_counts": condition_counts,
            "validation_history": validation_history,
            "deterministic_sentinel_history": deterministic_sentinel_history,
            "best_validation_loss": best_test_loss,
            "best_checkpoint_update": best_checkpoint_update,
            "last_validation_loss": last_test_loss,
            "gate2_consecutive_passes": gate2_consecutive_passes,
        }

    def validate_protocol3_checkpoint(*, count_gate2_pass):
        nonlocal best_test_loss
        nonlocal best_checkpoint_update
        nonlocal gate2_consecutive_passes
        nonlocal gate2_passed
        nonlocal last_test_loss

        validation_metrics = _run_validation(
            policy,
            hp,
            env_dict,
            network_noise=not single_condition,
            return_metrics=True,
        )
        if not all(np.isfinite(value) for value in validation_metrics.values()):
            raise RuntimeError("protocol3 validation produced NaN or Inf")
        test_loss = validation_metrics["phase_normalized_position_l1"]
        last_test_loss = test_loss
        test_losses.append(test_loss)
        validation_row = {
            "completed_updates": completed_updates,
            **validation_metrics,
        }
        validation_history.append(validation_row)
        np.savetxt(os.path.join(model_path, "test_losses.txt"), test_losses)
        _append_jsonl(
            os.path.join(model_path, "test_position_metrics.jsonl"),
            validation_row,
        )
        _append_jsonl(
            os.path.join(model_path, "training_condition_counts.jsonl"),
            {
                "completed_updates": completed_updates,
                **condition_counts,
            },
        )
        if not single_condition:
            sentinel_hp = dict(hp)
            sentinel_hp["deterministic_evaluation"] = True
            sentinel_metrics = _run_validation(
                policy,
                sentinel_hp,
                env_dict,
                network_noise=False,
                return_metrics=True,
            )
            if not all(np.isfinite(value) for value in sentinel_metrics.values()):
                raise RuntimeError("protocol3 deterministic sentinel produced NaN or Inf")
            sentinel_row = {
                "completed_updates": completed_updates,
                **sentinel_metrics,
            }
            deterministic_sentinel_history.append(sentinel_row)
            _append_jsonl(
                os.path.join(model_path, "deterministic_sentinel_metrics.jsonl"),
                sentinel_row,
            )

        if gate2 and count_gate2_pass:
            passed_now = bool(
                validation_metrics["normalized_mean_error"] <= 0.08
                and validation_metrics["normalized_endpoint_error"] <= 0.05
                and 0.85 <= validation_metrics["path_length_ratio"] <= 1.15
            )
            gate2_consecutive_passes = (
                gate2_consecutive_passes + 1 if passed_now else 0
            )
            gate2_passed = gate2_consecutive_passes >= 2

        is_best = test_loss <= best_test_loss
        if is_best:
            best_test_loss = test_loss
            best_checkpoint_update = completed_updates
            torch.save(
                _checkpoint_payload(
                    policy,
                    optimizer,
                    hp,
                    completed_updates,
                    test_loss,
                    training_state=protocol3_training_state(),
                ),
                os.path.join(model_path, model_file),
            )
            print("Model Saved!")
            print(f"Directory: {model_path}/{model_file}")
            print("\n")

    if protocol3 and completed_updates == 0:
        validate_protocol3_checkpoint(count_gate2_pass=False)
        initial_model_file = hp.get("initial_model_file")
        if initial_model_file:
            torch.save(
                _checkpoint_payload(
                    policy,
                    optimizer,
                    hp,
                    0,
                    last_test_loss,
                    training_state=protocol3_training_state(),
                ),
                os.path.join(model_path, initial_model_file),
            )

    for batch in range(completed_updates, stop_after_updates):

        # initialize batch
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

        if protocol3:
            (
                env_class,
                delay_index,
                environment_seed,
                reset_options,
            ) = _protocol3_training_condition(env_list, hp)
            env = env_class(effector=effector, **hp.get("env_kwargs", {}))
            _record_protocol3_condition(
                condition_counts,
                int(env_class.FIXED_DIGIT),
                delay_index,
            )
        else:
            rand_env = random.choices(env_list, probs)
            env = rand_env[0](effector=effector, **hp.get("env_kwargs", {}))
            reset_options = {"batch_size": hp["batch_size"]}

        # Get first timestep
        obs, info = env.reset(
            seed=environment_seed if protocol3 else None,
            options=reset_options,
        )
        terminated = False

        xy = []
        tg = []
        muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hs = [h.unsqueeze(1)]

        timestep = 0
        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached

            x, h, action = policy(obs, x, h)
            obs, reward, terminated, info = env.step(timestep, action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hs.append(h.unsqueeze(1))

            timestep += 1

        # concatenate into a (batch_size, n_timesteps, xy) tensor
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        muscle_acts = torch.cat(muscle_acts, axis=1)
        hs = torch.cat(hs, axis=1)

        position_metrics = position_l1_metrics(xy, tg, env.epoch_bounds)
        loss = position_metrics["phase_normalized_position_l1"]
        loss += l1_rate(hs, hp["l1_rate"])
        loss += l1_weight(policy, hp["l1_weight"])
        loss += l1_muscle_act(muscle_acts, hp["l1_muscle_act"])
        if hp["activation_name"] != "tanh":
            loss += simple_dynamics(hs, policy.mrnn, weight=hp["simple_dynamics_weight"])
        losses.append(loss.item())
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=hp.get("grad_clip_norm", 1.0)
        )  # important!
        optimizer.step()
        completed_updates = batch + 1

        log_training_metrics = (
            protocol3 and completed_updates % interval == 0
        ) or (
            not protocol3 and batch % interval == 0 and batch != 0
        )
        if log_training_metrics:
            mean_loss = sum(losses[-interval:]) / interval
            progress_target = (
                stop_after_updates
                if single_condition and resume_checkpoint is not None
                else hp["epochs"]
            )
            print(
                "Batch {}/{} Done, mean policy loss: {}".format(
                    batch, progress_target, mean_loss
                )
            )
            _append_jsonl(
                os.path.join(model_path, "training_position_metrics.jsonl"),
                {
                    "update": completed_updates if protocol3 else batch,
                    **detached_position_metrics(position_metrics),
                },
            )

        if protocol3 and completed_updates % hp["save_iter"] == 0:
            validate_protocol3_checkpoint(count_gate2_pass=True)
            if gate2_passed and not experimental_disable_gate2_early_stopping:
                break
        elif not protocol3 and (batch % hp["save_iter"] == 0):
            # Get test loss
            validation_metrics = _run_validation(
                policy,
                hp,
                env_dict,
                network_noise=True,
                return_metrics=True,
            )
            test_loss = validation_metrics["phase_normalized_position_l1"]
            last_test_loss = test_loss
            test_losses.append(test_loss)
            np.savetxt(os.path.join(model_path, "test_losses.txt"), test_losses)
            _append_jsonl(
                os.path.join(model_path, "test_position_metrics.jsonl"),
                {"update": batch, **validation_metrics},
            )
            # If current test loss is better than previous, save model and update best loss
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    _checkpoint_payload(
                        policy,
                        optimizer,
                        hp,
                        batch,
                        test_loss,
                    ),
                    os.path.join(model_path, model_file),
                )
                print("Model Saved!")
                print(f"Directory: {model_path}/{model_file}")
                print("\n")
    final_model_file = hp.get("final_model_file")
    if final_model_file:
        training_state = protocol3_training_state() if protocol3 else None
        torch.save(
            _checkpoint_payload(
                policy,
                optimizer,
                hp,
                completed_updates if protocol3 else completed_updates - 1,
                last_test_loss,
                training_state=training_state,
            ),
            os.path.join(model_path, final_model_file),
        )
    early_audit_pending = bool(
        protocol3
        and not gate2
        and completed_updates == 5_000
        and stop_after_updates == 5_000
    )
    if early_audit_pending:
        _write_json(
            os.path.join(model_path, "early_audit_pending.json"),
            {
                "completed_updates": completed_updates,
                "continuation_checkpoint": final_model_file,
                "early_audit_pending": True,
                "automatic_resume_allowed": False,
            },
        )
        print("EARLY_AUDIT_PENDING=1")
    if hp.get("protocol_config") is not None:
        return {
            "variant": hp.get("variant"),
            "seed": hp.get("seed"),
            "updates": completed_updates,
            "best_validation_loss": best_test_loss,
            "last_validation_loss": last_test_loss,
            "gate2_passed": gate2_passed if protocol3 else None,
            "condition_counts": condition_counts,
            "best_checkpoint_update": best_checkpoint_update,
            "validation_history": validation_history if protocol3 else None,
            "early_audit_pending": early_audit_pending,
        }






def train_subsets_held_out_base_model(
    load_model_path,
    load_model_file,
    save_model_path,
    save_model_file,
    hp=None,
    env_dict=None,
    retained_env_dict=None,
    rule_index=5,
):
    create_dir(save_model_path)
    def_hp = DEF_HP.copy()
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp
    if "seed" in hp:
        random.seed(hp["seed"])
        np.random.seed(hp["seed"])
        torch.manual_seed(hp["seed"])
    save_hp(hp, save_model_path)

    if env_dict is None:
        env_dict = {"digit_5": DlyFullReach}
    if retained_env_dict is None:
        retained_env_dict = {
            f"digit_{digit}": DIGIT_ENV_CLASSES[digit]
            for digit in HELDOUT5_DIGITS
        }

    policy, _ = load_digit_policy_checkpoint(
        os.path.join(load_model_path, load_model_file),
        expected_variant="heldout5",
    )
    source_state = {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
    }
    retained_before = _run_validation(
        policy,
        hp,
        retained_env_dict,
        network_noise=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(hp["seed"])
    rule_parameter = policy.prepare_rule_column_transfer(
        rule_index, generator=generator
    )
    optimizer = torch.optim.Adam(
        policy.transfer_trainable_parameters(), lr=hp["lr"]
    )

    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
    losses = []
    test_losses = []
    interval = 100
    best_test_loss = np.inf
    last_test_loss = None
    env_list = list(env_dict.values())
    probs = [1/len(env_list)] * len(env_list)

    for batch in range(hp["epochs"]):
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        env_class = random.choices(env_list, probs)[0]
        env = env_class(effector=effector, **hp.get("env_kwargs", {}))
        obs, info = env.reset(options={"batch_size": hp["batch_size"]})
        terminated = False
        xy = []
        tg = []
        timestep = 0
        while not terminated:
            x, h, action = policy(obs, x, h)
            obs, _, terminated, info = env.step(timestep, action=action)
            xy.append(info["states"]["fingertip"][:, None, :])
            tg.append(info["goal"][:, None, :])
            timestep += 1
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        position_metrics = position_l1_metrics(xy, tg, env.epoch_bounds)
        loss = position_metrics["phase_normalized_position_l1"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (rule_parameter,), max_norm=hp.get("grad_clip_norm", 1.0)
        )
        optimizer.step()
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            mean_loss = sum(losses[-interval:]) / interval
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], mean_loss))
            np.savetxt(os.path.join(save_model_path, "losses.txt"), losses)
            _append_jsonl(
                os.path.join(save_model_path, "training_position_metrics.jsonl"),
                {"update": batch, **detached_position_metrics(position_metrics)},
            )

        if (batch % hp["save_iter"] == 0):
            validation_metrics = _run_validation(
                policy, hp, env_dict, return_metrics=True
            )
            test_loss = validation_metrics["phase_normalized_position_l1"]
            last_test_loss = test_loss
            test_losses.append(test_loss)
            np.savetxt(os.path.join(save_model_path, "test_losses.txt"), test_losses)
            _append_jsonl(
                os.path.join(save_model_path, "test_position_metrics.jsonl"),
                {"update": batch, **validation_metrics},
            )
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    _checkpoint_payload(
                        policy, optimizer, hp, batch, test_loss
                    ),
                    os.path.join(save_model_path, save_model_file),
                )
                print("Model Saved!")
                print(f"Directory: {save_model_path}/{save_model_file}")
                print("\n")

    freeze_audit = audit_rule_column_only_change(
        policy, source_state, rule_index=rule_index
    )
    retained_after = _run_validation(
        policy,
        hp,
        retained_env_dict,
        network_noise=False,
    )
    max_retained_difference = abs(retained_before - retained_after)
    if max_retained_difference > 1e-12:
        raise RuntimeError("a retained-digit deterministic output changed")

    final_model_file = hp.get("final_model_file")
    if final_model_file:
        torch.save(
            _checkpoint_payload(
                policy,
                optimizer,
                hp,
                hp["epochs"] - 1,
                last_test_loss,
            ),
            os.path.join(save_model_path, final_model_file),
        )
    audit = {
        "best_target_validation_loss": best_test_loss,
        "last_target_validation_loss": last_test_loss,
        "retained_before": retained_before,
        "retained_after": retained_after,
        "max_retained_difference": max_retained_difference,
        "freeze_audit": freeze_audit,
    }
    _write_json(os.path.join(save_model_path, "transfer_audit.json"), audit)
    return audit





def train_compositional_env_base_model(
    load_model_path, 
    load_model_file, 
    save_model_path, 
    save_model_file, 
    hp=None
):

    # create model path for saving model and hp
    create_dir(save_model_path)

    def_hp = DEF_HP
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp

    # save hyperparameters
    save_hp(hp, save_model_path)

    device = torch.device("cpu")
    effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())

    policy = RNNPolicy(
        hp["inp_size"],
        hp["hid_size"],
        effector.n_muscles, 
        activation_name=hp["activation_name"],
        noise_level_act=hp["noise_level_act"], 
        noise_level_inp=hp["noise_level_inp"], 
        constrained=hp["constrained"], 
        dt=hp["dt"],
        t_const=hp["t_const"],
        device=device,
        add_new_rule_inputs=True
    )

    checkpoint = torch.load(os.path.join(load_model_path, load_model_file), map_location=torch.device('cpu'))
    policy.load_state_dict(checkpoint['agent_state_dict'], strict=False)

    for name, param in policy.named_parameters():
        if name != "mrnn.input_new_rules_region":
            param.requires_grad = False

    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["lr"])

    losses = []
    interval = 100
    best_loss = np.inf

    for batch in range(hp["epochs"]):

        # initialize batch
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

        env = ComposableEnv(effector=effector)

        # Get first timestep
        obs, info = env.reset(options={"batch_size": hp["batch_size"]})
        terminated = False

        # initial positions and targets
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]
        muscle_acts = [info["states"]["muscle"][:, 0].unsqueeze(1)]
        hs = [h.unsqueeze(1)]

        timestep = 0
        # simulate whole episode
        while not terminated:  # will run until `max_ep_duration` is reached

            x, h, action = policy(obs, x, h)
            obs, reward, terminated, info = env.step(timestep, action=action)

            xy.append(info["states"]["fingertip"][:, None, :])  # trajectories
            tg.append(info["goal"][:, None, :])  # targets
            muscle_acts.append(info["states"]["muscle"][:, 0].unsqueeze(1))
            hs.append(h.unsqueeze(1))

            timestep += 1

        # concatenate into a (batch_size, n_timesteps, xy) tensor
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        muscle_acts = torch.cat(muscle_acts, axis=1)
        hs = torch.cat(hs, axis=1)

        # Implement loss function
        loss = l1_dist(xy, tg)  # L1 loss on position
        
        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], sum(losses[-interval:])/interval))
            np.savetxt(os.path.join(save_model_path, "losses.txt"), losses)
            torch.save({
                'agent_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_model_path + "/" + save_model_file)
            print("Model Saved!")
            print(f"Directory: {save_model_path}/{save_model_file}")
            print("\n")





def _require_cpu(config):
    if config["device"] != "cpu":
        raise ValueError("the project-2 baseline device must be cpu")


def _digit_env_dict(digits):
    return {
        f"digit_{digit}": DIGIT_ENV_CLASSES[digit]
        for digit in digits
    }


def _validate_base_config(config):
    protocol = config.get("protocol")
    if protocol not in {
        "digit_writing_original_protocol2",
        "digit_writing_original_protocol3",
    }:
        raise ValueError("unsupported digit-writing protocol")
    if config["run_kind"] != "base_training":
        raise ValueError("base training requires run_kind=base_training")
    if config["variant"] not in {"full10", "heldout5"}:
        raise ValueError("base variant must be full10 or heldout5")
    expected = (
        FULL10_DIGITS if config["variant"] == "full10" else HELDOUT5_DIGITS
    )
    if tuple(config["train_digits"]) != expected:
        raise ValueError(f"{config['variant']} has an invalid training digit set")
    if "source_checkpoint" in config:
        raise ValueError("base training must not load a checkpoint")
    _require_cpu(config)

    model = config["model"]
    if (
        model["network"] != "rnn"
        or model["input_size"] != 28
        or model["hidden_size"] != 256
        or model["output_size"] != 6
        or model["activation"] != "softplus"
        or model["recurrent_noise_std"] != 0.1
        or model["input_noise_std"] != 0.01
        or model["constrained"]
        or model["rnn_dt_ms"] != 10
        or model["rnn_tau_ms"] != 20
        or not model["batch_first"]
    ):
        raise ValueError("base model does not match the frozen protocol")
    optimizer = config["optimizer"]
    if (
        optimizer["name"] != "Adam"
        or optimizer["learning_rate"] != 0.001
        or optimizer["grad_clip_norm"] != 1.0
    ):
        raise ValueError("base optimizer does not match the project-2 protocol")
    training = config["training"]
    if protocol == "digit_writing_original_protocol2":
        if (
            training["batch_size"] != 32
            or training["max_updates"] != 75_000
            or training["validation_interval"] != 500
        ):
            raise ValueError("base training schedule does not match the protocol")
    else:
        if config["variant"] != "full10":
            raise ValueError("protocol3 currently authorizes only full10 base training")
        if (
            training["batch_size"] != 32
            or training["max_updates"] != 75_000
            or training["validation_interval"] != 500
            or training.get("stop_after_updates") != 5_000
        ):
            raise ValueError("protocol3 full10 must stop after the first 5000 updates")
        if config.get("timing_mode") != "fixed_segment_timing":
            raise ValueError("protocol3 requires fixed_segment_timing")
        if config.get("selected_reference_steps") not in {50, 100}:
            raise ValueError("protocol3 selected_reference_steps must be 50 or 100")
        if config.get("direction_mode") != "balanced_8":
            raise ValueError("protocol3 requires balanced_8 training directions")
        if config.get("condition_schedule") != "independent_random_digit_delay_v1":
            raise ValueError("protocol3 condition schedule differs from the protocol")
        if config.get("digit_sampling") != "uniform_independent":
            raise ValueError("protocol3 digit sampling must be uniform independent")
        if config.get("delay_sampling") != "uniform_independent":
            raise ValueError("protocol3 delay sampling must be uniform independent")
        expected_scale = (
            2.5 if "scale2p50" in config["geometry_config"] else 2.25
        )
        if config.get("scale_multiplier") != expected_scale:
            raise ValueError("protocol3 scale identity does not match its geometry config")
        output = config["output"]
        if (
            output.get("best_checkpoint") != "best_checkpoint.pt"
            or output.get("initial_checkpoint") != "initial_checkpoint.pt"
            or output.get("final_checkpoint")
            != "early_continuation_checkpoint.pt"
            or "digit_writing_original_protocol3" not in output["directory"]
            or "protocol2" in output["directory"]
        ):
            raise ValueError("protocol3 full10 checkpoint/output identity is invalid")
    if config["regularization"] != {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }:
        raise ValueError("base regularization does not match the protocol")
    if config["position_loss"] != {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }:
        raise ValueError("base position loss does not match the final protocol")


def _base_hp_from_config(config):
    model = config["model"]
    optimizer = config["optimizer"]
    training = config["training"]
    regularization = config["regularization"]
    output = config["output"]
    hp = {
        "network": model["network"],
        "inp_size": model["input_size"],
        "hid_size": model["hidden_size"],
        "activation_name": model["activation"],
        "noise_level_act": model["recurrent_noise_std"],
        "noise_level_inp": model["input_noise_std"],
        "constrained": model["constrained"],
        "dt": model["rnn_dt_ms"],
        "t_const": model["rnn_tau_ms"],
        "batch_first": model["batch_first"],
        "lr": optimizer["learning_rate"],
        "grad_clip_norm": optimizer["grad_clip_norm"],
        "batch_size": training["batch_size"],
        "epochs": training["max_updates"],
        "save_iter": training["validation_interval"],
        "l1_rate": regularization["l1_rate"],
        "l1_weight": regularization["l1_weight"],
        "l1_muscle_act": regularization["l1_muscle_act"],
        "simple_dynamics_weight": regularization["simple_dynamics_weight"],
        "seed": config["seed"],
        "validation_seed": config["validation_seed"],
        "variant": config["variant"],
        "env_kwargs": {"geometry_config_path": config["geometry_config"]},
        "final_model_file": output["final_checkpoint"],
        "protocol_config": config,
        "protocol": config["protocol"],
    }
    if config["protocol"] == "digit_writing_original_protocol3":
        hp.update(
            {
                "condition_schedule": config["condition_schedule"],
                "git_identity": current_git_identity(
                    Path(__file__).resolve().parent
                ),
                "initial_model_file": output.get("initial_checkpoint"),
                "stop_after_updates": training.get(
                    "stop_after_updates",
                    training["max_updates"],
                ),
            }
        )
    return hp


def train_digit_base_model(config):
    """Map one digit configuration onto the original base-training core."""
    _validate_base_config(config)
    output = config["output"]
    output_dir = Path(output["directory"])
    if (
        config["protocol"] == "digit_writing_original_protocol3"
        and output_dir.exists()
    ):
        raise FileExistsError(f"Protocol3 output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    return train_subsets_base_model(
        str(output_dir),
        output["best_checkpoint"],
        hp=_base_hp_from_config(config),
        env_dict=_digit_env_dict(tuple(config["train_digits"])),
    )


def _validate_protocol3_gate2_config(config):
    if config.get("protocol") != "digit_writing_original_protocol3":
        raise ValueError("Gate 2 requires protocol3")
    if config.get("run_kind") != "protocol3_gate2":
        raise ValueError("Gate 2 requires run_kind=protocol3_gate2")
    if config.get("timing_mode") != "fixed_segment_timing":
        raise ValueError("Gate 2 requires fixed_segment_timing")
    if config.get("selected_reference_steps") not in {50, 100}:
        raise ValueError("Gate 2 reference must be 50 or 100")
    expected_scale = (
        2.5 if "scale2p50" in config["geometry_config"] else 2.25
    )
    if config.get("scale_multiplier") != expected_scale:
        raise ValueError("Gate 2 scale identity does not match its geometry config")
    _require_cpu(config)
    if config["training"] != {
        "batch_size": 8,
        "max_updates": 3000,
        "validation_interval": 100,
    }:
        raise ValueError("Gate 2 training schedule differs from the protocol")
    expected_cases = (
        ("o1_digit1", 1, 0, 50),
        ("o2_digit5", 5, 0, 50),
        ("o3_digit8", 8, 0, 50),
    )
    actual_cases = tuple(
        (
            case["label"],
            case["digit"],
            case["direction_index"],
            case["delay_steps"],
        )
        for case in config["cases"]
    )
    if actual_cases != expected_cases:
        raise ValueError("Gate 2 cases must be O1 digit1, O2 digit5, O3 digit8")
    model = config["model"]
    if model != {
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
    }:
        raise ValueError("Gate 2 model differs from the frozen protocol")
    if config["optimizer"] != {
        "name": "Adam",
        "learning_rate": 0.001,
        "grad_clip_norm": 1.0,
    }:
        raise ValueError("Gate 2 optimizer differs from the frozen protocol")
    if config["regularization"] != {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }:
        raise ValueError("Gate 2 regularization differs from the frozen protocol")
    if config["position_loss"] != {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }:
        raise ValueError("Gate 2 position loss differs from the frozen protocol")


def train_digit_protocol3_gate2(config):
    _validate_protocol3_gate2_config(config)
    from digit_writing.geometry import load_geometry_config
    from digit_writing.geometry_audit import build_workspace_audit
    from digit_writing.protocol3_gate2 import audit_gate2_case

    output = config["output"]
    output_root = Path(output["directory"])
    if output_root.exists():
        raise FileExistsError(f"Gate 2 output already exists: {output_root}")
    output_root.mkdir(parents=True)
    _write_json(output_root / "resolved_config.json", config)
    workspace_audit = build_workspace_audit(
        load_geometry_config(config["geometry_config"])
    )
    _write_json(output_root / "workspace_audit.json", workspace_audit)
    summaries = []
    for case in config["cases"]:
        case_directory = output_root / case["label"]
        hp = _base_hp_from_config(config)
        hp.update(
            {
                "variant": case["label"],
                "condition_schedule": "protocol3_gate2_single_condition",
                "gate2_direction_index": case["direction_index"],
                "gate2_delay_index": PROTOCOL3_DELAYS.index(case["delay_steps"]),
                "stop_after_updates": config["training"]["max_updates"],
            }
        )
        training_summary = train_subsets_base_model(
            str(case_directory),
            output["best_checkpoint"],
            hp=hp,
            env_dict=_digit_env_dict((case["digit"],)),
        )
        final_checkpoint = case_directory / output["final_checkpoint"]
        policy, checkpoint = load_digit_policy_checkpoint(
            final_checkpoint,
            expected_variant=case["label"],
        )
        checkpoint_error = None
        try:
            checkpoint_state = validate_protocol3_resume_checkpoint(
                checkpoint,
                config,
            )
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
            case_directory,
            workspace_audit,
        )
        engineering_passed = bool(
            checkpoint_complete and candidate_audit["engineering_passed"]
        )
        behavior_passed = bool(
            training_summary["gate2_passed"]
            and candidate_audit["behavior_passed"]
        )
        if not engineering_passed:
            classification = "engineering_or_safety_failure"
        elif not behavior_passed:
            classification = "behavior_failure"
        else:
            classification = "provisional_pass_pending_qualitative_review"
        summaries.append(
            {
                **case,
                **training_summary,
                "checkpoint_complete": checkpoint_complete,
                "checkpoint_error": checkpoint_error,
                "candidate_audit": candidate_audit,
                "engineering_passed": engineering_passed,
                "behavior_passed": behavior_passed,
                "classification": classification,
            }
        )
    engineering_passed = all(case["engineering_passed"] for case in summaries)
    behavior_passed = all(case["behavior_passed"] for case in summaries)
    automatic_metrics_passed = bool(engineering_passed and behavior_passed)
    if not engineering_passed:
        classification = "engineering_or_safety_failure"
    elif not behavior_passed:
        classification = "behavior_failure"
    else:
        classification = "provisional_pass_pending_qualitative_review"
    result = {
        "protocol": "digit_writing_original_protocol3",
        "run_kind": "protocol3_gate2",
        "git_identity": hp["git_identity"],
        "scale_multiplier": config["scale_multiplier"],
        "selected_reference_steps": config["selected_reference_steps"],
        "cases": summaries,
        "engineering_passed": engineering_passed,
        "behavior_passed": behavior_passed,
        "automatic_metrics_passed": automatic_metrics_passed,
        "gradient_metrics_are_diagnostic_only": True,
        "qualitative_overlay_review_required": automatic_metrics_passed,
        "classification": classification,
        "medium_fallback_allowed": bool(
            config["selected_reference_steps"] == 50
            and classification == "behavior_failure"
        ),
        "passed": False,
    }
    _write_json(output_root / "gate2_summary.json", result)
    return result


def load_digit_policy_checkpoint(checkpoint_path, expected_variant=None):
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    hp = checkpoint.get("hp")
    if hp is None:
        raise ValueError("checkpoint does not contain the project-2 hyperparameters")
    variant = checkpoint.get("variant")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"expected {expected_variant} checkpoint, got {variant}")
    policy = _build_policy(hp, 6, torch.device("cpu"))
    policy.load_state_dict(checkpoint["agent_state_dict"])
    return policy, checkpoint


def audit_rule_column_only_change(policy, before_state, rule_index=5):
    """Fail unless every state value except one rule column is unchanged."""
    after_state = policy.state_dict()
    input_parameter = policy.rule_input_weight()
    input_key = next(
        name
        for name, parameter in policy.named_parameters()
        if parameter is input_parameter
    )
    for name, before in before_state.items():
        after = after_state[name]
        if name != input_key:
            if not torch.equal(before, after):
                raise RuntimeError(f"frozen transfer parameter changed: {name}")
            continue
        if not torch.equal(before[:, :rule_index], after[:, :rule_index]):
            raise RuntimeError("a rule column before digit 5 changed")
        if not torch.equal(before[:, rule_index + 1 :], after[:, rule_index + 1 :]):
            raise RuntimeError("a rule column after digit 5 changed")
        if torch.equal(before[:, rule_index], after[:, rule_index]):
            raise RuntimeError("the digit-5 rule column did not change")
    return {"input_parameter": input_key, "changed_rule_column": rule_index}


def _validate_transfer_config(config):
    if config["run_kind"] != "transfer5" or config["target_digit"] != 5:
        raise ValueError("transfer requires run_kind=transfer5 and target_digit=5")
    _require_cpu(config)
    optimizer = config["optimizer"]
    if (
        optimizer["name"] != "Adam"
        or optimizer["learning_rate"] != 0.001
        or optimizer["grad_clip_norm"] != 1.0
    ):
        raise ValueError("transfer optimizer does not match the original entry")
    training = config["training"]
    if (
        training["batch_size"] != 32
        or training["max_updates"] != 75_000
        or training["validation_interval"] != 500
    ):
        raise ValueError("transfer schedule does not match the original entry")
    if config["position_loss"] != {
        "type": "phase_normalized_l1",
        "stable_weight": 0.1,
        "delay_weight": 0.1,
        "movement_weight": 0.6,
        "hold_weight": 0.2,
    }:
        raise ValueError("transfer position loss does not match the final protocol")


def train_digit_transfer5(config):
    """Map digit-5 transfer onto the original held-out training core."""
    _validate_transfer_config(config)
    source_path = Path(config["source_checkpoint"])
    source = torch.load(
        source_path, map_location=torch.device("cpu"), weights_only=False
    )
    if source.get("variant") != "heldout5":
        raise ValueError("digit-5 transfer requires a heldout5 checkpoint")
    source_config = source.get("protocol_config")
    if source_config is None:
        raise ValueError("heldout5 checkpoint is missing its protocol config")
    if source_config.get("protocol") != "digit_writing_original_protocol2":
        raise ValueError("transfer requires a project-2 heldout5 checkpoint")
    if config["geometry_config"] != source_config["geometry_config"]:
        raise ValueError("transfer geometry must match the heldout5 checkpoint")
    if config["position_loss"] != source_config["position_loss"]:
        raise ValueError("transfer position loss must match the heldout5 checkpoint")

    training = config["training"]
    output = config["output"]
    hp = source["hp"].copy()
    hp.update(
        {
            "lr": config["optimizer"]["learning_rate"],
            "grad_clip_norm": config["optimizer"]["grad_clip_norm"],
            "batch_size": training["batch_size"],
            "epochs": training["max_updates"],
            "save_iter": training["validation_interval"],
            "seed": config["seed"],
            "variant": "transfer5",
            "env_kwargs": {
                "geometry_config_path": config["geometry_config"]
            },
            "final_model_file": output["final_checkpoint"],
            "protocol_config": config,
        }
    )
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    return train_subsets_held_out_base_model(
        str(source_path.parent),
        source_path.name,
        str(output_dir),
        output["best_checkpoint"],
        hp=hp,
        env_dict=_digit_env_dict((5,)),
        retained_env_dict=_digit_env_dict(HELDOUT5_DIGITS),
        rule_index=5,
    )


if __name__ == "__main__":
    pass
