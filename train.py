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
from losses import l1_dist, l1_rate, l1_weight, l1_muscle_act, simple_dynamics
from envs import DlyHalfReach, DlyHalfCircleClk, DlyHalfCircleCClk, DlySinusoid, DlySinusoidInv
from envs import DlyFullReach, DlyFullCircleClk, DlyFullCircleCClk, DlyFigure8, DlyFigure8Inv
from envs import ComposableEnv
from utils import save_hp, create_dir, load_hp
from itertools import product

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


def _checkpoint_payload(policy, optimizer, hp, update, validation_loss):
    return {
        "agent_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hp": hp,
        "protocol_config": hp.get("protocol_config"),
        "variant": hp.get("variant"),
        "update": update,
        "validation_loss": validation_loss,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def _run_validation(policy, hp, env_dict, network_noise=True):
    seed = hp.get("validation_seed")
    if seed is None:
        return do_eval(
            policy,
            hp,
            env_dict=env_dict,
            network_noise=network_noise,
        )
    with _fixed_rng(seed):
        return do_eval(
            policy,
            hp,
            env_dict=env_dict,
            network_noise=network_noise,
        )

def do_eval(policy, hp, env_dict=None, network_noise=True):

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

    total_test_loss = 0
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

            # initial positions and targets
            xy = [info["states"]["fingertip"][:, None, :]]
            tg = [info["goal"][:, None, :]]

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

            # Implement loss function
            loss = l1_dist(xy, tg)  # L1 loss on position
            condition_loss += loss.item()
        condition_loss /= len(speed_conds)
        condition_losses[env] = condition_loss
        total_test_loss += condition_loss
    total_test_loss /= len(env_dict)

    print("\n")
    print("Eval Results:")
    for env in condition_losses:
        print(f"Total Loss for Environment {env}| {condition_losses[env]}")
    print(f"Total Testing Loss: {total_test_loss}")
    print("\n")
    
    return total_test_loss





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





def train_subsets_base_model(model_path, model_file, hp=None, env_dict=None):

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

    for batch in range(hp["epochs"]):

        # initialize batch
        x = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))
        h = torch.zeros(size=(hp["batch_size"], hp["hid_size"]))

        rand_env = random.choices(env_list, probs)
        env = rand_env[0](effector=effector, **hp.get("env_kwargs", {}))

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

        if (batch % interval == 0) and (batch != 0):
            mean_loss = sum(losses[-interval:]) / interval
            print("Batch {}/{} Done, mean policy loss: {}".format(batch, hp["epochs"], mean_loss))

        if (batch % hp["save_iter"] == 0):
            # Get test loss
            test_loss = _run_validation(policy, hp, env_dict)
            last_test_loss = test_loss
            test_losses.append(test_loss)
            np.savetxt(os.path.join(model_path, "test_losses.txt"), test_losses)
            # If current test loss is better than previous, save model and update best loss
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    _checkpoint_payload(
                        policy, optimizer, hp, batch, test_loss
                    ),
                    os.path.join(model_path, model_file),
                )
                print("Model Saved!")
                print(f"Directory: {model_path}/{model_file}")
                print("\n")

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
            os.path.join(model_path, final_model_file),
        )
    if hp.get("protocol_config") is not None:
        return {
            "variant": hp.get("variant"),
            "seed": hp.get("seed"),
            "updates": hp["epochs"],
            "best_validation_loss": best_test_loss,
            "last_validation_loss": last_test_loss,
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
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]
        timestep = 0
        while not terminated:
            x, h, action = policy(obs, x, h)
            obs, _, terminated, info = env.step(timestep, action=action)
            xy.append(info["states"]["fingertip"][:, None, :])
            tg.append(info["goal"][:, None, :])
            timestep += 1
        xy = torch.cat(xy, axis=1)
        tg = torch.cat(tg, axis=1)
        loss = l1_dist(xy, tg)
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

        if (batch % hp["save_iter"] == 0):
            test_loss = _run_validation(policy, hp, env_dict)
            last_test_loss = test_loss
            test_losses.append(test_loss)
            np.savetxt(os.path.join(save_model_path, "test_losses.txt"), test_losses)
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
        raise ValueError("the original-protocol baseline device must be cpu")


def _digit_env_dict(digits):
    return {
        f"digit_{digit}": DIGIT_ENV_CLASSES[digit]
        for digit in digits
    }


def _validate_base_config(config):
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
        raise ValueError("base optimizer does not match the original protocol")
    training = config["training"]
    if (
        training["batch_size"] != 32
        or training["max_updates"] != 75_000
        or training["validation_interval"] != 500
    ):
        raise ValueError("base training schedule does not match the protocol")
    if config["regularization"] != {
        "l1_rate": 0.001,
        "l1_weight": 0.001,
        "l1_muscle_act": 0.01,
        "simple_dynamics_weight": 0.001,
    }:
        raise ValueError("base regularization does not match the protocol")


def _base_hp_from_config(config):
    model = config["model"]
    optimizer = config["optimizer"]
    training = config["training"]
    regularization = config["regularization"]
    output = config["output"]
    return {
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
    }


def train_digit_base_model(config):
    """Map one digit configuration onto the original base-training core."""
    _validate_base_config(config)
    output = config["output"]
    output_dir = Path(output["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)
    return train_subsets_base_model(
        str(output_dir),
        output["best_checkpoint"],
        hp=_base_hp_from_config(config),
        env_dict=_digit_env_dict(tuple(config["train_digits"])),
    )


def load_digit_policy_checkpoint(checkpoint_path, expected_variant=None):
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    hp = checkpoint.get("hp")
    if hp is None:
        raise ValueError("checkpoint does not contain the Phase D hyperparameters")
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
    if config["geometry_config"] != source_config["geometry_config"]:
        raise ValueError("transfer geometry must match the heldout5 checkpoint")

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
