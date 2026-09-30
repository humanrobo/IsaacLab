# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a skrl checkpoint over a fixed number of episodes."""
from __future__ import annotations
import argparse
import os
import random
import sys
from collections import Counter
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate in parallel.")
parser.add_argument("--num_episodes", type=int, default=1000, help="Number of completed episodes to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import skrl
import torch
from packaging import version

if version.parse(skrl.__version__) < version.parse("2.0.0"):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}")

if args_cli.ml_framework.startswith("torch"):
    from isaaclab_tasks.direct.unicycle.rnd_runner import RNDRunner as Runner
else:
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

def _to_bool_tensor(x, num_envs, device):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.to(device=device).reshape(-1)
    else:
        x = torch.as_tensor(x, device=device).reshape(-1)
    if x.numel() != num_envs:
        return None
    return x.bool()

def _find_info_tensor(info, names, num_envs, device):
    if not isinstance(info, dict):
        return None
    for name in names:
        if name in info:
            value = _to_bool_tensor(info[name], num_envs, device)
            if value is not None:
                return value
    for container_name in ("final_info", "episode"):
        nested = info.get(container_name)
        if isinstance(nested, dict):
            for name in names:
                if name in nested:
                    value = _to_bool_tensor(nested[name], num_envs, device)
                    if value is not None:
                        return value
    return None

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    obs, _ = env.reset()
    states = env.state()
    completed = 0
    success = 0
    timeout = 0
    failure = 0
    info_keys_printed = False
    terminal_types = Counter()

    print(f"[INFO] Evaluation: {args_cli.num_episodes} episodes, {args_cli.num_envs} parallel environments")

    while simulation_app.is_running() and completed < args_cli.num_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, info = env.step(actions)
            rewards_t = torch.as_tensor(rewards, device=env.unwrapped.device).reshape(-1)
            states = env.state()

        num_envs = env.num_envs
        device = env.device
        terminated_t = _to_bool_tensor(terminated, num_envs, device)
        truncated_t = _to_bool_tensor(truncated, num_envs, device)
        if terminated_t is None:
            terminated_t = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if truncated_t is None:
            truncated_t = torch.zeros(num_envs, dtype=torch.bool, device=device)

        done = terminated_t | truncated_t
        done_ids = torch.nonzero(done, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        if not info_keys_printed and isinstance(info, dict):
            print(f"[INFO] info keys: {list(info.keys())}")
            info_keys_printed = True

        reached = _find_info_tensor(info, ("reached_goal", "success", "is_success"), num_envs, device)
        timeouts = _find_info_tensor(info, ("time_out", "timeout", "TimeLimit.truncated"), num_envs, device)

        for env_id in done_ids.tolist():
            if completed >= args_cli.num_episodes:
                break
            is_timeout = bool(truncated_t[env_id].item())
            if timeouts is not None:
                is_timeout = bool(timeouts[env_id].item()) or is_timeout

            if reached is not None:
                is_success = bool(reached[env_id].item())
            else:
                is_success = False

            if is_success:
                success += 1
                terminal_types["success"] += 1
            elif is_timeout:
                timeout += 1
                terminal_types["timeout"] += 1
            else:
                failure += 1
                terminal_types["failure"] += 1
            completed += 1

        if completed > 0 and completed % max(args_cli.num_envs, 1) == 0:
            print(f"[INFO] Episodes {completed}/{args_cli.num_episodes} | Success {success} | Timeout {timeout} | Failure {failure}")

    print()
    print("========== Evaluation ==========")
    print(f"Episodes       : {completed}")
    print(f"Success        : {success}")
    print(f"Timeout        : {timeout}")
    print(f"Failure        : {failure}")
    if completed > 0:
        print(f"Success Rate   : {100.0 * success / completed:.2f} %")
        print(f"Timeout Rate   : {100.0 * timeout / completed:.2f} %")
        print(f"Failure Rate   : {100.0 * failure / completed:.2f} %")
    else:
        print("Success Rate   : N/A")
        print("Timeout Rate   : N/A")
        print("Failure Rate   : N/A")
    print("================================")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
