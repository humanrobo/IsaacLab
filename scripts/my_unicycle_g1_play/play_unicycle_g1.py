import argparse
import sys
sys.path.insert(0, "/home/matsuno/WBC-AGILE")
import os
import time
import torch
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Unicycle policy -> G1 locomotion policy")
parser.add_argument("--unicycle_checkpoint", type=str, required=True)
parser.add_argument("--g1_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--num_steps", type=int, default=1000000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import agile.rl_env.tasks
import agile.isaaclab_extras.monkey_patches

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper

def load_g1_policy(checkpoint, device):
    print(f"[INFO] Loading G1 policy: {checkpoint}")
    policy = torch.jit.load(checkpoint, map_location=device)
    policy.eval()
    return policy

def main():
    device = args_cli.device
    # ============================================================
    # G1 Env
    # ============================================================
    g1_cfg = parse_env_cfg(
        "Velocity-G1-History-v0",
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(g1_cfg, "eval"):
        g1_cfg.eval()
    g1_env = gym.make(
        "Velocity-G1-History-v0",
        cfg=g1_cfg,
        render_mode=None,
    )
    if isinstance(g1_env.unwrapped, DirectMARLEnv):
        g1_env = multi_agent_to_single_agent(g1_env)
    g1_env = RslRlVecEnvWrapper(g1_env)
    g1_env.reset()
    # ============================================================
    # G1 policy
    # ============================================================
    g1_policy = load_g1_policy(
        args_cli.g1_checkpoint,
        device,
    )
    # ============================================================
    # Reset
    # ============================================================
    g1_obs, _ = g1_env.get_observations()
    print("[INFO] G1 environment started.")
    print(f"[INFO] G1 observation type: {type(g1_obs)}")
    print(g1_obs)
    # ============================================================
    # Main loop
    # ============================================================
    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break
        start_time = time.time()
        with torch.inference_mode():
            # Unicycle policy output
            v = torch.full((args_cli.num_envs,), 0.5, device=device)
            omega = torch.zeros(args_cli.num_envs, device=device)

            # Unicycle [v, omega] -> G1 [vx, vy, wz]
            command = g1_env.unwrapped.command_manager.get_command("base_velocity")
            command[:, 0] = 0.0#v
            command[:, 1] = 0.0
            command[:, 2] = 0.0#omega
            g1_obs, _ = g1_env.get_observations()
            # TensorDict -> Tensor
            g1_obs_tensor = torch.cat([
                g1_obs["velocity_commands"],
                g1_obs["base_ang_vel"],
                g1_obs["projected_gravity"],
                g1_obs["controlled_joint_pos"],
                g1_obs["controlled_joint_vel"],
                g1_obs["actions"],
            ], dim=-1)
            g1_obs_tensor = g1_obs_tensor.reshape(g1_obs_tensor.shape[0], -1)
            # print("G1 obs shape:", g1_obs_tensor.shape)
            # print("G1 obs[0]:", g1_obs_tensor[0])
            # G1 policy
            g1_actions = g1_policy(g1_obs_tensor)
            # G1 step
            g1_obs, _, _, _ = g1_env.step(g1_actions)

        if step % 100 == 0:
            print(
                f"step={step} "
                f"command=[{command[0,0].item():.3f}, "
                f"{command[0,1].item():.3f}, "
                f"{command[0,2].item():.3f}]"
            )
        if args_cli.real_time:
            sleep_time = g1_env.unwrapped.step_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    g1_env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()