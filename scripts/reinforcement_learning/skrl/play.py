# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

#追加
import omni
from pxr import UsdGeom
import omni.usd
from pxr import Gf, UsdGeom
from isaacsim.core.utils.viewports import set_camera_view
import carb
import omni.appwindow
import numpy as np
import pygame
import numpy as np
from isaacsim.util.debug_draw import _debug_draw
from isaaclab.utils.math import euler_xyz_from_quat
from omni.kit.viewport.utility import create_viewport_window
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility import capture_viewport_to_file

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    # runner.agent.set_running_mode("eval")
    for model in runner.agent.models.values():
        model.eval()

    # reset environment
    # obs, _ = env.reset()
    # timestep = 0

    #追加-----------------------------------------------------------------------------------------------------------
    #目標方向矢印+++++++++++++++++++++++++++++++++++++++++++++++++++++
    draw = _debug_draw.acquire_debug_draw_interface()
    head_index = env.unwrapped.robot.find_bodies("head")[0][0]
    #手足先位置軌跡+++++++++++++++++++++++++++++++++++++++++++++++++++
    # reset environment
    pygame.init()
    screen = pygame.display.set_mode((200, 200))  # 重要
    obs, _ = env.reset()
    timestep = 0
    trajectory = []
    foot_trajectory = []
    hand_trajectory = []
    robot = env.unwrapped.robot
    body_names = robot.data.body_names
    print(body_names)
    # 保存したいbody index
    left_foot_id = body_names.index("left_foot")
    right_foot_id = body_names.index("right_foot")
    left_hand_id = body_names.index("left_hand")
    right_hand_id = body_names.index("right_hand")
    # カメラウィンドウ初期化+++++++++++++++++++++++++++++++++++++++++++++++++++++
    create_viewport_window("Follow")
    follow_view = get_viewport_from_window_name("Follow")
    follow_view.set_active_camera("/World/FollowCamera")
    #目標方向更新++++++++++++++++++++++++++++++++++++++++++++++++++++++
    episode_length_s = 20.0
    sim_dt = env.unwrapped.sim.get_physics_dt()
    episode_steps = int(episode_length_s / sim_dt)
    # Top Camera View
    # create_viewport_window("Top")
    # top_view = get_viewport_from_window_name("Top")
    # top_view.set_active_camera("/World/TopCamera")
    # 録画設定+++++++++++++++++++++++++++++++++++++++++++++++++++++
    frame_id = 0
    yaw_index = 0
    yaw_list = torch.deg2rad(
        torch.arange(
            0,
            360,
            18,
            device=env.device
        )
    )
    env.unwrapped.goal_yaw[:] = yaw_list
    recording = False
    # 追従カメラ+++++++++++++++++++++++++++++++++++++++++++++++++++++
    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, "/World/FollowCamera")
    camera_path = "/World/FollowCamera"
    camera_offset = [6.0, 30.0, 8.5]
    app_window = omni.appwindow.get_default_app_window()
    follow_view.set_active_camera("/World/FollowCamera")
    #真上カメラ+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    top_camera = UsdGeom.Camera.Define(stage, "/World/TopCamera")
    top_camera_path = "/World/TopCamera"
    set_camera_view(
        eye=[0.0, 0.0, 80.0],      # 真上20m
        target=[0.0, 0.0, 0.0],    # Ground中心
        camera_prim_path=top_camera_path,
    )
    # viewport作成
    create_viewport_window("Top")
    top_view = get_viewport_from_window_name("Top")
    top_view.set_active_camera(
        top_camera_path
    )
    #キーボード操作+++++++++++++++++++++++++++++++++++++++++++++++++++
    keyboard = app_window.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    def on_keyboard_event(event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input == carb.input.KeyboardInput.W:
            camera_offset[0] -= 1
        elif event.input == carb.input.KeyboardInput.S:
            camera_offset[0] += 1
        elif event.input == carb.input.KeyboardInput.A:
            camera_offset[1] += 1
        elif event.input == carb.input.KeyboardInput.D:
            camera_offset[1] -= 1
        elif event.input == carb.input.KeyboardInput.Q:
            camera_offset[2] -= 1
        elif event.input == carb.input.KeyboardInput.E:
            camera_offset[2] += 1
        print(camera_offset)
        return True
    keyboard_sub = input_iface.subscribe_to_keyboard_events(
        keyboard,
        on_keyboard_event,
    )

    #-----------------------------------------------------------------------------------------------------------
    #  simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        #追加+++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #追加キーボード+++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulation_app.close()
            if event.type == pygame.KEYDOWN:
                delta_yaw = np.deg2rad(10.0)
                if event.key == pygame.K_a:
                    env.unwrapped.goal_yaw[:] += delta_yaw
                if event.key == pygame.K_d:
                    env.unwrapped.goal_yaw[:] -= delta_yaw

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
            #追加+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # body位置取得
            # root XY位置取得（20環境分）
            root_xy = (
                robot.data.root_pos_w[:, :2]
                .cpu()
                .numpy()
            )
            trajectory.append(root_xy)
            foot_pos = (robot.data.body_link_pos_w[:,[left_foot_id,right_foot_id]].cpu().numpy())
            foot_trajectory.append(foot_pos)
            hand_pos = (robot.data.body_link_pos_w[:,[left_hand_id,right_hand_id]].cpu().numpy())
            hand_trajectory.append(hand_pos)
            #最新のロボット状態データ保存
            if timestep == episode_steps - 1:
                #追加
                trajectory_array = np.array(trajectory)
                foot_array = np.array(foot_trajectory)
                print("trajectory shape:", trajectory_array.shape)
                print("foot shape:", foot_array.shape)
                np.save(
                    "/home/matsuno/IsaacLab/scripts/my_evaluate_project/output/root_xy_trajectory.npy",
                    trajectory_array
                )
                np.save(
                    "/home/matsuno/IsaacLab/scripts/my_evaluate_project/output/foot_trajectory.npy",
                    foot_array
                )
                np.save(
                    "/home/matsuno/IsaacLab/scripts/my_evaluate_project/output/hand_trajectory.npy",
                    np.array(hand_trajectory)
                )
                print(hasattr(env.unwrapped.robot.data, "body_pos_w"))
                torch.save({
                    "root_pos": env.unwrapped.robot.data.root_pos_w.cpu(),
                    "root_quat": env.unwrapped.robot.data.root_quat_w.cpu(),
                    "root_lin_vel": env.unwrapped.robot.data.root_lin_vel_w.cpu(),
                    "root_ang_vel": env.unwrapped.robot.data.root_ang_vel_w.cpu(),
                    "joint_pos": env.unwrapped.robot.data.joint_pos.cpu(),
                    "joint_vel": env.unwrapped.robot.data.joint_vel.cpu(),
                    "key_pos": env.unwrapped.robot.data.body_pos_w[
                        :, env.unwrapped.motion_key_body_indexes
                    ].cpu(),
                    "motion_ref_body_index": env.unwrapped.motion_ref_body_index,
                    "motion_key_body_indexes": env.unwrapped.motion_key_body_indexes,
                    "motion_dof_indexes": env.unwrapped.motion_dof_indexes,
                    "goal_yaw": env.unwrapped.goal_yaw.cpu(),
                }, "scripts/my_evaluate_project/data/robot_state_latest.pt")
                print("saved robot_state.pt")
            robot_pos = env.unwrapped.robot.data.root_pos_w[0].cpu().numpy()
            #カメラ位置
            root_quat = env.unwrapped.robot.data.root_quat_w[0]
            _, _, robot_yaw = euler_xyz_from_quat(root_quat.unsqueeze(0))
            robot_yaw = robot_yaw.item()
            distance = 8.0
            height = -1.0
            # ロボットの左真横
            eye = [
                robot_pos[0] - distance * np.sin(robot_yaw),
                robot_pos[1] + distance * np.cos(robot_yaw),
                robot_pos[2] + height,
            ]
            target = [
                robot_pos[0],
                robot_pos[1],
                robot_pos[2] ,
            ]
            set_camera_view(
                eye=eye,
                target=target,
                camera_prim_path=camera_path,
            )
        # ==========================================
        # Goal方向の矢印を頭の上に描画
        # ==========================================
        draw.clear_lines()

        root_pos = env.unwrapped.robot.data.body_pos_w[0, head_index]
        goal_yaw = env.unwrapped.goal_yaw[0]
        root_quat = env.unwrapped.robot.data.root_quat_w[0]
        _, _, robot_yaw = euler_xyz_from_quat(root_quat.unsqueeze(0))
        robot_yaw = robot_yaw[0]
        start = (
            root_pos[0].item(),
            root_pos[1].item(),
            root_pos[2].item() + 0.35,
        )
        length = 0.5
        end = (
            (root_pos[0] + length * torch.cos(goal_yaw)).item(),
            (root_pos[1] + length * torch.sin(goal_yaw)).item(),
            root_pos[2].item() + 0.35,
        )
        end_robot = (
            float(root_pos[0] + length * torch.cos(robot_yaw)),
            float(root_pos[1] + length * torch.sin(robot_yaw)),
            float(root_pos[2] + 0.35),
        )
        draw.draw_lines(
            [start],
            [end],
            [(0.0, 1.0, 0.0, 1.0)],
            [4.0],
        )
        draw.draw_lines(
            [start],
            [end_robot],
            [(0.0, 0.0, 1.0, 1.0)],   # 青
            [4.0],
        )

        if recording:
            capture_viewport_to_file(
                follow_view,
                f"./videos/yaw_{yaw_index}/side/{frame_id:06d}.png"
            )
            capture_viewport_to_file(
                top_view,
                f"./videos/yaw_{yaw_index}/top/{frame_id:06d}.png"
            )

        timestep += 1
        frame_id += 1
        # exit the play loop after recording one video
        # if timestep == args_cli.video_length:
        #     break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    #-----------------------------------------------------------------------------------------------------------

    print("saved trajectory:")

    keyboard_sub = None
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()