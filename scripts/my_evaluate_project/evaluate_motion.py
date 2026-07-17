from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
import signal
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import compute_obs
from isaaclab.utils.math import euler_xyz_from_quat

device = "cuda"
motion_file = "/home/matsuno/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_walk.npz"
state_file = "scripts/my_evaluate_project/data/robot_state_latest.pt"

print(f"Loading state file: {state_file}")
data = torch.load(state_file, map_location=device)
print(data.keys())

joint_pos = data["joint_pos"]
joint_vel = data["joint_vel"]
root_pos = data["root_pos"]
root_rot = data["root_quat"]
root_lin_vel = data["root_lin_vel"]
root_ang_vel = data["root_ang_vel"]
key_pos = data["key_pos"]

num_envs = joint_pos.shape[0]

motion_ref_body_index = data["motion_ref_body_index"]
motion_key_body_indexes = data["motion_key_body_indexes"]
motion_dof_indexes = data["motion_dof_indexes"]

robot_now = compute_obs(
    joint_pos,
    joint_vel,
    root_pos,
    root_rot,
    root_lin_vel,
    root_ang_vel,
    key_pos,
    torch.zeros(num_envs, device=device),
    torch.ones(num_envs, device=device),
)

_, _, yaw = euler_xyz_from_quat(root_rot)

yaw_bin = (((yaw + np.pi) / (2*np.pi) * 10).long())
yaw_bin = torch.clamp(yaw_bin,0,9)

motion = MotionLoader(motion_file, device)

best_error = torch.full(
    (num_envs,),
    float("inf"),
    device=device
)

best_frame = torch.zeros(
    num_envs,
    dtype=torch.long,
    device=device
)

times = np.linspace(
    0,
    motion.duration,
    motion.num_frames
)

print("Searching best matching frame...")

for frame,t in enumerate(tqdm(times)):
    dof_pos,dof_vel,body_pos,body_rot,body_lin,body_ang = motion.sample(
        num_samples=num_envs,
        times=np.full(num_envs,t),
    )

    motion_now = compute_obs(
        dof_pos[:,motion_dof_indexes],
        dof_vel[:,motion_dof_indexes],
        body_pos[:,motion_ref_body_index],
        body_rot[:,motion_ref_body_index],
        body_lin[:,motion_ref_body_index],
        body_ang[:,motion_ref_body_index],
        body_pos[:,motion_key_body_indexes],
        torch.zeros(num_envs,device=device),
        torch.ones(num_envs,device=device),
    )

    error = ((robot_now-motion_now)**2).mean(dim=1)

    mask = error < best_error
    best_error[mask] = error[mask]
    best_frame[mask] = frame

angle_labels = [
    "-162°","-126°","-90°","-54°","-18°",
    "18°","54°","90°","126°","162°"
]

counts=[]
errors=[]

print("\n===== Motion Error by Yaw =====")

for b in range(10):
    idx = yaw_bin == b
    count = idx.sum().item()
    counts.append(count)

    if count == 0:
        errors.append(0.0)
        print(f"{angle_labels[b]} N={count} error=0")
    else:
        mean_error = best_error[idx].mean().item()
        errors.append(mean_error)
        print(
            f"{angle_labels[b]} "
            f"N={count} "
            f"error={mean_error:.5f}"
        )

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.bar(angle_labels,counts)
plt.title("Robot Count by Yaw")
plt.xlabel("Yaw angle")
plt.ylabel("Count")

plt.subplot(1,2,2)
plt.bar(angle_labels,errors)
plt.title("Motion Error by Yaw")
plt.xlabel("Yaw angle")
plt.ylabel("MSE")

plt.tight_layout()

output_plot = "scripts/my_evaluate_project/output/yaw_gait_stats.png"
plt.savefig(output_plot)

print(f"saved: {output_plot}")

print("\nCreating adaptive yaw distribution...")

error_tensor = torch.tensor(
    errors,
    dtype=torch.float32,
    device=device
)

sym_error = torch.zeros_like(error_tensor)

pairs=[
    (0,9),
    (1,8),
    (2,7),
    (3,6),
    (4,5)
]

for a,b in pairs:
    value=(error_tensor[a]+error_tensor[b])/2
    sym_error[a]=value
    sym_error[b]=value

print("Symmetric error:")
print(sym_error.tolist())


# =====================================
# 難しい方向ほど確率を上げる
# =====================================

# 0除算防止 + 未評価方向にも少し確率を残す
epsilon = 1.0

difficulty = sym_error + epsilon

# 難しい方向を強調
temperature = 1.5

difficulty = difficulty ** temperature


# 確率化
yaw_probabilities = difficulty / difficulty.sum()


print("\nFinal probabilities:")

for i,p in enumerate(yaw_probabilities):
    print(
        f"{angle_labels[i]} : {p.item():.4f}"
    )

distribution_file="scripts/my_evaluate_project/output/yaw_prob_distribution.pt"

torch.save(
    {
        "probabilities":yaw_probabilities.cpu(),
        "num_bins":10
    },
    distribution_file
)

print(f"Saved: {distribution_file}")
simulation_app.close()
os.kill(os.getpid(),signal.SIGKILL)