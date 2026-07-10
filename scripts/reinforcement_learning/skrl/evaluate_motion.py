# ----------------------------------------------------------------
# 重要: ほかのライブラリをインポートする前に、必ずシミュレータを起動する
# ----------------------------------------------------------------
from isaaclab.app import AppLauncher

# simulation_app を起動（headless=True）
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ----------------------------------------------------------------
# シミュレータ起動後に、通常のインポートを行う
# ----------------------------------------------------------------
import os
import signal
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import compute_obs
from isaaclab.utils.math import euler_xyz_from_quat

# ------------------------
# 設定
# ------------------------
device = "cuda"
motion_file = "/home/matsuno/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_walk.npz"
state_file = "scripts/reinforcement_learning/skrl/robot_state_100000.pt"

# ------------------------
# robot状態・インデックス読み込み
# ------------------------
print(f"Loading state file: {state_file}")
data = torch.load(state_file, map_location=device)

joint_pos = data["joint_pos"]
joint_vel = data["joint_vel"]
root_pos = data["root_pos"]
root_rot = data["root_rot"]
root_lin_vel = data["root_lin_vel"]
root_ang_vel = data["root_ang_vel"]
key_pos = data["key_pos"]

num_envs = joint_pos.shape[0]

motion_ref_body_index = data["motion_ref_body_index"]
motion_key_body_indexes = data["motion_key_body_indexes"]
motion_dof_indexes = data["motion_dof_indexes"]

# ------------------------
# robot feature & yaw
# ------------------------
robot_now = compute_obs(
    joint_pos, joint_vel, root_pos, root_rot, root_lin_vel, root_ang_vel, key_pos,
    torch.zeros(num_envs, device=device), torch.ones(num_envs, device=device),
)

_, _, yaw = euler_xyz_from_quat(root_rot)
yaw_bin = ((yaw + np.pi) / (2 * np.pi) * 10).long()
yaw_bin = torch.clamp(yaw_bin, 0, 9)

# ------------------------
# motion
# ------------------------
motion = MotionLoader(motion_file, device)
best_error = torch.full((num_envs,), float("inf"), device=device)
best_frame = torch.zeros(num_envs, dtype=torch.long, device=device)
times = np.linspace(0, motion.duration, motion.num_frames)

print("Searching best matching frame...")
for frame, t in enumerate(tqdm(times)):
    dof_pos, dof_vel, body_pos, body_rot, body_lin, body_ang = motion.sample(
        num_samples=num_envs, times=np.full(num_envs, t),
    )
    motion_now = compute_obs(
        dof_pos[:, motion_dof_indexes], dof_vel[:, motion_dof_indexes],
        body_pos[:, motion_ref_body_index], body_rot[:, motion_ref_body_index],
        body_lin[:, motion_ref_body_index], body_ang[:, motion_ref_body_index],
        body_pos[:, motion_key_body_indexes],
        torch.zeros(num_envs, device=device), torch.ones(num_envs, device=device),
    )
    error = ((robot_now - motion_now) ** 2).mean(dim=1)
    mask = error < best_error
    best_error[mask] = error[mask]
    best_frame[mask] = frame

# ------------------------
# 各Binの中央値（角度°）をラベルにする設定
# ------------------------
# -180 ~ 180度を10等分したときの中央の角度
angle_labels = [
    "-162°", "-126°", "-90°", "-54°", "-18°", 
    "18°", "54°", "90°", "126°", "162°"
]
counts = []
errors = []

print("\n===== Motion error by Yaw Angle =====")
for b in range(10):
    idx = yaw_bin == b
    count = idx.sum().item()
    counts.append(count)
    
    if count == 0:
        errors.append(0.0)
        continue
    
    mean_err = best_error[idx].mean().item()
    errors.append(mean_err)
    
    # ターミナル表示も角度の範囲がわかりやすいように出力
    low_deg = -180 + b * 36
    high_deg = low_deg + 36
    print(f"Angle {low_deg:4d}° to {high_deg:4d}° (Center: {angle_labels[b]:>5s}): N={count:4d} error={mean_err:7.4f}")

# ------------------------
# グラフ描画と保存
# ------------------------
print("\nGenerating and saving plots with angle labels...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 1. データの分布グラフ（角度表示）
ax1.bar(angle_labels, counts, color='skyblue', edgecolor='black')
ax1.set_title("Robot Count by Yaw Angle")
ax1.set_xlabel("Yaw Angle (Center of 36° bins)")
ax1.set_ylabel("Number of Environments")
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 2. 方向ごとのエラーの大きさグラフ（角度表示）
ax2.bar(angle_labels, errors, color='salmon', edgecolor='black')
ax2.set_title("Average Motion Error by Yaw Angle")
ax2.set_xlabel("Yaw Angle (Center of 36° bins)")
ax2.set_ylabel("Mean Squared Error")
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# 保存パス
output_plot = "scripts/reinforcement_learning/skrl/yaw_gait_stats.png"
plt.savefig(output_plot)
print(f"Graph successfully saved to: {output_plot}")

# ------------------------
# 苦手な方向ほど選ばれやすい確率分布の作成と保存
# ------------------------
print("\nCalculating error-based probability distribution...")

# 各Binのエラー（負の方向など未サンプルの場所は0になっているため、最低値を1e-5にする）
error_tensor = torch.tensor(errors, dtype=torch.float32, device=device)
error_tensor = torch.clamp(error_tensor, min=1e-5)

# 【確率の調整（温度パラメータ）】
# 1.0 だとエラーに単純比例。数値を大きくする（例: 2.0）と苦手な方向がより極端に選ばれやすくなります
temperature = 2.0 
weighted_errors = error_tensor ** temperature

# 合計が 1.0 になるように正規化（離散確率分布の作成）
yaw_probabilities = weighted_errors / weighted_errors.sum()

# 環境（Env）側で読み込めるように辞書として保存
distribution_file = "scripts/reinforcement_learning/skrl/yaw_prob_distribution.pt"
torch.save({
    "probabilities": yaw_probabilities.cpu(), # 各Bin(0~9)の選ばれる確率
    "num_bins": 10
}, distribution_file)

print(f"Probability distribution saved to: {distribution_file}")
print(f"Probabilities per bin: {yaw_probabilities.tolist()}")

# ------------------------
# 後処理
# ------------------------
print("\nProcessing finished. Terminating backend...")
simulation_app.close()
os.kill(os.getpid(), signal.SIGKILL)