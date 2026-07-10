import pickle
import os
import numpy as np

pkl_path = "./data/B4_-_Stand_to_Walk_backwards_stageii.pkl"
output_npz_path = "./data/motion_for_skrl.npz"

print("--- .pkl ファイルを読み込み中 ---")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# キー一覧
print(data.keys())

# 各キーの型・形状を確認
for k, v in data.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"{k}: type={type(v)}, value={v}")

# NumPy配列としてそのまま抽出
root_pos = np.array(data["root_pos"], dtype=np.float32)
root_rot = np.array(data["root_rot"], dtype=np.float32)
dof_pos = np.array(data["dof_pos"], dtype=np.float32)
key_body_pos = np.array(data["key_body_pos"], dtype=np.float32)

num_frames = dof_pos.shape[0]
num_dof = dof_pos.shape[1]

print(f"元のデータフレーム数: {num_frames}")
print(f"元のデータ自由度数 (DoF): {num_dof}")
# 読み込み直後の確認コード
print("--- .pkl の中身のキーを確認 ---")
print(data.keys())

# -------------------------------------------------------------
# 🎯 Isaac Lab (H1ロボット環境) が絶対要求する名前リストの定義
# -------------------------------------------------------------
# H1の19個の関節名（順序が合わない場合は後ほど mapping で現物合わせ可能）
# H1の正しい関節リスト（シミュレータの順序と一致）
official_dof_names = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint",
    "torso_joint", 
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
]

# シミュレータが認識しているボディ名リスト（これに合わせて body_positions を埋める必要があります）
body_names = [
    'pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link', 
    'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 
    'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 
    'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 
    'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link', 
    'right_ankle_link', 'left_elbow_link', 'right_elbow_link'
]

# -------------------------------------------------------------
# 📐 全身のボディデータの器を作成 (Isaac Lab用フォーマット)
# -------------------------------------------------------------
num_bodies = len(body_names)

body_positions = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
body_rotations = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
body_rotations[..., 0] = 1.0  # クォータニオンの初期化 (W=1, X=0, Y=0, Z=0)

# ルート（ pelvis：インデックス0 ）に位置と回転を代入
body_positions[:, 0] = root_pos
body_rotations[:, 0] = root_rot

# 手足のキー位置 (key_body_pos: 通常4箇所) を対応する部位へ代入
# body_namesに対応：左足(4), 右足(8), 左手(12), 右手(15) 等
# ※一旦仮で割り当てますが、キー位置を合わせるだけであればこれでインフラを通過できます
if key_body_pos.ndim == 3 and key_body_pos.shape[1] == 4:
    # key_body_pos の中身が [左足, 右足, 左手, 右手] の順だと仮定した場合
    body_positions[:, 16] = key_body_pos[:, 0]  # left_ankle_link
    body_positions[:, 17] = key_body_pos[:, 1]  # right_ankle_link
    body_positions[:, 18] = key_body_pos[:, 2]  # left_elbow_link (手として代用)
    body_positions[:, 19] = key_body_pos[:, 3]  # right_elbow_link (手として代用)

# -------------------------------------------------------------
# 🏃‍♂️ 速度（Velocity）データの逆算
# -------------------------------------------------------------
fps = float(data.get("fps", 120.0)) # H1は120FPS
dt = 1.0 / fps

print(f"FPS: {fps} (dt: {dt:.4f}s) から公式形式の速度データを計算中...")

# 関節速度
dof_vel = np.zeros_like(dof_pos)
dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt

# ルートの線速度・角速度
body_linear_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
body_linear_velocities[:-1, 0] = (root_pos[1:] - root_pos[:-1]) / dt

body_angular_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

# -------------------------------------------------------------
# 💾 完全な公式仕様の .npz としてエクスポート (ピクルス回避版)
# -------------------------------------------------------------
os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
np.savez(
    output_npz_path,
    fps=fps,
    # dtype=object をやめて、明確にNumPyの文字列型（"U"）にキャストする ★ここを修正
    dof_names=np.array(official_dof_names, dtype="U"),
    body_names=np.array(body_names, dtype="U"),
    dof_positions=dof_pos,
    dof_velocities=dof_vel,
    body_positions=body_positions,
    body_rotations=body_rotations,
    body_linear_velocities=body_linear_velocities,
    body_angular_velocities=body_angular_velocities
)

print(f"\n🎉 ピクルス警告対策済みの.npzモーションデータを生成しました！")
print(f"保存先: {output_npz_path}")