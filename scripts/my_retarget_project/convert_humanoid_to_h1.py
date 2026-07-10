"""
convert_humanoid_to_h1.py

Isaac Lab純正ヒューマノイド(28dof)のAMPモーションnpzを、
Unitree H1(19dof)用のAMPモーションnpzにretargetするスクリプト。

やっていること:
  1. H1のURDFから見た目メッシュ(STL参照)を取り除いた「IK計算専用URDF」を生成
  2. 手足4点(左右の手→肘, 左右の足→足首)をIKターゲットにして、
     mink で各フレームのH1関節角度(19dof)を求める
     - 腕/脚の長さの違いは、肩/股関節を基準にスケーリングして吸収
     - 中立姿勢への軽い引き戻し(PostureTask)で余剰自由度の暴れを抑制
     - 関節可動域をIKに明示的に渡して限界張り付きを防止
  3. 求めた関節角度でFKし、H1の全リンクのワールド座標(pelvisの実際の
     位置・向きに元データをコピーした上でそれを基準に)を計算
  4. 速度(dof_velocities, body_linear/angular_velocities)を有限差分で計算
  5. Isaac Lab AMP形式のnpzとして保存

使い方:
  python convert_humanoid_to_h1.py \
      --src data/humanoid_walk.npz \
      --urdf data/h1.urdf \
      --out output/h1_walk_for_isaaclab.npz

必要なパッケージ: numpy, mujoco, mink  (pip install numpy mujoco mink)
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import mink


# ============================================================
# 1. H1側の定義(URDFの構造に合わせた固定値)
# ============================================================

H1_DOF_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

H1_BODY_NAMES = [
    "pelvis",
    "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link", "left_knee_link", "left_ankle_link",
    "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link", "right_ankle_link",
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
]

# 元ヒューマノイド(source)の body_name -> H1側でIKターゲットにするリンク名
KEY_BODY_MAP = {
    "right_hand": "right_elbow_link",
    "left_hand": "left_elbow_link",
    "right_foot": "right_ankle_link",
    "left_foot": "left_ankle_link",
}

# IK位置合わせの重み(足は特に重要なので高め)
POSITION_WEIGHT = {
    "right_hand": 1.0,
    "left_hand": 1.0,
    "right_foot": 2.0,
    "left_foot": 2.0,
}

# 各キーボディの「付け根」となる source 側の親リンク(スケーリングの基準)
SOURCE_ANCHOR = {
    "right_hand": "right_upper_arm",
    "left_hand": "left_upper_arm",
    "right_foot": "right_thigh",
    "left_foot": "left_thigh",
}

SOURCE_ROOT_BODY = "pelvis"


# ============================================================
# 2. クォータニオン/回転行列ユーティリティ (wxyz規約)
# ============================================================

def quat_to_rotmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


# ============================================================
# 3. URDFの軽量化(見た目メッシュを除去し、IK専用にする)
# ============================================================

def make_ik_only_urdf(src_urdf_path, dst_urdf_path):
    tree = ET.parse(src_urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)

        if link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            origin = ET.SubElement(inertial, "origin")
            origin.set("xyz", "0 0 0")
            origin.set("rpy", "0 0 0")
            mass = ET.SubElement(inertial, "mass")
            mass.set("value", "0.001")
            inertia = ET.SubElement(inertial, "inertia")
            for k in ("ixx", "iyy", "izz"):
                inertia.set(k, "1e-6")
            for k in ("ixy", "ixz", "iyz"):
                inertia.set(k, "0")

        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            geom = ET.SubElement(el, "geometry")
            sphere = ET.SubElement(geom, "sphere")
            sphere.set("radius", "0.02")

    tree.write(dst_urdf_path, encoding="utf-8", xml_declaration=True)


# ============================================================
# 4. IK retarget (source npz -> H1 dof_positions)
# ============================================================

def retarget_ik(src_npz_path, ik_urdf_path):
    src = np.load(src_npz_path, allow_pickle=True)
    body_names = list(src["body_names"])
    body_positions = src["body_positions"]
    body_rotations = src["body_rotations"]
    fps = float(src["fps"])
    T = body_positions.shape[0]

    root_idx = body_names.index(SOURCE_ROOT_BODY)
    root_pos = body_positions[:, root_idx, :]
    root_quat = body_rotations[:, root_idx, :]
    root_R = quat_to_rotmat(root_quat)

    model = mujoco.MjModel.from_xml_path(ik_urdf_path)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)  # q=0でのFK(付け根位置の実測に使う)

    def h1_body_pos(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid].copy()

    h1_anchor_pos = {
        "right_hand": h1_body_pos("right_shoulder_pitch_link"),
        "left_hand": h1_body_pos("left_shoulder_pitch_link"),
        "right_foot": h1_body_pos("right_hip_pitch_link"),
        "left_foot": h1_body_pos("left_hip_pitch_link"),
    }

    rel_targets = {}
    for src_name, h1_body in KEY_BODY_MAP.items():
        anchor_name = SOURCE_ANCHOR[src_name]
        idx_target = body_names.index(src_name)
        idx_anchor = body_names.index(anchor_name)

        vec_world = body_positions[:, idx_target, :] - body_positions[:, idx_anchor, :]
        vec_local = np.einsum("tij,tj->ti", root_R.transpose(0, 2, 1), vec_world)

        source_len = np.linalg.norm(vec_local, axis=-1).mean()
        h1_anchor = h1_anchor_pos[src_name]
        h1_target = h1_body_pos(h1_body)
        h1_len = np.linalg.norm(h1_target - h1_anchor)

        scale = h1_len / source_len
        rel_targets[src_name] = h1_anchor[None, :] + vec_local * scale

        print(f"  {src_name:10s}: source_len={source_len:.3f}  h1_len={h1_len:.3f}  scale={scale:.3f}")

    configuration = mink.Configuration(model)

    tasks = []
    task_by_srcname = {}
    for src_name, h1_body in KEY_BODY_MAP.items():
        task = mink.FrameTask(
            frame_name=h1_body,
            frame_type="body",
            position_cost=POSITION_WEIGHT[src_name],
            orientation_cost=0.0,
            lm_damping=1e-2,
        )
        tasks.append(task)
        task_by_srcname[src_name] = task

    posture_task = mink.PostureTask(model, cost=0.05)
    posture_task.set_target(np.zeros(model.nq))
    tasks.append(posture_task)

    joint_limits = mink.ConfigurationLimit(model)

    dt = 1.0 / fps
    q_history = np.zeros((T, model.nq), dtype=np.float32)
    configuration.update(np.zeros(model.nq))

    for t in range(T):
        for src_name, task in task_by_srcname.items():
            se3 = mink.SE3.from_rotation_and_translation(
                mink.SO3.identity(), rel_targets[src_name][t]
            )
            task.set_target(se3)

        for _ in range(20):
            vel = mink.solve_ik(
                configuration, tasks, dt=dt, solver="daqp", limits=[joint_limits]
            )
            configuration.integrate_inplace(vel, dt)

        q_history[t] = configuration.q.copy()

    return q_history, fps


# ============================================================
# 5. FK export (H1 dof_positions -> Isaac Lab AMP形式npz)
# ============================================================

def export_isaaclab_npz(src_npz_path, ik_urdf_path, q_history, fps, out_npz_path):
    src = np.load(src_npz_path, allow_pickle=True)
    src_body_names = list(src["body_names"])
    pelvis_idx = src_body_names.index("pelvis")

    root_pos = src["body_positions"][:, pelvis_idx, :].astype(np.float64)
    root_quat = src["body_rotations"][:, pelvis_idx, :].astype(np.float64)
    root_R = quat_to_rotmat(root_quat)

    T = q_history.shape[0]
    num_bodies = len(H1_BODY_NAMES)

    model = mujoco.MjModel.from_xml_path(ik_urdf_path)
    data = mujoco.MjData(model)

    body_positions = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_rotations = np.zeros((T, num_bodies, 4), dtype=np.float32)
    body_positions[:, 0, :] = root_pos
    body_rotations[:, 0, :] = root_quat

    for t in range(T):
        data.qpos[: model.nq] = q_history[t]
        mujoco.mj_kinematics(model, data)

        for bi, name in enumerate(H1_BODY_NAMES[1:], start=1):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            local_pos = data.xpos[bid].copy()
            local_quat = data.xquat[bid].copy()

            world_pos = root_pos[t] + root_R[t] @ local_pos
            world_quat = quat_mul(root_quat[t], local_quat)

            body_positions[t, bi, :] = world_pos
            body_rotations[t, bi, :] = world_quat

    dt = 1.0 / fps
    dof_vel = np.zeros_like(q_history, dtype=np.float32)
    dof_vel[:-1] = (q_history[1:] - q_history[:-1]) / dt

    body_lin_vel = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_lin_vel[:-1] = (body_positions[1:] - body_positions[:-1]) / dt

    body_ang_vel = np.zeros((T, num_bodies, 3), dtype=np.float32)
    for bi in range(num_bodies):
        q_t = body_rotations[:-1, bi, :]
        q_next = body_rotations[1:, bi, :]
        q_conj = q_t.copy()
        q_conj[:, 1:] *= -1
        dq = quat_mul(q_next, q_conj)
        body_ang_vel[:-1, bi, :] = (dq[:, 1:] * 2.0 / dt) * np.sign(dq[:, :1] + 1e-8)

    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez(
        out_npz_path,
        fps=fps,
        dof_names=np.array(H1_DOF_NAMES, dtype="U"),
        body_names=np.array(H1_BODY_NAMES, dtype="U"),
        dof_positions=q_history.astype(np.float32),
        dof_velocities=dof_vel,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_lin_vel,
        body_angular_velocities=body_ang_vel,
    )


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Isaac Lab humanoid AMP npz -> H1 AMP npz")
    parser.add_argument("--src", required=True, help="元ヒューマノイドのAMP npz (例: humanoid_walk.npz)")
    parser.add_argument("--urdf", required=True, help="H1のURDF (例: h1.urdf, meshes無くてOK)")
    parser.add_argument("--out", required=True, help="出力先npzパス")
    parser.add_argument("--tmp-dir", default="./_tmp_retarget", help="中間ファイル置き場")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    ik_urdf_path = os.path.join(args.tmp_dir, "h1_ik_only.urdf")

    print("[1/3] IK専用URDFを生成中...")
    make_ik_only_urdf(args.urdf, ik_urdf_path)

    print("[2/3] IK retarget中...")
    q_history, fps = retarget_ik(args.src, ik_urdf_path)

    print("[3/3] FKで全リンク座標を計算し、npz保存中...")
    export_isaaclab_npz(args.src, ik_urdf_path, q_history, fps, args.out)

    print(f"\n完了: {args.out}")


if __name__ == "__main__":
    main()
