#関節角度から位置、姿勢、速度を計算して.npzを作るスクリプト
import numpy as np
import mujoco
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

URDF_FILE = ROOT / "data/h1_ik_only.urdf"
Q_FILE = ROOT / "output/h1_dof_positions.npy"

OUTPUT = ROOT / "output/h1_walk_for_isaaclab.npz"


# -------------------------
# load
# -------------------------

q_seq = np.load(Q_FILE)
# q_seq[:,10] = 0

print("q shape:", q_seq.shape)


model = mujoco.MjModel.from_xml_path(
    str(URDF_FILE)
)
print("MuJoCo bodies:")
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    ))

data = mujoco.MjData(model)


# -------------------------
# body names
# -------------------------

body_names = [
    "torso_link",

    "left_hip_yaw_link",
    "right_hip_yaw_link",

    "left_hip_roll_link",
    "right_hip_roll_link",

    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",

    "left_hip_pitch_link",
    "right_hip_pitch_link",

    "left_shoulder_roll_link",
    "right_shoulder_roll_link",

    "left_knee_link",
    "right_knee_link",

    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",

    "left_ankle_link",
    "right_ankle_link",

    "left_elbow_link",
    "right_elbow_link",
]
body_ids=[]
for name in body_names:
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )

    print(name,bid)

    assert bid >= 0, f"{name} not found"

    body_ids.append(bid)
print(body_names)

# -------------------------
# FK
# -------------------------
num_frames = q_seq.shape[0]
body_pos = np.zeros(
    (num_frames, len(body_names), 3)
)
body_rot = np.zeros(
    (num_frames, len(body_names), 4)
)
# root trajectory
root_positions = np.zeros(
    (num_frames, 3)
)
root_rotations = np.zeros(
    (num_frames, 4)
)
for t in range(num_frames):
    data.qpos[:] = q_seq[t]
    mujoco.mj_forward(
        model,
        data
    )
    root_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "torso_link"
    )
    root_pos = data.xpos[root_id].copy()
    root_R = data.xmat[root_id].reshape(3,3)
    yaw = np.arctan2(
        root_R[1,0],
        root_R[0,0]
    )

    if t % 10 == 0:
        print(t, "yaw(deg)", np.degrees(yaw))
    # save root trajectory
    root_positions[t] = root_pos
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        quat,
        data.xmat[root_id]
    )
    root_rotations[t] = quat


    for i,bid in enumerate(body_ids):
        # -----------------
        # position
        # world -> pelvis local
        # -----------------
        body_pos[t,i] = (
            data.xpos[bid] - root_pos
        )
        # -----------------
        # rotation
        # world -> pelvis local
        # -----------------
        body_R = data.xmat[bid].reshape(3,3)
        rel_R = root_R.T @ body_R
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(
            quat,
            rel_R.flatten()
        )
        body_rot[t,i] = quat
        root_positions = np.zeros(
            (num_frames, 3)
        )
        root_rotations = np.zeros(
            (num_frames, 4)
        )
print("FK done")

# -------------------------
# velocities
# -------------------------
dt = 1/30
body_lin_vel = np.gradient(
    body_pos,
    dt,
    axis=0
)
body_ang_vel = np.zeros_like(
    body_lin_vel
)
dof_vel = np.gradient(
    q_seq,
    dt,
    axis=0
)

# -------------------------
# convert MuJoCo order -> IsaacLab order
# -------------------------
q_isaaclab = np.zeros_like(q_seq)
q_isaaclab[:,0]  = q_seq[:,0]   # left_hip_yaw
q_isaaclab[:,1]  = q_seq[:,5]   # right_hip_yaw
q_isaaclab[:,2]  = q_seq[:,10]  # torso
q_isaaclab[:,3]  = q_seq[:,1]   # left_hip_roll
q_isaaclab[:,4]  = q_seq[:,6]   # right_hip_roll
q_isaaclab[:,5]  = q_seq[:,11]  # left_shoulder_pitch
q_isaaclab[:,6]  = q_seq[:,15]  # right_shoulder_pitch
q_isaaclab[:,7]  = q_seq[:,2]   # left_hip_pitch
q_isaaclab[:,8]  = q_seq[:,7]   # right_hip_pitch
q_isaaclab[:,9]  = q_seq[:,12]  # left_shoulder_roll
q_isaaclab[:,10] = q_seq[:,16]  # right_shoulder_roll
q_isaaclab[:,11] = q_seq[:,3]   # left_knee
q_isaaclab[:,12] = q_seq[:,8]   # right_knee
q_isaaclab[:,13] = q_seq[:,13]  # left_shoulder_yaw
q_isaaclab[:,14] = q_seq[:,17]  # right_shoulder_yaw
q_isaaclab[:,15] = q_seq[:,4]   # left_ankle
q_isaaclab[:,16] = q_seq[:,9]   # right_ankle
q_isaaclab[:,17] = q_seq[:,14]  # left_elbow
q_isaaclab[:,18] = q_seq[:,18]  # right_elbow

# -------------------------
# save
# -------------------------

np.savez(
    OUTPUT,

    fps=30,

    dof_names=np.array([
        "left_hip_yaw",
        "right_hip_yaw",
        "torso",
        "left_hip_roll",
        "right_hip_roll",
        "left_shoulder_pitch",
        "right_shoulder_pitch",
        "left_hip_pitch",
        "right_hip_pitch",
        "left_shoulder_roll",
        "right_shoulder_roll",
        "left_knee",
        "right_knee",
        "left_shoulder_yaw",
        "right_shoulder_yaw",
        "left_ankle",
        "right_ankle",
        "left_elbow",
        "right_elbow",
    ]),

    dof_positions=q_isaaclab,
    dof_velocities=np.gradient(
        q_isaaclab,
        dt,
        axis=0
    ),

    body_names=np.array(body_names),

    body_positions=body_pos,

    body_rotations=body_rot,

    body_linear_velocities=body_lin_vel,

    body_angular_velocities=body_ang_vel,

    root_positions=root_positions,
    root_rotations=root_rotations,
)


print("saved:", OUTPUT)