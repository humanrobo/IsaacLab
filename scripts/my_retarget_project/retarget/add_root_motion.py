#h1のnpzにhumanoidのpelvis root motionを追加するスクリプト
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


human_file = ROOT / "data/humanoid_walk.npz"
h1_file = ROOT / "output/h1_walk_for_isaaclab.npz"

output = ROOT / "output/h1_walk_for_isaaclab_root.npz"


human = np.load(human_file)
h1 = np.load(h1_file)


# -----------------------
# human pelvis trajectory
# -----------------------

human_names = human["body_names"].tolist()

pelvis_id = human_names.index("pelvis")

human_root_pos = human["body_positions"][:, pelvis_id]
human_root_rot = human["body_rotations"][:, pelvis_id]


print("human pelvis")
print(human_root_pos[:3])
print(human_root_pos[-1]-human_root_pos[0])
print("root rot")
print(human_root_rot[:5])
print(human_root_rot[-5:])

# -----------------------
# scale
# -----------------------

# human pelvis height
human_z = np.mean(human_root_pos[:,2])

# H1 pelvis height
h1_z = 0.85
scale = h1_z / human_z
root_positions = human_root_pos * scale
# 初期位置を0にする
root_positions = root_positions - root_positions[0]
root_positions[:,2] += 1.05

# -----------------------
# apply root motion
# -----------------------

body_positions = h1["body_positions"].copy()
body_rotations = h1["body_rotations"].copy()


for t in range(len(body_positions)):

    # root rotation
    q = human_root_rot[t]

    # scipy quaternionへ変換
    # wxyz -> xyzw
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(
        [q[1], q[2], q[3], q[0]]
    ).as_matrix()


    for b in range(len(body_positions[t])):

        # local -> world
        body_positions[t,b] = (
            R @ body_positions[t,b]
            + root_positions[t]
        )

        # rotationも合成
        qb = body_rotations[t,b]

        Rb = Rotation.from_quat(
            [qb[1], qb[2], qb[3], qb[0]]
        )

        Rnew = Rotation.from_matrix(
            R @ Rb.as_matrix()
        )

        qnew = Rnew.as_quat()

        body_rotations[t,b] = [
            qnew[3],
            qnew[0],
            qnew[1],
            qnew[2],
        ]


# -----------------------
# save
# -----------------------

np.savez(
    output,

    fps=h1["fps"],

    dof_names=h1["dof_names"],
    dof_positions=h1["dof_positions"],
    dof_velocities=h1["dof_velocities"],

    body_names=h1["body_names"],
    body_positions=body_positions,
    body_rotations=body_rotations,

    body_linear_velocities=h1["body_linear_velocities"],
    body_angular_velocities=h1["body_angular_velocities"],

    root_positions=root_positions,
    root_rotations=human_root_rot,
)

print("saved",output)