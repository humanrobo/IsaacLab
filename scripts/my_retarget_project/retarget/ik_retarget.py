#humanoid.npzからh1.urdfに対応する関節角度を計算するスクリプト
import numpy as np
import pinocchio as pin
import mink
from mink import SE3
import mujoco

from key_body_mapping import HUMANOID_TO_H1


MOTION_FILE = "data/humanoid_walk.npz"
URDF_FILE = "data/h1_ik_only.urdf"

OUTPUT_FILE = "output/h1_dof_positions.npy"


# -----------------------
# Load motion
# -----------------------

motion = np.load(
    MOTION_FILE,
    allow_pickle=True
)


body_names = motion["body_names"]

body_pos = motion["body_positions"]
body_rot = motion["body_rotations"]


num_frames = body_pos.shape[0]


print("frames:", num_frames)



# body index
def get_body_id(name):
    return np.where(body_names == name)[0][0]


pelvis_id = get_body_id("pelvis")



# -----------------------
# Load H1 model
# -----------------------

model = mujoco.MjModel.from_xml_path(
    str(URDF_FILE)
)

configuration = mink.Configuration(
    model
)
print(configuration.q)
print(configuration.model.nq)

print("H1 nq:", configuration.model.nq)



# -----------------------
# Create tasks
# -----------------------

tasks = []


def create_task(target_body):

    h1_frame = HUMANOID_TO_H1[target_body]

    task = mink.FrameTask(
        h1_frame,
        "body",
        position_cost=1.0,
        orientation_cost=0.0,
    )

    tasks.append(task)

    return task



task_dict = {}


for body in HUMANOID_TO_H1.keys():

    task_dict[body] = create_task(body)



# -----------------------
# IK loop
# -----------------------

dt = 1.0 / 30.0


q_list = []


for t in range(num_frames):


    pelvis = body_pos[
        t,
        pelvis_id
    ]


    # target update
    for human_body, task in task_dict.items():


        idx = get_body_id(human_body)


        target_pos = (
            body_pos[t, idx]
            -
            pelvis
        )


        T = SE3(
            np.array([
                1.0, 0.0, 0.0, 0.0,
                target_pos[0],
                target_pos[1],
                target_pos[2],
            ])
        )


        task.set_target(T)



    # solve IK

    vel = mink.solve_ik(
        configuration,
        tasks,
        dt,
        solver="quadprog"
    )


    configuration.integrate_inplace(
        vel,
        dt
    )


    q_list.append(
        configuration.q.copy()
    )


    print(
        f"{t+1}/{num_frames}",
        end="\r"
    )



q_array = np.array(q_list)


print()
print("result:", q_array.shape)


np.save(
    OUTPUT_FILE,
    q_array
)


print(
    "saved:",
    OUTPUT_FILE
)