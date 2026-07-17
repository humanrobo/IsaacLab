import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument("--output_dir",default="output")
args=parser.parse_args()
os.makedirs(args.output_dir,exist_ok=True)
foot_dir=os.path.join(
    args.output_dir,
    "foot"
)
hand_dir=os.path.join(
    args.output_dir,
    "hand"
)
os.makedirs(foot_dir,exist_ok=True)
os.makedirs(hand_dir,exist_ok=True)
root_traj=np.load("output/root_xy_trajectory.npy")
foot_traj=np.load("output/foot_trajectory.npy")
hand_traj=np.load("output/hand_trajectory.npy")
print("root:",root_traj.shape)
print("foot:",foot_traj.shape)
print("hand:",hand_traj.shape)

# ==========================
# root XY trajectory
# ==========================
# 初期位置を原点化
root_traj = root_traj - root_traj[0:1]
plt.figure(figsize=(8,8))
for i in range(20):
    plt.plot(
        root_traj[:,i,0],
        root_traj[:,i,1],
        label=f"{i*18} deg"
    )
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Root Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend(
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)
plt.savefig(
    os.path.join(
        args.output_dir,
        "root_xy_20direction.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================
# foot cycle each direction
# ==========================
for i in range(20):
    left_z = foot_traj[:,i,0,2]
    right_z = foot_traj[:,i,1,2]
    plt.figure(figsize=(10,4))
    plt.plot(
        left_z,
        label="Left foot"
    )
    plt.plot(
        right_z,
        label="Right foot"
    )
    plt.xlabel("Step")
    plt.ylabel("Foot height [m]")
    plt.title(
        f"Foot Cycle ({i*18} deg)"
    )
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(
            foot_dir,
            f"{i*18}deg_foot_cycle.png"
        ),
        dpi=300
    )
    plt.close()

# ==========================
# hand cycle each direction
# ==========================
for i in range(20):
    left_hand_x = hand_traj[:,i,0,0]
    right_hand_x = hand_traj[:,i,1,0]
    plt.figure(figsize=(10,4))
    plt.plot(
        left_hand_x,
        label="Left hand"
    )
    plt.plot(
        right_hand_x,
        label="Right hand"
    )
    plt.xlabel("Step")
    plt.ylabel("Hand X [m]")

    plt.title(
        f"Hand Cycle ({i*18} deg)"
    )
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(
            hand_dir,
            f"{i*18}deg_hand_cycle.png"
        ),
        dpi=300
    )
    plt.close()

print("Done")