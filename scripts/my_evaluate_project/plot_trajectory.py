import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# データ読み込み
traj = np.load("output/h1_body_trajectory.npy")

print("shape:", traj.shape)

body_names = [
    'torso',
    'pelvis',
    'head',
    'right_upper_arm',
    'left_upper_arm',
    'right_thigh',
    'left_thigh',
    'right_lower_arm',
    'left_lower_arm',
    'right_shin',
    'left_shin',
    'right_hand',
    'left_hand',
    'right_foot',
    'left_foot'
]

left_id = body_names.index("left_foot")
right_id = body_names.index("right_foot")

left = traj[:, left_id, :]
right = traj[:, right_id, :]
left_hand = traj[:, body_names.index("left_hand"), :]
right_hand = traj[:, body_names.index("right_hand"), :]
left_foot = traj[:, body_names.index("left_foot"), :]
right_foot = traj[:, body_names.index("right_foot"), :]

# 足のXY軌跡
plt.figure(figsize=(6, 6))
plt.plot(left[:, 0], left[:, 1], label="Left foot")
plt.plot(right[:, 0], right[:, 1], label="Right foot")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Foot Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(args.output_dir, "foot_xy.png"), dpi=300)
#足の上下
plt.figure(figsize=(10,4))
plt.plot(left_foot[:,2], label="Left foot")
plt.plot(right_foot[:,2], label="Right foot")
plt.xlabel("Step")
plt.ylabel("Height [m]")
plt.title("Foot Height")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(args.output_dir, "foot_height.png"), dpi=300)
#手の前後
plt.figure(figsize=(10,4))
plt.plot(left_hand[:,0], label="Left hand")
plt.plot(right_hand[:,0], label="Right hand")
plt.xlabel("Step")
plt.ylabel("X position [m]")
plt.title("Hand Swing")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(args.output_dir, "hand_x.png"), dpi=300)
plt.show()