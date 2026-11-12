import os
import csv
import matplotlib.pyplot as plt

# CSVファイル
CSV_PATH = "/tmp/humanoid_debug/kinematic_true_debug.csv"

# グラフ保存先
OUTPUT_DIR = "/home/matsuno/IsaacLab/scripts/my_evaluate_project/plots_true"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV読み込み
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# step
steps = [float(row["step"]) for row in rows]

# 全データ列
# data_columns = [
#     col for col in rows[0].keys()
#     if col not in ["step", "time"]
# ]
data_columns = [
    "left_foot_pos_z",
    "right_foot_pos_z",
    "left_foot_linvel_z",
    "right_foot_linvel_z",
    "torso_pos_z",
]

data_columns = [
    col for col in data_columns
    if col in rows[0]
]

# 各データを散布図化
for col in data_columns:
    values = [float(row[col]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.scatter(steps, values, s=5)

    plt.xlabel("Step")
    plt.ylabel(col)
    plt.title(col)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{col}.png"),
        dpi=150
    )

    plt.close()

print(f"Generated {len(data_columns)} scatter plots.")
print(f"Saved to: {OUTPUT_DIR}")