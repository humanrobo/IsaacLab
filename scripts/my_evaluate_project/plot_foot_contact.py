import csv
import matplotlib.pyplot as plt
import os

# 保存先
output_dir = "scripts/my_evaluate_project/contact_force_false"
os.makedirs(output_dir, exist_ok=True)

csv_path = "/tmp/humanoid_debug/foot_contact_force_false.csv"

steps = []

left_fx, left_fy, left_fz = [], [], []
right_fx, right_fy, right_fz = [], [], []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        steps.append(int(row["step"]))

        left_fx.append(float(row["left_fx"]))
        left_fy.append(float(row["left_fy"]))
        left_fz.append(float(row["left_fz"]))

        right_fx.append(float(row["right_fx"]))
        right_fy.append(float(row["right_fy"]))
        right_fz.append(float(row["right_fz"]))


# X・Y・Z方向の接触力
components = [
    ("Fx", left_fx, right_fx, "X Contact Force [N]"),
    ("Fy", left_fy, right_fy, "Y Contact Force [N]"),
    ("Fz", left_fz, right_fz, "Z Contact Force [N]"),
]

for name, left_data, right_data, ylabel in components:
    plt.figure(figsize=(14, 5))

    plt.plot(steps, left_data, label="Left Foot")
    plt.plot(steps, right_data, label="Right Foot")

    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.title(f"Foot Contact Force {name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, f"foot_contact_{name.lower()}.png"),
        dpi=200
    )
    plt.show()
    plt.close()