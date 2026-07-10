import pickle

pkl_path = "./data/B4_-_Stand_to_Walk_backwards_stageii.pkl"

try:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"エラー: {pkl_path} が見つかりません。")
    import os
    print("現在のデータフォルダの中身:", os.listdir("./data") if os.path.exists("./data") else "dataフォルダ自体がありません")
    exit()

print("--- pklデータ内のキーの確認 ---")
for k in data.keys():
    print(k)

if "joint_names" in data:
    print("\n【pkl内の関節順】:")
    for i, name in enumerate(data["joint_names"]):
        print(f" {i:02d}: {name}")
elif "dof_names" in data:
    print("\n【pkl内の関節順】:")
    for i, name in enumerate(data["dof_names"]):
        print(f" {i:02d}: {name}")
else:
    print("\n[判定] pklデータ内に関節名のテキスト情報が含まれていません。")
