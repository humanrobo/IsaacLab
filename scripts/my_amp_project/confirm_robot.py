import torch
import pickle
from isaaclab.app import AppLauncher

# 1. シミュレータの起動準備 (headless=Trueで画面なし起動)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 2. 必要なライブラリのインポート（AppLauncherの後に呼ぶ必要があります）
from isaaclab_assets import H1_CFG
from isaaclab.assets import Articulation

def main():
    print("\n=== Isaac Lab の H1 の関節順を確認中 ===")
    # H1のConfigから関節名のリストを取得
    # ※バージョンによってCFGの構造が少し異なる場合、以下で名前のリストが取れます
    if hasattr(H1_CFG, "actuators"):
        for name, actuator in H1_CFG.actuators.items():
            print(f"Actuator Group: {name}, Joints: {actuator.joint_names_expr}")
            
    # もしくは環境を簡易起動して直接順番（dof_names）を引っこ抜くのが最も確実です
    print("\n--- 注意: これより下はシミュレータ上で確認する名前です ---")
    #（※もし上記で名前が出れば、まずはそれをメモしてください）

if __name__ == "__main__":
    main()
    simulation_app.close()