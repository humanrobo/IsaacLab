import sys
import os

# IsaacLab/source へのパスを通す
sys.path.append(os.path.abspath("../../source"))

# エラーを避けるため、大元のパッケージからインポートを試みる
try:
    import isaaclab_assets
    # H1_CFGの場所を自動探索
    if hasattr(isaaclab_assets, "H1_CFG"):
        H1_CFG = isaaclab_assets.H1_CFG
    else:
        from isaaclab_assets import H1_CFG
except ImportError:
    print("エラー: isaaclab_assets が見つかりません。")
    print("sys.path に追加したパスが正しいか確認してください。")
    sys.exit(1)

def main():
    print("\n==============================================")
    print("   Isaac Lab 内の H1 関節順（DoF Order）確認")
    print("==============================================")
    
    # H1_CFG内の主要なデータを出力
    cfg = H1_CFG
    
    # パターン1: cfg.data.ordered_joint_names を探す
    if hasattr(cfg, "data") and hasattr(cfg.data, "ordered_joint_names") and cfg.data.ordered_joint_names:
        print("\n【確定ジョイント順】:")
        for i, name in enumerate(cfg.data.ordered_joint_names):
            print(f" {i:02d}: {name}")
        return

    # パターン2: 自由度のデフォルト値（dof_posの初期値など）のキー名から抽出する
    # Isaac LabのConfigでは、多くの場合ここに29個の関節名が辞書型で定義されています
    init_state = None
    if hasattr(cfg, "init_state"):
        init_state = cfg.init_state
    elif hasattr(cfg, "default_joint_angles"):
        init_state = cfg
        
    if init_state and hasattr(init_state, "dof_pos"):
        # dof_pos が辞書型、またはオブジェクトの場合、そのキー名が関節名です
        dof_pos_data = init_state.dof_pos
        if isinstance(dof_pos_data, dict):
            print("\n【Configの初期姿勢から抽出した29関節名（順番通り）】:")
            for i, name in enumerate(dof_pos_data.keys()):
                print(f" {i:02d}: {name}")
            return
        elif hasattr(dof_pos_data, "__dict__"):
            print("\n【Configの初期姿勢から抽出した29関節名（順番通り）】:")
            for i, name in enumerate(dof_pos_data.__dict__.keys()):
                print(f" {i:02d}: {name}")
            return

    # パターン3: それでもダメならActuatorグループの規則を表示
    print("\n[情報] 直接のリストが見つからないため、グループ規則を表示します:")
    for k, v in cfg.actuators.items():
        print(f"  - グループ '{k}': {v.joint_names_expr}")

if __name__ == "__main__":
    main()