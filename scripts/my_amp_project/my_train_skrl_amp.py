import sys
import os
import torch

# IsaacLabへのパスを通す
sys.path.append(os.path.abspath("../../source"))

# 1. Isaac Lab のランチャー初期化（必ず最初に呼ぶ必要があります）
from isaaclab.app import AppLauncher
# ローカルでの挙動確認用に画面を表示したい場合は headless=False にしてください
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 2. 必要なライブラリのインポート
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

# skrl (強化学習ライブラリ) のコンポーネント
from skrl.agents.torch.amp import AMP
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# 3. 事前確認
motion_data_path = "./data/motion_for_skrl.pt"
if not os.path.exists(motion_data_path):
    print(f"エラー: {motion_data_path} が見つかりません。")
    simulation_app.close()
    sys.exit(1)

# =============================================================================
# skrl 用のネットワーク（ポリシー、バリュー、ディスクリミネータ）の定義
# =============================================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, self.num_actions)
        )
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# AMP特有のディスクリミネータ（識別器）: 「本物のモーションデータか、ポリシーが作った動きか」を見分ける
class Discriminator(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, amp_observation_space, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(amp_observation_space.shape[0], 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# =============================================================================
# メイン学習ループ
# =============================================================================
def main():
    set_seed(42)

    # 4. Isaac Lab の H1 AMP用タスクをロード (Isaac Lab公式が提供しているタスク名)
    # ※もし独自タスクを作っている場合は、その名前に書き換えてください
    task_name = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    print(f"--- Isaac Lab 環境を起動中: {task_name} ---")
    
    try:
        env = gym.make(task_name)
    except gym.error.NameNotFound:
        print(f"エラー: タスク '{task_name}' が見つかりません。")
        print("お使いのIsaacLabで登録されているAMP用タスク名を確認してください。")
        simulation_app.close()
        return

    # 5. モーションデータのロードとskrl用のバッファ（AMP Buffer）への詰め込み
    print(f"--- モーションデータをロード中: {motion_data_path} ---")
    motion_dataset = torch.load(motion_data_path, map_location=env.device)
    
    # 6. skrl AMPエージェントの設定
    device = env.device
    # デフォルトConfigを使わず、直接辞書として定義する
    cfg = {
        "rollouts": 16,
        "mini_batches": 4,
        "epochs": 5,
        "learning_rate": 5e-5,
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
        "state_preprocessor": None,
        
        # AMP特有の設定
        "amp_motion_dataset": motion_dataset,
        "amp_task_reward_weight": 0.5,
        "amp_style_reward_weight": 0.5,
        
        "experiment": {
            "directory": "./runs",
            "experiment_name": "h1_amp_skrl",
        }
    }
    # 7. メモリとネットワークの初期化
    memory = RandomMemory(memory_size=16384, num_environments=env.num_envs, device=device)
    
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device),
        "discriminator": Discriminator(env.observation_space, env.action_space, device, env.amp_observation_space)
    }

    # 8. エージェントの生成
    agent = AMP(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        amp_observation_space=env.amp_observation_space
    )

    # 9. 学習の実行（Trainer）
    print("🚀 学習を開始します...")
    trainer = SequentialTrainer(cfg={"timesteps": 24000000, "headless": True}, env=env, agents=agent)
    trainer.train()

    # 環境の終了
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()