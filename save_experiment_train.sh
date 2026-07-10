#!/bin/bash

# ===============================
# 実験名
# 使い方:
# ./run_experiment.sh heading_reward
# ===============================
EXP_NAME=$1
DATE=$(date +"%Y%m%d_%H%M%S")

SAVE_DIR="experiments/train/${DATE}_${EXP_NAME}"

echo "======================================="
echo "Experiment : $EXP_NAME"
echo "======================================="

#--------------------------------------------------
# 保存先作成
#--------------------------------------------------
mkdir -p "$SAVE_DIR"/code/humanoid_amp
mkdir -p "$SAVE_DIR"/code/agents
mkdir -p "$SAVE_DIR"/code/skrl
mkdir -p "$SAVE_DIR"/code/output
mkdir -p "$SAVE_DIR"/videos
mkdir -p "$SAVE_DIR"/checkpoints

#--------------------------------------------------
# コード保存（学習前）
#--------------------------------------------------
echo "Saving source code..."

cp scripts/my_evaluate_project/output/yaw_prob_distribution.pt \
   "$SAVE_DIR"/code/output/

cp scripts/reinforcement_learning/skrl/train.py \
   "$SAVE_DIR"/code/skrl/

cp source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env.py \
   "$SAVE_DIR"/code/humanoid_amp/

cp source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env_cfg.py \
   "$SAVE_DIR"/code/humanoid_amp/

cp source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/__init__.py \
   "$SAVE_DIR"/code/humanoid_amp/

cp source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/agents/skrl_walk_amp_cfg.yaml \
   "$SAVE_DIR"/code/agents/

#--------------------------------------------------
# Git情報
#--------------------------------------------------
git rev-parse HEAD > "$SAVE_DIR"/git_commit.txt
git status > "$SAVE_DIR"/git_status.txt
git diff > "$SAVE_DIR"/git_diff.patch

#--------------------------------------------------
# 学習
# ↓ここを普段使っているコマンドに置き換える
#--------------------------------------------------
echo "Start training..."

if [ -z "$2" ]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
        --task Isaac-Humanoid-AMP-Walk-Direct-v0 \
        --algorithm AMP  \
        --num_envs 1024 \
        --headless \
        --video
else
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
        --task Isaac-Humanoid-AMP-Walk-Direct-v0 \
        --checkpoint "$2" \
        --algorithm AMP   \
        --num_envs 1024 \
        --headless \
        --video
fi


echo "Training finished."

#--------------------------------------------------
# 最新run取得
#--------------------------------------------------
LATEST_RUN=$(ls -td logs/skrl/humanoid_amp_walk/* | head -n 1)

echo "Latest run:"
echo "$LATEST_RUN"

#--------------------------------------------------
# 最新動画コピー
#--------------------------------------------------
LATEST_VIDEO=$(find "$LATEST_RUN/videos/train" -name "*.mp4" | sort -V | tail -n 1)

if [ -f "$LATEST_VIDEO" ]; then
    cp "$LATEST_VIDEO" "$SAVE_DIR/videos/"
    echo "Copied video."
fi

#--------------------------------------------------
# best checkpoint
#--------------------------------------------------
if [ -f "$LATEST_RUN/checkpoints/best_agent.pt" ]; then
    cp "$LATEST_RUN/checkpoints/best_agent.pt" \
       "$SAVE_DIR/checkpoints/"
    echo "Copied best_agent.pt"
fi

echo "======================================="
echo "Experiment saved!"
echo "$SAVE_DIR"
echo "======================================="