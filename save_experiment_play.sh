#!/bin/bash

# ===============================
# 実験名
# 使い方:
# ./run_experiment.sh heading_reward
# ===============================
EXP_NAME=$1
TASK_NAME=$2
CHECKPOINT=$3
NUM_ENVS=$4


DATE=$(date +"%Y%m%d_%H%M%S")

SAVE_DIR="experiments/train/${DATE}_${EXP_NAME}"

echo "======================================="
echo "Task         : $TASK_NAME"
echo "Experiment   : $EXP_NAME"
echo "Checkpoint   : $CHECKPOINT"
echo "======================================="


#--------------------------------------------------
# 保存先作成
#--------------------------------------------------
mkdir -p "$SAVE_DIR"/code/humanoid_amp
mkdir -p "$SAVE_DIR"/code/agents
mkdir -p "$SAVE_DIR"/videos
mkdir -p "$SAVE_DIR"/checkpoints
mkdir -p "$SAVE_DIR"/code/skrl

#--------------------------------------------------
# コード保存（学習前）
#--------------------------------------------------
echo "Saving source code..."

cp scripts/reinforcement_learning/skrl/play.py \
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

if [ $# -ne 4 ]; then
    echo "Usage: $0 <task_name> <experiment_name> <checkpoint> <num_envs>"
    exit 1
fi
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task "$TASK_NAME" \
    --algorithm AMP \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" 

echo "Play finished."

echo "======================================="
echo "Experiment saved!"
echo "$SAVE_DIR"
echo "======================================="