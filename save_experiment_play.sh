#!/bin/bash

# 使い方
# GUIあり
# ./run_experiment.sh EXP_NAME TASK_NAME CHECKPOINT NUM_ENVS
#
# headless
# ./run_experiment.sh EXP_NAME TASK_NAME CHECKPOINT NUM_ENVS --headless

EXP_NAME=$1
TASK_NAME=$2
CHECKPOINT=$3
NUM_ENVS=$4
HEADLESS=$5

DATE=$(date +"%Y%m%d_%H%M%S")

SAVE_DIR="experiments/train/${DATE}_${EXP_NAME}"

echo "======================================="
echo "Task         : $TASK_NAME"
echo "Experiment   : $EXP_NAME"
echo "Checkpoint   : $CHECKPOINT"
echo "Num envs     : $NUM_ENVS"
echo "Headless     : $HEADLESS"
echo "======================================="

#--------------------------------------------------
# 引数チェック
#--------------------------------------------------
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "Usage:"
    echo "  $0 <experiment_name> <task_name> <checkpoint> <num_envs> [--headless]"
    exit 1
fi

#--------------------------------------------------
# 保存先作成
#--------------------------------------------------
mkdir -p "$SAVE_DIR"/code/humanoid_amp
mkdir -p "$SAVE_DIR"/code/agents
mkdir -p "$SAVE_DIR"/videos
mkdir -p "$SAVE_DIR"/checkpoints
mkdir -p "$SAVE_DIR"/code/skrl

# （コード保存などはそのまま）

#--------------------------------------------------
# 学習
#--------------------------------------------------
echo "Start play..."

CMD=(
./isaaclab.sh
-p scripts/reinforcement_learning/skrl/play.py
--task "$TASK_NAME"
--algorithm AMP
--checkpoint "$CHECKPOINT"
--num_envs "$NUM_ENVS"
)

# headless指定があれば追加
if [ "$HEADLESS" = "--headless" ]; then
    CMD+=(--headless)
fi

echo "Running:"
echo "${CMD[@]}"

"${CMD[@]}"

echo "Play finished."