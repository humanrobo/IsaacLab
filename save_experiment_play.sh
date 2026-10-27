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
HEADLESS=false
MANUAL=false

for arg in "${@:5}"; do
    if [ "$arg" = "--headless" ]; then
        HEADLESS=true
    elif [ "$arg" = "--manual" ]; then
        MANUAL=true
    fi
done

DATE=$(date +"%Y%m%d_%H%M%S")

SAVE_DIR="experiments/train/${DATE}_${EXP_NAME}"

echo "======================================="
echo "Task         : $TASK_NAME"
echo "Experiment   : $EXP_NAME"
echo "Checkpoint   : $CHECKPOINT"
echo "Num envs     : $NUM_ENVS"
echo "Headless     : $HEADLESS"
echo "Manual       : $MANUAL"
echo "======================================="

#--------------------------------------------------
# 引数チェック
#--------------------------------------------------
if [ $# -lt 4 ] || [ $# -gt 6 ]; then
    echo "Usage:"
    echo "  $0 <experiment_name> <task_name> <checkpoint> <num_envs> [--headless] [--manual]"
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
-p scripts/reinforcement_learning/skrl/play_defo.py
--task "$TASK_NAME"
--algorithm AMP
--checkpoint "$CHECKPOINT"
--num_envs "$NUM_ENVS"
)

# headless指定があれば追加
if [ "$HEADLESS" = true ]; then
    CMD+=(--headless)
fi

# manual指定があれば追加
if [ "$MANUAL" = true ]; then
    CMD+=(--manual)
fi

echo "Running:"
echo "${CMD[@]}"

"${CMD[@]}"

echo "Play finished."