#!/bin/bash

# ===============================
# 使い方:
# ./run_experiment.sh <task_name> <experiment_name> [checkpoint]
#
# 例:
# ./run_experiment.sh Isaac-Humanoid-AMP-Walk-Direct-v0 heading_reward
# ===============================

if [ -z "$1" ]; then
    echo "Error: task name is required."
    echo "Usage: $0 <task_name> "
    exit 1
fi

EXP_NAME=$1

DATE=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="output/${DATE}_${EXP_NAME}"
mkdir -p "$SAVE_DIR"

echo "======================================="
echo "Task         : $TASK_NAME"
echo "======================================="

#--------------------------------------------------
# 学習
# ↓ここを普段使っているコマンドに置き換える
#--------------------------------------------------
echo "Start..."

python plot_trajectory.py --output_dir "$SAVE_DIR"


echo "======================================="
echo "Experiment saved!"
echo "$SAVE_DIR"
echo "======================================="