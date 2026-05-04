#!/bin/bash
#
# TriPlaneMamba-UNet Training Script
# Run from the directory containing train.py
#

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
SAVE_DIR="./checkpoints_triplane"
LOG_FILE="training_triplane_output.log"

echo "[TriPlaneMamba] Starting fresh training..."
echo "[TriPlaneMamba] Data:  $DATA_DIR"
echo "[TriPlaneMamba] Save:  $SAVE_DIR"
echo "[TriPlaneMamba] Log:   $LOG_FILE"

nohup python train.py \
    --data_dir   "$DATA_DIR"  \
    --epochs     500          \
    --batch_size 1            \
    --lr         2e-4         \
    --base_ch    32           \
    --d_state    16           \
    --patch_size 32 128 128     \
    --num_workers 4           \
    --save_dir   "$SAVE_DIR"  \
    > "$LOG_FILE" 2>&1 &

echo "[TriPlaneMamba] PID: $!"
echo "[TriPlaneMamba] Monitor: tail -f $LOG_FILE"