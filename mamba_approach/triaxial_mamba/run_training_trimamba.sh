#!/bin/bash
#
# TriMamba-UNet V2 Training Script
# Run from the triaxial_mamba/ directory
#

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
SAVE_DIR="./checkpoints_trimamba"

echo "[TriMamba] Starting fresh training..."
echo "[TriMamba] Data: $DATA_DIR"
echo "[TriMamba] Save: $SAVE_DIR"

nohup python train.py \
    --data_dir "$DATA_DIR" \
    --epochs 500 \
    --batch_size 1 \
    --lr 2e-4 \
    --base_ch 32 \
    --patch_size 32 128 128 \
    --num_workers 4 \
    --save_dir "$SAVE_DIR" \
    > training_trimamba_output.log 2>&1 &

echo "[TriMamba] PID: $!"
echo "[TriMamba] Monitor: tail -f training_trimamba_output.log"
