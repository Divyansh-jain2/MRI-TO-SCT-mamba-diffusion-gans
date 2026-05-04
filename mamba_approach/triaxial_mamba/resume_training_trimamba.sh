#!/bin/bash
#
# TriMamba-UNet V2 — Resume Training Script
# Resumes from the best checkpoint (or epoch 50 if available)
#

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
SAVE_DIR="./checkpoints_trimamba"

# Auto-detect best checkpoint to resume from
if [ -f "$SAVE_DIR/trimamba_epoch50.pth" ]; then
    RESUME_CKPT="$SAVE_DIR/trimamba_epoch50.pth"
elif [ -f "$SAVE_DIR/trimamba_best.pth" ]; then
    RESUME_CKPT="$SAVE_DIR/trimamba_best.pth"
else
    echo "[ERROR] No checkpoint found in $SAVE_DIR"
    echo "  Looked for: trimamba_epoch50.pth, trimamba_best.pth"
    exit 1
fi

echo "[TriMamba] Resuming training from: $RESUME_CKPT"
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
    --resume "$RESUME_CKPT" \
    > training_trimamba_output.log 2>&1 &

echo "[TriMamba] PID: $!"
echo "[TriMamba] Monitor: tail -f training_trimamba_output.log"
