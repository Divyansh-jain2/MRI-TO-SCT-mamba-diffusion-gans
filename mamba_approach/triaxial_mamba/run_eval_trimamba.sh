#!/bin/bash
#
# TriMamba-UNet V2 Evaluation Script
#

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
CKPT="./checkpoints_trimamba/trimamba_best.pth"

echo "[Eval] Running with TTA..."
python evaluate.py \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CKPT" \
    --base_ch 32 \
    --tta \
    --save_preds
