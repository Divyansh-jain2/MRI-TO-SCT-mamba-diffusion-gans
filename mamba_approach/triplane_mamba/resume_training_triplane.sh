#!/bin/bash
#
# TriPlaneMamba-UNet — Resume Training Script
#
# Checkpoint priority:
#   1. Latest periodic checkpoint (triplane_epochN.pth) — most training preserved
#   2. Best checkpoint (triplane_best.pth)              — fallback if no periodic exists
#
# NOTE: This script only resumes from checkpoints created by this same model
#       (TriPlaneMamba-UNet). Old TriAxial checkpoints are NOT compatible.
#

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
SAVE_DIR="./checkpoints_triplane"
LOG_FILE="training_triplane_output.log"

# ── Auto-detect best checkpoint to resume from ──────────────────────────────
# Priority 1: latest periodic epoch checkpoint
LATEST_EPOCH=$(ls "$SAVE_DIR"/triplane_epoch*.pth 2>/dev/null | sort -V | tail -1)

if [ -n "$LATEST_EPOCH" ]; then
    RESUME_CKPT="$LATEST_EPOCH"
    echo "[TriPlaneMamba] Found latest epoch checkpoint: $RESUME_CKPT"

elif [ -f "$SAVE_DIR/triplane_best.pth" ]; then
    RESUME_CKPT="$SAVE_DIR/triplane_best.pth"
    echo "[TriPlaneMamba] No epoch checkpoint found, using best: $RESUME_CKPT"

else
    echo "[ERROR] No TriPlaneMamba checkpoint found in $SAVE_DIR"
    echo "        Looked for: triplane_epoch*.pth, triplane_best.pth"
    echo "        Run run_training.sh first to create checkpoints."
    exit 1
fi
# ────────────────────────────────────────────────────────────────────────────

echo "[TriPlaneMamba] Resuming from: $RESUME_CKPT"
echo "[TriPlaneMamba] Data:          $DATA_DIR"
echo "[TriPlaneMamba] Save:          $SAVE_DIR"
echo "[TriPlaneMamba] Log:           $LOG_FILE"

nohup python train.py \
    --data_dir   "$DATA_DIR"    \
    --epochs     500            \
    --batch_size 1              \
    --lr         2e-4           \
    --base_ch    32             \
    --d_state    16             \
    --patch_size 32 128 128       \
    --num_workers 4             \
    --save_dir   "$SAVE_DIR"    \
    --resume     "$RESUME_CKPT" \
    >> "$LOG_FILE" 2>&1 &

echo "[TriPlaneMamba] PID: $!"
echo "[TriPlaneMamba] Monitor: tail -f $LOG_FILE"