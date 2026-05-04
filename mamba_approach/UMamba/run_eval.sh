#!/bin/bash

# ─────────────────────────────────────────────
# Evaluation script for U-Mamba MRI-to-CT Synthesis
# Usage: bash run_eval_umamba.sh
# ─────────────────────────────────────────────

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
CHECKPOINT="./checkpoints_umamba/umamba_best.pth"
MODEL="umamba"
PRED_DIR="./predictions_umamba"

echo "======================================"
echo " U-Mamba MRI-to-CT Evaluation"
echo "======================================"
echo "Data dir   : $DATA_DIR"
echo "Checkpoint : $CHECKPOINT"
echo "Model      : $MODEL"
echo "Predictions: $PRED_DIR"
echo "======================================"

python ../src/evaluate.py \
    --data_dir   $DATA_DIR \
    --checkpoint $CHECKPOINT \
    --model      $MODEL \
    --pred_dir   $PRED_DIR \
    --save_preds

echo ""
echo "======================================"
echo " Evaluation complete!"
echo " Results saved to: $(dirname $CHECKPOINT)/${MODEL}_test_results.txt"
echo "======================================"
