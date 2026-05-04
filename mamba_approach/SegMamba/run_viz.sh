#!/bin/bash

# ─────────────────────────────────────────────
# Visualization script for Mamba MRI-to-CT
# Usage: bash run_viz.sh
# ─────────────────────────────────────────────

DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
CHECKPOINT="./checkpoints/segmamba_best.pth"
MODEL="segmamba"
NUM_CASES=37
OUT_DIR="./visualizations"

echo "======================================"
echo " Mamba MRI-to-CT Visualization"
echo "======================================"
echo "Data dir    : $DATA_DIR"
echo "Checkpoint  : $CHECKPOINT"
echo "Model       : $MODEL"
echo "Cases       : $NUM_CASES"
echo "Output dir  : $OUT_DIR"
echo "======================================"

python ../src/visualize.py \
    --data_dir   $DATA_DIR \
    --checkpoint $CHECKPOINT \
    --model      $MODEL \
    --num_cases  $NUM_CASES \
    --out_dir    $OUT_DIR

echo ""
echo "======================================"
echo " Done! PNG files saved to: $OUT_DIR"
echo "======================================"