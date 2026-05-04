#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# TriPlaneMamba-UNet Evaluation Script
# Metrics: MAE, RMSE, PSNR, SSIM  |  Sliding-window inference + optional TTA
#
# IMPORTANT: must run in the mamba_ct conda env (has mamba-ssm installed).
# The checkpoint was trained with real Mamba SSM blocks, not the GRU fallback.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Activate conda env that has mamba-ssm ─────────────────────────────────────
CONDA_ENV="mamba_ct"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "[Env] Using conda env: ${CONDA_ENV} ($(which python))"

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR="/DATA/divyansh/mc_ddpm_data/brain_npy"
CKPT_DIR="./checkpoints_triplane"

# Auto-select best checkpoint (or override with a specific one, e.g. triplane_epoch500.pth)
CKPT="${CKPT_DIR}/triplane_best.pth"

BASE_CH=32          # Must match training config
PRED_DIR="./predictions_triplane"

# Set TTA=1 to enable test-time augmentation (4 flips averaged), 0 to disable
TTA=1
# ─────────────────────────────────────────────────────────────────────────────

# Resolve absolute path to checkpoint
CKPT="$(realpath "${CKPT}")"

echo "============================================================"
echo "  TriPlaneMamba-UNet Evaluation"
echo "============================================================"
echo "  Data dir  : ${DATA_DIR}"
echo "  Checkpoint: ${CKPT}"
echo "  Pred dir  : ${PRED_DIR}"
echo "  base_ch   : ${BASE_CH}"
echo "  TTA       : $([ "${TTA}" -eq 1 ] && echo ON || echo OFF)"
echo "============================================================"
echo ""

# Build extra flags
EXTRA_FLAGS="--save_preds --pred_dir ${PRED_DIR}"
[ "${TTA}" -eq 1 ] && EXTRA_FLAGS="${EXTRA_FLAGS} --tta"

python evaluate.py \
    --data_dir   "${DATA_DIR}" \
    --checkpoint "${CKPT}"    \
    --base_ch    "${BASE_CH}" \
    ${EXTRA_FLAGS}

echo ""
echo "[Done] Results written to: ${CKPT_DIR}/triplane_test_results.txt"
