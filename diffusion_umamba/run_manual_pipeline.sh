#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL RUN PIPELINE
#  This script trains the perfectly optimized UMamba Diffusion Model and then
#  evaluates the dosimetric accuracy locally.
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Activate conda env
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/teaching/miniconda3)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate diffusion_mamba_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================="
echo " 🚀 STEP 1: TRAINING OPTIMIZED UMAMBA DIFFUSION"
echo "======================================================="
python3 main_umamba_diffusion.py

echo "======================================================="
echo " 📊 STEP 2: DOSIMETRIC ANALYSIS & VISUALIZATION"
echo "======================================================="
python3 evaluate_dosimetry.py

echo "✅ Pipeline Complete! Check 'final_run/results_brain_umamba_diffusion' for results."
