#!/bin/bash
# Script to run UMamba Diffusion model training in the background

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion_mamba_env

# Run using nohup to prevent it from getting killed when the laptop is closed
echo "Starting UMamba Diffusion training..."
nohup python main_umamba_diffusion.py > diffusion_training.log 2>&1 &

echo "Training is now running in the background."
echo "You can view the logs at any time by running: tail -f diffusion_training.log"
