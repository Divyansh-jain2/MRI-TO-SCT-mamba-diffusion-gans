#!/bin/bash
# Script to run UMamba Diffusion model testing in the background

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion_mamba_env

# Run using nohup to prevent it from getting killed when the laptop is closed
echo "Starting UMamba Diffusion testing..."
nohup python test_umamba_diffusion.py > test_umamba.log 2>&1 &

echo "Testing is now running in the background."
echo "You can view the logs at any time by running: tail -f test_umamba.log"
