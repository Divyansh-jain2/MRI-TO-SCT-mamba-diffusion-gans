#!/bin/bash
echo "Starting Mamba Training in the background..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python ../src/train.py --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy --model segmamba --patch_size 32 128 128 > training_output.log 2>&1 &

echo "Training started! View logs with: tail -f training_output.log"