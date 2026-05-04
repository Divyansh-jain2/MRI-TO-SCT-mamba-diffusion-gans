#!/bin/bash
echo "Starting U-Mamba Training in the background..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python ../src/train.py \
    --data_dir /DATA/divyansh/mc_ddpm_data/brain_npy \
    --model umamba \
    --patch_size 32 128 128 \
    --save_dir ./checkpoints_umamba \
    > training_umamba_output.log 2>&1 &

echo "Training started! View logs with: tail -f training_umamba_output.log"
