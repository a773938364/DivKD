#!/bin/bash

# Learning rates to try
seeds="6 7 8"

# Loop over learning rates
for seed in $seeds; do
    echo "Training with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python -u run_Ro_graph2tree_cvae_kd.py --seed $seed --gamma 0.01 --use_soft_kd --student_model_path models/model_student_vae_soft_$seed
done
