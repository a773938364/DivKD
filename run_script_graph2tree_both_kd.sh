#!/bin/bash

# Learning rates to try
seeds="2 3 4"

# Loop over learning rates
for seed in $seeds; do
    echo "Training with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python -u run_Ro_graph2tree_cvae_kd.py --seed $seed   --beta 0.1 --gamma 0.01 --use_hard_kd --use_soft_kd --student_model_path models/model_student_vae_both_$seed
done
