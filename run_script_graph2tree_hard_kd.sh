#!/bin/bash

# Learning rates to try
seeds="5 6 7"

# Loop over learning rates
for seed in $seeds; do
    echo "Training with seed: $seed"
    CUDA_VISIBLE_DEVICES=1 python -u run_Ro_graph2tree_cvae_kd.py --seed $seed   --beta 0.1 --use_hard_kd --student_model_path models/model_student_vae_hard_$seed
done
