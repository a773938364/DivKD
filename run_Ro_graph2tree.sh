#!/bin/bash

# Learning rates to try
seeds="0 1 2 3 4"

# Loop over learning rates
for seed in $seeds; do
    echo "Training with seed: $seed"
    CUDA_VISIBLE_DEVICES=1 python -u run_Ro_graph2tree.py --seed $seed
done
