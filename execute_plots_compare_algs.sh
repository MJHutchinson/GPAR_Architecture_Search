#!/usr/bin/env bash
source venv/bin/activate

datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm" "wine-quality-red" "yacht")
experiment="weight_pruning_hyperprior3-3-output_incremental_long"
datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm" "wine-quality-red" "yacht")
experiment="weight_pruning_hyperprior3-3-output_long"
#datasets=("synthetic")
#experiment=incremental_synthetic_3_output_incremental_long
#datasets=("synthetic")
#experiment=incremental_synthetic_3_output_long

for ds in "${datasets[@]}"; do
    python compare_algorithms.py --experiment "$experiment" --data "$ds"
done