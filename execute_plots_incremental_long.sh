#!/usr/bin/env bash
source venv/bin/activate

. sys.config

datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm" "wine-quality-red" "protein-tertiary-structure" "yacht")
experiment="weight_pruning_hyperprior3-3-output_incremental_long"
#datasets=("protein-tertiary-structure" "yacht")

#datasets=("synthetic")
#experiment="incremental_synthetic_3_output_incremental_long"

names=("50_sample_per_thompson-no_refit" "10_sample_per_thompson-no_refit" "3_sample_per_thompson-no_refit")

for ds in "${datasets[@]}"; do
    for name in "${names[@]}"; do
        python search_analysis_incremental.py  --experiment_extra _incremental_long --experiment "$experiment" --data "$ds" -n "$name" --outdir "$OUTDIR"
    done
done