#!/usr/bin/env bash
source venv/bin/activate

. sys.config

#datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
datasets=("synthetic")
#acquisitions=("EI" "PI" "SD")
#seeds=("0" "1" "2" "3" "4" "5")
#experiment="weight_pruning_hyperprior3"
experiment="synthetic"
#initial_random_point=3
name="1_sample_per_thompson"

for ds in "${datasets[@]}"; do
    python search_analysis.py --experiment "$experiment" --data "$ds" -n "$name" --outdir "$OUTDIR"
done