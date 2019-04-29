#!/usr/bin/env bash
source venv/bin/activate

. sys.config

datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
#datasets=("synthetic")
#acquisitions=("EI" "PI" "SD")
#seeds=("0" "1" "2" "3" "4" "5")
experiment="weight_pruning_hyperprior3"
#experiment="synthetic"
#initial_random_point=3
names=("1_sample_per_thompson" "3_sample_per_thompson" "5_sample_per_thompson" "10_sample_per_thompson" "20_sample_per_thompson" "all_sample_per_thompson")

for ds in "${datasets[@]}"; do
    for name in "${names[@]}"; do
        python search_analysis.py --experiment "$experiment" --data "$ds" -n "$name" --outdir "$OUTDIR"
    done
done