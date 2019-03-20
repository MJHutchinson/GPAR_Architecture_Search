#!/usr/bin/env bash
source venv/bin/activate

#DATADIR="data"
#OUTDIR="output"
#datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
datasets=("synthetic")
#acquisitions=("EI" "PI" "SD")
#seeds=("0" "1" "2" "3" "4" "5")
#experiment="weight_pruning_hyperprior3"
experiment="synthetic"
#initial_random_point=3
version=0.2.0

for ds in "${datasets[@]}"; do
    python search_analysis.py --experiment "$experiment" --data "$ds" -v "$version"
done