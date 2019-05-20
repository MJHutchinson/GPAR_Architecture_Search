#!/usr/bin/env bash

source venv/bin/activate

. sys.config

#datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
#datasets=("kin8nm" "wine" "yacht")
datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm" "wine" "yacht")
#datasets=("synthetic")
acquisitions=("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")
experiment="weight_pruning_hyperprior3-3-output"
initial_random_point=3
thompson_samples=4
samples_per_thompsons=(0 20 10 5 3 1)
names=("50_sample_per_thompson-no_refit" "20_sample_per_thompson-no_refit" "10_sample_per_thompson-no_refit" "5_sample_per_thompson-no_refit" "3_sample_per_thompson-no_refit" "1_sample_per_thompson-no_refit")

for index in in ${!names[*]}; do
    samples_per_thompson="${samples_per_thompsons[$index]}"
    name="${names[$index]}"
    for ds in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
                python incremental_search.py --name "$name"  --data "$ds" --experiment "$experiment" --seed "$seed" --thompson_samples "$thompson_samples" --samples_per_thompson "$samples_per_thompson" --random --datadir "$DATADIR" --outdir "$OUTDIR"

            for acquisition in "${acquisitions[@]}"; do
                python incremental_search.py --name "$name" --data "$ds" --experiment "$experiment" --seed "$seed" --thompson_samples "$thompson_samples" --samples_per_thompson "$samples_per_thompson" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
    #            python dummy_search.py --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.02 --seed "$seed" --thompson_samples "$thompson_samples" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
            done
        done
    done
done