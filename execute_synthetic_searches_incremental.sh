#!/usr/bin/env bash

source venv/bin/activate

. sys.config

#datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
datasets=("synthetic")
acquisitions=("PI")  #("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")
function_seed="7"
experiment="incremental_synthetic"
initial_random_point=3
thompson_samples=4
samples_per_thompson=3
name="3_sample_per_thompson"

for ds in "${datasets[@]}"; do

    for seed in "${seeds[@]}"; do
            python incremental_search.py --name "$name"  --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.01 --seed "$seed" --thompson_samples "$thompson_samples" --samples_per_thompson "$samples_per_thompson" --random --datadir "$DATADIR" --outdir "$OUTDIR"

        for acquisition in "${acquisitions[@]}"; do
            python incremental_search.py --name "$name" --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.01 --seed "$seed" --thompson_samples "$thompson_samples" --samples_per_thompson "$samples_per_thompson" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
#            python dummy_search.py --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.02 --seed "$seed" --thompson_samples "$thompson_samples" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
        done
    done
done