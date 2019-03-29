#!/usr/bin/env bash

source venv/bin/activate

. sys.config

#datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
datasets=("synthetic")
acquisitions=("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")
function_seed="9"
experiment="synthetic"
initial_random_point=3
thompson_samples=1

for ds in "${datasets[@]}"; do

    for seed in "${seeds[@]}"; do
            python dummy_search.py --name "1_sample_per_thompson" --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.01 --seed "$seed" --thompson_samples "$thompson_samples" --random --final --datadir "$DATADIR" --outdir "$OUTDIR"

        for acquisition in "${acquisitions[@]}"; do
            python dummy_search.py --name "1_sample_per_thompson" --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.01 --seed "$seed" --thompson_samples "$thompson_samples" --acquisition "$acquisition" --final --datadir "$DATADIR" --outdir "$OUTDIR"
#            python dummy_search.py --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.02 --seed "$seed" --thompson_samples "$thompson_samples" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
        done
    done
done