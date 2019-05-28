#!/usr/bin/env bash

source venv/bin/activate

. sys.config

#datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
datasets=("synthetic")
acquisitions=("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")
function_seeds=("1" "3" "6")
experiment="incremental_synthetic_3_output"
initial_random_point=5
thompson_samples=4
samples_per_thompson=0
names=("50_sample_per_thompson-no_refit" "10_sample_per_thompson-no_refit" "3_sample_per_thompson-no_refit")
samples_per_thompsons=(0 10 3)
#nameles_per_thompsons=(3)
synthetic_scales="2. 0.5"


for function_seed in "${function_seeds[@]}"; do
    for index in ${!names[*]}; do
        samples_per_thompson="${samples_per_thompsons[$index]}"
        name="${names[$index]}"

        for ds in "${datasets[@]}"; do

            for seed in "${seeds[@]}"; do
                    python incremental_search.py --long --name "$name"  --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.01 --seed "$seed" --thompson_samples "$thompson_samples" --samples_per_thompson "$samples_per_thompson" --random --datadir "$DATADIR" --outdir "$OUTDIR"

                for acquisition in "${acquisitions[@]}"; do
                    python incremental_search.py --long --name "$name" --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.01 --seed "$seed" --thompson_samples "$thompson_samples" --samples_per_thompson "$samples_per_thompson" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
        #            python dummy_search.py --data "$ds" --experiment "$experiment" --function_seed "$function_seed" --noise 0.02 --seed "$seed" --thompson_samples "$thompson_samples" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
                done
            done
        done
    done
done