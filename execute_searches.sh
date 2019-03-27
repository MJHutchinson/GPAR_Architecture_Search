#!/usr/bin/env bash

source venv/bin/activate

DATADIR="data"
OUTDIR="output"
datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
acquisitions=("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")
experiment="weight_pruning_hyperprior3"
initial_random_point=3
thompson_samples=1


for ds in "${datasets[@]}"; do

    for seed in "${seeds[@]}"; do
            python dummy_search.py --data "$ds" --experiment "$experiment" --thompson_samples "$thompson_samples" --seed "$seed" --random --final --datadir "$DATADIR" --outdir "$OUTDIR"

        for acquisition in "${acquisitions[@]}"; do
            python dummy_search.py --data "$ds" --experiment "$experiment" --thompson_samples "$thompson_samples" --seed "$seed" --acquisition "$acquisition" --final --datadir "$DATADIR" --outdir "$OUTDIR"
#            python dummy_search.py --data "$ds" --experiment "$experiment" --seed "$seed" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
        done
    done
done