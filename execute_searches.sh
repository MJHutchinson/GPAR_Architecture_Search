#!/usr/bin/env bash

source venv/bin/activate

DATADIR="data"
OUTDIR="output"
datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm")
acquisitions=("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5")
experiment="weight_pruning_hyperprior3"
initial_random_point=3

for ds in "${datasets[@]}"; do

    for seed in "${seeds[@]}"; do
            python dummy_search.py --data "$ds" --experiment "$experiment" --seed "$seed" --random --final --datadir "$DATADIR" --outdir "$OUTDIR"

        for acquisition in "${acquisitions[@]}"; do
            python dummy_search.py --data "$ds" --experiment "$experiment" --seed "$seed" --acquisition "$acquisition" --final --datadir "$DATADIR" --outdir "$OUTDIR"
            python dummy_search.py --data "$ds" --experiment "$experiment" --seed "$seed" --acquisition "$acquisition" --datadir "$DATADIR" --outdir "$OUTDIR"
        done
    done
done



#python examples/dummy_search.py --data bostonHousing --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data concrete --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data energy --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data kin8nm --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data power-plant --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data protein-tertiary-structure --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data wine-quality-red --experiment weight_pruning_hyperprior3 -n 3 # --joint
#python examples/dummy_search.py --data yacht --experiment weight_pruning_hyperprior3 -n 3 # --joint
#
#python examples/dummy_search.py --data bostonHousing --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data concrete --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data energy --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data kin8nm --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data power-plant --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data protein-tertiary-experiment --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data wine-quality-red --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint
#python examples/dummy_search.py --data yacht --experiment weight_pruning_hyperprior3 --rmse -n 3 # --joint