#!/usr/bin/env bash

source venv/bin/activate

#python fit_gpar.py --data bostonHousing --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data concrete --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data energy --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data kin8nm --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data power-plant --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data protein-tertiary-structure --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data wine-quality-red --experiment weight_pruning_hyperprior --joint
#python fit_gpar.py --data yacht --experiment weight_pruning_hyperprior --joint

#python fit_gpar.py --data bostonHousing --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data concrete --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data energy --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data kin8nm --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data power-plant --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data protein-tertiary-structure --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data wine-quality-red --experiment weight_pruning_hyperprior --joint --rmse
#python fit_gpar.py --data yacht --experiment weight_pruning_hyperprior --joint --rmse


#python fit_gpar.py --data bostonHousing --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data concrete --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data energy --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data kin8nm --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data power-plant --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data protein-tertiary-structure --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data wine-quality-red --experiment weight_pruning_hyperprior2 # --joint
#python fit_gpar.py --data yacht --experiment weight_pruning_hyperprior2 # --joint

#python fit_gpar.py --data bostonHousing --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data concrete --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data energy --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data kin8nm --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data power-plant --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data protein-tertiary-experiment --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data wine-quality-red --experiment weight_pruning_hyperprior2 # --joint --rmse
#python fit_gpar.py --data yacht --experiment weight_pruning_hyperprior2 # --joint --rmse


#python fit_gpar.py --data bostonHousing --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data concrete --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data energy --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data kin8nm --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data power-plant --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data protein-tertiary-structure --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data wine-quality-red --experiment weight_pruning_hyperprior3 # --joint
#python fit_gpar.py --data yacht --experiment weight_pruning_hyperprior3 # --joint

#python fit_gpar.py --data bostonHousing --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data concrete --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data energy --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data kin8nm --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data power-plant --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data protein-tertiary-experiment --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data wine-quality-red --experiment weight_pruning_hyperprior3 --rmse # --joint
#python fit_gpar.py --data yacht --experiment weight_pruning_hyperprior3 --rmse # --joint

#!/usr/bin/env bash

. sys.config

#datasets=("concrete")
datasets=("power-plant" "bostonHousing" "concrete" "energy" "kin8nm" "naval-propulsion-plant" "protein-tertiary-structure" "wine-quality-red" "yacht")
#acquisitions=("EI" "PI" "SD")
seeds=("0" "1" "2" "3" "4" "5")
#experiment="weight_pruning_hyperprior3"
#initial_random_point=3

#for ds in "${datasets[@]}"; do
##    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --rmse --final
##    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --rmse
##    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --final
##    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds"
#
#
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear                     --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear --joint             --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear         --scale_tie --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear --joint --scale_tie --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear                     --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear --joint             --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear         --scale_tie --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear --joint --scale_tie --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear                     --markov 0
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear --joint             --markov 0
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear         --scale_tie --markov 0
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --output_linear --joint --scale_tie --markov 0
#
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear                     --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear --joint             --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear         --scale_tie --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear --joint --scale_tie --markov 3
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear                     --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear --joint             --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear         --scale_tie --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear --joint --scale_tie --markov 1
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear                     --markov 0
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear --joint             --markov 0
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear         --scale_tie --markov 0
#    python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --subexperiment hypersetting_comparison --validsmall --output_linear --joint --scale_tie --markov 0
#
#done

for seed in "${seeds[@]}"; do
    for ds in "${datasets[@]}"; do

        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison                                                   --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --input_linear                                    --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison                --output_linear                    --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --input_linear --output_linear                    --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison                --output_linear --output_nonlinear --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --input_linear --output_linear --output_nonlinear --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison                                --output_nonlinear --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --input_linear                 --output_nonlinear --markov 0

        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall                                                   --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall --input_linear                                    --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall                --output_linear                    --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall --input_linear --output_linear                    --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall                --output_linear --output_nonlinear --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall --input_linear --output_linear --output_nonlinear --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall                                --output_nonlinear --markov 0
        python fit_gpar.py --experiment weight_pruning_hyperprior3 --data "$ds" --seed "$seed" --markov 1 --joint --subexperiment kernal_comparison --validsmall --input_linear                 --output_nonlinear --markov 0

    done
done


