#!/usr/bin/env bash

seed="7"
noises=("0.01" "0.02" "0.05" "0.1" "0.2" "0.5" "0.8" "1." "1.5")

python test_synthetic_draw.py -s "$seed" -l -f

for noise in "${noises[@]}"; do
    python test_synthetic_draw.py -s "$seed" -n "$noise" -f
done