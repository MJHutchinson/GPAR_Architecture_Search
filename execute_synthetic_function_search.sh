#!/usr/bin/env bash

seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")

for seed in "${seeds[@]}"; do
    python test_synthetic_draw.py -s "$seed" -l
done