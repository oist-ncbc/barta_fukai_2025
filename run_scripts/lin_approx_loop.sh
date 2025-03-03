#!/bin/bash

conditions=("hebb_recharge_plus_strong")
patterns=(1000 1200 1400 1600 1800 2000)

for condition in "${conditions[@]}"; do
    for num in "${patterns[@]}"; do

        python activations.py --system "$condition" --patterns "$num" &

    done
done
