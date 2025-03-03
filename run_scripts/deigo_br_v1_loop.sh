#!/bin/bash

conditions=("hebb_br_v1r" "hebb_br_v1o")
patterns=(1000 1200 1400 1600 1800 2000)
run_id=$$

for condition in "${conditions[@]}"; do
    for num in "${patterns[@]}"; do
        mkdir -p logs

        log_file="logs/${run_id}_${condition}_${num}.log"

        ./run_scripts/run_all_deigo.sh "$num" "$condition" "$log_file" &
    done
done
