#!/bin/bash

conditions=("rate" "hebb")
patterns=(2500 3000 3500 4000 4500 5000 6000 7000 8000 9000 10000)
run_id=$$

for condition in "${conditions[@]}"; do
    for num in "${patterns[@]}"; do
        mkdir -p logs

        log_file="logs/${run_id}_${condition}_${num}.log"

        ./run_scripts/run_all_deigo.sh "$num" "$condition" "$log_file" &
    done
done
