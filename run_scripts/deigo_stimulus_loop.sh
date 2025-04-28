#!/bin/bash

conditions=(
    "hebb"
    "patpacsize_minus"
    "patpacsize_plus")
runs=("stimuli100ms")
patterns=(800 1000 1200 1400 1600 1800 2000)
run_id=$$

for condition in "${conditions[@]}"; do
    for num in "${patterns[@]}"; do
        for run in "${runs[@]}"; do
            mkdir -p logs

            log_file="logs/${run_id}_${run}_${condition}_${num}.log"

            sbatch --output "$log_file" --error "$log_file" run_scripts/simulation_job.slurm "$num" "$condition" "$run"
        done
    done
done
