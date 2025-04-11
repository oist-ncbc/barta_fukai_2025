#!/bin/bash

conditions=(
    "hebb"
    "rate")
runs=("stimuli100ms")
patterns=(10000)
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
