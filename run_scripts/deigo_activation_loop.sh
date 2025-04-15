#!/bin/bash

systems=(
    "rate_ii")
runs=("spontaneous" "spontaneous_learning")
patterns=(800 1000 1200 1400 1600 1800 2000)
run_id=$$

for system in "${systems[@]}"; do
    for run in "${runs[@]}"; do
        for num in "${patterns[@]}"; do
            mkdir -p logs

            log_file="logs/${run_id}_${system}_${run}_activation_${num}.log"

            # sbatch --output "$log_file" --error "$log_file" run_scripts/spontaneous_learning_job.slurm "$num" "$condition"
            sbatch --output "$log_file" --error "$log_file" run_scripts/activation_job.slurm "$num" "$run" "$system"
        done
    done
done