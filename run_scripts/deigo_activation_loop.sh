#!/bin/bash

systems=(
    "hebb"
    "rate")
runs=("spontaneous" "spontaneous_learning")
patterns=(800 2200 2400 2600 2800 3000)
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