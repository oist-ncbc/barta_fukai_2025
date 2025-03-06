#!/bin/bash

systems=(
    "hebb_recharge_minus_strong_tr2.0"
    "hebb_recharge_minus_strong_tr2.1"
    "hebb_recharge_minus_strong_tr2.2"
    "hebb_recharge_minus_strong_tr2.3"
    "hebb_recharge_minus_strong_tr2.4"
    "hebb_recharge_minus_strong_tr2.5"
    "hebb_recharge_minus_strong_tr2.6"
    "hebb_recharge_minus_strong_tr2.7"
    "hebb_recharge_plus_strong_tr1.5"
    "hebb_recharge_plus_strong_tr1.6"
    "hebb_recharge_plus_strong_tr1.7"
    "hebb_recharge_plus_strong_tr1.8"
    "hebb_recharge_plus_strong_tr1.9"
    "hebb_recharge_plus_strong_tr2.0"
    "hebb_recharge_plus_strong_tr2.1"
    "hebb_recharge_plus_strong_tr2.2"
    "hebb_recharge_plus_strong_tr2.3"
    "hebb_recharge_plus_strong_tr2.4"
    "hebb_recharge_plus_strong_tr2.5")
runs=("spontaneous")
patterns=(1000 1200 1400 1600 1800 2000)
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