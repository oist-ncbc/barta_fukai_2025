#!/bin/bash

# Ensure arguments are provided
if [[ -z "$1" ]]; then
    echo "Error: Missing argument for patterns." >&2
    exit 1
fi

if [[ -z "$2" ]]; then
    echo "Error: Missing argument for plasticity." >&2
    exit 1
fi

if [[ -z "$3" ]]; then
    echo "Error: Missing argument for log file."
    exit 1
fi

num="$1"
condition="$2"
log_file="$3"

# Ensure log directory exists
mkdir -p logs

# Redirect both stdout and stderr to the same log file

# Submit SLURM jobs and capture job IDs
job_id1=$(sbatch --parsable --wait --output="$log_file" --error="$log_file" run_scripts/train_and_cond_job.slurm "$num" "$condition")

job_id2=$(sbatch --parsable --wait --output="$log_file" --error="$log_file" run_scripts/gstats_job.slurm "$num" "$condition")

job_id3=$(sbatch --parsable --output="$log_file" --error="$log_file" run_scripts/perturbation_job.slurm "$num" "$condition")