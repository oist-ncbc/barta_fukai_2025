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
run_id="$3"

# Ensure log directory exists
mkdir -p logs
mkdir -p "logs/${run_id}"
mkdir -p "logs/${run_id}/${condition}"
mkdir -p "logs/${run_id}/${condition}/${num}"

# Redirect both stdout and stderr to the same log file

# Submit SLURM jobs and capture job IDs

log_file_gstats="logs/${run_id}/${condition}/${num}/gstats.log"
job_id_gstats=$(sbatch --parsable --output="$log_file_gstats" --error="$log_file_gstats" run_scripts/gstats_job.slurm "$num" "$condition")
echo "SLURM JOB ID: $job_id_gstats" >> "$log_file_gstats"

log_file_spont="logs/${run_id}/${condition}/${num}/spont.log"
job_id_spont=$(sbatch --parsable --output "$log_file_spont" --error "$log_file_spont" run_scripts/simulation_job.slurm "$num" "$condition" spontaneous)
echo "SLURM JOB ID: $job_id_spont" >> "$log_file_spont"

log_file_spont_learn="logs/${run_id}/${condition}/${num}/spont_learn.log"
job_id_spont_learn=$(sbatch --parsable --output "$log_file_spont_learn" --error "$log_file_spont_learn" run_scripts/simulation_job.slurm "$num" "$condition" spontaneous_learning)
echo "SLURM JOB ID: $job_id_spont_learn" >> "$log_file_spont_learn"

log_file_perturbation="logs/${run_id}/${condition}/${num}/perturbation.log"
job_id_perturbation=$(sbatch --parsable --dependency=afterok:"$job_id_gstats" --output="$log_file_perturbation" --error="$log_file_perturbation" run_scripts/perturbation_job.slurm "$num" "$condition")
echo "SLURM JOB ID: $job_id_perturbation" >> "$log_file_perturbation"