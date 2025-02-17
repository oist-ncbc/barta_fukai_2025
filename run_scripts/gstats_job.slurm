#!/bin/bash
#SBATCH -p short
#SBATCH -t 1:00:00
#SBATCH --mem=64G
#SBATCH -c 64
#SBATCH --open-mode=append

# Load Conda if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StructuredInhibition  # Replace with your Conda environment name

# Run gstats_multiproc.py
python gstats_multiproc.py --name "$2_conductances" --folder lognormal --patterns "$1"
