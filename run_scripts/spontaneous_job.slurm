#!/bin/bash
#SBATCH -p short
#SBATCH -t 1:00:00
#SBATCH --mem=128G
#SBATCH -c 10
#SBATCH -C xeon
#SBATCH --open-mode=append

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StructuredInhibition

python simulation.py --system "config/systems/$2.yml" --run config/runtypes/spontaneous.yml --patterns "$1"