#!/bin/bash
#SBATCH -p short
#SBATCH -t 1:00:00
#SBATCH --mem=128G
#SBATCH -c 5
#SBATCH -C xeon
#SBATCH --open-mode=append

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StructuredInhibition

python get_activations.py --patterns "$1" --run "$2" --system "$3"
