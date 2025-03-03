#!/bin/bash
#SBATCH -p compute
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -C xeon
#SBATCH --open-mode=append

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate StructuredInhibition

python simulation.py --system "config/systems/$2.yml" --run "config/runtypes/$3.yml" --patterns "$1"