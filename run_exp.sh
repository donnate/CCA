#!/bin/sh

#SBATCH --partition=broadwl
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive

# Load the default version of GNU Parallel
module load python/anaconda-2021.05

python3 experiments-link-prediction-all-methods.py --epochs 300 --patience 30
