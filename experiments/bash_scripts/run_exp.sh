#!/bin/sh

#SBATCH --partition=caslake
#SBATCH --account=pi-cdonnat
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1

# Load the default version of GNU Parallel
module load python
module load pytorch

python3 comparison-all-methods.py --epochs 1000 --patience 10 --lr 0.01 --dataset Cora
