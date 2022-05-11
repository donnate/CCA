#!/bin/bash

#SBATCH --job-name=array-job
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-100
#SBATCH --time=24:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem=4G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
module load python/anaconda-2021.05

python3 experiments-link-prediction-all-methods.py --epochs 300 --patience 30 --dataset Cora --result_file $SLURM_ARRAY_TASK_ID
~
~
