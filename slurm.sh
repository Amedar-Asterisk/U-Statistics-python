#!/bin/bash

#SBATCH --job-name=V_stats_compleity
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=16G
source ~/.bashrc

python -u test_com.py