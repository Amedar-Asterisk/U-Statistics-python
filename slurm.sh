#!/bin/bash

#SBATCH --job-name=graph_proof
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --output=%x.out
#SBATCH --error=%x.err

source ~/.bashrc

python -u test_com.py