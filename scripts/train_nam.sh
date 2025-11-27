#!/bin/bash
#SBATCH --job-name=train-nam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --qos=normal
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err

python ../reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/train_nam.py
