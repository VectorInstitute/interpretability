#!/bin/bash

#SBATCH --job-name=train-nih
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --mem=100G
#SBATCH --qos=normal
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err

source .venv/bin/activate

srun torchrun --standalone --nproc-per-node=4 --nnodes=1 ../interpretability-bootcamp/reference_implementations/Intepretable-models/Imaging/SelfAttention/train_nih.py
