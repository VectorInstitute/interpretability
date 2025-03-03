#!/bin/bash
#SBATCH --job-name=train-nih
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --mem=50G
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err

source .venv/bin/activate
torchrun --standalone --nproc-per-node=4 --nnodes=1 scripts/train_nih.py