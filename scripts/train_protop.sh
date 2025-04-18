#!/bin/bash
#SBATCH --job-name=single-node-multiple-gpus
#SBATCH --partition=rtx6000
#SBATCH --qos=m
#SBATCH --nodes=1
#SBATCH --mem=128gb

#SBATCH --gres=gpu:rtx6000:4
#SBATCH --output=imagenet.%j.out
#SBATCH --error=imagenet.%j.err
#SBATCH --time=12:00:00
# SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1

# export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # set to 1 for NCCL backend
# export CUDA_LAUNCH_BLOCKING=1

export PYTHONPATH="."
nvidia-smi

torchrun --nproc-per-node=4 --nnodes=1 main_proto.py 