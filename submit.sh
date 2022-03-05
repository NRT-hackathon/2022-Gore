#!/bin/bash -l

# SLURM SUBMIT SCRIPT

# SLURM resource allocation flags
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=0-02:00:00
#SBATCH --signal=SIGUSR1@90

# Activate the appropriate conda env
# source activate <path_to_your_python_with_pytorch_and_lightning>

# Debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# On your cluster you might need these to set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Load the appropriate CUDA module
# module load NCCL/2.4.7-1-cuda.10.0

# Run the training script
# You will need to add the appropriate command line flags necessary 
# for training
srun python3 lpdgen/train.py