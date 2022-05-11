#!/bin/bash

#SBATCH --job-name=gpugan2
#SBATCH --time=600
#SBATCH --ntasks=4
# #SBATCH --partition=standard
#SBATCH --partition=gpu-v100
#SBATCH --gpus=2

source /etc/profile.d/valet.sh
source ~/.bashrc
vpkg_require anaconda
conda activate env_pytorch
cd /home/2578/gore1/867-team-gore/lpdgen

python3 train.py dcgan3d --data_dir ./data --image_size 64 --batch_size 16 --ngf 64 --ndf 64 --nz 64 --learning_rate 1e-5 --output_dir ./models_done --latent_dim 32
