#!/bin/bash

#SBATCH --job-name=cpuBCE
#SBATCH --time=1200
#SBATCH --ntasks=4
# #SBATCH --partition=standard
#SBATCH --partition=large-mem
# #SBATCH --gpus=3

echo STARTING

echo TO USE: ./any_BCEonCPUrun.sh LEARNING_RATE BATCH_SIZE

echo Learning Rate $1
echo Batch Size $2

source /etc/profile.d/valet.sh
source ~/.bashrc
vpkg_require anaconda
conda activate env_pytorch
cd /home/2578/gore1/867-team-gore/lpdgen

python3 BCEtrain.py dcgan3d --data_dir ./data --image_size 64 --batch_size $2 --ngf 64 --ndf 64 --nz 64 --learning_rate $1 --output_dir ./models_done --latent_dim 32

echo DONE BATCH JOB

