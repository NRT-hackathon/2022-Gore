#!/bin/bash

#SBATCH --job-name=rstarLM
#SBATCH --time=1200
#SBATCH --ntasks=4
# #SBATCH --partition=large-mem
#SBATCH --partition=standard

echo TO USE: ./restart_CPUrun.sh LEARNING_RATE BATCH_SIZE FILE_LOCATION_OF_SAVED_MODEL

echo =============== Running on standard partition. ===================

echo Learning rate $1
echo Batch size $2
echo Location of model $3

source /etc/profile.d/valet.sh
source ~/.bashrc
vpkg_require anaconda
conda activate env_pytorch
cd /home/2578/gore1/867-team-gore/lpdgen

python3 train.py dcgan3d --data_dir ./data --image_size 64 --batch_size $2 --ngf 64 --ndf 64 --nz 64 --learning_rate $1 --output_dir ./models_done --latent_dim 32 --load_model_pth $3

# #python3 train.py dcgan3d --data_dir ./data --image_size 64 --batch_size $2 --ngf 64 --ndf 64 --nz 64 --learning_rate $1 --output_dir ./models_done --latent_dim 32

