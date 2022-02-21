#!/bin/bash

usage() {
cat << EOF
Usage: setup_environments.sh CONDA_ROOT DEVICE_OPTION
    ACTION ........... Root folder of your anaconda installation
    DEVICE_OPTION .... Device option, either "cpu" or "gpu"
EOF
}

if [ $# -lt 2 ]
then
    usage
    exit 0
fi

CONDA_ROOT=$1
GPU_YES=$2

CWD=`pwd`
CONDA_PATH=${CONDA_ROOT}/condabin
CONDA_ENVS=${CONDA_ROOT}/envs
CONDA_EXE=${CONDA_PATH}/conda
GIT_EXE=`which git`
CONDA_ENV_NAME=py3_lpdgen
CONDA_ENV_PATH=${CONDA_ENVS}/${CONDA_ENV_NAME}

${CONDA_EXE} create -n ${CONDA_ENV_NAME} -c conda-forge python=3 numpy scipy matplotlib scikit-image --yes
cp .condarc ${CONDA_ENV_PATH}
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

conda install -n ${CONDA_ENV_NAME} pytest pandas pep8 numba imageio --yes
conda install -n ${CONDA_ENV_NAME} ipykernel sphinx scikit-learn --yes
conda install -n ${CONDA_ENV_NAME} pillow opencv --yes

if [ ${GPU_YES} == 'gpu' ]
then
    conda install -n ${CONDA_ENV_NAME} -c pytorch pytorch torchvision torchaudio --yes
elif [ ${GPU_YES} == 'cpu' ]
then
    conda install -n ${CONDA_ENV_NAME} -c pytorch pytorch torchvision torchaudio cpuonly --yes
else
    echo "Unknown device entered. Only cpu or gpu are allowed as options."
fi

conda install -n ${CONDA_ENV_NAME} pytorch-lightning --yes
