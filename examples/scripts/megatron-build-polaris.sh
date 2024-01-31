#!/bin/bash

cd /home/yuke/lyd

# mkdir conda3
# cd conda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh -b -f -p $(pwd)

# conda create --name pytorchNCCL-hao python=3.9.12

cd conda3

source /home/yuke/lyd/conda.sh

conda activate pytorchNCCL-hao

#module load cudatoolkit-standalone/11.8.0

module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
#ml nvhpc-mixed/22.11
ml gcc/10.3.0
ml cudatoolkit-standalone/11.8.0



# export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
# #export CUDA_NVCC_EXECUTABLE=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/nvcc
# export USE_DISTRIBUTED=1
# export MPI_HOME=/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1
export USE_SYSTEM_NCCL=1
export NCCL_ROOT=/home/yuke/ncclPG/CCL-LYD/msccl-lyd
export NCCL_INCLUDE_DIR=/home/yuke/ncclPG/CCL-LYD/msccl-lyd/build/include
export NCCL_LIB_DIR=/home/yuke/ncclPG/CCL-LYD/msccl-lyd/build/lib

# cd /home/yuke/lyd/conda3/envs/pytorchNCCL-hao

# # git clone --recursive --branch v2.0.0 https://github.com/pytorch/pytorch.git

# cd /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/pytorch

# pip install cmake==3.27.5
# pip install ninja==1.11.1
#conda install cmake=3.27.5 ninja=1.10.2

# cd /home/yuke/lyd/cudnn

# wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz

# tar -xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz 

export USE_CUDA=1
export USE_CUDNN=1
export CUDNN_LIBRARY=/home/yuke/lyd/cudnn/cudnn-linux-x86_64-8.7.0.84_cuda11-archive
export CUDNN_LIB_DIR=/home/yuke/lyd/cudnn/cudnn-linux-x86_64-8.7.0.84_cuda11-archive/lib
export CUDNN_INCLUDE_DIR=/home/yuke/lyd/cudnn/cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include

# pip install -r requirements.txt

# pip install mkl==2023.2.0
# pip install mkl-include==2023.2.0
#conda install mkl mkl-include

# conda install -c pytorch magma-cuda118

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# pip install numpy==1.19.3 pyyaml==6.0.1 setuptools==68.0.0 cffi==1.15.1 typing_extensions==4.8.0 future==0.18.3 six==1.16.0 requests==2.31.0 dataclasses==0.6

export CPLUS_INCLUDE_PATH=/home/yuke/lyd/cudnn/cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/home/yuke/lyd/cudnn/cudnn-linux-x86_64-8.7.0.84_cuda11-archive/include:$C_INCLUDE_PATH

# pip install regex==2022.7.9
# pip install numpy==1.19.3
# pip install pybind11==2.11.1

# python setup.py clean

# python setup.py install

# cd /home/yuke/lyd

# git clone https://github.com/NVIDIA/apex apex
# cd apex
# git checkout master
# git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0

# cd /home/yuke/lyd/apex

# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# pip install -r requirements.txt

# cd /home/yuke/lyd/conda3/envs/pytorchNCCL-hao/lib/python3.9/site-packages/apex/amp

# cp /home/yuke/lyd/tmp-file/_amp_state.py .

# cp /home/yuke/lyd/tmp-file/_initialize.py .

# git clone --branch v3.0.2 --depth 1 https://github.com/NVIDIA/Megatron-LM.git
# to do steps


# pip install torchvision==0.15.1
# pip install torchaudio==2.0.1
# pip uninstall torch

cd /home/yuke/lyd/conda3/envs/pytorchNCCL/pytorch

python setup.py clean

python setup.py install

