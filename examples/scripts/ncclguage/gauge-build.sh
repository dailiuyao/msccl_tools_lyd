#!/bin/bash

# Set environment variables

export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

source /home/liuyao/sbatch_sh/.nvccrc

export CUDA_HOME=/home/liuyao/software/cuda-11.7
export PATH=/home/liuyao/software/cuda-11.7/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/liuyao/software/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDACXX=/home/liuyao/software/cuda-11.7/bin/nvcc
export CUDNN_LIBRARY=/home/liuyao/software/cuda-11.7/lib64
export CUDNN_INCLUDE_DIR=/home/liuyao/software/cuda-11.7/include

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# NCCL source location
NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"

# Compilation command. Ensure to link against the MPI and NCCL libraries correctly.
# nvcc $NVCC_GENCODE -ccbin g++ -I${NCCL_SRC_LOCATION}/build/include -I${MPI_HOME}/include -L${NCCL_SRC_LOCATION}/build/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -lnccl -lcudart -lmpi $1 -o ${1%.cu}.exe
nvcc $NVCC_GENCODE -ccbin g++ -I${NCCL_SRC_LOCATION}/build/include -I${MPI_HOME}/include -L${NCCL_SRC_LOCATION}/build/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -lnccl -lcudart -lmpi $1 -o ${1%.cu}.exe

# Verification of the output
if [ -f ${1%.cu}.exe ]; then
    echo "Compilation successful. Output file: ${1%.cu}.exe"
else
    echo "Compilation failed."
fi