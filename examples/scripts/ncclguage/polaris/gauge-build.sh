#!/bin/bash

# Set environment variables

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.25/ofi/nvidia/20.7
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# NCCL source location
NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"

# Compilation command. Ensure to link against the MPI and NCCL libraries correctly.
# nvcc $NVCC_GENCODE -ccbin g++ -I${NCCL_SRC_LOCATION}/build/include -I${MPI_HOME}/include -L${NCCL_SRC_LOCATION}/build/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -lnccl -lcudart -lmpi $1 -o ${1%.cu}.exe
nvcc $NVCC_GENCODE -ccbin g++ -I${NCCL_SRC_LOCATION}/build/include -I${MPI_HOME}/include -L${NCCL_SRC_LOCATION}/build/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -lnccl -lcudart -lmpi $1 -o ${1%.cu}.exe

# Verification of the output
if [ -f ${1%.cu}.exe ]; then
    echo "Compilation successful. Output file: ${1%.cu}.exe"
else
    echo "Compilation failed."
fi