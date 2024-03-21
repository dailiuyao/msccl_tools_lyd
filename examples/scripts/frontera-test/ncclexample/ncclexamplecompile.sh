#!/bin/bash

# Load necessary modules
module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3

# Set environment variables
export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"

# NCCL source location
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"

LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# Compilation command
nvcc $NVCC_GENCODE -I${NCCL_SRC_LOCATION}/build/include -I${MPI_HOME}/include -L${NCCL_SRC_LOCATION}/build/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -lnccl -lcudart -lmpi $1 -o ${1%.cu}

# Verification of the output
if [ -f ${1%.cu} ]; then
    echo "Compilation successful. Output file: ${1%.cu}"
else
    echo "Compilation failed."
fi
