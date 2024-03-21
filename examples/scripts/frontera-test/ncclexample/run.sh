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



# ./oneProcessOneThread4GPUs

$MPI_HOME/bin/mpirun -np 4 ./oneGPUPerProcessOrThread

$MPI_HOME/bin/mpirun -np 2 ./multipleGpusPerThread

# ##################################### NCCL TEST #####################################
# echo "##################################### NCCL TEST #####################################"
# NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"
# export NCCL_SRC_LOCATION

# NCCLTESTS_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests"
# export NCCLTESTS_SRC_LOCATION

# export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple

# # export NCCL_MIN_NCHANNELS=1
# # export NCCL_MAX_NCHANNELS=1

# # # export NCCL_NTHREADS=256

# $MPI_HOME/bin/mpirun -np 4 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 2 -e 1M -w 0 -f 2 -g 1 -n 10
