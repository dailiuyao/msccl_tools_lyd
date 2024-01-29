#!/bin/bash

set -e

module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3

export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

export WORK_DIR=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test

NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"
export NCCL_PROFILE_SRC_LOCATION

NCCLTESTS_NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile"
export NCCLTESTS_NCCL_PROFILE_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_PROFILE_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple




# Determine the MPI rank and calculate the GPU ID
MPI_RANK=${OMPI_COMM_WORLD_RANK:-$PMI_RANK}

# Run nsys profile for the specific GPU
nsys profile -o nccl-output/nsys_test_nccl_profile_rank${MPI_RANK} --stats=true $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60
