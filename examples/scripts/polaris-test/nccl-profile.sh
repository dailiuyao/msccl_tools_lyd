#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -q prod
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N nccl-profile-31
#PBS -o nccl-profile-31.out
#PBS -e nccl-profile-31.error


module load gcc
module load cudatoolkit-standalone/11.4.4
export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# Set location to store NCCL_TEST source/repository


export NCCL_TEST_HOME="/home/yuke/ncclPG/nccl-tests-profile"

# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/yuke/ncclPG/nccl_profile"
export NCCL_SRC_LOCATION

export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Tree


mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 16MB -e 16MB -f 2 -g 1
