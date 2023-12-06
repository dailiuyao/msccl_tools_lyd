#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N nccl-profile-125-2
#PBS -o nccl-profile-125-2.out
#PBS -e nccl-profile-125-2.error


module load gcc/11.2.0
module load cudatoolkit-standalone/11.4.4
export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# Set location to store NCCL_TEST source/repository


export NCCL_TEST_HOME="/home/yuke/ncclPG/nccl-tests"

# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/yuke/ncclPG/nccl"
export NCCL_SRC_LOCATION

export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple


mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 2 -e 512MB -w 0 -n 1 -f 2 -g 1
