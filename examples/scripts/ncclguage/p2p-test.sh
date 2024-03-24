#!/bin/bash

# Set environment variables

module load mpich/3.4.2-gcc-8.4.1 

source /home/liuyao/sbatch_sh/.nvccrc

export CUDA_HOME=/home/liuyao/software/cuda-11.6

# export MPI_HOME="/home/liuyao/software/mpich_4_1_1_pgcc"

export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_gcc-8.4.1"
export NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Additional compiler flags for NVCC
# export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

##################################### NCCL #####################################
echo "##################################### NCCL #####################################"

NCCLTESTS_SRC_LOCATION="/home/liuyao/scratch/deps/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=RING
export NCCL_PROTO=Simple
# export NCCL_NTHREADS=192

export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4

# $MPI_HOME/bin/mpirun -np 2 -hosts node03:2 $NCCLTESTS_SRC_LOCATION/build/sendrecv_perf -b 2MB -e 2MB -f 2 -g 1 > output.log 2>&1

# $MPI_HOME/bin/mpirun -np 2 -hosts node03:1,node04:1 $NCCLTESTS_SRC_LOCATION/build/sendrecv_perf -b 2MB -e 2MB -f 2 -g 1 > output.log 2>&1

$MPI_HOME/bin/mpirun -np 4 -hosts node03:2,node04:2 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 64MB -e 64MB -f 2 -g 1 > output.log 2>&1

