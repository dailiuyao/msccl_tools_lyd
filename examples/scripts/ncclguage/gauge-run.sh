#!/bin/bash

source /home/liuyao/sbatch_sh/.mpich_ucx

export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# $MPI_HOME/bin/mpirun -np 2 -hosts node04:1,node05:1 ./gauge-run.sh 

export CUDA_HOME=/home/liuyao/software/cuda-11.7
export PATH=/home/liuyao/software/cuda-11.7/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/liuyao/software/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDACXX=/home/liuyao/software/cuda-11.7/bin/nvcc
export CUDNN_LIBRARY=/home/liuyao/software/cuda-11.7/lib64
export CUDNN_INCLUDE_DIR=/home/liuyao/software/cuda-11.7/include

source /home/liuyao/sbatch_sh/.nvccrc

export NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=1

export NCCL_NTHREADS=256

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

cd /home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage

./gauge/gauge.exe


# ##################################### NCCL #####################################
# echo "##################################### NCCL #####################################"

# NCCLTESTS_SRC_LOCATION="/home/liuyao/scratch/deps/nccl-tests"
# export NCCLTESTS_SRC_LOCATION

# export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=RING
# export NCCL_PROTO=Simple
# # export NCCL_NTHREADS=192

# export NCCL_MIN_NCHANNELS=1
# export NCCL_MAX_NCHANNELS=1

# $MPI_HOME/bin/mpirun -np 2 -hosts node05:2 $NCCLTESTS_SRC_LOCATION/build/sendrecv_perf -b 4MB -e 4MB -f 2 -g 1 -n 20 > output.log 2>&1

