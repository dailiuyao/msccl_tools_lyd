#!/bin/bash

# Set environment variables

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

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

export MPIEXEC_HOME=/opt/cray/pals/1.3.4
export NCCL_NET_PLUGIN_HOME="/home/yuke/ncclPG/aws-ofi-nccl-1.7.4-aws/build"     
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_IB_HCA=cxi0,cxi1
export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:$LD_LIBRARY_PATH

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# NCCL source location
NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# export NCCL_MIN_NCHANNELS=1
# export NCCL_MAX_NCHANNELS=1

# export NCCL_NTHREADS=256

cd /home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/scripts/ncclguage

./gauge/gauge.exe


# export MPIEXEC_HOME=/opt/cray/pals/1.3.4
# $MPIEXEC_HOME/bin/mpirun -np 2 -hosts x3006c0s1b1n0:2 gauge-run.sh

