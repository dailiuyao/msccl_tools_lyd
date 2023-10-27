#!/bin/bash

### Load Modules (DIFFERENT DEPENDING ON SYSTEM) ###
module load cuda
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
# CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda"
export CUDA_HOME
#module load mpich  # [FIXME] Why does this cause NCCL build to error? Very strange...
# module load mpich/3.4.2-nvidiahpc-21.9-0
MPI_HOME=/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0 
export MPI_HOME
export PATH="${MPI_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"

# Set location to store NCCL/MSCCL source/repository
MSCCL_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/msccl_lyd"
export MSCCL_SRC_LOCATION

NCCL_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl"
export NCCL_SRC_LOCATION

### Set environment variables ###

node01=gnode001
node02=gnode002

echo $node01,$node02

echo ""

# Get the current timestamp
timestamp=$(date "+%Y-%m-%d %H:%M:%S")

echo "############################################################# MSCCL ########################################################################"

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
MSCCL_HOME="${MSCCL_SRC_LOCATION}/build" 
export MSCCL_HOME
echo "[DEBUG] MSCCL_HOME has been set to: ${MSCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${MSCCL_HOME}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
PATH="${MSCCL_HOME}/include:${PATH}"
export PATH
echo ""

export LD_LIBRARY_PATH=${MSCCL_HOME}/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=/home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/xml/allreduce_ring_Simple_gpu4_ch2_ins1.xml
export NCCL_ALGO=MSCCL,Tree,Ring
export NCCL_PROTO=Simple

mpirun -np 4 -host $node01:2,$node02:2\
 /home/ldai8/scratch/msccl_build/deps/nccl-tests-msccl-lyd/build/all_reduce_perf -w 0 -b 64MB -e 64MB -f 2 -g 1 -n 1 


echo "############################################################# NCCL ########################################################################"

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
export NCCL_HOME
echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
PATH="${NCCL_HOME}/include:${PATH}"
export PATH
echo ""

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

mpiexec -np 4 -host $node01:2,$node02:2\
 /home/ldai8/scratch/msccl_build/deps/nccl-tests/build/all_reduce_perf -w 0 -b 64MB -e 64MB -f 2 -g 1 -n 1