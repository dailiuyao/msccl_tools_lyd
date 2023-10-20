#!/bin/bash


### Load Modules (DIFFERENT DEPENDING ON SYSTEM) ###
module load cuda
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
# CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda"
export CUDA_HOME
#module load mpich  # [FIXME] Why does this cause NCCL build to error? Very strange...
# module load mpich/3.4.2-nvidiahpc-21.9-0
MPI_HOME=/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0 
# MPI_HOME=/opt/apps/mpi/mpich-3.4.2_gcc-8.4.1
export MPI_HOME
export PATH="${MPI_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"

# Set location to store NCCL source/repository
NCCL_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/msccl_lyd"
export NCCL_SRC_LOCATION

### Set environment variables ###

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

# Set location of CUDA on the machine
# CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
# export CUDA_HOME

# # set multiple nodes name
# scontrol show hostname > nccl_2node.list

# # Extract node names from the list
# node01=`sed -n '1p' nccl_2node.list`
# node02=`sed -n '2p' nccl_2node.list`

node01=gnode005.cluster
node02=gnode008.cluster

echo $node01,$node02

echo ""

# mpirun -np 8 -host $node01:2,$node02:2,$node03:2,$node04:2 -x LD_LIBRARY_PATH=/home/ldai8/scratch/msccl_build/deps/msccl/build/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV -x MSCCL_XML_FILES=/home/ldai8/bash/msccl/allreduce_allpairs_multinode_4gpus_2.xml -x NCCL_ALGO=MSCCL,RING,TREE  /home/ldai8/scratch/msccl_build/deps/nccl-tests/build/all_reduce_perf -b 128 -e 32MB -f 2 -g 1
#mpirun -np 2 -host $node01,$node02 -x LD_LIBRARY_PATH=/home/ldai8/scratch/msccl_build/deps/msccl/build/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV -x  NCCL_ALGO=MSCCL,RING,TREE  /home/ldai8/scratch/msccl_build/deps/nccl-tests/build/all_reduce_perf -b 128 -e 32MB -f 2 -g 2 -c 1 -n 100 -w 100 -G 100 -z 0
#mpirun -np 2 -host $node01,$node02 -npernode 1 hostname

export LD_LIBRARY_PATH=${NCCL_HOME}/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=/home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/xml/allreduce_binary_tree_Simple_gpu4_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpirun -np 4 -host $node01:2,$node02:2 /home/ldai8/scratch/msccl_build/deps/nccl-tests-msccl-lyd/build/all_reduce_perf -w 1 -b 512MB -e 512MB -f 2 -g 1 -n 1\
 >> /home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/scripts/pinnacles-test/2nodes_test2.output 
