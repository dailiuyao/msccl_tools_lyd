#!/bin/bash

set -e



export CUDA_HOME=
export MPI_HOME=

##################################### MSCCL #####################################
echo "##################################### MSCCL #####################################"
MSCCL_SRC_LOCATION=""
export MSCCL_SRC_LOCATION

NCCLTESTS_MSCCL_SRC_LOCATION=""
export NCCLTESTS_MSCCL_SRC_LOCATION

export LD_LIBRARY_PATH="${MSCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_DEBUG=TRACE
export NCCL_PROTO=Simple
# export NCCL_NTHREADS=256

export GENMSCCLXML=

export MSCCL_XML_FILES=/home/liuyao/scratch/deps/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree_p_gpu01/allreduce_binary_tree_p_gpu01_1ch_1chunk.xml

$MPI_HOME/bin/mpirun -np 32 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 32K -e 512M -f 2 -g 1
