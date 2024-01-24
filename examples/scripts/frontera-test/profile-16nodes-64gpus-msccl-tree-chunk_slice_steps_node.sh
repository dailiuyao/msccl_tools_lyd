#!/bin/bash

module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

##################################### MSCCL #####################################
echo "##################################### MSCCL #####################################"
MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/msccl-lyd"
export MSCCL_SRC_LOCATION

NCCLTESTS_MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION

export LD_LIBRARY_PATH="${MSCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_DEBUG=TRACE
export NCCL_PROTO=Simple



export NCCL_NTHREADS=64

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1 -n 60 >> output.log 2>&1


export NCCL_NTHREADS=128

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1 -n 60 >> output.log 2>&1


export NCCL_NTHREADS=256

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1 -n 60 >> output.log 2>&1


export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1 -n 60 >> output.log 2>&1
