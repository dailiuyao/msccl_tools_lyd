#!/bin/bash -l
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:05:00
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N nccl-tests-msccl
#PBS -o nccl-tests-msccl.out
#PBS -e nccl-tests-msccl.error

set -x

module load cudatoolkit-standalone/11.4.4

export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl"
MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl-tools-lyd"

echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch1_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch8_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch1_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch8_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch1_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch8_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch1_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch8_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: Simple ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch1_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: Simple ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch1_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL128_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL128_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1



echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins1_hierarchical.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins2_hierarchical.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins1_hierarchical.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins2_hierarchical.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: LL ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL128_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL128_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=LL128

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins2.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

