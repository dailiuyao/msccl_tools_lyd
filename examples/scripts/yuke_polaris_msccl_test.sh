#!/bin/bash -l
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N nccl-tests-msccl
#PBS -o nccl-tests-msccl.out
#PBS -e nccl-tests-msccl.error

set -x



# echo "########################################   MSCCL TEST  #####################################################"

# module load cudatoolkit-standalone/11.4.4

# export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
# export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
# export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl-test

# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
# MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl-tools-lyd"

# echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch1_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch8_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1 


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch1_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch8_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch1_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch8_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch1_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch8_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: Simple ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch1_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: Simple ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch1_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL128_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL128_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1



# echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins1_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1




# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins1_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1



# echo "########################################   NCCL TEST  #####################################################"

# cd /home/yuke/ncclPG/nccl-tests-cu116

# #module swap PrgEnv-nvhpc PrgEnv-gnu
# #module load nvhpc-mixed
# #source env.sh
# module load nvhpc/23.1
# module load cudatoolkit-standalone/11.4.4

# export NCCL_MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/cray/10.0
# export NCCL_CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4
# export NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/comm_libs/nccl

# #export PATH=${NCCL_MPI_HOME}/bin:$PATH
# export LD_LIBRARY_PATH=${NCCL_CUDA_HOME}/lib64:${NCCL_MPI_HOME}/lib:${NCCL_HOME}/lib:$LD_LIBRARY_PATH

# #make MPI=1 NCCL_MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1 NCCL_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/comm_libs/nccl

# echo "######################### LIBRARY: NCCL ALGORITHM: RING PROTOCOL: LL ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Ring
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# echo "######################### LIBRARY: NCCL ALGORITHM: RING PROTOCOL: SIMPLE ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Ring
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# # echo "######################### LIBRARY: NCCL ALGORITHM: RING PROTOCOL: LL128 ##############################################"

# # export NCCL_DEBUG=INFO
# # export NCCL_ALGO=Ring
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: LL ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Tree
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: SIMPLE ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# # echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: LL128 ##############################################"

# # export NCCL_DEBUG=INFO
# # export NCCL_ALGO=Tree
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1



# # echo "########################################   MSCCL TEST  #####################################################"

# # module load cudatoolkit-standalone/11.4.4

# # export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
# # export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
# # export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl-test

# # export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# # MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
# # MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl-tools-lyd"


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL128_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL128_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1



echo "########################################   MSCCL TEST  #####################################################"

module load cudatoolkit-standalone/11.4.4

export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl-test

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl-tools-lyd"


# echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST RING time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: AllPAIRS INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST AllPAIRS time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_allpairs_v2_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY-PIPE INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY-PIPE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY-PIPE INSTANCE: 1 CHANNEL: 4 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY-PIPE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_ch4_h_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY-PIPE INSTANCE: 1 CHANNEL: 8 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY-PIPE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_ch8_h_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINOMIAL time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINOMIAL INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINOMIAL time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_h_Simple_gpus_64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: H-4-NOMIAL INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-4-NOMIAL time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_4_nomial_tree_h_Simple_gpus_64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 1-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 1-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_1ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-4-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-4-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_4ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 4-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 4-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_4ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: HIE-ALLREDUCE INSTANCE: 1 PROTOCOL: Simple ##############################################"

# Print the current time
echo "MSCCL TEST HIE-ALLREDUCE time: $(date)"


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_allreduce_hierarchical_allreduce_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: REC-DOUB-HALV INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST REC-DOUB-HALV time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINO-H INSTANCE: 1 PROTOCOL: Simple ##############################################"

# Print the current time
echo "MSCCL TEST BINO-H time: $(date)"


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_h_ch4_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 1-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 1-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_1ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-4-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-4-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_4ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# Print the current time
echo "MSCCL TEST BINARY-H-P 2-INTRA-1-INTER time: $(date)"


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 4-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# Print the current time
echo "MSCCL TEST BINARY-H-P 4-INTRA-2-INTER time: $(date)"


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_4ch_intra_2ch_inter_Simple_gpu64_ins1.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1
