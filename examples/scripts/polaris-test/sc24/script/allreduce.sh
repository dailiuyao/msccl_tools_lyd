#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:19:59
#PBS -q prod
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-16nodes
#PBS -o ../log/ccl-16nodes-trinomial-1.out
#PBS -e ../log/ccl-16nodes-trinomial-1.error

export MPI_HOME=/opt/cray/pe/mpich/8.1.25/ofi/nvidia/20.7
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda

export MPIEXEC_HOME=/opt/cray/pe/pals/1.2.11
export NCCL_NET_PLUGIN_HOME="/home/yuke/ncclPG/aws-ofi-nccl-1.7.4-aws/build"          

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# export OFI_NCCL_PROTOCOL=RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_IB_HCA=cxi0,cxi1
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple

################################### MSCCL TEST #########################################################

echo "NCCL TEST with MSCCL"

export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
export MSCCL_SRC_LOCATION

export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple
export NCCL_NTHREADS=512

export GENMSCCLXML=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/scripts/gen_msccl_xml_frontera.sh



echo "##################################### MSCCL #####################################" 

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_1ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_2ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_1ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_2ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling/allreduce_recursive_doubling_1ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling/allreduce_recursive_doubling_2ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling_halving/allreduce_recursive_doubling_halving_1ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling_halving/allreduce_recursive_doubling_halving_2ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/trinomial_tree/allreduce_trinomial_tree_1ch_64chunk_64gpus.xml

# $MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 768K -e 256M -f 2 -g 1 -n 60

export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/trinomial_tree/allreduce_trinomial_tree_2ch_32chunk_64gpus.xml

$MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 768K -e 256M -f 2 -g 1 -n 60

export MSCCL_XML_FILES=" "

export NCCL_ALGO=TREE

$MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

export NCCL_ALGO=RING

$MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60
