#!/bin/bash -l
#PBS -l select=16:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -q demand
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-16nodes
#PBS -o log/ccl-16nodes.out
#PBS -e log/ccl-16nodes.error

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

# dool --time --mem --cpu --net -N bond0,hsn0,hsn1,lo,total --output /home/yuke/ncclPG/msccl_tools_lyd/examples/scripts/polaris-test/dool.csv 1 &
# DOOL_PID=$!

# sh /home/yuke/lyd/megatron_run_scripts/rtop.sh -d hsn0 > /home/yuke/ncclPG/msccl_tools_lyd/examples/scripts/polaris-test/hsn0.csv &
# RTOP1_PID=$!

# sh /home/yuke/lyd/megatron_run_scripts/rtop.sh -d hsn1 > /home/yuke/ncclPG/msccl_tools_lyd/examples/scripts/polaris-test/hsn1.csv &
# RTOP2_PID=$!

# kill $DOOL_PID
# kill $RTOP1_PID
# kill $RTOP2_P

################################### NCCL TEST Original ##########################################################

echo "NCCL TEST with Original NCCL"

export NCCL_TEST_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests"

NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl"
export NCCL_SRC_LOCATION

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_NTHREADS=512
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=2


$MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32K -e 512MB -f 2 -g 1

# ################################### NCCL TEST Profile ##########################################################

# echo "NCCL TEST with NCCL Profile"

# export NCCL_TEST_PROFILE_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"

# NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
# export NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_PROFILE_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# $MPIEXEC_HOME/bin/mpiexec -n 16 --ppn 4 --cpu-bind core ${NCCL_TEST_PROFILE_HOME}/build/all_reduce_perf -b 32K -e 512MB -f 2 -g 1

################################### MSCCL TEST #########################################################

echo "NCCL TEST with MSCCL"

export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
:q
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
export MSCCL_SRC_LOCATION

export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_p_gpu01_16nodes_channel2_chunk128.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple
export NCCL_NTHREADS=256

$MPIEXEC_HOME/bin/mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 32K -e 512MB -f 2 -g 1
