#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-2nodes
#PBS -o log/ccl-2nodes.out
#PBS -e log/ccl-2nodes.error

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

export MPIEXEC_HOME=/opt/cray/pals/1.3.4
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
# export NCCL_NTHREADS=192
# export NCCL_MIN_NCHANNELS=2
# export NCCL_MAX_NCHANNELS=2

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_HOME}/build/sendrecv_perf -b 2 -e 512MB -f 2 -g 1

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 2 --cpu-bind core ${NCCL_TEST_HOME}/build/sendrecv_perf -b 2 -e 512MB -f 2 -g 1

$MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 2 -e 1MB -f 2 -g 1

$MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 2 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 2 -e 1MB -f 2 -g 1

# ################################### NCCL TEST Profile ##########################################################

# echo "NCCL TEST with NCCL Profile"

# export NCCL_TEST_PROFILE_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"

# NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
# export NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_PROFILE_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_PROFILE_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

# ################################### MSCCL TEST #########################################################


# export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
# MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
# export MSCCL_SRC_LOCATION

# export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export NCCL_PROTO=Simple

# # echo "NCCL TEST with NCCL_MSCCL 1 channel 2gpus"

# # export NCCL_ALGO=TREE
# # export NCCL_MIN_NCHANNELS=1
# # export NCCL_MAX_NCHANNELS=1

# # export NCCL_NTHREADS=64

# # $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# # export NCCL_NTHREADS=128

# # $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# # export NCCL_NTHREADS=256

# # $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# # export NCCL_NTHREADS=512

# # $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60


# # echo "NCCL TEST with NCCL_MSCCL 1 channel 4gpus"

# # export NCCL_ALGO=TREE
# # export NCCL_MIN_NCHANNELS=1
# # export NCCL_MAX_NCHANNELS=1

# # export NCCL_NTHREADS=64

# # $MPIEXEC_HOME/bin/mpiexec -n 4 --ppn 2 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# # export NCCL_NTHREADS=128

# # $MPIEXEC_HOME/bin/mpiexec -n 4 --ppn 2 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# # export NCCL_NTHREADS=256

# # $MPIEXEC_HOME/bin/mpiexec -n 4 --ppn 2 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# # export NCCL_NTHREADS=512

# # $MPIEXEC_HOME/bin/mpiexec -n 4 --ppn 2 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60


# # unset NCCL_MAX_NCHANNELS
# # unset NCCL_MIN_NCHANNELS


# echo "NCCL TEST with MSCCL 1 channel 2gpus 128 steps"

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel2_reverse_chunk2.xml
# export NCCL_ALGO=MSCCL,TREE,RING

# export NCCL_NTHREADS=64

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 256MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=128

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 256MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=256

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 256MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=512

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 256MB -f 2 -g 1 -n 60

# echo "NCCL TEST with MSCCL 1 channel 2gpus 4 steps"

# export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
# MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
# export MSCCL_SRC_LOCATION

# export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk4.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple


# export NCCL_NTHREADS=64

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=128

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=256

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=512

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60


##########################################   4 CHANNELS   ############################################

# ################################### NCCL TEST Original ##########################################################
# 
# echo "NCCL TEST with Original NCCL"
# 
# export NCCL_TEST_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests"
# 
# NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl"
# export NCCL_SRC_LOCATION
# 
# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# 
# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple
# export NCCL_NTHREADS=192
# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4
# 
# 
# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

# ################################### NCCL TEST Profile ##########################################################

# echo "NCCL TEST with NCCL Profile"

# export NCCL_TEST_PROFILE_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"

# NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
# export NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_PROFILE_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_PROFILE_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

################################### MSCCL TEST #########################################################

# echo "NCCL TEST with MSCCL"

# export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
# MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
# export MSCCL_SRC_LOCATION

# export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_2nodes_channel4_reverse_chunk8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple
# export NCCL_NTHREADS=128


# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1








