#!/bin/bash -l
#PBS -l select=32:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q prod
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-32nodes
#PBS -o log/ccl-32nodes.out
#PBS -e log/ccl-32nodes.error

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
# # kill $RTOP2_P

# ################################### NCCL TEST Original ##########################################################

# echo "NCCL TEST with Original NCCL"

# export NCCL_TEST_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests"

# NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl"
# export NCCL_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple
# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4

# export NCCL_NTHREADS=64

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=128
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=256
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60

# export NCCL_NTHREADS=512
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 60


# ################################### NCCL TEST Profile ##########################################################

# echo "NCCL TEST with NCCL Profile"

# export NCCL_TEST_PROFILE_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"

# NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
# export NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_PROFILE_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# $MPIEXEC_HOME/bin/mpiexec -n 16 --ppn 4 --cpu-bind core ${NCCL_TEST_PROFILE_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

# ################################### MSCCL TEST #########################################################
# 
# echo "NCCL TEST with MSCCL"
# 
# export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
# MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
# export MSCCL_SRC_LOCATION
# 
# export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"
# 
# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel2_chunk256.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple
# export NCCL_NTHREADS=256
# 
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100


##########################################   4 CHANNELS   ############################################


# ################################### NCCL TEST Original ##########################################################

# echo "NCCL TEST with Original NCCL"

# export NCCL_TEST_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests"

# NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl"
# export NCCL_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple
# export NCCL_NTHREADS=64
# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4


# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 200

# export NCCL_NTHREADS=128
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

# export NCCL_NTHREADS=256
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

# export NCCL_NTHREADS=512
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100


# ################################### NCCL TEST Profile ##########################################################

# echo "NCCL TEST with NCCL Profile"

# export NCCL_TEST_PROFILE_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"

# NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
# export NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_PROFILE_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# $MPIEXEC_HOME/bin/mpiexec -n 16 --ppn 4 --cpu-bind core ${NCCL_TEST_PROFILE_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

# ################################### MSCCL TEST #########################################################

echo "NCCL TEST with MSCCL"

export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
export MSCCL_SRC_LOCATION

export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=TREE
export NCCL_PROTO=Simple


# echo "NCCL TEST with NCCL(MSCCL) Ring"
# 
# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4
# 
# export NCCL_ALGO=RING
# 
# export NCCL_NTHREADS=64
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=128
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=256
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=512
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# echo "NCCL TEST with MSCCL based tree"
# 
# unset NCCL_MAX_NCHANNELS
# unset NCCL_MIN_NCHANNELS
# 
# export NCCL_ALGO=MSCCL,TREE,RING
# 
# export NCCL_NTHREADS=64
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/allredcue_basic_binary_tree_128gpus.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=128
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/allredcue_basic_binary_tree_128gpus.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=256
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/allredcue_basic_binary_tree_128gpus.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=512
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/allredcue_basic_binary_tree_128gpus.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# echo "NCCL TEST with MSCCL hierarchical tree 2 channels"

# export NCCL_ALGO=MSCCL,TREE,RING

# export NCCL_NTHREADS=64
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk2.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=128
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk2.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1
# 
# export NCCL_NTHREADS=256
# 
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk2.xml
# 
# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=512

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk2.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# echo "NCCL TEST with MSCCL hierarchical tree 4 channels 256 chunks"

# export NCCL_ALGO=MSCCL,TREE,RING

# export NCCL_NTHREADS=64

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk1024.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=128

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk1024.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=256

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk1024.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=512

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk1024.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# echo "NCCL TEST with MSCCL hierarchical tree"

# export NCCL_ALGO=MSCCL,TREE,RING

# export NCCL_NTHREADS=64

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_chunk4.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=128

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_chunk4.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=256

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_chunk4.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

# export NCCL_NTHREADS=512

# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_channel4_chunk4.xml

# $MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1


echo "NCCL TEST with MSCCL hierarchical tree 2,4,8 channel 2 chunks"

export NCCL_ALGO=MSCCL,TREE,RING

export NCCL_NTHREADS=64

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel2_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel2_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel2_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel2_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_ALGO=MSCCL,TREE,RING

export NCCL_NTHREADS=64

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel4_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel4_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel4_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel4_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1


export NCCL_ALGO=MSCCL,TREE,RING

export NCCL_NTHREADS=64

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel8_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel8_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel8_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_32nodes_4gpus_channel8_chunk2.xml

$MPIEXEC_HOME/bin/mpiexec -n 128 --ppn 4 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -n 60 -g 1