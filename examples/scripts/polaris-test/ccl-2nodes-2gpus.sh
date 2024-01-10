#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -q demand
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-2nodes
#PBS -o log/ccl-2nodes.out
#PBS -e log/ccl-2nodes.error

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
export NCCL_NTHREADS=384
# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4


$MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

# ################################### NCCL TEST Profile ##########################################################

# echo "NCCL TEST with NCCL Profile"

# export NCCL_TEST_PROFILE_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"

# NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
# export NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_PROFILE_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

# $MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_PROFILE_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

################################### MSCCL TEST #########################################################

echo "NCCL TEST with MSCCL"

export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
export MSCCL_SRC_LOCATION

export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_p_gpu01_2nodes_channel2_chunk4.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple
export NCCL_NTHREADS=256

# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4

$MPIEXEC_HOME/bin/mpiexec -n 2 --ppn 1 --cpu-bind core ${NCCL_TEST_MSCCL_HOME}/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1






















##################################### NCCL #####################################
echo "##################################### NCCL #####################################"
NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl"
export NCCL_SRC_LOCATION

NCCLTESTS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_NTHREADS=384

# export NCCL_MIN_NCHANNELS=4
# export NCCL_MAX_NCHANNELS=4

$MPIEXEC_HOME/bin/mpiexec -np 2 -ppn 1 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

# ##################################### NCCL PROFILE #####################################
# echo "##################################### NCCL PROFILE #####################################"
# NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"
# export NCCL_PROFILE_SRC_LOCATION

# NCCLTESTS_NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile"
# export NCCLTESTS_NCCL_PROFILE_SRC_LOCATION

# export LD_LIBRARY_PATH="${NCCL_PROFILE_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple

# $MPI_HOME/bin/mpirun -np 2 -ppn 1 $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

##################################### MSCCL #####################################
echo "##################################### MSCCL #####################################"
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
export MSCCL_SRC_LOCATION

NCCLTESTS_MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/allreduce_binary_tree_p_gpu01_2nodes_channel4_chunk8.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_DEBUG=TRACE
export NCCL_PROTO=Simple
export NCCL_NTHREADS=256

$MPI_HOME/bin/mpirun -np 2 -ppn 1 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1