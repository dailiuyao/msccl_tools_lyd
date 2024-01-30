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
# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_4nodes_16gpus_channel2_reverse_chunk32.xml
export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/allredcue_basic_binary_tree_16gpus.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

num_gpus_per_node=4
total_num_gpus=16

hostfile="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes"

# $MPI_HOME/bin/mpirun -np 16 --hostfile $hostfile -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 &>> output.log

# Run the application with MPI and profile each process
$MPI_HOME/bin/mpirun -np 16 --hostfile $hostfile -ppn 4 \
bash -c 'nsys profile --force-overwrite true -o msccl-output/nsys_test_msccl_profile_rank${OMPI_COMM_WORLD_RANK:-$PMI_RANK} --trace=cuda,nvtx,osrt --stats=true $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 2' \
&>> output.log


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


# hostfile="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes"

# $MPI_HOME/bin/mpirun -np 16 --hostfile $hostfile -ppn 4 $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 &>> output.log 

# # Run the application with MPI and profile each process
# $MPI_HOME/bin/mpirun -np 16 --hostfile $hostfile -ppn 4 \
# bash -c 'nsys profile --force-overwrite true -o nccl-output/nsys_test_nccl_profile_rank${OMPI_COMM_WORLD_RANK:-$PMI_RANK} --stats=true $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60' \
# &>> output.log
