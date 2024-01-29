#!/bin/bash

module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3

export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

##################################### NCCL PROFILE #####################################
echo "##################################### NCCL PROFILE #####################################"
NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"
export NCCL_PROFILE_SRC_LOCATION

NCCLTESTS_NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile"
export NCCLTESTS_NCCL_PROFILE_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_PROFILE_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple

export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4

num_gpus_per_node=2
total_num_gpus=4
hostfile="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes"

# Loop over the number of GPUs and create a profile for each
for ((i=0; i<total_num_gpus; i++)); do
    gpu_id=$((i % num_gpus_per_node))
    node_id=$((i / num_gpus_per_node))
    export CUDA_VISIBLE_DEVICES=$gpu_id
    nsys profile -o nsys_test_nccl_profile_gpu${gpu_id}_node${node_id} --stats=true $MPI_HOME/bin/mpirun -np 1 -ppn 1 -hostfile $hostfile -genv CUDA_VISIBLE_DEVICES $gpu_id $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 2 -e 1M -f 2 -g 1 -n 100 &>> output_gpu${gpu_id}_node${node_id}.log &
done

# Wait for all background processes to complete
wait
