#!/bin/bash

#SBATCH -J ccl-run-4nodes-16gpus           # Job name
#SBATCH -o ../log/paper0/ccl-run-4nodes-16gpus.o%j       # Name of stdout output file
#SBATCH -e ../log/paper0/ccl-run-4nodes-16gpus.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 4               # Total # of nodes (must be 1 for serial)
#SBATCH -n 16               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --exclude=c197-072,c196-102
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-4nodes-16gpus       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

set -e

module load gcc/9.1.0
module load impi/19.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64

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

export GENMSCCLXML=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/gen_msccl_xml_frontera.sh

export MSCCL_TOOLS_XML="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd"

nchunks_values=(1 4 16 64 256)
nchannel_values=(1 2 4)
trees_values=(2)
# nodes_values=(4 8 16 32 64)
nodes_values=4

export ngpus=4

for nnodes in "${nodes_values[@]}"; do
    for nchannel in "${nchannel_values[@]}"; do
        for nchunks in "${nchunks_values[@]}"; do
            for trees in "${trees_values[@]}"; do
                echo "Running MSCCL tree test with ${nnodes} nodes, ${nchannel} channels, ${nchunks} chunks, ${trees} trees"
                export MSCCL_XML_FILES=${MSCCL_TOOLS_XML}/binary_tree/allreduce_binary_tree_${nchannel}ch_${trees}tree_${nchunks}chunk_${nnodes}node_$((nnodes*ngpus))gpu.xml
                ibrun -n $((nnodes*ngpus)) --ntasks-per-node=$ngpus $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60 \
                > /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/sc24/log/paper0/chunk_step_2/tree/all-reduce_sum_float_binary-tree_node${nnodes}_gpu$((nnodes*ngpus))_mcl${nchannel}_mck${nchunks}_i0.out
            done
        done
    done
done


