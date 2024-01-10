#!/bin/bash

#SBATCH -J ccl-run-2nodes-2gpus           # Job name
#SBATCH -o ./log/ccl-run-2nodes-2gpus.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-2nodes-2gpus.e%j       # Name of stderr error file
#SBATCH -p rtx-dev           # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 2               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:19:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-2nodes-2gpus       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

set -e

module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

##################################### NCCL #####################################
echo "##################################### NCCL #####################################"
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"
export NCCL_SRC_LOCATION

NCCLTESTS_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_NTHREADS=192

export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4

$MPI_HOME/bin/mpirun -np 2 -ppn 1 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1

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
MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/msccl-lyd"
export MSCCL_SRC_LOCATION

NCCLTESTS_MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION

export LD_LIBRARY_PATH="${MSCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/allreduce_binary_tree_p_gpu01_2nodes_channel4_chunk8.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_DEBUG=TRACE
export NCCL_PROTO=Simple
export NCCL_NTHREADS=128

$MPI_HOME/bin/mpirun -np 2 -ppn 1 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1