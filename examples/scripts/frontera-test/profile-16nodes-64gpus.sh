#!/bin/bash

#SBATCH -J ccl-run-16nodes-64gpus           # Job name
#SBATCH -o ./log/ccl-run-16nodes-64gpus.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-16nodes-64gpus.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 16               # Total # of nodes (must be 1 for serial)
#SBATCH -n 64               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:19:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-16nodes-64gpus       # Project/Allocation name (req'd if you have more than 1)
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
export NCCL_PROTO=Simple

export NCCL_ALGO=TREE
export NCCL_NTHREADS=64

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_NTHREADS=128

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_NTHREADS=256

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_NTHREADS=512

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_ALGO=RING
export NCCL_NTHREADS=64

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_NTHREADS=128

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_NTHREADS=256

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

export NCCL_NTHREADS=512

$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 1K -e 512MB -f 2 -g 1 -n 100

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

# export NCCL_MIN_NCHANNELS=1
# export NCCL_MAX_NCHANNELS=1

# # export NCCL_NTHREADS=256

# # export NCCL_COMM_BLOCKING=1

# # $MPI_HOME/bin/mpirun -np 32 -ppn 4 $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 1M -e 1M -w 0 -f 2 -g 1 -n 1

# $MPI_HOME/bin/mpirun -np 32 -ppn 4 $NCCLTESTS_NCCL_PROFILE_SRC_LOCATION/build/all_reduce_perf -b 128M -e 128M -f 2 -g 1
