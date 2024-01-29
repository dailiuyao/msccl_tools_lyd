#!/bin/bash

#SBATCH -J ccl-run-4nodes-8gpus           # Job name
#SBATCH -o ./log/ccl-run-4nodes-8gpus.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-4nodes-8gpus.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 4               # Total # of nodes (must be 1 for serial)
#SBATCH -n 8               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --exclude=c199-121,c199-051,c197-022,c199-012
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-4nodes-16gpus       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

set -e

module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

export WORK_DIR=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test

# Create the hostfile with allocated node names (shortened)
srun --nodes=$SLURM_NNODES hostname | cut -d'.' -f1 > $WORK_DIR/myhostfile_slurm


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

num_gpus_per_node=2
total_num_gpus=8

hostfile="$WORK_DIR/myhostfile_slurm"

# Loop over the number of GPUs and create a profile for each
$MPI_HOME/bin/mpirun -np 8 -ppn 2 --hostfile $hostfile -genvall ./gpu_profile_wrapper.sh
