#!/bin/bash
#SBATCH -J ncclgauge           # Job name
#SBATCH -o ./log/intra/test.o%j       # Name of stdout output file
#SBATCH -e ./log/intra/test.e%j       # Name of stderr error file
#SBATCH -p rtx-dev           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 2               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:19:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ncclgauge       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

# Set environment variables
source /home/liuyao/sbatch_sh/.nvccrc

export CUDA_HOME=/home/liuyao/software/cuda-11.6

MPI_HOME="/home/liuyao/software/mpich_4_1_1_pgcc"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# NCCL source location
NCCL_SRC_LOCATION="/home/liuyao/NCCL/deps-nccl/nccl"

$MPI_HOME/bin/mpirun -np 4 -ppn 2 -hosts node03,node04 /home/liuyao/sbatch_sh/nccl-example/OneDevicePerThread.exe