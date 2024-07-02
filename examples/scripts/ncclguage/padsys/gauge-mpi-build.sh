#!/bin/bash

# Set environment variables

# module load mpich

# export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0"

source /home/liuyao/sbatch_sh/.mpich_ucx

export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export NCCL_GAUGE_HOME="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage"

for ((i = 1; i <= 32; i *= 32)); do
    for mode in pping ; do
        # Use proper variable expansion and quoting in the command
        mpicc -I"${MPI_HOME}/include" \
            -L"${MPI_HOME}/lib" -lmpi \
            -D N_ITERS=${i} \
            -D GAUGE_D=0 \
            "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi.cc" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi_${i}.exe"

        # Verification of the output
        if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi_${i}.exe" ]; then
            echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi_${i}.exe"
        else
            echo "Compilation failed."
        fi
    done
done