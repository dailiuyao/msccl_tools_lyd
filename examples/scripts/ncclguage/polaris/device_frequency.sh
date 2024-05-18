#!/bin/bash -l

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda


cat /proc/cpuinfo | grep "cpu MHz" >> device_frequency.txt 2>&1


nvidia-smi --query-gpu=clocks.gr --format=csv >> device_frequency.txt 2>&1

