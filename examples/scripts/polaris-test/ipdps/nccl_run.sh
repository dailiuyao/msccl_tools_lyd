#!/bin/bash -l
#PBS -l select=64:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:29:59
#PBS -q prod
#PBS -l filesystems=home
#PBS -A SR_APPFL 
#PBS -k doe
#PBS -N ccl-64nodes-chunk
#PBS -o log/ccl-64nodes-chunk.out
#PBS -e log/ccl-64nodes-chunk.error

# Set environment variables

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

# Install and load libxml2 using Spack
spack load libxml2

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

export MPIEXEC_HOME=/opt/cray/pals/1.3.4
export NCCL_NET_PLUGIN_HOME="/home/ldai8/ccl/aws-ofi-nccl-1.7.4-aws/build"     
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_IB_HCA=cxi0,cxi1
export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:$LD_LIBRARY_PATH

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"


# NCCL source location
NCCL_SRC_LOCATION="/home/ldai8/ccl/nccl"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

NCCL_TEST_HOME="/home/ldai8/ccl/nccl-tests"
export NCCL_TEST_HOME


export NCCL_DEBUG="TRACE"
export NCCL_PROTO="Simple"
export NCCL_ALGO=TREE

$MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60


export NCCL_ALGO=Ring

$MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60


echo "NCCL TEST WITH ALLREDUCE_SLICESTEPS = NCCL_STEPS/8"

# NCCL source location
NCCL_SRC_LOCATION="/home/ldai8/ccl/nccl_step8"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

NCCL_TEST_HOME="/home/ldai8/ccl/nccl-tests"
export NCCL_TEST_HOME


export NCCL_DEBUG="TRACE"
export NCCL_PROTO="Simple"
export NCCL_ALGO=TREE

$MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60


export NCCL_ALGO=Ring

$MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60


echo "NCCL TEST WITH ALLREDUCE_SLICESTEPS = NCCL_STEPS/2"

# NCCL source location
NCCL_SRC_LOCATION="/home/ldai8/ccl/nccl_step2"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

NCCL_TEST_HOME="/home/ldai8/ccl/nccl-tests"
export NCCL_TEST_HOME


export NCCL_DEBUG="TRACE"
export NCCL_PROTO="Simple"
export NCCL_ALGO=TREE

$MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60


export NCCL_ALGO=Ring

$MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60
