#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A SR_APPFL
#PBS -k doe
#PBS -N ccl-build
#PBS -o log/nccl-build.out
#PBS -e log/nccl-build.error

# mkdir -p ./log

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/ldai8/ccl/nccl"
export NCCL_SRC_LOCATION
export NCCL_COMMIT="nccl_origin"

# Set location to store NCCL_TEST source/repository
NCCLTESTS_SRC_LOCATION="/home/ldai8/ccl/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

### NCCL-Section ###

export PROFAPI=1
# Download NCCL
if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL repository..."
	git clone https://github.com/NVIDIA/nccl.git "${NCCL_SRC_LOCATION}"
elif [ -d "${NCCL_SRC_LOCATION}" ]; then 
	echo "[INFO] NCCL repository already exists."
fi
echo ""

# Enter NCCL dir
pushd "${NCCL_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_COMMIT}"

# Build NCCL
echo "[INFO] Building NCCL..."
make clean
make -j src.build
echo ""

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
export NCCL_HOME
echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
export LD_LIBRARY_PATH
PATH="${PATH}:${NCCL_HOME}/include"
export PATH
echo ""

# Exit NCCL dir
popd || exit
echo ""


### NCCL Tests Section ###

# Download NCCL Tests
if [ ! -d "${NCCLTESTS_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL Tests repository..."
	git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_SRC_LOCATION}"
elif [ -d "${NCCLTESTS_SRC_LOCATION}" ]; then
	echo "[INFO] NCCL Tests repository already exists."
fi
echo ""

# Enter NCCL Tests dir
pushd "${NCCLTESTS_SRC_LOCATION}" || exit
echo ""
make clean

# Build NCCL Tests
echo "[INFO] Building NCCL tests (nccl-tests)"
make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${NCCL_SRC_LOCATION}/build"  


# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${MSCCL_HOME}"

# Exit NCCL Tests dir
popd || exit
echo ""

### NCCL_step2-Section ###


# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/ldai8/ccl/nccl_step2"
export NCCL_SRC_LOCATION
export NCCL_COMMIT="nccl_origin"

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

### NCCL-Section ###

export PROFAPI=1
# Download NCCL
if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL repository..."
	git clone https://github.com/NVIDIA/nccl.git "${NCCL_SRC_LOCATION}"
elif [ -d "${NCCL_SRC_LOCATION}" ]; then 
	echo "[INFO] NCCL repository already exists."
fi
echo ""

# Enter NCCL dir
pushd "${NCCL_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_COMMIT}"

# Build NCCL
echo "[INFO] Building NCCL..."
make clean
make -j src.build
echo ""

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
export NCCL_HOME
echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
export LD_LIBRARY_PATH
PATH="${PATH}:${NCCL_HOME}/include"
export PATH
echo ""

# Exit NCCL dir
popd || exit
echo ""


### NCCL_step8-Section ###


# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/ldai8/ccl/nccl_step8"
export NCCL_SRC_LOCATION
export NCCL_COMMIT="nccl_origin"

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

### NCCL-Section ###

export PROFAPI=1
# Download NCCL
if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL repository..."
	git clone https://github.com/NVIDIA/nccl.git "${NCCL_SRC_LOCATION}"
elif [ -d "${NCCL_SRC_LOCATION}" ]; then 
	echo "[INFO] NCCL repository already exists."
fi
echo ""

# Enter NCCL dir
pushd "${NCCL_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_COMMIT}"

# Build NCCL
echo "[INFO] Building NCCL..."
make clean
make -j src.build
echo ""

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
export NCCL_HOME
echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
export LD_LIBRARY_PATH
PATH="${PATH}:${NCCL_HOME}/include"
export PATH
echo ""

# Exit NCCL dir
popd || exit
echo ""