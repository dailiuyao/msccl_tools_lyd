#!/bin/bash

module load gcc
module load cudatoolkit-standalone/11.4.4
export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/

# Set location to store MSCCL_TEST source/repository
MSCCL_TEST_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
export MSCCL_SRC_LOCATION
export MSCCL_TEST_COMMIT="v0.7.4"

export MSCCL_TEST_HOME=${MSCCL_TEST_SRC_LOCATION}/build

# Set location to store NCCL-Tests-MSCCL-LYD source/repository
NCCLTESTS_MSCCL_TEST_SRC_LOCATION="/home/yuke/ncclPG/nccl-tests-msccl-test"
export NCCLTESTS_MSCCL_TEST_SRC_LOCATION

### MSCCL_TEST Core Section ###

# Download MSCCL_TEST
if [ ! -d "${MSCCL_TEST_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading MSCCL_TEST repository..."
	git clone https://github.com/dailiuyao/msccl-lyd.git "${MSCCL_TEST_SRC_LOCATION}"
elif [ -d "${MSCCL_TEST_SRC_LOCATION}" ]; then
	echo "[INFO] MSCCL_TEST repository already downloaded; will not re-download."
fi
echo ""

# Enter MSCCL_TEST directory
pushd "${MSCCL_TEST_SRC_LOCATION}" || exit

# Fetch latest changes
git fetch --all

# Checkout the correct commit
git checkout "${MSCCL_TEST_COMMIT}"

# Build MSCCL_TEST
echo "[INFO] Building MSCCL_TEST..."
make -j src.build
echo ""

# Create install package
# [TODO]

# Exist MSCCL_TEST directory
popd || exit
echo ""


### NCCL-Tests-MSCCL-LYD Section ###

# Download NCCL-Tests-MSCCL-LYD
if [ ! -d "${NCCLTESTS_MSCCL_TEST_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL Tests with MSCCL support repository..."
	git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_MSCCL_TEST_SRC_LOCATION}"
elif [ -d "${NCCLTESTS_MSCCL_TEST_SRC_LOCATION}" ]; then
	echo "[INFO] NCCL Tests with MSCCL support repository already exists."
fi
echo ""

# Enter NCCL-Tests-MSCCL-TEST dir
pushd "${NCCLTESTS_MSCCL_TEST_SRC_LOCATION}" || exit
echo ""

# Build NCCL Tests with MSCCL support
echo "[INFO] Building NCCL tests (nccl-tests) with MSCCL support..."
make clean
make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${MSCCL_TEST_HOME}" -j  # Note: Use MSCCL's "version" of NCCL to build nccl-tests

# Exit NCCL Tests dir
popd || exit
echo ""