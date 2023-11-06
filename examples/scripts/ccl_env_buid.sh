#!/bin/bash 

module load gcc
module load cudatoolkit-standalone/11.4.4
export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/

# Set location to store MSCCL_TEST source/repository
MSCCL_TEST_SRC_LOCATION="/home/yuke/ncclPG/msccl_test_profile"
export MSCCL_SRC_LOCATION
export MSCCL_TEST_COMMIT="algorithm_test_profile"

# Set location to store NCCL_TEST_PROFILE source/repository
NCCLTESTS_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/nccl-tests-profile"
export NCCLTESTS_PROFILE_SRC_LOCATION

export MSCCL_TEST_HOME=${MSCCL_TEST_SRC_LOCATION}/build

# Set location to store NCCL-Tests-MSCCL-LYD source/repository
NCCLTESTS_MSCCL_TEST_SRC_LOCATION="/home/yuke/ncclPG/nccl-tests-msccl-test_profile"
export NCCLTESTS_MSCCL_TEST_SRC_LOCATION
export NCCL_TEST_PROFILE_COMMIT="nccl-test-profile"

# Set location to store NCCL-PROFILE source/repository
NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/nccl_profile"
export NCCL_PROFILE_SRC_LOCATION
export NCCL_PROFILE_COMMIT="profile_steps"

# ### NCCL-PROFILE Section ###

# export PROFAPI=1
# # Download NCCL
# if [ ! -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL repository..."
# 	git clone git@github.com:dailiuyao/NCCL_profile.git "${NCCL_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then 
# 	echo "[INFO] NCCL repository already exists."
# fi
# echo ""

# # Enter NCCL dir
# pushd "${NCCL_PROFILE_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_PROFILE_COMMIT}"

# # Build NCCL
# echo "[INFO] Building NCCL_PROFILE..."
# make clean
# make -j src.build
# echo ""

# # Set environment variables that other tasks will use
# echo "[INFO] Setting NCCL-related environment variables for other tasks..."
# NCCL_HOME="${NCCL_PROFILE_SRC_LOCATION}/build" 
# export NCCL_HOME
# echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

# echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
# LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
# export LD_LIBRARY_PATH
# PATH="${PATH}:${NCCL_HOME}/include"
# export PATH
# echo ""

# # Exit NCCL dir
# popd || exit
# echo ""


# ### NCCL-Tests-PROFILE Section ###

# # Download NCCL-Tests-PROFILE
# if [ ! -d "${NCCLTESTS_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL Tests with profile support repository..."
# 	git clone git@github.com:dailiuyao/nccl-tests.git "${NCCLTESTS_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] NCCL Tests with PROFILE support repository already exists."
# fi
# echo ""

# # Enter NCCL-Tests-MSCCL-TEST dir
# pushd "${NCCLTESTS_PROFILE_SRC_LOCATION}" || exit
# echo ""

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_TEST_PROFILE_COMMIT}"


# # Build NCCL Tests with MSCCL support
# echo "[INFO] Building NCCL tests (nccl-tests) with PROFILE support..."
# make clean
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${NCCL_HOME}" -j  

# # Exit NCCL Tests dir
# popd || exit
# echo ""


### MSCCL_TEST Core Section ###

rm -rf "${MSCCL_TEST_SRC_LOCATION}" 

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

rm -rf "${NCCLTESTS_MSCCL_TEST_SRC_LOCATION}" 

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
