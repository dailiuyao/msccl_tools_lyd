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

# Compilation command. Ensure to link against the MPI and NCCL libraries correctly.
nvcc $NVCC_GENCODE -ccbin g++ -I${NCCL_SRC_LOCATION}/build/include -I${MPI_HOME}/include -L${NCCL_SRC_LOCATION}/build/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -lnccl -lcudart -lmpi $1 -o ${1%.cu}.exe

# Verification of the output
if [ -f ${1%.cu}.exe ]; then
    echo "Compilation successful. Output file: ${1%.cu}.exe"
else
    echo "Compilation failed."
fi