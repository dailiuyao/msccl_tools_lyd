#!/bin/bash
#SBATCH -N 2 # request 2 nodes
#SBATCH --nodelist=node01,node02
#SBATCH --output=my_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "ncclgauge"    # this is your job’s name
#SBATCH --gpus-per-node=1
##SBATCH --exclusive
#SBATCH --time=23:59:59

source /home/liuyao/sbatch_sh/.mpich_ucx

export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# module load mpich

# export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0"

# $MPI_HOME/bin/mpirun -np 2 -hosts node04:1,node05:1 ./gauge-run.sh 

export CUDA_HOME=/home/liuyao/software/cuda-11.7
export PATH=/home/liuyao/software/cuda-11.7/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/liuyao/software/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDACXX=/home/liuyao/software/cuda-11.7/bin/nvcc
export CUDNN_LIBRARY=/home/liuyao/software/cuda-11.7/lib64
export CUDNN_INCLUDE_DIR=/home/liuyao/software/cuda-11.7/include

source /home/liuyao/sbatch_sh/.nvccrc

export NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export NCCL_NTHREADS=256

# export NCCL_DEBUG=INFO

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

cd /home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage/padsys

export NCCL_GAUGE_HOME="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage"
export GAUGE_OUT_DIRE="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage/padsys"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

# benchmarks for G g o

for ((itr = 0; itr < 2; itr += 1)); do
    for sync_mode in sync group; do
        for ((n = 1; n <= 8; n *= 8)); do
            for ((nch = 1; nch <= 1; nch *= 2)); do
                for mode in pping; do
                    for ((d = 0; d <= 100*1000; d += 20*1000)); do
                        for ((msize=32; msize<=256*1024; msize*=2)); do
                            export GAUGE_MESSAGE_SIZE=${msize}
                            export GAUGE_ITERATION=${itr} 
                            export GAUGE_NCHANNELS=${nch}
                            export GAUGE_MODE=${mode}
                            export NCCL_MIN_NCHANNELS=${nch}
                            export NCCL_MAX_NCHANNELS=${nch}
                            export GAUGE_D_HOST=${d}
                            $MPI_HOME/bin/mpirun -n 2 --ppn 1 -hosts node01,node02 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}.exe
                            # ibrun -n 2 --ntasks-per-node=2 \
                            # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                            # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                        done
                    done
                done
            done 
        done
    done
done

# # benchmarks for L

for ((itr = 0; itr < 2; itr += 1)); do
    for sync_mode in sync group; do
        for ((n = 1; n <= 1; n *= 8)); do
            for ((nch = 1; nch <= 1; nch *= 2)); do
                for mode in pping; do
                    for ((d = 0; d <= 0; d += 100*1000)); do
                        for ((msize=1; msize<=1; msize*=2)); do
                            export GAUGE_MESSAGE_SIZE=${msize}
                            export GAUGE_ITERATION=${itr} 
                            export GAUGE_NCHANNELS=${nch}
                            export GAUGE_MODE=${mode}
                            export NCCL_MIN_NCHANNELS=${nch}
                            export NCCL_MAX_NCHANNELS=${nch}
                            export GAUGE_D_HOST=${d}
                            $MPI_HOME/bin/mpirun -n 2 --ppn 1 -hosts node01,node02 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}.exe
                        done
                    done
                done
            done 
        done
    done
done