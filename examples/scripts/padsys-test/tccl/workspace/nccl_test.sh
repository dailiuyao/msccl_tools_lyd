#!/bin/bash
#SBATCH -N 2 # request 1 nodes
#SBATCH --nodelist=node01,node02
#SBATCH --output=nccl-test.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "nccl_test"    # this is your job’s name
#SBATCH --gpus-per-node=1


spack load gcc@10.4.0 

source /home/liuyao/sbatch_sh/.nvccrc

spack load openmpi@5.0.3

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/openmpi-5.0.3-ltv5k5ckeuhvwzb2dnjqsb22eggfhmwh"

##################################### NCCL #####################################
echo "##################################### NCCL #####################################"
NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"
export NCCL_SRC_LOCATION

NCCLTESTS_SRC_LOCATION="/home/liuyao/scratch/deps/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
# export NCCL_NTHREADS=192

# export NCCL_MIN_NCHANNELS=1
# export NCCL_MAX_NCHANNELS=1

# dool --time --mem --cpu --net -N ib0,ens786f1,lo,total 1 > /home/liuyao/sbatch_sh/nccl-test/output/CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/liuyao/sbatch_sh/nccl-test/output/GPU.csv &
#         sh /home/liuyao/sbatch_sh/nccl-test/rtop.sh -d ib0 > /home/liuyao/sbatch_sh/nccl-test/output/RTOP.csv  &

# UCX_NET_DEVICES=mlx5_0:1  $MPI_HOME/bin/mpirun -np 2 -npernode 1 -host node01,node02 $NCCLTESTS_SRC_LOCATION/build/sendrecv_perf -b 128K -e 2MB -f 2 -g 1

# kill %1
# kill %2
# kill %3

$MPI_HOME/bin/mpirun -np 1 $NCCLTESTS_SRC_LOCATION/build/sendrecv_perf -b 128K -e 2MB -f 2 -g 1
