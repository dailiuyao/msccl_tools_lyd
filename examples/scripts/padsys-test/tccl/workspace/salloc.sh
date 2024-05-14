#!/bin/bash
#SBATCH -N 3 # request 2 nodes
#SBATCH --nodelist=node04,node05,node06
#SBATCH --output=my_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "tccl"    # this is your job’s name
#SBATCH --gpus-per-node=2
#SBATCH --exclusive
#SBATCH --time=23:59:59


source /home/liuyao/sbatch_sh/.mpich_ucx

source /home/liuyao/sbatch_sh/.nvccrc



mpirun --bind-to none \
-hosts node04:11,node05:11,node06:11 \
/home/liuyao/scratch/deps/tccl/tools/build/pathfinder -o /home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/padsys-test/tccl/workspace