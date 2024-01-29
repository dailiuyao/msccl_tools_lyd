#!/bin/sh


module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3

export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

#---- USER CONFIG PARAMS----
#MPI_RUN=/act/mpich/gcc/bin/mpirun
MPI_RUN=$MPI_HOME/bin/mpirun
#-------------------------

export WORKDIR=`pwd`
# scontrol show hostnames > ~/myhostnames # store the hostnames
temp=($(wc /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes))
HOST_NUM=${temp[0]} # Count the number of hosts

# Clean up
# for host in `cat $HOME/myhostnames`; do
# 	ssh $host killall -9 python
# 	ssh $host killall -9 mpirun
# done

# module load cuda/11.4 mvapich2/2.3.3-gcc-8.4.1 
# $MPI_RUN -f ~/myhostnames -np $HOST_NUM ./yolov5_dist_train.sh
$MPI_RUN -hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -np $HOST_NUM -ppn 1 ./yolov5_dist_train.sh
# $MPI_RUN -hostfile ~/myhostnames -np $HOST_NUM -npernode 1 ./test.sh

