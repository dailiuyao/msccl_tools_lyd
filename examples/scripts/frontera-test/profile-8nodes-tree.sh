#!/bin/bash

#SBATCH -J ccl-run-8nodes-32gpus           # Job name
#SBATCH -o ./log/ccl-run-8nodes-32gpus.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-8nodes-32gpus.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 8               # Total # of nodes (must be 1 for serial)
#SBATCH -n 32               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:19:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-8nodes-32gpus       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

set -e





module load gcc/9.1.0
module load impi/19.0.5
module load cuda/11.3



export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

# ##################################### NCCL #####################################
# echo "##################################### NCCL #####################################"
# NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"
# export NCCL_SRC_LOCATION

# NCCLTESTS_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests"
# export NCCLTESTS_SRC_LOCATION

# export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple

# echo "##################################### NCCL Tree 8 nodes #####################################" >> output.log 2>&1

# export NCCL_NTHREADS=64
# $MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

# export NCCL_NTHREADS=128
# $MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

# export NCCL_NTHREADS=256
# $MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

# export NCCL_NTHREADS=512
# $MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1


##################################### MSCCL #####################################
echo "##################################### MSCCL #####################################"
MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/msccl-lyd"
export MSCCL_SRC_LOCATION

NCCLTESTS_MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION

export LD_LIBRARY_PATH="${MSCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_DEBUG=INFO
export NCCL_PROTO=Simple

export GENMSCCLXML=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/gen_msccl_xml_frontera.sh



echo "##################################### MSCCL Tree 8 nodes #####################################" 

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_1ch_32chunk.xml

ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_2ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_1ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_2ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling/allreduce_recursive_doubling_1ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling/allreduce_recursive_doubling_2ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling_halving/allreduce_recursive_doubling_halving_1ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling_halving/allreduce_recursive_doubling_halving_2ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_ring_8ch_1chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/trinomial_tree/allreduce_trinomial_tree_1ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1M -e 256M -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/trinomial_tree/allreduce_trinomial_tree_2ch_32chunk.xml

# ibrun -n 32 --ntasks-per-node=4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 768K -e 256M -f 2 -g 1 -n 60


