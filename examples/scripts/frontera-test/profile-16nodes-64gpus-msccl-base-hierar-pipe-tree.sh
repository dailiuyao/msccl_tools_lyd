#!/bin/bash

#SBATCH -J ccl-run-16nodes-64gpus           # Job name
#SBATCH -o ./log/ccl-run-16nodes-64gpus-hie-pipe.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-16nodes-64gpus-hie-pipe.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 16               # Total # of nodes (must be 1 for serial)
#SBATCH -n 64               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-8nodes-32gpus       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu


module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

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
export NCCL_DEBUG=TRACE
export NCCL_PROTO=Simple



export NCCL_NTHREADS=64

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk4.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk8.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk16.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk32.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk64.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk128.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk256.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk512.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1


export NCCL_NTHREADS=128

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk4.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk8.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk16.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk32.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk64.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk128.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk256.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk512.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk4.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk8.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk16.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk32.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk64.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk128.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk256.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk512.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk4.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk8.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk16.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk32.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk64.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk128.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk256.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk512.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk1024.xml
$MPI_HOME/bin/mpirun -np 64 -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 1K -e 256M -f 2 -g 1
