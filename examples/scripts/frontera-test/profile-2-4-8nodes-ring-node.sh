#!/bin/bash




module load gcc/9.1.0
module load impi/18.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64

##################################### NCCL #####################################
echo "##################################### NCCL #####################################"
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"
export NCCL_SRC_LOCATION

NCCLTESTS_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

export NCCL_NTHREADS=64
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1


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

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1



##################################### NCCL #####################################
echo "##################################### NCCL #####################################"
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"
export NCCL_SRC_LOCATION

NCCLTESTS_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

export NCCL_DEBUG=TRACE
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

echo "##################################### NCCL Ring 2 nodes #####################################" >> output.log 2>&1

export NCCL_NTHREADS=64
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

echo "##################################### NCCL Ring 4 nodes #####################################" >> output.log 2>&1

export NCCL_NTHREADS=64
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

echo "##################################### NCCL Ring 8 nodes #####################################" >> output.log 2>&1

export NCCL_NTHREADS=64
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1


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

echo "##################################### MSCCL Ring 2 nodes #####################################" >> output.log 2>&1

export NCCL_NTHREADS=64

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_2nodes_4gpus_8chunks_8channels_frontera.xml
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_2nodes_4gpus_8chunks_8channels_frontera.xml
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_2nodes_4gpus_8chunks_8channels_frontera.xml
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_2nodes_4gpus_8chunks_8channels_frontera.xml
$MPI_HOME/bin/mpirun -np 8 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1


echo "##################################### MSCCL Ring 4 nodes #####################################" >> output.log 2>&1

export NCCL_NTHREADS=64

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_4nodes_4gpus_16chunks_16channels_frontera.xml
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_4nodes_4gpus_16chunks_16channels_frontera.xml
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_4nodes_4gpus_16chunks_16channels_frontera.xml
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_4nodes_4gpus_16chunks_16channels_frontera.xml
$MPI_HOME/bin/mpirun -np 16 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_4nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1




echo "##################################### MSCCL Ring 8 nodes #####################################" >> output.log 2>&1

export NCCL_NTHREADS=64

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=128

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=256

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1

export NCCL_NTHREADS=512

export MSCCL_XML_FILES=/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml
$MPI_HOME/bin/mpirun -np 32 --hostfile /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_8nodes -ppn 4 $NCCLTESTS_MSCCL_SRC_LOCATION/build/all_reduce_perf -b 16M -e 16M -f 2 -g 1 -n 60 >> output.log 2>&1
