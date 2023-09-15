### experiment set: 16nodes, one gpu per node
### tested algorithm: nccl ring, nccl tree, msccl ring, msccl double binary tree, msccl double binomial tree, msccl triple trinomial tree, msccl resursive doubling halving

### Set environment variables ###
# CUDA HOME
CUDA_HOME="/where/is/cuda"
export CUDA_HOME

# MPI HOME
MPI_HOME="/where/is/mpi"
export MPI_HOME
export PATH="${MPI_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"

# NCCL HOME
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="/where/is/nccl" 
export NCCL_HOME
echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
PATH="${NCCL_HOME}/include:${PATH}"
export PATH
echo ""

# NCCL TESTS HOME
NCCLTESTS_SRC_LOCATION="/where/is/nccltests"
export NCCLTESTS_SRC_LOCATION

#MSCCL HOME
MSCCL_SRC_LOCATION="/where/is/msccl"
export MSCCL_SRC_LOCATION

# MSCCL-Tools HOME
export MSCCLTOOLS_SRC_LOCATION="/where/is/msccl-tools"

# MSCCL XML HOME
export MSCCLTOOLS_XML_LOCATION="${MSCCLTOOLS_SRC_LOCATION}/examples/xml"

# NCCL TESTS WITH MSCCL HOME
NCCLTESTS_MSCCL_SRC_LOCATION="/where/is/nccl-tests-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION

######### TEST IN AWS #########
echo " "
echo "------ Library: MSCCL, Algorithm: Ring_TEST, Protocol: Simple, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_gpu16_ins1_test.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring_TEST, Protocol: Simple, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_gpu16_ins2_test.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring_TEST, Protocol: Simple, Instance: 8 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_gpu16_ins8_test.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring_TEST, Protocol: Simple, Instance: 16 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_gpu16_ins16_test.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1




### run nccl ###

echo " "
echo "------ Library: NCCL, Algorithm: Tree, Protocol: Simple ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: NCCL, Algorithm: Tree, Protocol: LL ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Tree
export NCCL_PROTO=LL
mpirun -np 16 -npernode 1 ${NCCLTESTS_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: NCCL, Algorithm: Tree, Protocol: LL128 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Tree
export NCCL_PROTO=LL128
mpirun -np 16 -npernode 1 ${NCCLTESTS_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: NCCL, Algorithm: Ring, Protocol: Simple ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
mpirun -np 16 -npernode 1 ${NCCLTESTS_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: NCCL, Algorithm: Ring, Protocol: LL ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Ring
export NCCL_PROTO=LL
mpirun -np 16 -npernode 1 ${NCCLTESTS_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: NCCL, Algorithm: Ring, Protocol: LL128 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=Ring
export NCCL_PROTO=LL128
mpirun -np 16 -npernode 1 ${NCCLTESTS_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1



### run msccl ###

### case 01: instance = 1

export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib:$LD_LIBRARY_PATH
export PATH="${MSCCL_SRC_LOCATION}/include:${PATH}"

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: Simple, Channel: 1, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_Simple_gpu16_ch1_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL, Channel: 1, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL_gpu16_ch1_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL128, Channel: 1, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL128_gpu16_ch1_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: Simple, Channel: 8, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_Simple_gpu16_ch8_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL, Channel: 8, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL_gpu16_ch8_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL128, Channel: 8, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL128_gpu16_ch8_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binary Tree, Protocol: Simple, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binary_tree_Simple_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binary Tree, Protocol: LL, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binary_tree_LL_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binary Tree, Protocol: LL128, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binary_tree_LL128_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binomial Tree, Protocol: Simple, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binomial_tree_Simple_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binomial Tree, Protocol: LL, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binomial_tree_LL_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binomial Tree, Protocol: LL128, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binomial_tree_LL128_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Trinomial Tree, Protocol: Simple, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_trinomial_tree_Simple_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Trinomial Tree, Protocol: LL, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_trinomial_tree_LL_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Trinomial Tree, Protocol: LL128, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_trinomial_tree_LL128_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Recursive Halving Doubling, Protocol: Simple, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_rec_doub_halv_Simple_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Recursive Halving Doubling, Protocol: LL, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_rec_doub_halv_LL_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Recursive Halving Doubling, Protocol: LL128, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_rec_doub_halv_LL128_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Fibonacci Tree, Protocol: Simple, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_fibonacci_tree_Simple_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Fibonacci Tree, Protocol: LL, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_fibonacci_tree_LL_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Fibonacci Tree, Protocol: LL128, Instance: 1 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_fibonacci_tree_LL128_gpu16_ins1.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

### case 02: instance = 2

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: Simple, Channel: 1, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_Simple_gpu16_ch1_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL, Channel: 1, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL_gpu16_ch1_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL128, Channel: 1, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL128_gpu16_ch1_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: Simple, Channel: 8, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_Simple_gpu16_ch8_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL, Channel: 8, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL_gpu16_ch8_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Ring, Protocol: LL128, Channel: 8, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_ring_LL128_gpu16_ch8_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binary Tree, Protocol: Simple, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binary_tree_Simple_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binary Tree, Protocol: LL, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binary_tree_LL_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binary Tree, Protocol: LL128, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binary_tree_LL128_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binomial Tree, Protocol: Simple, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binomial_tree_Simple_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binomial Tree, Protocol: LL, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binomial_tree_LL_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Binomial Tree, Protocol: LL128, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_binomial_tree_LL128_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Trinomial Tree, Protocol: Simple, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_trinomial_tree_Simple_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Trinomial Tree, Protocol: LL, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_trinomial_tree_LL_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Trinomial Tree, Protocol: LL128, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_trinomial_tree_LL128_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Recursive Halving Doubling, Protocol: Simple, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_rec_doub_halv_Simple_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Recursive Halving Doubling, Protocol: LL, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_rec_doub_halv_LL_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Recursive Halving Doubling, Protocol: LL128, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_rec_doub_halv_LL128_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Fibonacci Tree, Protocol: Simple, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_fibonacci_tree_Simple_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=Simple

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Fibonacci Tree, Protocol: LL, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_fibonacci_tree_LL_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1

echo " "
echo "------ Library: MSCCL, Algorithm: Fibonacci Tree, Protocol: LL128, Instance: 2 ------"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCLTOOLS_XML_LOCATION}/allreduce_fibonacci_tree_LL128_gpu16_ins2.xml
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_PROTO=LL128

mpirun -np 16 -npernode 1 ${NCCLTESTS_MSCCL_SRC_LOCATION}/build/all_reduce_perf -b 8 -e 128MB -f 2 -g 1