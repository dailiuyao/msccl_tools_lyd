#!/bin/bash -l
#PBS -l select=64:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:59:59
#PBS -q prod
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-64nodes
#PBS -o ../log/paper0/ccl-64nodes-tree.out
#PBS -e ../log/paper0/ccl-64nodes-tree.error

export MPI_HOME=/opt/cray/pe/mpich/8.1.25/ofi/nvidia/20.7
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda

export MPIEXEC_HOME=/opt/cray/pe/pals/1.2.11
export NCCL_NET_PLUGIN_HOME="/home/yuke/ncclPG/aws-ofi-nccl-1.7.4-aws/build"          

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# export OFI_NCCL_PROTOCOL=RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_IB_HCA=cxi0,cxi1


################################### MSCCL TEST #########################################################

echo "NCCL TEST with MSCCL"

export NCCL_TEST_MSCCL_HOME="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd"
export MSCCL_SRC_LOCATION

export MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd"

export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple
# export NCCL_NTHREADS=512

export GENMSCCLXML=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/scripts/gen_msccl_xml_frontera.sh

export MSCCL_TOOLS_XML="/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd" 

nchunks_values=(64)
nchannel_values=(2 4)
trees_values=(2)
nodes_values=(64)

export ngpus=4

for nnodes in "${nodes_values[@]}"; do
    for nchannel in "${nchannel_values[@]}"; do
        for nchunks in "${nchunks_values[@]}"; do
            for trees in "${trees_values[@]}"; do
                echo "Running MSCCL tree test with ${nnodes} nodes, ${nchannel} channels, ${nchunks} chunks, ${trees} trees"
                export MSCCL_XML_FILES=${MSCCL_TOOLS_XML}/ring/allreduce_ring_node${nnodes}_gpu$((nnodes*ngpus))_mcl${nchannel}_mck${nchunks}_gan0.xml
                $MPIEXEC_HOME/bin/mpiexec -n $((nnodes*ngpus)) --ppn ${ngpus} --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 64K -e 256MB -f 2 -g 1 -n 60 \
                > /home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/scripts/polaris-test/sc24/log/paper0/buffer_size_2/ring/all-reduce_sum_float_ring_node${nnodes}_gpu$((nnodes*ngpus))_mcl${nchannel}_mck${nchunks}_i0.out
            done
        done
    done
done




# echo "##################################### MSCCL #####################################" 


# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/4_nomial_tree/allreduce_4_nomial_tree_2ch_32chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/4_nomial_tree/allreduce_4_nomial_tree_2ch_64chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4--cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/4_nomial_tree/allreduce_4_nomial_tree_2ch_128chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_2ch_32chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60


# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_2ch_64chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binary_tree/allreduce_binary_tree_2ch_128chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_2ch_32chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_2ch_64chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_2ch_128chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling/allreduce_recursive_doubling_2ch_4chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling/allreduce_recursive_doubling_2ch_32chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60


# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling_halving/allreduce_recursive_doubling_halving_2ch_2chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60

# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/recursive_doubling_halving/allreduce_recursive_doubling_halving_2ch_4chunk.xml

# $MPIEXEC_HOME/bin/mpiexec -n 256 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 512K -e 512MB -f 2 -g 1 -n 60


# export MSCCL_XML_FILES=/home/yuke/ncclPG/CCL-LYD/msccl_tools_lyd/examples/xml/xml_lyd/trinomial_tree/allreduce_trinomial_tree_2ch_64chunk_27nodes.xml

# $MPIEXEC_HOME/bin/mpiexec -n 324 --ppn 4 --cpu-bind core $NCCL_TEST_MSCCL_HOME/build/all_reduce_perf -b 768K -e 768MB -f 2 -g 1 -n 60
