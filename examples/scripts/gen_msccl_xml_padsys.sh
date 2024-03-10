#!/bin/bash
# source /home/ldai8/bash/.conda_msccl

# source /home/ldai8/bash/.msccltoolrc  

source /home/liuyao/scratch/deps/conda/etc/profile.d/conda.sh

conda activate param_msccl

# export PATH="/home/liuyao/scratch/venv/bin:$PATH"

export MSCCL_TOOLS_ALGORITHMS='/home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/home/liuyao/scratch/deps/msccl_tools_lyd/examples/xml/xml_lyd'


############################# Algo for SC24 ################################################
# multiple chunks (done), two trees/rings (done), channels (not done, some algos only support to 2 channels)


###################### recursive_having_doubling ######################
# only support up to 2 channels
python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_p.py --protocol=Simple --num_gpus=4 --num_nodes=8 --nchunks=32 --nchannel=2 --instances=1 > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/allreduce_recursive_doubling_halving.xml

###################### ring ######################
# the num_chunks is the original number of chunks per channel for each gpu
# total chunks = 2 * num_chunks * size * channels * number of rings

python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allgather_ring_p.py --num_nodes=4 --num_chunks=8 --channels=2 --protocol=Simple --num_gpus=4 --instances=1 > ${MSCCL_TOOLS_XML}/ring/allgather_ring.xml

# num_chunks = 2 * num_nodes * num_gpus

# channles not used to multiply the chunks, only to divide the chunks into multiple parallelism

python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_ring_p.py --num_nodes=4 --channels=2 --protocol=Simple --num_gpus=4 --instances=1 > ${MSCCL_TOOLS_XML}/ring/allreduce_ring.xml


###################### double_binary_tree ######################
# num_total_chunks = num_chunks * num_channel * trees
# only support up to 2 channels
python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binary_tree_p_gpu01.py --protocol=Simple --num_gpus=4 --num_nodes=4 --nchunk=64 --nchannel=2 --instances=1 > ${MSCCL_TOOLS_XML}/binary_tree_p_gpu01/allreduce_binary_tree_p_gpu01.xml

###################### double_binomial_tree ######################
# num_total_chunks = num_chunks * num_channel * trees
# only support up to 2 channels
python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binomial_tree_p.py --protocol=Simple --num_gpus=4 --num_nodes=32 --nchunk=8 --nchannel=2 --instances=1 > ${MSCCL_TOOLS_XML}/binomial_tree/allreduce_binomial_tree.xml

###################### triple_trinomial_tree ######################
# num_total_chunks = num_chunks * num_channel * trees
# only support up to 2 channels
python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_trinomial_tree_p.py --protocol=Simple --num_nodes=9 --num_gpus=4 --nchunks=64 --nchannel=2 --instances=1 > ${MSCCL_TOOLS_XML}/trinomial_tree/allreduce_trinomial_tree.xml