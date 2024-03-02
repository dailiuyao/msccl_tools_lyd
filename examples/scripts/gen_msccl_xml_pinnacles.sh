#!/bin/bash

export PATH="/home/ldai8/scratch/msccl_build/venv/bin:$PATH"

export MSCCL_TOOLS_ALGORITHMS='/home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/xml/xml_lyd'

# python ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_trinomial_tree_p.py --protocol=Simple --num_gpus=4 --num_nodes=9 --nchunks=216 --nchannel=6 --instances=1 > ${MSCCL_TOOLS_XML}/trinomial_tree/trinomial_tree_p.xml


# python ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_trinomial_tree_p.py --protocol=Simple --num_nodes=9 --num_gpus=4 --nchunks=216 --nchannel=6 --instances=1 > ${MSCCL_TOOLS_XML}/trinomial_tree/trinomial_tree_p.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binomial_tree_p.py --protocol=Simple --num_nodes=8 --num_gpus=4 --nchunks=128 --nchannel=4 --instances=1 > ${MSCCL_TOOLS_XML}/binomial_tree/binomial_tree_p.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_v2.py --protocol=Simple --num_gpus=16 --instances=1 > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/recursive_doubling_halving.xml

python ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_p.py --protocol=Simple --num_gpus=4 --num_nodes=8 --nchunks=2 --nchannel=2 --instances=1 > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/recursive_doubling_halving_4gpus_4nodes_4channels_4chunks.xml


# python ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_ring_4_gpus_n_nodes_n_channels.py --num_nodes=16 --channels=2 --protocol=Simple --num_gpus=4 --instances=1 > ${MSCCL_TOOLS_XML}/ring/allreduce_ring_4_gpus_64_nodes_2_channels.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/ring/allgather_ring_4_gpus_n_nodes_n_channels.py --num_nodes=16 --channels=2 --protocol=Simple --num_gpus=4 --instances=1 > ${MSCCL_TOOLS_XML}/ring/allgather_ring_4_gpus_16_nodes_2_channels.xml