#!/bin/bash

source ~/.bashrc

export MSCCL_TOOLS_ALGORITHMS='/Users/liuyaodai/github/msccl_tools_lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/Users/liuyaodai/github/msccl_tools_lyd/examples/xml/xml_lyd'

# ##################### ring ######################
# # the num_chunks is the original number of chunks per channel for each gpu
# # total chunks = 2 * num_chunks * size * channels * number of rings

nchunks_values=(1 2)
nchannel_values=(1 2 4)
trees_values=(2)
nodes_values=(4 8 16 32 64)

export ngpus=4

for nnodes in "${nodes_values[@]}"; do
    for nchannel in "${nchannel_values[@]}"; do
        for nchunks in "${nchunks_values[@]}"; do
            for trees in "${trees_values[@]}"; do
                python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_ring_h_p.py  \
                --protocol=Simple --num_gpus=$ngpus --num_nodes=$nnodes --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
                > ${MSCCL_TOOLS_XML}/ring/allreduce_ring_node${nnodes}_gpu$((nnodes*ngpus))_mcl${nchannel}_mck$((nchunks))_gan0.xml
            done
        done
    done
done