#!/bin/bash

source /home1/09168/ldai1/anaconda3/etc/profile.d/conda.sh

conda activate msccl_tools

export MSCCL_TOOLS_ALGORITHMS='/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/xml/xml_lyd/binary_h_gpu01_reverse_p'


/home1/09168/ldai1/anaconda3/envs/msccl_tools/bin/python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 2 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk2_frontera.xml