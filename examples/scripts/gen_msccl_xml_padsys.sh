#!/bin/bash
# source /home/ldai8/bash/.conda_msccl

# source /home/ldai8/bash/.msccltoolrc  

source /home/liuyao/scratch/deps/conda/etc/profile.d/conda.sh

conda activate param_msccl

# export PATH="/home/liuyao/scratch/venv/bin:$PATH"

export MSCCL_TOOLS_ALGORITHMS='/home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/home/liuyao/scratch/deps/msccl_tools_lyd/examples/xml/xml_lyd'


# +++++++++++++++++++++++++++++++++++++++++++++++++++ Algos for SC24 +++++++++++++++++++++++++++++++++++++++++++++++++++++
# multiple chunks (done), two trees/rings (done), channels (not done, some algos only support to 2 channels)


# ###################### recursive_having_doubling ######################
# # only support up to 2 channels

nchunks_values=(32)
nchannel_values=(2)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_p.py \
        --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/allreduce_recursive_doubling_halving_${nchannel}ch_${nchunks}chunk.xml
    done
done



# ##################### ring ######################
# # the num_chunks is the original number of chunks per channel for each gpu
# # total chunks = 2 * num_chunks * size * channels * number of rings

# nchunks_values=(1 2 4)
# nchannel_values=(1 2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allgather_ring_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/ring/allgather_ring_${nchannel}ch_${nchunks}chunk.xml
#     done
# done

# # num_chunks = 2 * num_nodes * num_gpus

# # channles not used to multiply the chunks, only to divide the chunks into multiple parallelism

# # at least 2 channels

nchunks_values=(4)
nchannel_values=(8)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_ring_p.py \
        --protocol=Simple --num_gpus=1 --num_nodes=8 --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/ring/allreduce_ring_${nchannel}ch_${nchunks}chunk.xml
    done
done


##################### double_binary_tree ######################
# num_total_chunks = num_chunks * num_channel * trees
# only support up to 2 channels


nchunks_values=(256)
nchannel_values=(2)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binary_tree_p_gpu01.py \
        --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/binary_tree/allreduce_binary_tree_${nchannel}ch_${nchunks}chunk.xml
    done
done

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allgather_ring_p.py \
#         --protocol=Simple --num_gpus=4 --num_nodes=8 --nchunks=4 --nchannel=1 --instances=1 \
#         > ${MSCCL_TOOLS_XML}/ring/allgather_ring_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allgather_binary_tree_p_gpu01.py --protocol=Simple --num_gpus=2 --num_nodes=8 --nchunk=4 --nchannel=1 --instances=1 > ${MSCCL_TOOLS_XML}/binary_tree/allgather_binary_tree.xml


# ###################### double_binomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels

nchunks_values=(128)
nchannel_values=(2)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binomial_tree_p.py \
        --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/binomial_tree/allreduce_binomial_tree_${nchannel}ch_${nchunks}chunk.xml
    done
done


# ###################### triple_trinomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels


nchunks_values=(128)
nchannel_values=(1 2)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_trinomial_tree_p.py \
        --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/trinomial_tree/allreduce_trinomial_tree_${nchannel}ch_${nchunks}chunk.xml
    done
done


# ###################### recursive_doubling ######################
# # only support up to 2 channels

nchunks_values=(32)
nchannel_values=(2)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_doubling/allreduce_recursive_doubling_p.py \
        --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/recursive_doubling/allreduce_recursive_doubling_${nchannel}ch_${nchunks}chunk.xml
    done
done




