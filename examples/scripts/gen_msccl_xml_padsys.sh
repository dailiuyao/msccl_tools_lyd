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

# nchunks_values=(8)
# nchannel_values=(2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_p.py \
#         --protocol=Simple --num_gpus=4 --num_nodes=64 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/allreduce_recursive_doubling_halving_${nchannel}ch_${nchunks}chunk.xml
#     done
# done



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

# at least 2 channels

# nchannel_values=(2)

# for nchannel in "${nchannel_values[@]}"; do
#     python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_ring_p.py \
#     --protocol=Simple --num_gpus=4 --num_nodes=64 --nchannel=$nchannel --instances=1 \
#     > ${MSCCL_TOOLS_XML}/ring/allreduce_ring_${nchannel}ch_2ring_512chunk_64node_256gpu.xml
# done


##################### double_binary_tree ######################
# num_total_chunks = num_chunks * num_channel * trees
# only support up to 2 channels


nchunks_values=(1 2 4)
nchannel_values=(8)
trees_values=(2)
nodes_values=(2)

export ngpus=8

for nnodes in "${nodes_values[@]}"; do
    for nchannel in "${nchannel_values[@]}"; do
        for nchunks in "${nchunks_values[@]}"; do
            for trees in "${trees_values[@]}"; do
                python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binary_tree_p.py \
                --protocol=Simple --num_gpus=$ngpus --num_nodes=$nnodes --nchunks=$nchunks --nchannel=$nchannel --instances=1 --trees=$trees \
                > ${MSCCL_TOOLS_XML}/aws-test/8nic/16gpus/allreduce_binary_tree_node${nnodes}_gpu$((nnodes*ngpus))_mcl${nchannel}_mck${nchunks}_gan0.xml
            done
        done
    done
done


# nchunks_values=(16 32 64)
# nchannel_values=(8)
# trees_values=(1)
# nodes_values=(16)

# export ngpus=1

# for nnodes in "${nodes_values[@]}"; do
#     for nchannel in "${nchannel_values[@]}"; do
#         for nchunks in "${nchunks_values[@]}"; do
#             for trees in "${trees_values[@]}"; do
#                 python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binary_tree_p_gpu.py \
#                 --protocol=Simple --num_gpus=$ngpus --num_nodes=$nnodes --nchunks=$nchunks --nchannel=$nchannel --instances=1 --trees=$trees \
#                 > ${MSCCL_TOOLS_XML}/aws-test/8nic/16gpus/allreduce_binary_tree_node${nnodes}_gpu$((nnodes*ngpus))_mcl${nchannel}_mck${nchunks}_gan1.xml
#             done
#         done
#     done
# done




# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         for trees in "${trees_values[@]}"; do
#             python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binary_tree_p_gpu01.py \
#             --protocol=Simple --num_gpus=4 --num_nodes=64 --nchunks=$nchunks --nchannel=$nchannel --instances=1 --trees=$trees \
#             > ${MSCCL_TOOLS_XML}/binary_tree/allreduce_binary_tree_$((nchannel*2))ch_${trees}tree_${nchunks}chunk_64node_256gpu.xml
#         done
#     done
# done

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allgather_ring_p.py \
#         --protocol=Simple --num_gpus=4 --num_nodes=8 --nchunks=4 --nchannel=1 --instances=1 \
#         > ${MSCCL_TOOLS_XML}/ring/allgather_ring_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allgather_binary_tree_p_gpu01.py --protocol=Simple --num_gpus=2 --num_nodes=8 --nchunk=4 --nchannel=1 --instances=1 > ${MSCCL_TOOLS_XML}/binary_tree/allgather_binary_tree.xml


# ###################### double_binomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels

# nchunks_values=(128)
# nchannel_values=(2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binomial_tree_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=8 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/binomial_tree/allreduce_binomial_tree_${nchannel}ch_${nchunks}chunk.xml
#     done
# done


# ###################### triple_trinomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels


# nchunks_values=(64)
# nchannel_values=(2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_trinomial_tree_p.py \
#         --protocol=Simple --num_gpus=4 --num_nodes=81 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/trinomial_tree/allreduce_trinomial_tree_${nchannel}ch_${nchunks}chunk_27nodes.xml
#     done
# done


# ###################### recursive_doubling ######################
# # only support up to 2 channels

# nchunks_values=(4)
# nchannel_values=(2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_doubling/allreduce_recursive_doubling_p.py \
#         --protocol=Simple --num_gpus=4 --num_nodes=64 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/recursive_doubling/allreduce_recursive_doubling_${nchannel}ch_${nchunks}chunk.xml
#     done
# done


# ###################### quadruple_quadrinomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels


# nchunks_values=(8 16 32 64)
# nchannel_values=(4)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_4_nomial_tree_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 --trees=4\
#         > ${MSCCL_TOOLS_XML}/aws-test/8nic/16gpus/allreduce_4_nomial_tree_${nchannel}ch_${nchunks}chunk.xml
#     done
# done


# ###################### basic_msccl ######################
# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_ring.py \
# --num_gpus=64 --instances=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_ring_64gpus.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binomial_tree.py \
# --protocol=Simple --num_gpus=64 --instances=1 --trees=2 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binomial_tree_64gpus_2tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binomial_tree.py \
# --protocol=Simple --num_gpus=64 --instances=1 --trees=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binomial_tree_64gpus_1tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_recursive_doubling_halving.py \
# --protocol=Simple --num_gpus=64 --instances=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_rec_hv_db_64gpus.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binary_tree.py \
# --protocol=Simple --num_gpus=64 --instances=1 --trees=2 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binary_tree_64gpus_2tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binary_tree.py \
# --protocol=Simple --num_gpus=64 --instances=1 --trees=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binary_tree_64gpus_1tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_ring.py \
# --num_gpus=128 --instances=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_ring_128gpus.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binomial_tree.py \
# --protocol=Simple --num_gpus=128 --instances=1 --trees=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binomial_tree_128gpus_1tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binomial_tree.py \
# --protocol=Simple --num_gpus=128 --instances=1 --trees=2 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binomial_tree_128gpus_2tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_recursive_doubling_halving.py \
# --protocol=Simple --num_gpus=128 --instances=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_rec_hv_db_128gpus.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binary_tree.py \
# --protocol=Simple --num_gpus=128 --instances=1 --trees=2 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binary_tree_128gpus_2tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binary_tree.py \
# --protocol=Simple --num_gpus=128 --instances=1 --trees=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binary_tree_128gpus_1tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_ring.py \
# --num_gpus=256 --instances=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_ring_256gpus.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binomial_tree.py \
# --protocol=Simple --num_gpus=256 --instances=1 --trees=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binomial_tree_256gpus_1tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binomial_tree.py \
# --protocol=Simple --num_gpus=256 --instances=1 --trees=2 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binomial_tree_256gpus_2tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_recursive_doubling_halving.py \
# --protocol=Simple --num_gpus=256 --instances=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_rec_hv_db_256gpus.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binary_tree.py \
# --protocol=Simple --num_gpus=256 --instances=1 --trees=2 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binary_tree_256gpus_2tree.xml

# python3 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang/basic_msccl/allreduce_binary_tree.py \
# --protocol=Simple --num_gpus=256 --instances=1 --trees=1 \
# > ${MSCCL_TOOLS_XML}/basic_msccl/allreduce_basic_binary_tree_256gpus_1tree.xml






