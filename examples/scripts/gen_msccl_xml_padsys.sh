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

# nchunks_values=(8 16 32 64 128 256)
# nchannel_values=(1 2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/allreduce_recursive_doubling_halving_${nchannel}ch_${nchunks}chunk_16gpus.xml
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
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/ring/allgather_ring_${nchannel}ch_${nchunks}chunk_16gpus.xml
#     done
# done

# # num_chunks = 2 * num_nodes * num_gpus

# # channles not used to multiply the chunks, only to divide the chunks into multiple parallelism

# # at least 2 channels

# nchunks_values=(1 2 4)
# nchannel_values=(2 4 8)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_ring_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/ring/allreduce_ring_${nchannel}ch_${nchunks}chunk_16gpus.xml
#     done
# done


##################### double_binary_tree ######################
# num_total_chunks = num_chunks * num_channel * trees
# only support up to 2 channels


# nchunks_values=(8 16 32 64 128 256)
# nchannel_values=(1 2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binary_tree_p_gpu01.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/binary_tree/allreduce_binary_tree_${nchannel}ch_${nchunks}chunk_16gpus.xml
#     done
# done



# python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allgather_binary_tree_p_gpu01.py --protocol=Simple --num_gpus=4 --num_nodes=4 --nchunk=4 --nchannel=2 --instances=1 > ${MSCCL_TOOLS_XML}/binary_tree/allgather_binary_tree.xml


# ###################### double_binomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels

# nchunks_values=(8 16 32 64 128 256)
# nchannel_values=(1 2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_binomial_tree_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/binomial_tree/allreduce_binomial_tree_${nchannel}ch_${nchunks}chunk_16gpus.xml
#     done
# done


# ###################### triple_trinomial_tree ######################
# # num_total_chunks = num_chunks * num_channel * trees
# # only support up to 2 channels


# nchunks_values=(8 16 32 64 128 256)
# nchannel_values=(1 2)

# for nchannel in "${nchannel_values[@]}"; do
#     for nchunks in "${nchunks_values[@]}"; do
#         python3 ${MSCCL_TOOLS_ALGORITHMS}/tree/allreduce_trinomial_tree_p.py \
#         --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
#         > ${MSCCL_TOOLS_XML}/trinomial_tree/allreduce_trinomial_tree_${nchannel}ch_${nchunks}chunk_16gpus.xml
#     done
# done


# ###################### recursive_doubling ######################
# # only support up to 2 channels

nchunks_values=(8 16 32 64 128 256)
nchannel_values=(1 2)

for nchannel in "${nchannel_values[@]}"; do
    for nchunks in "${nchunks_values[@]}"; do
        python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_doubling/allreduce_recursive_doubling_p.py \
        --protocol=Simple --num_gpus=1 --num_nodes=16 --nchunks=$nchunks --nchannel=$nchannel --instances=1 \
        > ${MSCCL_TOOLS_XML}/recursive_doubling/allreduce_recursive_doubling_${nchannel}ch_${nchunks}chunk_16gpus.xml
    done
done





# mpirun --hostfile ~/hostfile --map-by ppr:1:node git -C /home/ec2-user/deps/msccl pull

# mpirun --hostfile ~/hostfile --map-by ppr:8:node \
#     -x CUDA_HOME="/usr/local/cuda" \
#     -x CUDA_PATH="/usr/local/cuda" \
#     -x NCCL_HOME="/home/ec2-user/deps/msccl/build" \
#     -x MPI_HOME="/opt/amazon/openmpi" \
#     -x NCCL_ALGO="TREE" \
#     -x NCCL_MIN_NCHANNELS= "4" \
#     -x LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/amazon/openmpi/lib64:/home/ec2-user/deps/msccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
#     -x NCCL_DEBUG="INFO" \
#     -x FI_EFA_FORK_SAFE=1 \
#     -x MSCCL_XML_FILES="/home/ec2-user/deps/msccl-tools-lyd/examples/xml/xml_lyd/binomial_tree/allreduce_binomial_tree_2ch_64chunk_16gpus.xml" \
#     -x MSCCL_XML_FILES=" " \
#     -x GENMSCCLXML=1 \
#     --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
#     /home/ec2-user/deps/nccl-tests-lyd/build/all_reduce_perf \
#     --nthreads 1 \
#     --ngpus 1 \
#     --minbytes 512 \
#     --maxbytes 128M \
#     --stepfactor 2 \
#     --op sum \
#     --datatype float \
#     --iters 20 \
#     --warmup_iters 5

mpirun --hostfile ~/hostfile --map-by ppr:8:node \
    -x CUDA_HOME="/usr/local/cuda" \
    -x CUDA_PATH="/usr/local/cuda" \
    -x NCCL_HOME="/home/ec2-user/deps/msccl/build" \
    -x MPI_HOME="/opt/amazon/openmpi" \
    -x LD_LIBRARY_PATH="/opt/aws-ofi-nccl/lib:/opt/amazon/openmpi/lib64:/home/ec2-user/deps/msccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    -x NCCL_DEBUG="INFO" \
    -x FI_EFA_FORK_SAFE=1 \
    -x OFI_NCCL_NIC_DUP_CONNS=2 \
    -x MSCCL_XML_FILES="/home/ec2-user/deps/msccl-tools-lyd/examples/xml/xml_lyd/aws-test/1nic/allreduce_binary_tree_1ch_16chunk.xml" \
    -x GENMSCCLXML=1 \
    --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    /home/ec2-user/deps/nccl-tests-lyd/build/all_reduce_perf \
    --nthreads 1 \
    --ngpus 1 \
    --minbytes 384 \
    --maxbytes 384M \
    --stepfactor 2 \
    --op sum \
    --datatype float \
    --iters 20 \
    --warmup_iters 5 
    # > output.txt 2>&1