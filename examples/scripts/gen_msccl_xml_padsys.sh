#!/bin/bash
# source /home/ldai8/bash/.conda_msccl

# source /home/ldai8/bash/.msccltoolrc  

source /home/liuyao/scratch/deps/conda/etc/profile.d/conda.sh

conda activate param_msccl

# export PATH="/home/liuyao/scratch/venv/bin:$PATH"

export MSCCL_TOOLS_ALGORITHMS='/home/liuyao/scratch/deps/msccl_tools_lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/home/liuyao/scratch/deps/msccl_tools_lyd/examples/xml/xml_lyd'

# ### generated algorithm for 16 gpus: msccl ring, msccl double binary tree, msccl double binomial tree, msccl triple trinomial tree, msccl resursive doubling halving
# ### ring
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 1 1 --protocol=LL > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch1_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 1 1 --protocol=LL128 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch1_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 1 1 --protocol=Simple > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch1_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 64 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch1_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 64 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch1_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 64 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch1_ins2.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch8_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch8_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch8_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 64 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch8_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 64 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch8_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 64 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch8_ins2.xml

# ### binary tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL128 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL128_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL128 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL128_gpu64_ins2.xml


# ### binomial tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL128 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL128 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu64_ins2.xml


# ### hierarchical binomial tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL128 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins1_hierarchical.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=Simple 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL128 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins2_hierarchical.xml


# ### hierarchical 4_nomial tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL128 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins1_hierarchical.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=Simple 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL128 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins2_hierarchical.xml



# ### trinomial tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=Simple 64 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_Simple_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL 64 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL128 64 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL128_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=Simple 64 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_Simple_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL 64 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL128 64 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL128_gpu64_ins2.xml


# ### resursive doubling halving
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu64_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL128 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL128_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 64 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 64 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu64_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL128 64 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL128_gpu64_ins2.xml


# ### a100 resursive doubling halving
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=Simple 1 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_Simple_ways1_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL 1 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL_ways1_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL128 1 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL128_ways1_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=Simple 2 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_Simple_ways2_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL 2 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL_ways2_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL128 2 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL128_ways2_ins1.xml

# ### fibonacci tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=Simple 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_Simple_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL128 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL128_gpu16_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=Simple 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_Simple_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL128 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL128_gpu16_ins2.xml

# ### msccl ring tree test
# python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins1_test.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins2_test.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 8 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins8_test.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 16 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins16_test.xml


############################# Algo for AWS GPU64 ################################################

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 8 1 --protocol=Simple > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch8_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_allpairs_v2_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_h_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_4_nomial_tree_h_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_pipeline.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_pipeline_4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_ch4_h_p_Simple_gpu64_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_pipeline_8.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_ch8_h_p_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_1ch_intra_2ch_inter.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_1ch_intra_2ch_inter_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_2nic_Simple_gpu64_ins1.xml

# ###### test for msccl-tools ######
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_2_gpus_tree.py 2 1 1 > ${MSCCL_TOOLS_XML}/allreduce_2_gpus_tree_gpu2_ins1_test.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_2_gpus_tree.py 2 1 2 > ${MSCCL_TOOLS_XML}/allreduce_2_gpus_tree_gpu2_ins2_test.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_2_gpus_tree.py -h

# python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py -h

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 2 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu2_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 8 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu8_ins2.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 4 2 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu4_ch2_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 4 2 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu4_ch2_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 4 2 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu4_ch2_ins2.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 2 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu2_ins1.xml







# ### a100 allpairs_v2
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_allpairs_v2_gpus64_ins1.xml

### binary_hierarchical tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_hierarchical_Simple_gpu64_ins1.xml

# hierarchical binomial tree ch3
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_ch4_hierarchical.xml


# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins1_ch4_hierarchical.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_hierarchical_Simple_gpu64_ins1_ch4.xml




python3 ${MSCCL_TOOLS_ALGORITHMS}/recursive_having_doubling/allreduce_recursive_doubling_halving_p.py --protocol=Simple --num_gpus=4 --num_nodes=8 --nchunks=32 --nchannel=4 --instances=1 > ${MSCCL_TOOLS_XML}/recursive_doubling_halving/recursive_doubling_halving_4gpus_4nodes_4channels_4chunks.xml


