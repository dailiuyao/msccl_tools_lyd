#!/bin/bash

export MSCCL_TOOLS_ALGORITHMS='/Users/liuyaodai/github/msccl_tools_lyd/examples/mscclang'

export MSCCL_TOOLS_XML='/Users/liuyaodai/github/msccl_tools_lyd/examples/xml'

# ### generated algorithm for 16 gpus: msccl ring, msccl double binary tree, msccl double binomial tree, msccl triple trinomial tree, msccl resursive doubling halving
# ### ring
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py -h
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 1 1 --protocol=LL > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch1_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 1 1 --protocol=LL128 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch1_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 1 1 --protocol=Simple > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch1_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 64 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch1_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 64 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch1_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 64 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch1_ins2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch8_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch8_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch8_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 64 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu64_ch8_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 64 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu64_ch8_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 64 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch8_ins2.xml

# ### binary tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL128 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL128_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL128 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL128_gpu64_ins2.xml


# ### binomial tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL128 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL128 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 64 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu64_ins2.xml


# ### hierarchical binomial tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL128 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins1_hierarchical.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=Simple 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=LL128 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins2_hierarchical.xml


# ### hierarchical 4_nomial tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL128 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins1_hierarchical.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=Simple 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=LL128 4 16 2 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins2_hierarchical.xml



# ### trinomial tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=Simple 64 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_Simple_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL 64 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL128 64 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL128_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=Simple 64 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_Simple_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL 64 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL128 64 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL128_gpu64_ins2.xml


# ### resursive doubling halving
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py -h
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu64_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL128 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL128_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 64 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 64 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu64_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL128 64 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL128_gpu64_ins2.xml


# ### a100 resursive doubling halving
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py -h
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=Simple 1 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_Simple_ways1_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL 1 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL_ways1_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL128 1 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL128_ways1_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=Simple 2 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_Simple_ways2_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL 2 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL_ways2_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_recursive_doubling_halving.py --protocol=LL128 2 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_rec_doub_halv_LL128_ways2_ins1.xml

# ### fibonacci tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=Simple 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_Simple_gpu16_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL_gpu16_ins1.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL128 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL128_gpu16_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=Simple 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_Simple_gpu16_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL_gpu16_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL128 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL128_gpu16_ins2.xml

# ### msccl ring tree test
# python3 ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py -h
# python3 ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins1_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins2_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 8 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins8_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 16 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins16_test.xml


############################# Algo for GPU64 ################################################

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 64 8 1 --protocol=Simple > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_ch8_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_allpairs_v2_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_h_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_4_nomial_tree_h_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_pipeline.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_pipeline_4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_ch4_h_p_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_pipeline_8.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_ch8_h_p_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_p.py --protocol=Simple 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_1ch_intra_2ch_inter.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_1ch_intra_2ch_inter_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_2ch_intra_4ch_inter.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2ch_intra_4ch_inter_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_4_ch_intra_2ch_inter.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_4ch_intra_2ch_inter_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/hierarchical_allreduce.py --protocol=Simple --schedule=manual 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_allreduce_hierarchical_allreduce_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2_h.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_allreduce_a100_allpairs_v2_h_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_h_ch4_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_2nic_Simple_gpu64_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic_4gpu_pipe.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_2nic_4gpu_p_Simple_gpu64_ins1.xml


# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic_4gpu_pipe.py --protocol=Simple 4 16 256 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_256_nch_8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic_4gpu_pipe.py --protocol=Simple 4 16 128 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_128_nch_8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic_4gpu_pipe.py --protocol=Simple 4 16 64 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_64_nch_8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic_4gpu_pipe.py --protocol=Simple 4 16 32 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_32_nch_8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2_ch_intra_1ch_inter_2nic_4gpu_pipe.py --protocol=Simple 4 16 16 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_16_nch_8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_8_ch_2nic_4gpu_pipe.py --protocol=Simple 4 16 4 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_8_ch_2nic_4gpu_pipe.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_8_ch_intra_pipe_inter_2nicPtree.py --protocol=Simple 4 16 4 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_8_ch_intra_pipe_inter_2nicPtree.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_16_ch_intra_pipe_inter_2nicPtree.py --protocol=Simple 4 16 8 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_16_ch_intra_pipe_inter_2nicPtree.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2nicPtree_ch_4.py --protocol=Simple 4 16 8 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nicPtree_ch_4.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2nicPtree_ch_8_intra_8_inter_4.py --protocol=Simple 4 16 8 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nicPtree_ch_8_intra_8_inter_4.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_8_inter_2.py --protocol=Simple 4 16 16 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_8_inter_2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_8_inter_2_aggre.py --protocol=Simple 4 16 16 8 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_8_inter_2_aggre.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_2_stage.py --protocol=Simple 4 24 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_ring_ch24_manul_2_stage.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 4 4 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_ring_ch4_manul_1ins.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 4 4 4 > ${MSCCL_TOOLS_XML}/allreduce_a100_ring_ch24_manul_2ins.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 4 4 6 > ${MSCCL_TOOLS_XML}/allreduce_a100_ring_ch4_manul_6ins.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 4 8 4 > ${MSCCL_TOOLS_XML}/allreduce_a100_ring_ch8_manul_4ins.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_8links.py --protocol=Simple 4 32 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_ring_ch32_8links.xml

# ###### test for msccl-tools ######


# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_2_gpus_tree.py 2 1 1 > ${MSCCL_TOOLS_XML}/allreduce_2_gpus_tree_gpu2_ins1_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_2_gpus_tree.py 2 1 2 > ${MSCCL_TOOLS_XML}/allreduce_2_gpus_tree_gpu2_ins2_test.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_2_gpus_tree.py -h

# python3 ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py -h

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 2 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu2_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 8 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu8_ins2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 4 2 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu4_ch2_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 4 2 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu4_ch2_ins2.xml
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 4 4 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu4_ch4_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 4 1 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu4_ch2_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_2_stage.py --protocol=Simple 4 96 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu4_channel24_chunk96_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_64gpu_4channel_256chunk.py --protocol=Simple 64 256 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu64_channel4_chunk256_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_2nicPtree_channel4_chunk256.py --protocol=Simple 4 16 256 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_2nicPtree_channel4_chunk256.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel4.py --protocol=Simple 2 2 16 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_channel4_chunk16.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 1 2 4 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_channel2_chunk4.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 8 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_8nodes_channel2_chunk64.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 4 32 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_4nodes_channel2_chunk32.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 8 128 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_8nodes_channel4_chunk128.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 4 64 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_4nodes_channel4_chunk64.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 16 256 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_16nodes_channel4_chunk256.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 16 128 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_16nodes_channel2_chunk128.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 1 2 8 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_channel4_chunk8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 2 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu2_ins1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n.py --protocol=Simple 4 32 512 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_chunk512.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 512 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk512.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 16 256 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_16nodes_channel4_reverse_chunk256.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 8 128 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_8nodes_channel4_reverse_chunk128.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 4 64 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_4nodes_channel4_reverse_chunk64.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 256 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk256.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 128 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk128.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 64 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk64.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 32 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk32.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 16 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk16.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 8 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 4 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk4.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 8 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk8.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 4 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk4.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 2 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk16.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 32 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk32.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 64 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk64.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 128 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk128.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 256 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk256.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 512 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel2_reverse_chunk512.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 1024 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk1024.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 4 32 2048 4 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_32nodes_channel4_reverse_chunk2048.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 2 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_channel2_reverse_chunk2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 2 2 2 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_4gpus_channel2_reverse_chunk2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 1 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 2 2 1 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_4gpus_channel1_reverse_chunk1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 4 1 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_4nodes_4gpus_channel1_reverse_chunk1.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 2 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk2.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 4 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk4.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 128 1 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk128.xml


# ### a100 allpairs_v2
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2.py -h
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_allpairs_v2.py --protocol=Simple 64 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_allpairs_v2_gpus64_ins1.xml

### binary_hierarchical tree
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical.py -h
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_hierarchical_Simple_gpu64_ins1.xml

# hierarchical binomial tree ch3
# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_binomial_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_ch4_hierarchical.xml


# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_4_nomial_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins1_ch4_hierarchical.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree_hierarchical_ch4.py --protocol=Simple 4 16 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_hierarchical_Simple_gpu64_ins1_ch4.xml



######################################### the xml for HPDC 2024 #########################################


# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree_p_gpu01_channel_n_reverse.py --protocol=Simple 1 2 128 1 1 > ${MSCCL_TOOLS_XML}/xml_lyd/binary_h_gpu01_reverse_p/allreduce_binary_tree_p_gpu01_2nodes_2gpus_channel1_reverse_chunk128.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree.py --protocol=Simple 128 2 1 > ${MSCCL_TOOLS_XML}/xml_lyd/allredcue_basic_binary_tree_128gpus.xml


# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 64 64 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_basic_ring_16nodes_4gpus_64chunks_64channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 64 64 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_16nodes_4gpus_64chunks_polaris.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_frontera_16nodes_nchannels.py --protocol=Simple 64 4 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_16nodes_4gpus_64chunks_4channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_frontera_16nodes_nchannels.py --protocol=Simple 64 1 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_16nodes_4gpus_64chunks_1channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_frontera_16nodes_nchannels.py --protocol=Simple 64 8 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_16nodes_4gpus_64chunks_8channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_frontera_16nodes_nchannels.py --protocol=Simple 64 16 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_16nodes_4gpus_64chunks_16channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_frontera_16nodes_nchannels.py --protocol=Simple 64 32 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_16nodes_4gpus_64chunks_32channels_frontera.xml


# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_polaris_32nodes_nchannels.py --protocol=Simple 128 64 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_32nodes_4gpus_256chunks_64channels_polaris.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_polaris_32nodes_nchannels.py --protocol=Simple 128 32 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_32nodes_4gpus_256chunks_32channels_polaris.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_polaris_32nodes_nchannels.py --protocol=Simple 128 16 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_32nodes_4gpus_256chunks_16channels_polaris.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_polaris_32nodes_nchannels.py --protocol=Simple 128 8 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_32nodes_4gpus_256chunks_8channels_polaris.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring_polaris_32nodes_nchannels.py --protocol=Simple 128 4 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_ring_32nodes_4gpus_256chunks_4channels_polaris.xml


# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 32 32 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_basic_ring_8nodes_4gpus_32chunks_32channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 16 16 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_basic_ring_4nodes_4gpus_16chunks_16channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 8 8 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_basic_ring_2nodes_4gpus_8chunks_8channels_frontera.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree.py --protocol=Simple 32 2 1 > ${MSCCL_TOOLS_XML}/xml_lyd/allredcue_basic_binary_tree_32gpus.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree.py --protocol=Simple 16 2 1 > ${MSCCL_TOOLS_XML}/xml_lyd/allredcue_basic_binary_tree_16gpus.xml

# python3 ${MSCCL_TOOLS_ALGORITHMS}/binary/allreduce_binary_tree.py --protocol=Simple 8 2 1 > ${MSCCL_TOOLS_XML}/xml_lyd/allredcue_basic_binary_tree_8gpus.xml


python3 ${MSCCL_TOOLS_ALGORITHMS}/ring/allreduce_a100_ring.py --protocol=Simple 128 128 1 > ${MSCCL_TOOLS_XML}/xml_lyd/ring/allreduce_basic_ring_32nodes_128gpus_128chunks_32channels_polaris.xml