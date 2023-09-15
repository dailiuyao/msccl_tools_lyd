#!/bin/bash
# source /home/ldai8/bash/.conda_msccl

# source /home/ldai8/bash/.msccltoolrc  

export PATH="/home/ldai8/scratch/msccl_build/venv/bin:$PATH"

export MSCCL_TOOLS_ALGORITHMS='/home/ldai8/scratch/msccl_build/deps/msccl-tools/examples/mscclang'

export MSCCL_TOOLS_XML='/home/ldai8/scratch/msccl_build/deps/msccl-tools/examples/xml'

### generated algorithm for 16 gpus: msccl ring, msccl double binary tree, msccl double binomial tree, msccl triple trinomial tree, msccl resursive doubling halving
### ring
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 16 1 1 --protocol=LL > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu16_ch1_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 16 1 1 --protocol=LL128 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu16_ch1_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py 16 1 1 --protocol=Simple > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu16_ch1_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 16 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu16_ch1_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 16 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu16_ch1_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 16 1 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu16_ch1_ins2.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 16 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu16_ch8_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 16 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu16_ch8_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 16 8 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu16_ch8_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL 16 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL_gpu16_ch8_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=LL128 16 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_LL128_gpu16_ch8_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_a100_ring.py --protocol=Simple 16 8 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_Simple_gpu16_ch8_ins2.xml

# ### binary tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL128 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL128_gpu16_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=Simple 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_Simple_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binary_tree.py --protocol=LL128 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binary_tree_LL128_gpu16_ins2.xml


# ### binomial tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL128 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu16_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=LL128 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_LL128_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_binomial_tree.py --protocol=Simple 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_binomial_tree_Simple_gpu16_ins2.xml

# ### trinomial tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=Simple 16 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_Simple_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL 16 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL128 16 3 1 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL128_gpu16_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=Simple 16 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_Simple_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL 16 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_trinomial_tree.py --protocol=LL128 16 3 2 > ${MSCCL_TOOLS_XML}/allreduce_trinomial_tree_LL128_gpu16_ins2.xml


# ### resursive doubling halving
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py -h
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 16 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 16 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL128 16 1 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL128_gpu16_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=Simple 16 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_Simple_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL 16 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_recursive_doubling_halving.py --protocol=LL128 16 2 > ${MSCCL_TOOLS_XML}/allreduce_rec_doub_halv_LL128_gpu16_ins2.xml


# ### fibonacci tree
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=Simple 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_Simple_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL_gpu16_ins1.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL128 16 2 1 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL128_gpu16_ins1.xml

# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=Simple 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_Simple_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL_gpu16_ins2.xml
# python ${MSCCL_TOOLS_ALGORITHMS}/allreduce_fibonacci_tree.py --protocol=LL128 16 2 2 > ${MSCCL_TOOLS_XML}/allreduce_fibonacci_tree_LL128_gpu16_ins2.xml

### msccl ring tree test
python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py -h
python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 1 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins1_test.xml

python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 2 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins2_test.xml

python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 8 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins8_test.xml

python ${MSCCL_TOOLS_ALGORITHMS}/simple/allreduce_ring.py 16 16 > ${MSCCL_TOOLS_XML}/allreduce_ring_gpu16_ins16_test.xml

