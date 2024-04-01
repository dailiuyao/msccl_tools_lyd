# #!/bin/bash

# # Directory containing the files to process
# directory="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/frontera-test/sc24/log/paper0/chunk_step_4/tree"

# # Loop through each file in the directory
# for file in "$directory"/*; do
#   echo "Processing $file..."
#   # Use sed to find and replace all instances within each file
#   sed -i -r 's/allreduce_binary_tree_([0-9]+)ch_([0-9]+)tree_([0-9]+)chunk_([0-9]+)node_([0-9]+)gpu.xml/allreduce_binary-tree_node\4_gpu\5_mcl\1_mck\3_gan0.xml/g' "$file"
# done

# echo "Processing complete."

Directory containing the files to process
directory="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/frontera-test/sc24/log/paper0/chunk_step_4/ring"

for file in "$directory"/*; do
    mv "$file" "${file//i0\.out/buf4_gan0_i0\.log}"
done



# echo "Processing complete."

# all-reduce_sum_float_binary-tree_node4_gpu16_mcl1_mck1_i0

# all-reduce_binary-tree_node4_gpu32_mcl16_mck1_buf1_gan0_i0.log

