#!/bin/bash

# # Directory containing the files to process
# directory="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/polaris-test/sc24/log/paper0/buffer_size_1"

# # Loop through each file in the directory
# for file in "$directory"/*; do
#   echo "Processing $file..."
#   # Use sed to find and replace all instances within each file
#   sed -i -r 's/allreduce_binary_tree_([0-9]+)ch_([0-9]+)tree_([0-9]+)chunk_([0-9]+)node_([0-9]+)gpu.xml/allreduce_binary-tree_node\4_gpu\5_mcl\1_mck\3_gan0.xml/g' "$file"
# done

# echo "Processing complete."

# Directory containing the files to process
directory="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/sc24/log/paper0/basic_msccl"

for file in "$directory"/*; do
    mv "$file" "${file//basic-binary_tree/basic-binary-tree}"
done

echo "Processing complete."

