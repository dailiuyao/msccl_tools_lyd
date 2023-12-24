#!/bin/bash

# Base directory for the input files
base_dir="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/original-nccl-output"

# Output directories
output_host_dir="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/original-nccl-host"
output_device_dir="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/original-nccl-device"

# Loop through files from nccl-0.out to nccl-15.out
for i in {0..15}
do
    # Construct input file name
    input_file="$base_dir/nccl-$i.out"

    # Construct output file names for HOST and DEVICE
    output_host_file="$output_host_dir/nccl-$i.txt"
    output_device_file="$output_device_dir/nccl-$i.txt"

    # Use awk to process each file and write lines containing "HOST |" to the output file for HOST
    awk '/HOST \|/ { print > "'"$output_host_file"'" }' "$input_file"

    # Use awk to process each file and write lines containing "DEVICE |" to the output file for DEVICE
    awk '/DEVICE \|/ { print > "'"$output_device_file"'" }' "$input_file"
done
