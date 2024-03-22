import os
import re
import csv

def parse_and_split_to_csvs(text_file_path, output_dir):
    # Regular expression to match the start of a new block
    start_block_re = re.compile(r"NCCL version 2.12.12.MSCCL.0.7.4+cuda11.8")

    # Read the file content
    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    # Temporary storage for current block's data and filename
    current_block = []
    current_filename = None

    for line in lines:
        # Check if the line is the start of a new block
        match = start_block_re.match(line)
        if match:
            # If there's an existing block, write it to a CSV
            if current_block and current_filename:
                write_block_to_csv(current_block, current_filename, output_dir)

            # Start a new block
            current_block = []
            xml_file_path = match.group(1)
            current_filename = construct_new_filename(xml_file_path)

        # If it's a data line, add it to the current block
        elif line.strip() and not line.startswith('#'):
            # Split the line by whitespace and add it to the current block
            split_line = re.split(r'\s+', line.strip())
            current_block.append(split_line)

    # Write the last block if it exists
    if current_block and current_filename:
        write_block_to_csv(current_block, current_filename, output_dir)

def construct_new_filename(old_filename):
    # Extract parts of the old filename using regex
    match = re.search(r'allreduce_binary_tree_(\d+)ch_(\d+)tree_(\d+)chunk_(\d+)node_(\d+)gpu', old_filename)
    if not match:
        raise ValueError(f"Filename {old_filename} did not match the expected pattern.")

    # Construct the new filename
    channels, trees, chunks, nodes, gpus = match.groups()
    new_filename = f"all-reduce_sum_float_binary-tree_node{nodes}_gpu{gpus}_mcl{channels}_mck{chunks}_i0.csv"
    return new_filename

def write_block_to_csv(block, filename, output_dir):
    csv_file_path = os.path.join(output_dir, filename)
    # Write the data to a CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(block)
    print(f"Data written to {csv_file_path}")

# Paths to the input text file and the output directory for the CSV files


text_file_path = '/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/polaris-test/sc24/log/paper0/buffer_size_2/ccl-4nodes-tree.out'  # Replace with the actual path to your source file
output_dir = '/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/polaris-test/sc24/log/paper0/csv/tree'    # Replace with the actual path to your output directory

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Call the function
parse_and_split_to_csvs(text_file_path, output_dir)
