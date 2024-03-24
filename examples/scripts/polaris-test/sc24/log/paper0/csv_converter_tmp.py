import os
import re
import csv

def parse_and_split_to_csvs(text_file_path, output_dir):
    # Regular expression to match the start of a new block
    start_block_re = re.compile(r"The MSCCL XML file exists: (.+\.xml)")
    # Regular expression to match and skip the NCCL version line
    skip_line_re = re.compile(r"NCCL version 2\.12\.12\.MSCCL\.0\.7\.4\+cuda11\.8")
    # Regular expression to match and skip the specific unwanted line
    skip_running_test_line_re = re.compile(r"Running MSCCL tree test with")
    
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

        # If it's a line to be skipped, continue to the next iteration
        elif skip_line_re.match(line) or skip_running_test_line_re.match(line):
            continue
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


# import os
# import re
# import csv

# def parse_and_split_to_csvs(text_file_path, output_dir):
#     # Regular expression to match the start of a new block
#     start_block_re = re.compile(r"The MSCCL XML file exists: (.+\.xml)")
#     # Regular expression to find the starting line of headers
#     header_start_re = re.compile(r"#\s+out-of-place\s+in-place")

#     # Read the file content
#     with open(text_file_path, 'r') as file:
#         lines = file.readlines()

#     # Temporary storage for current block's data and filename
#     current_block = []
#     current_filename = None
#     # Flag to start capturing lines after the header line is found
#     capture_lines = False

#     for line in lines:
#         # Check if the line is the start of a new block
#         match = start_block_re.match(line)
#         if match:
#             # If there's an existing block, write it to a CSV
#             if current_block and current_filename:
#                 write_block_to_csv(current_block, current_filename, output_dir)

#             # Start a new block
#             current_block = []
#             xml_file_path = match.group(1)
#             current_filename = construct_new_filename(xml_file_path)
#             capture_lines = False  # Reset capturing for the new block

#         # Check for the header start line to begin capturing
#         elif header_start_re.search(line):
#             capture_lines = True
#             current_block.append(["#", "size", "count", "type", "redop", "root", "time", "algbw", "busbw", "#wrong", "time", "algbw", "busbw", "#wrong"])

#         # If it's a line to be captured, add it to the current block
#         elif capture_lines:
#             # Split the line by whitespace and add it to the current block
#             split_line = re.split(r'\s+', line.strip())
#             current_block.append(split_line)

#     # Write the last block if it exists
#     if current_block and current_filename:
#         write_block_to_csv(current_block, current_filename, output_dir)

# def construct_new_filename(old_filename):
#     # Extract parts of the old filename using regex
#     match = re.search(r'allreduce_binary_tree_(\d+)ch_(\d+)tree_(\d+)chunk_(\d+)node_(\d+)gpu', old_filename)
#     if not match:
#         raise ValueError(f"Filename {old_filename} did not match the expected pattern.")

#     # Construct the new filename
#     channels, trees, chunks, nodes, gpus = match.groups()
#     new_filename = f"all-reduce_sum_float_binary-tree_node{nodes}_gpu{gpus}_mcl{channels}_mck{chunks}_i0.csv"
#     return new_filename

# def write_block_to_csv(block, filename, output_dir):
#     csv_file_path = os.path.join(output_dir, filename)
#     # Write the data to a CSV file
#     with open(csv_file_path, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerows(block)
#     print(f"Data written to {csv_file_path}")

# # Paths to the input text file and the output directory for the CSV files
# text_file_path = '/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/polaris-test/sc24/log/paper0/buffer_size_2/ccl-4nodes-tree.out'  # Replace with the actual path to your source file
# output_dir = '/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/polaris-test/sc24/log/paper0/csv/tree'    # Replace with the actual path to your output directory

# # Create the output directory if it does not exist
# os.makedirs(output_dir, exist_ok=True)

# # Call the function
# parse_and_split_to_csvs(text_file_path, output_dir)

