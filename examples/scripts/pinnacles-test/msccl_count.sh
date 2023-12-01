#!/bin/bash

# The file containing the log lines
input_file="/home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/scripts/pinnacles-test/msccl-0.out"

# The file where the output will be saved
output_file="msccl_lines_one_iteration_pro_0.txt"

# Use awk to output lines after line 39972
awk 'NR >= 55338 && NR <= 83000 { print > "'"$output_file"'" }' "$input_file"


# The file containing the log lines
input_file="msccl_lines_one_iteration_pro_0.txt"

output_file="msccl_lines_one_iteration_pro_0.txt"

# Use awk to process the lines, remove the tid part, and count occurrences
awk '
  # Match lines that contain the pattern
  /MSCCL MSCCL_SEND |MSCCL MSCCL_RECV |MSCCL MSCCL_RECV |MSCCL MSCCL_RECV_COPY_SEND |MSCCL MSCCL_RECV_REDUCE_SEND |MSCCL MSCCL_RECV_REDUCE_COPY_SEND|MSCCL MSCCL_RECV_REDUCE_COPY |MSCCL MSCCL_LOCAL_COPY / {
    # Remove the tid=XXX part
    gsub(/, tid=[0-9]+/, "")
    # Increment the count for the modified line
    count[$0]++
    if (count[$0] == 1) {
      order[++n] = $0
    }
  }
  END {
    # Output the lines in the order they appeared without the tid part
    for (i = 1; i <= n; i++) {
      print order[i] " - " count[order[i]] " times" > "'"$output_file"'"
    }
  }
' "$input_file"