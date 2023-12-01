#!/bin/bash

# The file containing the log lines
input_file="/home/ldai8/scratch/msccl_build/deps/msccl-tools-lyd/examples/scripts/pinnacles-test/nccl-0.out"

# The file where the output will be saved
output_file="nccl_lines_one_iteration_bid0_32.txt"

# Use awk to output lines after line 39972
awk 'NR >= 14336 && NR <= 53512 { print > "'"$output_file"'" }' "$input_file"


# The file containing the log lines
input_file="nccl_lines_one_iteration_pro_0.txt"

output_file="nccl_lines_one_iteration_pro_0.txt"

# Use awk to process the lines, remove the tid part, and count occurrences
awk '
  # Match lines that contain the pattern
  /Reduce up, recv reduce send|Broadcast down, directRecvCopySend|Reduce up, sned |Broadcast down, directRecv |reduce and broadcast/ {
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