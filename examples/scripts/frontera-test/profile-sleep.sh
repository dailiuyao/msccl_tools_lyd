#!/bin/bash

#SBATCH -J ccl-run           # Job name
#SBATCH -o ./log/ccl-run-common.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-common.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 1             # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 23:59:59        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-common       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

# export IBRUN_TASKS_PER_NODE=1
# readarray -t nodes <<< "$(ibrun -np 2 hostname)"

# echo "${nodes[0]}" > /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes
# echo "${nodes[1]}" >> /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/frontera-test/myhostfile_2nodes

# echo "Running on nodes: ${nodes[0]} and ${nodes[1]}"



sleep 1000000000000000000
