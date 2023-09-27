# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_2_gpus_tree(size, instances, trees, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, trees, True)
    with MSCCLProgram("allreduce_2_gpus_tree", topology, collective, instances, protocol=protocol):
        # Reduce tree - reducing onto Rank 0
        c1 = chunk(1, Buffer.input, 0)
        chunk(0, Buffer.input, 0).reduce(c1, 0)
        # Broadcast tree - root is Rank 0
        chunk(0, Buffer.input, 0).copy(1, Buffer.input, 0)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_2_gpus_tree(args.num_gpus, args.instances, args.trees, args.protocol)