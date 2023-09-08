# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_trinomial_tree(size, instances, trees, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, trees, True)
    with MSCCLProgram("allreduce_trinomial_tree", topology, collective, instances, protocol=protocol):
        # reduce-tree0
        c1 = chunk(1, Buffer.input, 0)
        chunk(0, Buffer.input, 0).reduce(c1, 0)
        c1 = chunk(2, Buffer.input, 0)
        chunk(0, Buffer.input, 0).reduce(c1, 0)
        # broadcast-tee0
        chunk(0, Buffer.input, 0).copy(1, Buffer.input, 0)
        chunk(0, Buffer.input, 0).copy(2, Buffer.input, 0)
        
        if trees == 2:
            # reduce-tree1
            c1 = chunk(0, Buffer.input, 1)
            chunk(1, Buffer.input, 1).reduce(c1)
            c1 = chunk(2, Buffer.input, 1)
            chunk(1, Buffer.input, 1).reduce(c1)
            # broadcast-tree1
            chunk(1, Buffer.input, 1).copy(0, Buffer.input, 1)
            chunk(1, Buffer.input, 1).copy(2, Buffer.input, 1)

            # reduce-tree2
            c1 = chunk(0, Buffer.input, 2)
            chunk(2, Buffer.input, 2).reduce(c1)
            c1 = chunk(1, Buffer.input, 2)
            chunk(2, Buffer.input, 2).reduce(c1)
            # broadcast-tree1
            chunk(2, Buffer.input, 2).copy(0, Buffer.input, 2)
            chunk(2, Buffer.input, 2).copy(1, Buffer.input, 2)     
            # c1 = chunk(0, Buffer.input, 1)
            # chunk(2, Buffer.input, 1).reduce(c1)
            # c1 = chunk(1, Buffer.input, 1)
            # chunk(2, Buffer.input, 1).reduce(c1)
            # # broadcast-tree1
            # chunk(2, Buffer.input, 1).copy(0, Buffer.input, 1)
            # chunk(2, Buffer.input, 1).copy(1, Buffer.input, 1)         
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_trinomial_tree(args.num_gpus, args.instances, args.trees, args.protocol)