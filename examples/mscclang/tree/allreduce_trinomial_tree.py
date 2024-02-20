# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
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
        num_level_ori = math.log(size,3)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 1
        step_parent_parent = 3
        current_level = 1
        while current_level <= num_level:
            for rank in range (0, size, step_parent_parent):
                if rank + step_parent_child < size:
                    c1 = chunk(rank + step_parent_child, Buffer.input, 0)
                    chunk(rank, Buffer.input, 0).reduce(c1, 0)
                if rank + step_parent_child*2 < size:
                    c1 = chunk(rank + step_parent_child*2, Buffer.input, 0)
                    chunk(rank, Buffer.input, 0).reduce(c1, 0)
            step_parent_child *= 3
            step_parent_parent *= 3
            current_level += 1
        # broadcast-tee0
        num_level_ori = math.log(size,3)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 3**(num_level-1)
        step_parent_parent = 3**num_level
        current_level = num_level
        while current_level >= 1:
            for rank in range (0, size, int(step_parent_parent)):
                if rank + step_parent_child < size:
                    chunk(rank, Buffer.input, 0).copy(rank + int(step_parent_child), Buffer.input, 0)
                if rank + step_parent_child*2 < size:
                    chunk(rank, Buffer.input, 0).copy(rank + int(step_parent_child*2), Buffer.input, 0)
            step_parent_child /= 3
            step_parent_parent /= 3
            current_level -= 1
        
        if trees == 3:
            # reduce-tree1
            num_level_ori = math.log(size,3)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 1
            step_parent_parent = 3
            current_level = 1
            while current_level <= num_level:
                if current_level == 1:
                    for rank in range (1, size, step_parent_parent):
                        if rank - step_parent_child < size:
                            c1 = chunk(rank - step_parent_child, Buffer.input, 1)
                            chunk(rank, Buffer.input, 1).reduce(c1)
                        if rank + step_parent_child < size:
                            c1 = chunk(rank + step_parent_child, Buffer.input, 1)
                            chunk(rank, Buffer.input, 1).reduce(c1)
                else:
                    for rank in range (1, size, step_parent_parent):
                        if rank + step_parent_child < size:
                            c1 = chunk(rank + step_parent_child, Buffer.input, 1)
                            chunk(rank, Buffer.input, 1).reduce(c1)
                        if rank + step_parent_child*2 < size:
                            c1 = chunk(rank + step_parent_child*2, Buffer.input, 1)
                            chunk(rank, Buffer.input, 1).reduce(c1)
                    # hardcode for 16 gpus
                    if (size == 16) and (current_level == 2):
                        c1 = chunk(15, Buffer.input, 1)
                        chunk(10, Buffer.input, 1).reduce(c1)
                step_parent_child *= 3
                step_parent_parent *= 3
                current_level += 1


            # broadcast-tee1
            num_level_ori = math.log(size,3)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 3**(num_level-1)
            step_parent_parent = 3**num_level
            current_level = num_level
            while current_level >= 1:
                if current_level != 1:
                    for rank in range (1, size, int(step_parent_parent)):
                        if rank + step_parent_child < size:
                            chunk(rank, Buffer.input, 1).copy(rank + int(step_parent_child), Buffer.input, 1)
                        if rank + step_parent_child*2 < size:
                            chunk(rank, Buffer.input, 1).copy(rank + int(step_parent_child*2), Buffer.input, 1)
                    # hardcode for 16 gpus
                    if (size == 16) and (current_level == 2):
                        chunk(10, Buffer.input, 1).copy(15, Buffer.input, 1)
                else:
                    for rank in range (1, size, int(step_parent_parent)):
                        if rank - step_parent_child < size:
                            chunk(rank, Buffer.input, 1).copy(rank - int(step_parent_child), Buffer.input, 1)
                        if rank + step_parent_child < size:
                            chunk(rank, Buffer.input, 1).copy(rank + int(step_parent_child), Buffer.input, 1)
                step_parent_child /= 3
                step_parent_parent /= 3
                current_level -= 1


            # reduce-tree2
            num_level_ori = math.log(size,3)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 1
            step_parent_parent = 3
            current_level = 1
            while current_level <= num_level:
                if current_level == 1:
                    for rank in range (2, size, step_parent_parent):
                        c1 = chunk(rank - step_parent_child, Buffer.input, 2)
                        chunk(rank, Buffer.input, 2).reduce(c1)
                        c1 = chunk(rank - step_parent_child*2, Buffer.input, 2)
                        chunk(rank, Buffer.input, 2).reduce(c1)
                    # hardcode for 8 gpus
                    if size == 8:
                        c1 = chunk(6, Buffer.input, 2)
                        chunk(7, Buffer.input, 2).reduce(c1)
                else:
                    for rank in range (2, size, step_parent_parent):
                        if rank + step_parent_child < size:
                            c1 = chunk(rank + step_parent_child, Buffer.input, 2)
                            chunk(rank, Buffer.input, 2).reduce(c1)
                        if rank + step_parent_child*2 < size:
                            c1 = chunk(rank + step_parent_child*2, Buffer.input, 2)
                            chunk(rank, Buffer.input, 2).reduce(c1)
                    # hardcode for 8 gpus
                    if size == 8:
                        c1 = chunk(7, Buffer.input, 2)
                        chunk(2, Buffer.input, 2).reduce(c1)
                    # hardcode for 16 gpus
                    if (size == 16) and (current_level == 2):
                        c1 = chunk(15, Buffer.input, 2)
                        chunk(11, Buffer.input, 2).reduce(c1)
                step_parent_child *= 3
                step_parent_parent *= 3
                current_level += 1
            # broadcast-tee2
            num_level_ori = math.log(size,3)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 3**(num_level-1)
            step_parent_parent = 3**num_level
            current_level = num_level
            while current_level >= 1:
                if current_level != 1:
                    for rank in range (2, size, int(step_parent_parent)):
                        if rank + step_parent_child < size:
                            chunk(rank, Buffer.input, 2).copy(rank + int(step_parent_child), Buffer.input, 2)
                        if rank + step_parent_child*2 < size:
                            chunk(rank, Buffer.input, 2).copy(rank + int(step_parent_child*2), Buffer.input, 2)
                    # hardcode for 8 gpus
                    if size == 8:
                        chunk(2, Buffer.input, 2).copy(7, Buffer.input, 2)
                    # hardcode for 16 gpus
                    if (size == 16) and (current_level == 2):
                        chunk(11, Buffer.input, 2).copy(15, Buffer.input, 2)
                else:
                    for rank in range (2, size, int(step_parent_parent)):
                        chunk(rank, Buffer.input, 2).copy(rank - int(step_parent_child), Buffer.input, 2)
                        chunk(rank, Buffer.input, 2).copy(rank - int(step_parent_child*2), Buffer.input, 2)
                    # hardcode for 8 gpus
                    if size == 8:
                        chunk(7, Buffer.input, 2).copy(6, Buffer.input, 2)
                step_parent_child /= 3
                step_parent_parent /= 3
                current_level -= 1

        
                   
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('trees', type=int, choices=[1, 2, 3], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_trinomial_tree(args.num_gpus, args.instances, args.trees, args.protocol)