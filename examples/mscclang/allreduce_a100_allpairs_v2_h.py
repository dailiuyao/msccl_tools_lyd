# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce
import math

# intra node allreduce using ring algorithm
def ring_reduce(size, rank_offset=0, local_chunk_size=16):
    for step in range(0, size-1):
        other = chunk((size-1-step) +rank_offset, Buffer.input, 0, local_chunk_size)
        c = chunk((size-1-step-1)+rank_offset, Buffer.input, 0, local_chunk_size)
        c.reduce(other, sendtb=step)
        
def ring_broadcast(size, rank_offset=0, local_chunk_size=16):
    for step in range(0, size-1):
        c = chunk(step, Buffer.input, 0, local_chunk_size)
        c.copy(step+1, Buffer.input, 0, local_chunk_size, sendtb=step+1, recvtb=step)

def allreduce_allpairs(num_nodes, num_local_gpus, instances, protocol):
    num_nodes=num_nodes
    num_local_gpus=num_local_gpus
    size = num_nodes*num_local_gpus
    chunksperloop = num_nodes
    topology = fully_connected(size)
    collective = AllReduce(size, chunksperloop, True)
    with MSCCLProgram("allreduce_pairs", topology, collective, instances, protocol=protocol, 
        interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual, dependence_nop=False):
        
        # intra node reduce
        for n in range(num_nodes):
            ring_reduce(size=num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes)
        # Each rank sends the nth chunk to the nth rank into scratch space
        for r1 in range(num_nodes):
            for r2 in range(num_nodes):
                if r1 != r2:
                    index = r2 
                    c = chunk(r1*num_local_gpus, Buffer.input, index)
                    c.copy(r2*num_local_gpus, 'scratch', sendtb=r2, recvtb=r1)

        # Each rank performs a local reduction on the nth chunk
        # Utilize 8 threadblocks for this reduction for better parallelism
        for r in range(num_nodes):
            for k in range(1,int(math.log2(num_nodes)+1)):
              level = 2**k
              for index in range(0, num_nodes//level):
                    if index == 0:
                        c = chunk(r*num_local_gpus, Buffer.input, r)
                    else:
                        c = chunk(r*num_local_gpus, 'scratch', (index-1))
                    c.reduce(chunk(r*num_local_gpus, 'scratch', (index+num_nodes//level-1)), sendtb=index)
                    #c = chunk(r, Buffer.input, r*size + (index % size))
                    #c.reduce(chunk(r, 'scratch', index), sendtb=(index % size))
        
        # Each rank sends the fully reduced nth chunk to all other gpus
        for r1 in range(num_nodes):
            for r2 in range(num_nodes):
                if r1 != r2:
                    index = r1
                    c = chunk(r1*num_local_gpus, Buffer.input, index)
                    c.copy(r2*num_local_gpus, Buffer.input, index, sendtb=r2, recvtb=r1)
        
        # intra node broadcast
        for n in range(num_nodes):
            ring_broadcast(size=num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=num_nodes) 
        
                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL128', 'LL'], help='Protocol')

args = parser.parse_args()

allreduce_allpairs(args.num_nodes, args.num_gpus, args.instances, args.protocol)
