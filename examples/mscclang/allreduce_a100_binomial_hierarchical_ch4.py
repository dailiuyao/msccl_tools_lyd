# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# intra node allreduce using ring algorithm
def ring_reduce_scatter(size, rank_offset=0, local_chunk_size=1, pipe_size=2, ch_offset=0):
    for ch in range(0, size):
        index = ch * local_chunk_size
        for step in range(0, size-1):
            for pipe_step in range(0, pipe_size):
                other = chunk(((step+1+ch) % size) +rank_offset, Buffer.input, index*pipe_size+pipe_step, local_chunk_size)
                c = chunk(((step+2+ch) % size)+rank_offset, Buffer.input, index*pipe_size+pipe_step, local_chunk_size)
                c.reduce(other, ch=pipe_step+ch_offset)

def ring_all_gather(size, rank_offset=0, local_chunk_size=1, pipe_size=2, ch_offset=2):
    for ch in range(0, size):
        index = ch  * local_chunk_size
        for step in range(0, size-1):
            for pipe_step in range(0, pipe_size):
                c = chunk(((step+ch) % size) + rank_offset, Buffer.input, index*pipe_size+pipe_step, local_chunk_size)
                c.copy(((step+ch+1) % size) + rank_offset, Buffer.input, index*pipe_size+pipe_step, local_chunk_size, ch=pipe_step+ch_offset)
                        

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_binomial_tree(num_nodes, num_local_gpus, instances, protocol):
    num_gpus = num_nodes * num_local_gpus
    pipe_size = 2
    chunk_size = pipe_size * num_local_gpus
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, chunk_size, True)
    with MSCCLProgram("allreduce_binomial_tree", topology, collective, instances, protocol=protocol, interleaved_replication=False):
        
        # intra node reduce scatter
        for n in range(num_nodes):
            ring_reduce_scatter(size=num_local_gpus, rank_offset=n * num_local_gpus, pipe_size=pipe_size, ch_offset=0)
            
        
        # tree0
        # Reduce tree - reducing onto Rank 0
        distance = 1
        cross_node_chunk_size = int(num_local_gpus/2)           
        while distance <= num_nodes // 2:
            # Reduce onto the top neighbor that is distance away
            for rank in range(0, num_nodes, distance*2):
                peer = rank + distance
                for n in range(0, cross_node_chunk_size):                    
                    for pipe_step in range(0, pipe_size):    
                        c1 = chunk(peer*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step)
                        chunk(rank*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step).reduce(c1, ch=pipe_step+2*pipe_size)
            distance *= 2
             
        # Broadcast tree - root is Rank 0 
        distance = distance // 2
        while distance >= 1:
            # Copy to the bottom neighbor that is distance away
            for rank in range(0, num_nodes, distance*2):
                peer = rank + distance                   
                for n in range(0, cross_node_chunk_size):
                    for pipe_step in range(0, pipe_size):  
                        chunk(rank*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step).copy(peer*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step, ch=pipe_step+3*pipe_size) 
        
            distance = distance // 2
            
            
            
        # Mirrored version of the tree
        # Reduce tree - reducing onto Rank N-1
        distance = 1
        while distance <= num_nodes // 2:
            # Reduce onto the top neighbor that is distance away
            for rank in range(num_nodes-1, 0, -distance*2):
                peer = rank - distance
                for n in range(cross_node_chunk_size, cross_node_chunk_size*2):                    
                    for pipe_step in range(0, pipe_size):    
                            c1 = chunk(peer*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step)
                            chunk(rank*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step).reduce(c1, ch=pipe_step+4*pipe_size)
            distance *= 2
             
        # Broadcast tree - root is Rank N-1
        distance = distance // 2
        while distance >= 1:
            # Copy to the bottom neighbor that is distance away
            for rank in range(num_nodes-1, 0, -distance*2):
                peer = rank - distance
                for n in range(cross_node_chunk_size, cross_node_chunk_size*2):
                    for pipe_step in range(0, pipe_size):  
                        chunk(rank*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step).copy(peer*num_local_gpus+n, Buffer.input, n*pipe_size+pipe_step, ch=pipe_step+5*pipe_size) 
            distance = distance // 2
            
                      
        # intra node all gather
        for n in range(num_nodes):
            ring_all_gather(size=num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=1, pipe_size=pipe_size, ch_offset=pipe_size) 

        

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
# parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_binomial_tree(args.num_nodes, args.num_gpus, args.instances, args.protocol)
