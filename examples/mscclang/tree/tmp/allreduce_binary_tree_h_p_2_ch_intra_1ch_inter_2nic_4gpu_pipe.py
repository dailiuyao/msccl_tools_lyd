# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf

# intra node allreduce using ring algorithm
def ring_reduce_scatter(size, rank_offset=0, local_chunk_size=1, pipe_size=4, ch_offset=0, ch_size=2):
    pipe_size_ptree = int(pipe_size/2)
    for ch in range(0, size):
        index = ch * local_chunk_size
        for step in range(0, size-1):
            for pipe_step in range(0, pipe_size_ptree):
                other = chunk(((step+1+index) % size) +rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size)
                c = chunk(((step+2+index) % size)+rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size)
                c.reduce(other, ch=(pipe_step%ch_size)+ch_offset)
    for index in range(0, size, 2):
        for pipe_step in range(0,pipe_size_ptree):
            c = chunk((index % size) + rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size)
            c.copy(((index+1) % size) + rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size, ch=(pipe_step%ch_size)+ch_offset)



def ring_all_gather(size, rank_offset=0, local_chunk_size=1, pipe_size=4, ch_offset=2, ch_size=2):
    pipe_size_ptree = int(pipe_size/2)
    for index in range(0, size, 2):
        for pipe_step in range(0,pipe_size_ptree):
            c = chunk(((index+1) % size) + rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size)
            c.copy((index % size) + rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size, ch=(pipe_step%ch_size)+ch_offset)
        for pipe_step in range(0,pipe_size_ptree):
            c = chunk(((index+1) % size) + rank_offset, Buffer.input, (index+1)*pipe_size_ptree+pipe_step, local_chunk_size)
            c.copy(((index+2) % size) + rank_offset, Buffer.input, (index+1)*pipe_size_ptree+pipe_step, local_chunk_size, ch=(pipe_step%ch_size)+ch_offset)

    for ch in range(0, size):
        index = ch  * local_chunk_size
        for step in range(1, size-1):
            for pipe_step in range(0, pipe_size_ptree):
                c = chunk(((step+index) % size) + rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size)
                c.copy(((step+index+1) % size) + rank_offset, Buffer.input, index*pipe_size_ptree+pipe_step, local_chunk_size, ch=(pipe_step%ch_size)+ch_offset)


def allreduce_binary_tree_hierarchical(num_nodes, num_local_gpus, num_chunks, num_channel, instances, protocol):
    trees=2
    size = num_nodes * num_local_gpus
    pipe_size = int(num_chunks/trees)
    chunk_size = num_chunks
    num_channel_per_stage = int(num_channel/4)
    num_channel_inter_stage = int(num_channel_per_stage/2)
    topology = fully_connected(size)
    collective = AllReduce(size, chunk_size, True)
    with MSCCLProgram("allreduce_binary_tree_hierarchical", topology, collective, instances, protocol=protocol):

        # intra node reduce scatter
        for n in range(num_nodes):
            ring_reduce_scatter(size=num_local_gpus, rank_offset=n * num_local_gpus, pipe_size=pipe_size, ch_offset=0,ch_size=num_channel_per_stage)

        inter_ngpu_ptree = int(num_local_gpus/4) 
        # Reduce tree - reducing onto Rank 0
        inter_gpu_offset = 1
        num_level_ori = math.log(num_nodes,2)
        num_level = math.ceil(num_level_ori) - 1
        step = 2
        while step <= 2**num_level:       
            for rank in range(step, num_nodes, step*2):
                bit = 1
                while bit < num_nodes:            
                    if bit & rank:
                        break
                    bit *= 2   
                
                low_bit = bit // 2
                peer_0 = rank - low_bit
                for pipe_step in range(0, pipe_size):    
                    c1 = chunk(peer_0*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step)
                    chunk(rank*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step).reduce(c1, ch=0*num_channel_inter_stage+2*num_channel_per_stage)
                
                peer_1 = rank + low_bit
                while peer_1 >= num_nodes:
                    peer_1 = rank + low_bit
                    low_bit //= 2
                for pipe_step in range(0, pipe_size):     
                    c1 = chunk(peer_1*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step)
                    chunk(rank*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step).reduce(c1, ch=0*num_channel_inter_stage+2*num_channel_per_stage)
            step *= 2
            
        peer_0 = 1
        while peer_0 < num_nodes:            
            peer_0 *= 2
     
        peer_0 //= 2
        for pipe_step in range(0, pipe_size):   
            c1 = chunk(peer_0*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step)
            chunk(0*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step).reduce(c1, ch=0*num_channel_inter_stage+2*num_channel_per_stage)         
                
         # Broadcast tree - root is Rank 0
        for pipe_step in range(0, pipe_size):  
            chunk(0*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step).copy(peer_0*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step, ch=1*num_channel_inter_stage+2*num_channel_per_stage) 
        
        step = 2**num_level
        while step >= 2:       
            for rank in range(step, num_nodes, step*2):
                bit = 1
                while bit < num_nodes:            
                    if bit & rank:
                        break
                    bit *= 2   
                
                low_bit = bit // 2
                peer_0 = rank - low_bit
                for pipe_step in range(0, pipe_size):   
                    chunk(rank*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step).copy(peer_0*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step, ch=1*num_channel_inter_stage+2*num_channel_per_stage) 
                
                peer_1 = rank + low_bit
                while peer_1 >= num_nodes:
                    peer_1 = rank + low_bit
                    low_bit //= 2
                for pipe_step in range(0, pipe_size):   
                    chunk(rank*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step).copy(peer_1*num_local_gpus+inter_gpu_offset, Buffer.input, 0*pipe_size+pipe_step, ch=1*num_channel_inter_stage+2*num_channel_per_stage) 
            step //= 2

        # Mirrored version of the second tree for even ranks
        # Reduce tree - reducing onto Rank N-1
        
        inter_gpu_offset = 3
        if (trees == 2) and (num_nodes % 2 == 0):
            # Reduce tree - reducing onto Rank N-1
            num_level_ori = math.log(num_nodes,2)
            num_level = math.ceil(num_level_ori) - 1
            step = 2
            while step <= 2**num_level:       
                for rank in range(step, num_nodes, step*2):
                    bit = 1
                    while bit < num_nodes:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    for pipe_step in range(0, pipe_size):    
                        c1 = chunk((num_nodes-1-peer_0)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step)
                        chunk((num_nodes-1-rank)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step).reduce(c1, ch=0*num_channel_inter_stage+3*num_channel_per_stage)
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= num_nodes:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    for pipe_step in range(0, pipe_size):  
                        c1 = chunk((num_nodes-1-peer_1)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step)
                        chunk((num_nodes-1-rank)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step).reduce(c1, ch=0*num_channel_inter_stage+3*num_channel_per_stage)
                step *= 2
                
            peer_0 = 1
            while peer_0 < num_nodes:            
                peer_0 *= 2
        
            peer_0 //= 2
            for pipe_step in range(0, pipe_size):  
                c1 = chunk((num_nodes-1-peer_0)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step)
                chunk((num_nodes-1-0)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step).reduce(c1, ch=0*num_channel_inter_stage+3*num_channel_per_stage)         
                    
            # Broadcast tree - root is Rank N-1
            for pipe_step in range(0, pipe_size): 
                chunk((num_nodes-1-0)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step).copy((num_nodes-1-peer_0)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step, ch=1*num_channel_inter_stage+3*num_channel_per_stage) 
            
            step = 2**num_level
            while step >= 2:       
                for rank in range(step, num_nodes, step*2):
                    bit = 1
                    while bit < num_nodes:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    for pipe_step in range(0, pipe_size):  
                        chunk((num_nodes-1-rank)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step).copy((num_nodes-1-peer_0)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step, ch=1*num_channel_inter_stage+3*num_channel_per_stage) 
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= num_nodes:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    for pipe_step in range(0, pipe_size): 
                        chunk((num_nodes-1-rank)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step).copy((num_nodes-1-peer_1)*num_local_gpus+inter_gpu_offset, Buffer.input, 1*pipe_size+pipe_step, ch=1*num_channel_inter_stage+3*num_channel_per_stage) 
                step //= 2
        
        # intra node all gather
        for n in range(num_nodes):
            ring_all_gather(size=num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=1, pipe_size=pipe_size, ch_offset=num_channel_per_stage, ch_size=num_channel_per_stage) 
        
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
parser.add_argument('nchunk', type=int, help ='number of chunks')
parser.add_argument('nchannel', type=int, help ='number of channels')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_binary_tree_hierarchical(args.num_nodes, args.num_gpus, args.nchunk, args.nchannel ,args.instances, args.protocol)