# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def calculate_index_vector_halving_distance_doubling(group_index, count, size):
    # Calculate the number of groups based on the current count
    num_groups = size / (count*2)
    # Calculate the size of each group
    group_size = size // num_groups
    for rank in range(size):
        # Calculate the position within the group
        position_in_group = rank % group_size
        # Adjust the position based on the count to generate the pattern
        group_index[rank] += (position_in_group//count)*num_groups

def calculate_index_vector_doubling_distance_halving(group_index, count, size):
    # Calculate the number of groups based on the current count
    num_groups = size / (count*2)
    # Calculate the size of each group
    group_size = size // num_groups
    for rank in range(size):
        # Calculate the position within the group
        position_in_group = rank % group_size
        # Adjust the position based on the count to generate the pattern
        group_index[rank] -= (position_in_group//count)*num_groups

def reduce_scatter_vector_halving_distance_doubling(size, group_index, chunk_step_total, gpu_index, ch_idx, num_gpus):
    count = 1
    while count < size:
        calculate_index_vector_halving_distance_doubling(group_index, count, size)
        size_count = int((size // 2)/count)
        for rank in range(size):
            peer = rank ^ count
            index = int(group_index[peer])
            # print(index)
            c1 = chunk(rank*num_gpus+gpu_index[num_gpus-1], Buffer.input, index+chunk_step_total, size=size_count)
            chunk(peer*num_gpus+gpu_index[num_gpus-1], Buffer.input, index+chunk_step_total, size=size_count).reduce(c1, ch=int(ch_idx))
        count *= 2

def allgather_recursive_vector_doubling_distance_halving(size, group_index, chunk_step_total, gpu_index, ch_idx, num_gpus):
    count = size // 2
    size_count = int((size // 2) / count)
    while count >= 1:
        # print(group_index)
        size_count = int((size // 2) / count)
        for rank in range(size):
            peer = rank ^ count
            index = int(group_index[rank])
            chunk(rank*num_gpus+gpu_index[num_gpus-1], Buffer.input, chunk_step_total+index, size=size_count).copy(peer*num_gpus+gpu_index[num_gpus-1], Buffer.input, chunk_step_total+index, ch=ch_idx) 
        count //= 2
        if (count != 0):
            calculate_index_vector_doubling_distance_halving(group_index, count*2, size)

# reduce from 0 -> num_nodes-1
def intra_reduce(num_nodes=0, node_offset=0, num_local_gpus=4, gpu_index=[0,0,0,0], chunk_step_total=0, ch_idx=0):
    rank_offset = node_offset * num_local_gpus
    for index in range(0, num_local_gpus-1):
        other = chunk(int((gpu_index[index])+rank_offset), Buffer.input, chunk_step_total, size = int(num_nodes))
        c1 = chunk(int((gpu_index[index+1])+rank_offset), Buffer.input, chunk_step_total, size = int(num_nodes)) 
        c1.reduce(other, ch=int(ch_idx))

# broadcast from num_nodes-1 -> 0       
def intra_broadcast(num_nodes=0, node_offset=0, num_local_gpus=4, gpu_index=[0,0,0,0], chunk_step_total=0, ch_idx=0):    
    rank_offset = node_offset * num_local_gpus
    for index in range(0, num_local_gpus-1):
        c = chunk(int((gpu_index[num_local_gpus - 1 - index])%num_local_gpus + rank_offset), Buffer.input, chunk_step_total, size = int(num_nodes))
        c.copy(int((gpu_index[num_local_gpus - 2 - index])%num_local_gpus + rank_offset), Buffer.input, chunk_step_total, ch=ch_idx)


def chunk_reduce(group_index, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):    
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel 
        for node in range(0, num_nodes):
            intra_reduce(num_nodes=num_nodes, node_offset=node, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)

        reduce_scatter_vector_halving_distance_doubling(num_nodes, group_index, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total, num_gpus=num_gpus)
        

def chunk_broadcast(group_index, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel  
        for node in range(0, num_nodes):
            intra_broadcast(num_nodes=num_nodes, node_offset=node, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 

        allgather_recursive_vector_doubling_distance_halving(num_nodes, group_index, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total, num_gpus=num_gpus)
        
def allreduce_recursive_doubling_halving(num_gpus, num_nodes, nchunks, nchannel, instances, protocol):
    
    if (nchannel == 1):
        trees=1
    else:
        trees=2
    
    size = num_nodes * num_gpus
    num_chunks_per_channel = int(nchunks / nchannel) 
    num_channel_per_tree=int(nchannel/trees)
    nchunks_total = nchunks * num_nodes
    
    topology = fully_connected(size)
    collective = AllReduce(size, nchunks_total, True)
    with MSCCLProgram("allreduce_recursive_doubling_halving", topology, collective, instances, protocol):

        # channel 0: 0->1->2->3
        # channel 1: 3->2->1->0
        # each tree has one channel
        # Reduce tree - reducing onto Rank 0
        gpu_index0 = list(range(0, num_gpus, 1))
        # gpu_index1 = gpu_index0
        gpu_index1 = list(reversed(gpu_index0))
        combined_indices = [gpu_index0, gpu_index1]
        
        for chunk_step in range(0, num_chunks_per_channel):
            tree_id = 0
            
            group_index = [0] * num_nodes
                        
            chunk_reduce(group_index, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)
                        
            chunk_broadcast(group_index, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step) 
        
        if trees == 2:
            for chunk_step in range(0, num_chunks_per_channel):
                tree_id = 1
                
                group_index = [0] * num_nodes
                
                chunk_reduce(group_index, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)
                                
                chunk_broadcast(group_index, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step) 
                             
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--nchunks', type=int, help ='number of chunks')
parser.add_argument('--num_gpus', type=int, help='number of gpus per node')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--nchannel', type=int, help ='number of channels')

parser.add_argument('--instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allreduce_recursive_doubling_halving(args.num_gpus, args.num_nodes, args.nchunks, args.nchannel ,args.instances, args.protocol)