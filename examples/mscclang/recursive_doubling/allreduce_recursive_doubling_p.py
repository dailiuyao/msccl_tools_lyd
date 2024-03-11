# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


# reduce from 0 -> num_nodes-1
def intra_reduce(num_nodes=0, node_offset=0, num_local_gpus=4, gpu_index=[0,0,0,0], chunk_step_total=0, ch_idx=0, nchunks_per_pair=0):
    rank_offset = node_offset * num_local_gpus
    for ch_offset in range(nchunks_per_pair):
        for index in range(0, num_local_gpus-1):
            other = chunk(int((gpu_index[index])+rank_offset), Buffer.input, chunk_step_total*nchunks_per_pair+ch_offset, size = 1)
            c1 = chunk(int((gpu_index[index+1])+rank_offset), Buffer.input, chunk_step_total*nchunks_per_pair+ch_offset, size = 1) 
            c1.reduce(other, ch=ch_idx+ch_offset)

# broadcast from num_nodes-1 -> 0       
def intra_broadcast(num_nodes=0, node_offset=0, num_local_gpus=4, gpu_index=[0,0,0,0], chunk_step_total=0, ch_idx=0, nchunks_per_pair=0):    
    rank_offset = node_offset * num_local_gpus
    for ch_offset in range(nchunks_per_pair):
        for index in range(0, num_local_gpus-1):
            c = chunk(int((gpu_index[num_local_gpus - 1 - index])%num_local_gpus + rank_offset), Buffer.input, chunk_step_total*nchunks_per_pair+ch_offset, size = 1)
            c.copy(int((gpu_index[num_local_gpus - 2 - index])%num_local_gpus + rank_offset), Buffer.input, chunk_step_total*nchunks_per_pair+ch_offset, ch=ch_idx+ch_offset)


def reduce_distance_doubling(size, chunk_step_total, ch_idx, num_gpus, gpu_index=[0,0,0,0], nchunks_per_pair=0):
    count = 1
    while count < size:
        for rank in range(size):
            peer = rank ^ count

            if rank < peer:
                index_offset_in_pair = 0
                channel_offset_in_pair = 0
            else:
                index_offset_in_pair = 1
                channel_offset_in_pair = 1

            c1 = chunk(rank*num_gpus+gpu_index[num_gpus-1], Buffer.input, chunk_step_total*nchunks_per_pair+index_offset_in_pair, size=1)
            chunk(peer*num_gpus+gpu_index[num_gpus-1], Buffer.input, chunk_step_total*nchunks_per_pair+index_offset_in_pair, size=1).reduce(c1, ch=ch_idx+channel_offset_in_pair)

            chunk(peer*num_gpus+gpu_index[num_gpus-1], Buffer.input, chunk_step_total*nchunks_per_pair+index_offset_in_pair, size=1).copy(rank*num_gpus+gpu_index[num_gpus-1], Buffer.input, chunk_step_total*nchunks_per_pair+index_offset_in_pair, ch=ch_idx+channel_offset_in_pair) 


        count *= 2


def chunk_allreduce(num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0, nchunks_per_pair=0):    
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel 
        channel_index = nchunks_per_pair * channel_total
        for node in range(0, num_nodes):
            intra_reduce(num_nodes=num_nodes, node_offset=node, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_index, nchunks_per_pair=nchunks_per_pair)

        reduce_distance_doubling(size=num_nodes, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_index, num_gpus=num_gpus, nchunks_per_pair=nchunks_per_pair)

        for node in range(0, num_nodes):
            intra_broadcast(num_nodes=num_nodes, node_offset=node, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_index, nchunks_per_pair=nchunks_per_pair)

        
def allreduce_recursive_doubling(num_gpus: int, num_nodes: int, nchunks: int, nchannel: int, instances: int, protocol: str):
    
    if (nchannel == 1):
        trees=1
    else:
        trees=2
    
    size = num_nodes * num_gpus
    num_chunks_per_channel = nchunks
    num_channel_per_tree= nchannel
    nchunks_per_pair = 2
    nchunks_total = nchunks * nchannel * trees * nchunks_per_pair
    
    
    topology = fully_connected(size)
    collective = AllReduce(size, nchunks_total, True)
    with MSCCLProgram("allreduce_recursive_doubling", topology, collective, instances, protocol):

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
                        
            chunk_allreduce(num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step, nchunks_per_pair)
                                
        if trees == 2:
            for chunk_step in range(0, num_chunks_per_channel):
                tree_id = 1
                
                chunk_allreduce(num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step, nchunks_per_pair)
                                                             
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

allreduce_recursive_doubling(args.num_gpus, args.num_nodes, args.nchunks, args.nchannel ,args.instances, args.protocol)