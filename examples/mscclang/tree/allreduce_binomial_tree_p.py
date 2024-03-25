# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce


# reduce from 0 -> num_nodes-1
def intra_reduce(node_offset=0, num_local_gpus=4, gpu_index=[0,0,0,0], chunk_step_total=0, ch_idx=0):
    rank_offset = node_offset * num_local_gpus
    for index in range(0, num_local_gpus-1):
        other = chunk(int((gpu_index[index])+rank_offset), Buffer.input, chunk_step_total)
        c1 = chunk(int((gpu_index[index+1])+rank_offset), Buffer.input, chunk_step_total) 
        c1.reduce(other, ch=int(ch_idx))

# broadcast from num_nodes-1 -> 0       
def intra_broadcast(node_offset=0, num_local_gpus=4, gpu_index=[0,0,0,0], chunk_step_total=0, ch_idx=0):    
    rank_offset = node_offset * num_local_gpus
    for index in range(0, num_local_gpus-1):
        c = chunk(int((gpu_index[num_local_gpus - 1 - index])%num_local_gpus + rank_offset), Buffer.input, chunk_step_total)
        c.copy(int((gpu_index[num_local_gpus - 2 - index])%num_local_gpus + rank_offset), Buffer.input, chunk_step_total, ch=ch_idx)


def chunk_reduce(rank=0, child=0, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):    
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel 
        if child < num_nodes:
            intra_reduce(node_offset=child, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)
            # the children node child_0 is ready, reduce to the last gpu in the parent node rank. 
            c1 = chunk(int(child*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total)
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).reduce(c1, ch=channel_total)
        

def chunk_broadcast(rank=0, child=0, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel  
        if child < num_nodes:
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).copy(int(child*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total, ch=channel_total)
            intra_broadcast(node_offset=child, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 

        
        
# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_binomial_tree(num_gpus, num_nodes, nchunks, nchannel, instances, protocol, trees):
    
    # if (nchannel == 1):
    #     trees=1
    # else:
    #     trees=2
    
    size = num_nodes * num_gpus
    num_chunks_per_channel = nchunks
    num_channel_per_tree=nchannel
    num_total_chunks = nchunks * nchannel * trees 
    
    topology = fully_connected(size)
    collective = AllReduce(size, num_total_chunks, True)
    with MSCCLProgram("allreduce_binomial_tree", topology, collective, instances, protocol=protocol):
        
        # channel 0: 0->1->2->3
        # channel 1: 3->2->1->0
        # each tree has one channel
        # Reduce tree - reducing onto Rank 0
        # gpu_index0 = list(range(0, num_gpus, 1))
        # # gpu_index1 = gpu_index0
        # gpu_index1 = list(reversed(gpu_index0))
        # gpu_index2 = [2,3,0,1]
        # gpu_index3 = [1,0,3,2]
        # combined_indices_0 = [gpu_index0, gpu_index0]
        # combined_indices_1 = [gpu_index1, gpu_index1]
        # combined_indices_2 = [gpu_index2, gpu_index2]
        # combined_indices_3 = [gpu_index3, gpu_index3]

        gpu_indices = []
        gpu_indices.append([7,6,5,4,3,2,1,0])  # gpu_index0
        gpu_indices.append([0,7,6,5,4,3,2,1])  # gpu_index1
        
        
        combined_indices_0 = [gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1]]
        combined_indices_1 = [gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1], gpu_indices[0], gpu_indices[1]]

        
        for chunk_step in range(0, num_chunks_per_channel):
            tree_id = 0
            distance = 1
            # Reduce tree - reducing onto Rank 0
            while distance <= num_nodes // 2:
                # Reduce onto the left neighbor that is distance away
                for rank in range(0, num_nodes, distance*2):
                    child = rank + distance
                    chunk_reduce(rank, child, num_gpus, num_nodes, combined_indices_0, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    
                distance *= 2
            
            # root node reduce of tree 0
            for channel in range(num_channel_per_tree):
                channel_total = channel+tree_id*num_channel_per_tree
                chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                intra_reduce(node_offset=0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_0[channel], ch_idx=channel_total)
            
            # root node broadcast of tree 0
            for channel in range(num_channel_per_tree):
                channel_total = channel+tree_id*num_channel_per_tree
                chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                intra_broadcast(node_offset=0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_0[channel], ch_idx=channel_total) 
                         
            
            # Broadcast tree - root is Rank 0
            distance = distance // 2
            while distance >= 1:
                # Copy to the right neighbor that is distance away
                for rank in range(0, num_nodes, distance*2):
                    child = rank + distance
                    chunk_broadcast(rank, child, num_gpus, num_nodes, combined_indices_0, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)
                distance = distance // 2
            

        # Mirrored version of the tree
        # Reduce tree - reducing onto Rank N-1
        if trees == 2:
            for chunk_step in range(0, num_chunks_per_channel):
                tree_id = 1
                distance = 1
                while distance <= num_nodes // 2:
                    # Reduce onto the right neighbor that is distance away
                    for rank in range(num_nodes-1, 0, -distance*2):
                        child = rank - distance
                        chunk_reduce(rank, child, num_gpus, num_nodes, combined_indices_1, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    distance *= 2
                
                # root node reduce of tree 1
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                    intra_reduce(node_offset=num_nodes-1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_1[channel], ch_idx=channel_total)
                
                # root node broadcast of tree 0
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                    intra_broadcast(node_offset=num_nodes-1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_1[channel], ch_idx=channel_total) 
                         
                
                
                # Broadcast tree - root is Rank N-1
                distance = distance // 2
                while distance >= 1:
                    # Copy to the left neighbor that is distance away
                    for rank in range(num_nodes-1, 0, -distance*2):
                        child = rank - distance
                        chunk_broadcast(rank, child, num_gpus, num_nodes, combined_indices_1, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)
                    distance = distance // 2

        XML()
        Check()

parser = argparse.ArgumentParser()


parser.add_argument('--nchunks', type=int, help ='number of chunks')
parser.add_argument('--num_gpus', type=int, help='number of gpus per node')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--nchannel', type=int, help ='number of channels')
parser.add_argument('--trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('--instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allreduce_binomial_tree(args.num_gpus, args.num_nodes, args.nchunks, args.nchannel ,args.instances, args.protocol, args.trees)