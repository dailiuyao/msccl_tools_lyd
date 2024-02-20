# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
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


def chunk_reduce(rank=0, child_0=0, child_1=0, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):    
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel 
        if child_0 < num_nodes:
            intra_reduce(node_offset=child_0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)
            # the children node child_0 is ready, reduce to the last gpu in the parent node rank. 
            c1 = chunk(int(child_0*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total)
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).reduce(c1, ch=channel_total)
        
        if child_1 < num_nodes:
            intra_reduce(node_offset=child_1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)
            # the children node child_0 is ready, reduce to the last gpu in the parent node rank. 
            c1 = chunk(int(child_1*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total)
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).reduce(c1, ch=channel_total)

def chunk_broadcast(rank=0, child_0=0, child_1=0, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel  
        if child_0 < num_nodes:
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).copy(int(child_0*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total, ch=channel_total)
            intra_broadcast(node_offset=child_0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 

        if child_1 < num_nodes:
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).copy(int(child_1*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total, ch=channel_total)
            intra_broadcast(node_offset=child_1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 
                

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf



def allreduce_trinomial_tree(num_gpus, num_nodes, nchunks, nchannel, instances, protocol):
    
    if (nchannel == 1):
        trees=1
    else:
        trees=3

    size = num_nodes * num_gpus
    num_chunks_per_channel = int(nchunks / nchannel) 
    num_channel_per_tree=int(nchannel/trees)
    
    topology = fully_connected(size)
    collective = AllReduce(size, nchunks, True)
    with MSCCLProgram("allreduce_trinomial_tree", topology, collective, instances, protocol=protocol):

        # tree0: channel0 0->1->2->3
        # tree1: channel2 3->2->1->0
        # each tree has one channel
        # Reduce tree - reducing onto Rank 0
        gpu_index0 = list(range(0, num_gpus, 1))
        # gpu_index1 = gpu_index0
        gpu_index1 = list(reversed(gpu_index0))
        combined_indices = [gpu_index0, gpu_index1]


        for chunk_step in range(0, num_chunks_per_channel):
            # reduce-tree0
            tree_id = 0
            num_level_ori = math.log(num_nodes,3)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 1
            step_parent_parent = 3
            current_level = 1
            # level is the parent node level
            # current_level: 1 -> num_level
            # step_parent_child: 1 -> 3^(num_level-1)
            # step_parent_parent: 3 -> 3^num_level
            while current_level <= num_level:
                for rank in range (0, num_nodes, step_parent_parent):

                    child_0 = rank + step_parent_child
                    child_1 = rank + step_parent_child*2

                    chunk_reduce(rank, child_0, child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                        
                step_parent_child *= 3
                step_parent_parent *= 3
                current_level += 1
            
            # root node reduce of tree0
            for channel in range(num_channel_per_tree):
                channel_total = channel+tree_id*num_channel_per_tree
                chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                intra_reduce(node_offset=0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)

            # broadcast-tee0
            num_level_ori = math.log(num_nodes,3)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 3**(num_level-1)
            step_parent_parent = 3**num_level
            current_level = num_level

            # root node broadcast of tree0
            for channel in range(num_channel_per_tree):
                channel_total = channel+tree_id*num_channel_per_tree
                chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                intra_broadcast(node_offset=0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 
                            
            while current_level >= 1:
                for rank in range (0, num_nodes, int(step_parent_parent)):

                    child_0 = rank + step_parent_child
                    child_1 = rank + step_parent_child*2

                    chunk_broadcast(rank, child_0, child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)

                step_parent_child /= 3
                step_parent_parent /= 3
                current_level -= 1
        
        if trees == 3:
            for chunk_step in range(0, num_chunks_per_channel):
                # reduce-tree1
                tree_id = 1
                num_level_ori = math.log(num_nodes,3)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 1
                step_parent_parent = 3
                current_level = 1
                
                while current_level <= num_level:
                    if current_level == 1:
                        for rank in range (1, num_nodes, step_parent_parent):

                            level_1_child_0 = rank - step_parent_child
                            level_1_child_1 = rank + step_parent_child
                            
                            chunk_reduce(rank, level_1_child_0, level_1_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    else:
                        for rank in range (1, num_nodes, step_parent_parent):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            
                            chunk_reduce(rank, level_n_child_0, level_n_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    step_parent_child *= 3
                    step_parent_parent *= 3
                    current_level += 1
                
                # root node reduce of tree1
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                    intra_reduce(node_offset=1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)


                # broadcast-tee1
                num_level_ori = math.log(num_nodes,3)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 3**(num_level-1)
                step_parent_parent = 3**num_level
                current_level = num_level
                
                # root node broadcast of tree1
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                    intra_broadcast(node_offset=1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 
                            
                
                while current_level >= 1:
                    if current_level != 1:
                        for rank in range (1, num_nodes, int(step_parent_parent)):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            
                            chunk_broadcast(rank, level_n_child_0, level_n_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    else:
                        for rank in range (1, num_nodes, int(step_parent_parent)):
                            
                            level_1_child_0 = rank - step_parent_child
                            level_1_child_1 = rank + step_parent_child
                            
                            chunk_broadcast(rank, level_1_child_0, level_1_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    
     
                    step_parent_child /= 3
                    step_parent_parent /= 3
                    current_level -= 1


                # reduce-tree2
                tree_id = 2
                num_level_ori = math.log(num_nodes,3)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 1
                step_parent_parent = 3
                current_level = 1
                while current_level <= num_level:
                    if current_level == 1:
                        for rank in range (2, num_nodes, step_parent_parent):
                            
                            level_1_child_0 = rank - step_parent_child
                            level_1_child_1 = rank - step_parent_child*2
                            
                            chunk_reduce(rank, level_1_child_0, level_1_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    else:
                        for rank in range (2, num_nodes, step_parent_parent):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            
                            chunk_reduce(rank, level_n_child_0, level_n_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    step_parent_child *= 3
                    step_parent_parent *= 3
                    current_level += 1
                
                # root node reduce of tree2
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                    intra_reduce(node_offset=2, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)
    
                    
                # broadcast-tee2
                num_level_ori = math.log(num_nodes,3)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 3**(num_level-1)
                step_parent_parent = 3**num_level
                current_level = num_level
                
                # root node broadcast of tree2
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                    intra_broadcast(node_offset=2, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 
                      
                
                while current_level >= 1:
                    if current_level != 1:
                        for rank in range (2, num_nodes, int(step_parent_parent)):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            
                            chunk_broadcast(rank, level_n_child_0, level_n_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    else:
                        for rank in range (2, num_nodes, int(step_parent_parent)):
                            
                            level_1_child_0 = rank - step_parent_child
                            level_1_child_1 = rank - step_parent_child*2
                            
                            chunk_broadcast(rank, level_1_child_0, level_1_child_1, num_gpus, num_nodes, combined_indices, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    step_parent_child /= 3
                    step_parent_parent /= 3
                    current_level -= 1

        
                   
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

allreduce_trinomial_tree(args.num_gpus, args.num_nodes, args.nchunks, args.nchannel ,args.instances, args.protocol)
