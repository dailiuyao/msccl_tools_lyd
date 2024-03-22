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


def chunk_reduce(rank=0, child_0=0, child_1=0, child_2=0, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):    
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
        
        if child_2 < num_nodes:
            intra_reduce(node_offset=child_2, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total)
            # the children node child_0 is ready, reduce to the last gpu in the parent node rank. 
            c1 = chunk(int(child_2*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total)
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).reduce(c1, ch=channel_total)

def chunk_broadcast(rank=0, child_0=0, child_1=0, child_2=0, num_gpus=0, num_nodes=0, combined_indices=[[0,0,0,0],[0,0,0,0]], tree_id=0, num_chunks_per_channel=0, num_channel_per_tree=0, chunk_step_channel=0):
    for channel in range(num_channel_per_tree):
        channel_total = channel+tree_id*num_channel_per_tree
        chunk_step_total = chunk_step_channel+channel_total*num_chunks_per_channel  
        if child_0 < num_nodes:
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).copy(int(child_0*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total, ch=channel_total)
            intra_broadcast(node_offset=child_0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 

        if child_1 < num_nodes:
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).copy(int(child_1*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total, ch=channel_total)
            intra_broadcast(node_offset=child_1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 

        if child_2 < num_nodes:
            chunk(int(rank*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total).copy(int(child_2*num_gpus + combined_indices[channel][num_gpus-1]), Buffer.input, chunk_step_total, ch=channel_total)
            intra_broadcast(node_offset=child_2, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices[channel], ch_idx=channel_total) 

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf



def allreduce_4_nomial_tree(num_gpus, num_nodes, nchunks, nchannel, instances, protocol, trees):
    
    # if (nchannel == 1):
    #     trees=1
    # else:
    #     trees=4



    size = num_nodes * num_gpus
    num_chunks_per_channel = nchunks
    num_channel_per_tree=nchannel
    num_total_chunks = nchunks * nchannel * trees
    
    topology = fully_connected(size)
    collective = AllReduce(size, num_total_chunks, True)
    with MSCCLProgram("allreduce_4_nomial_tree", topology, collective, instances, protocol=protocol):

        # channel 0: 0->1->2->3
        # channel 1: 3->2->1->0
        # each tree has one channel
        # Reduce tree - reducing onto Rank 0
        
        gpu_indices = []
        gpu_indices.append(list(range(num_gpus)))  # gpu_index0
        gpu_indices.append(list(reversed(gpu_indices[0])))  # gpu_index1
        
        for i in range(1, nchannel*trees//2):
            gpu_indices.append([(x + i) % num_gpus for x in gpu_indices[0]])
            gpu_indices.append(list(reversed(gpu_indices[-1])))

        combined_indices_0 = [gpu_indices[0], gpu_indices[1], gpu_indices[2], gpu_indices[3]]
        combined_indices_1 = [gpu_indices[4], gpu_indices[5], gpu_indices[6], gpu_indices[7]]
        combined_indices_2 = [gpu_indices[8], gpu_indices[9], gpu_indices[10], gpu_indices[11]]
        combined_indices_3 = [gpu_indices[12], gpu_indices[13], gpu_indices[14], gpu_indices[15]]


        for chunk_step in range(0, num_chunks_per_channel):
            # reduce-tree0
            tree_id = 0
            num_level_ori = math.log(num_nodes,4)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 1
            step_parent_parent = 4
            current_level = 1
            # level is the parent node level
            # current_level: 1 -> num_level
            # step_parent_child: 1 -> 4^(num_level-1)
            # step_parent_parent: 4 -> 4^num_level
            while current_level <= num_level:
                for rank in range (0, num_nodes, step_parent_parent):

                    child_0 = rank + step_parent_child
                    child_1 = rank + step_parent_child*2
                    child_2 = rank + step_parent_child*3

                    chunk_reduce(rank, child_0, child_1, child_2, num_gpus, num_nodes, combined_indices_0, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                        
                step_parent_child *= 4
                step_parent_parent *= 4
                current_level += 1
            
            # root node reduce of tree0
            for channel in range(num_channel_per_tree):
                channel_total = channel+tree_id*num_channel_per_tree
                chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                intra_reduce(node_offset=0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_0[channel], ch_idx=channel_total)

            # broadcast-tee0
            num_level_ori = math.log(num_nodes,4)
            num_level = math.ceil(num_level_ori)
            step_parent_child = 4**(num_level-1)
            step_parent_parent = 4**num_level
            current_level = num_level

            # root node broadcast of tree0
            for channel in range(num_channel_per_tree):
                channel_total = channel+tree_id*num_channel_per_tree
                chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                intra_broadcast(node_offset=0, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_0[channel], ch_idx=channel_total) 
                            
            while current_level >= 1:
                for rank in range (0, num_nodes, int(step_parent_parent)):

                    child_0 = rank + step_parent_child
                    child_1 = rank + step_parent_child*2
                    child_2 = rank + step_parent_child*3

                    chunk_broadcast(rank, child_0, child_1, child_2, num_gpus, num_nodes, combined_indices_0, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)

                step_parent_child /= 4
                step_parent_parent /= 4
                current_level -= 1
        
        if trees == 2 or trees == 4:
            for chunk_step in range(0, num_chunks_per_channel):
                # reduce-tree1
                tree_id = 1
                num_level_ori = math.log(num_nodes,4)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 1
                step_parent_parent = 4
                current_level = 1
                
                while current_level <= num_level:
                    if current_level == 1:
                        for rank in range (1, num_nodes, step_parent_parent):

                            level_1_child_0 = rank + step_parent_child
                            level_1_child_1 = rank + step_parent_child*2
                            level_1_child_2 = rank - step_parent_child
                            
                            chunk_reduce(rank, level_1_child_0, level_1_child_1, level_1_child_2, num_gpus, num_nodes, combined_indices_1, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    else:
                        for rank in range (1, num_nodes, step_parent_parent):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            level_n_child_2 = rank + step_parent_child*3
                            
                            chunk_reduce(rank, level_n_child_0, level_n_child_1, level_n_child_2, num_gpus, num_nodes, combined_indices_1, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    step_parent_child *= 4
                    step_parent_parent *= 4
                    current_level += 1
                
                # root node reduce of tree1
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                    intra_reduce(node_offset=1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_1[channel], ch_idx=channel_total)


                # broadcast-tee1
                num_level_ori = math.log(num_nodes,4)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 4**(num_level-1)
                step_parent_parent = 4**num_level
                current_level = num_level
                
                # root node broadcast of tree1
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                    intra_broadcast(node_offset=1, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_1[channel], ch_idx=channel_total) 
                            
                
                while current_level >= 1:
                    if current_level != 1:
                        for rank in range (1, num_nodes, int(step_parent_parent)):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            level_n_child_2 = rank + step_parent_child*3
                            
                            chunk_broadcast(rank, level_n_child_0, level_n_child_1, level_n_child_2, num_gpus, num_nodes, combined_indices_1, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step) 
                            
                    else:
                        for rank in range (1, num_nodes, int(step_parent_parent)):
                            
                            level_1_child_0 = rank + step_parent_child
                            level_1_child_1 = rank + step_parent_child*2
                            level_1_child_2 = rank - step_parent_child
                            
                            chunk_broadcast(rank, level_1_child_0, level_1_child_1, level_1_child_2, num_gpus, num_nodes, combined_indices_1, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    
     
                    step_parent_child /= 4
                    step_parent_parent /= 4
                    current_level -= 1
                    
                    

        if trees == 4:
                # reduce-tree2
                tree_id = 2
                num_level_ori = math.log(num_nodes,4)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 1
                step_parent_parent = 4
                current_level = 1
                while current_level <= num_level:
                    if current_level == 1:
                        for rank in range (2, num_nodes, step_parent_parent):
                            
                            level_1_child_0 = rank + step_parent_child
                            level_1_child_1 = rank - step_parent_child*2
                            level_1_child_2 = rank - step_parent_child
                            
                            chunk_reduce(rank, level_1_child_0, level_1_child_1, level_1_child_2, num_gpus, num_nodes, combined_indices_2, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)
                        
                    else:
                        for rank in range (2, num_nodes, step_parent_parent):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            level_n_child_2 = rank + step_parent_child*3
                            
                            chunk_reduce(rank, level_n_child_0, level_n_child_1, level_n_child_2, num_gpus, num_nodes, combined_indices_2, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)
                            
                    step_parent_child *= 4
                    step_parent_parent *= 4
                    current_level += 1
                
                # root node reduce of tree2
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                    intra_reduce(node_offset=2, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_2[channel], ch_idx=channel_total)
    
                    
                # broadcast-tee2
                num_level_ori = math.log(num_nodes,4)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 4**(num_level-1)
                step_parent_parent = 4**num_level
                current_level = num_level
                
                # root node broadcast of tree2
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                    intra_broadcast(node_offset=2, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_2[channel], ch_idx=channel_total) 
                      
                
                while current_level >= 1:
                    if current_level != 1:
                        for rank in range (2, num_nodes, int(step_parent_parent)):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            level_n_child_2 = rank + step_parent_child*3
                            
                            chunk_broadcast(rank, level_n_child_0, level_n_child_1, level_n_child_2, num_gpus, num_nodes, combined_indices_2, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    

                    else:
                        for rank in range (2, num_nodes, int(step_parent_parent)):
                            
                            level_1_child_0 = rank + step_parent_child
                            level_1_child_1 = rank - step_parent_child*2
                            level_1_child_2 = rank - step_parent_child
                            
                            chunk_broadcast(rank, level_1_child_0, level_1_child_1, level_1_child_2, num_gpus, num_nodes, combined_indices_2, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)   
                            
                    step_parent_child /= 4
                    step_parent_parent /= 4
                    current_level -= 1
                
                # reduce-tree3
                tree_id = 3
                num_level_ori = math.log(num_nodes,4)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 1
                step_parent_parent = 4
                current_level = 1
                while current_level <= num_level:
                    if current_level == 1:
                        for rank in range (3, num_nodes, step_parent_parent):
                            
                            level_1_child_0 = rank - step_parent_child*3
                            level_1_child_1 = rank - step_parent_child*2
                            level_1_child_2 = rank - step_parent_child
                            
                            chunk_reduce(rank, level_1_child_0, level_1_child_1, level_1_child_2, num_gpus, num_nodes, combined_indices_3, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)

                    else:
                        for rank in range (3, num_nodes, step_parent_parent):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            level_n_child_2 = rank + step_parent_child*3
                            
                            chunk_reduce(rank, level_n_child_0, level_n_child_1, level_n_child_2, num_gpus, num_nodes, combined_indices_3, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)

                    step_parent_child *= 4
                    step_parent_parent *= 4
                    current_level += 1
                
                # root node reduce of tree3
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel 
                    intra_reduce(node_offset=3, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_3[channel], ch_idx=channel_total)
    
                    
                # broadcast-tee3
                num_level_ori = math.log(num_nodes,4)
                num_level = math.ceil(num_level_ori)
                step_parent_child = 4**(num_level-1)
                step_parent_parent = 4**num_level
                current_level = num_level
                
                # root node broadcast of tree3
                for channel in range(num_channel_per_tree):
                    channel_total = channel+tree_id*num_channel_per_tree
                    chunk_step_total = chunk_step+channel_total*num_chunks_per_channel  
                    intra_broadcast(node_offset=3, num_local_gpus=num_gpus, chunk_step_total=chunk_step_total, gpu_index=combined_indices_3[channel], ch_idx=channel_total) 
                      
                
                while current_level >= 1:
                    if current_level != 1:
                        for rank in range (3, num_nodes, int(step_parent_parent)):
                            
                            level_n_child_0 = rank + step_parent_child
                            level_n_child_1 = rank + step_parent_child*2
                            level_n_child_2 = rank + step_parent_child*3
                            
                            chunk_broadcast(rank, level_n_child_0, level_n_child_1, level_n_child_2, num_gpus, num_nodes, combined_indices_3, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)    
     
                    else:
                        for rank in range (3, num_nodes, int(step_parent_parent)):
                            
                            level_1_child_0 = rank - step_parent_child*3
                            level_1_child_1 = rank - step_parent_child*2
                            level_1_child_2 = rank - step_parent_child
                            
                            chunk_broadcast(rank, level_1_child_0, level_1_child_1, level_1_child_2, num_gpus, num_nodes, combined_indices_3, tree_id, num_chunks_per_channel, num_channel_per_tree, chunk_step)   
                            
                           

                    step_parent_child /= 4
                    step_parent_parent /= 4
                    current_level -= 1
        
                

        
                   
        XML()
        Check()

parser = argparse.ArgumentParser()

parser.add_argument('--nchunks', type=int, help ='number of chunks')
parser.add_argument('--num_gpus', type=int, help='number of gpus per node')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--nchannel', type=int, help ='number of channels')
parser.add_argument('--trees', type=int, choices=[1, 2, 4], help ='number of trees')
parser.add_argument('--instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

allreduce_4_nomial_tree(args.num_gpus, args.num_nodes, args.nchunks, args.nchannel ,args.instances, args.protocol, args.trees)
