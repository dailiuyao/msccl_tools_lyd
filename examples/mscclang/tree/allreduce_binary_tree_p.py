# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf


def intra_reduce(node_offset=0, num_local_gpus=4, chunk_step=0, gpu_index=[0,0,0,0], num_chunks_per_channel=1, ch_idx=0):
    rank_offset = node_offset * num_local_gpus
    for index in range(0, num_local_gpus-1):
        other = chunk((gpu_index[index])+rank_offset, Buffer.input, chunk_step+ch_idx*num_chunks_per_channel)
        c1 = chunk((gpu_index[index+1])+rank_offset, Buffer.input, chunk_step+ch_idx*num_chunks_per_channel)
        c1.reduce(other, ch=ch_idx)
        
def intra_broadcast_peer1(node_offset=0, num_local_gpus=4, chunk_step=0, gpu_index=[0,0,0,0], num_chunks_per_channel=1, ch_idx=0):    
    rank_offset = node_offset * num_local_gpus
    for index in range(0, num_local_gpus-1):
        c = chunk((gpu_index[num_local_gpus - 1 - index])%num_local_gpus + rank_offset, Buffer.input, chunk_step+ch_idx*num_chunks_per_channel)
        c.copy((gpu_index[num_local_gpus - 2 - index])%num_local_gpus + rank_offset, Buffer.input, chunk_step+ch_idx*num_chunks_per_channel, ch=ch_idx)


def allreduce_binary_tree_hierarchical(num_nodes:int, num_local_gpus:int, num_chunks:int, num_channel:int, instances:int, protocol:str, trees:int):
    # if (num_channel == 1):
    #     trees=1
    # else:
    #     trees=2
    size = num_nodes * num_local_gpus
    
    num_chunks_per_channel = num_chunks
    
    num_channel_per_tree=num_channel
    
    num_total_chunks = num_chunks * num_channel * trees
    topology = fully_connected(size)
    collective = AllReduce(size, num_total_chunks, True)

    #/* Btree which alternates leaves and nodes.
    #* Assumes root is 0, which conveniently builds a tree on powers of two,
    #* (because we have pow2-1 ranks) which lets us manipulate bits.
    #* Find first non-zero bit, then :
    #* Find the parent :
    #*   xx01[0] -> xx10[0] (1,5,9 below) or xx00[0] if xx10[0] is out of bounds (13 below)
    #*   xx11[0] -> xx10[0] (3,7,11 below)
    #* Find the children :
    #*   xx10[0] -> xx01[0] (2,4,6,8,10,12) or -1 (1,3,5,7,9,11,13)
    #*   xx10[0] -> xx11[0] (2,4,6,8,10) or xx101[0] (12) or xx1001[0] ... or -1 (1,3,5,7,9,11,13)
    #*
    #* Illustration :
    #* 0---------------8
    #*          ______/ \______
    #*         4               12
    #*       /   \            /  \
    #*     2       6       10      14
    #*    / \     / \     /  \    /  \
    #*   1   3   5   7   9   11  13   15
    #*   p0  p1  p0  p1  p0  p1  p0   p1
    #*/

    with MSCCLProgram("allreduce_binary_tree_hierarchical", topology, collective, instances, protocol=protocol):

        # tree0: channel0 3->2->1->0
        # tree0: channel2 0->1->2->3
        # each tree has one channel
        # Reduce tree - reducing onto Rank 0
        
        # gpu_indices = []
        # gpu_indices.append(list(range(num_local_gpus)))  # gpu_index0
        # gpu_indices.append(list(reversed(gpu_indices[0])))  # gpu_index1
        
        # for i in range(1, num_channel*trees//4):
        #     gpu_indices.append([(x + i) % num_local_gpus for x in gpu_indices[0]])
        #     gpu_indices.append(list(reversed(gpu_indices[-1])))
        
        gpu_indices = []
        gpu_indices.append([0,7,6,5,4,3,2,1])  # gpu_index0
        gpu_indices.append([7,6,5,4,3,2,1,0])  # gpu_index1
        
        
        combined_indices = [gpu_indices[0], gpu_indices[0], gpu_indices[0], gpu_indices[0], gpu_indices[1], gpu_indices[1], gpu_indices[1], gpu_indices[1]]

        # for channel in range(num_channel_per_tree):
        #     for gpuidx in range(num_local_gpus):
        #         print(f"channel is {channel}, gpuidx is {gpuidx}, combined_indices is {combined_indices[channel][gpuidx]}")


        # peer0 and peer 1 are children nodes
        for chunk_step in range(0, num_chunks_per_channel):
            
            num_level_ori = math.log(num_nodes,2)
            num_level = math.ceil(num_level_ori) - 1
            step = 2
            # the second bottom level is the first loop of this while
            # every iteration will move to upper level
            while step <= 2**num_level:       
                for rank in range(step, num_nodes, step*2):
                    bit = 1
                    while bit < num_nodes:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    
                    # peer_0 is one of the two children nodes, the first step is in the children node: reduce to the last gpu.
                    for channel in range(num_channel_per_tree):
                        intra_reduce(node_offset=peer_0, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel)

                    # the children node peer_0 is ready, reduce to the last-1 gpu in the parent node rank. 
                    for channel in range(num_channel_per_tree):
                        c1 = chunk(peer_0*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel)
                        chunk(rank*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel).reduce(c1, ch=channel)
                        
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= num_nodes:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    
                    if peer_1 > rank:
                        # peer_1 is another children node, the first step is in the children node: reduce to the last gpu.
                        for channel in range(num_channel_per_tree):
                            intra_reduce(node_offset=peer_1, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel)
                        
                        # the children node peer_1 is ready, reduce to the last gpu in the parent node rank.
                        for channel in range(num_channel_per_tree): 
                            c1 = chunk(peer_1*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel)
                            chunk(rank*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel).reduce(c1, ch=channel)
                step *= 2
                
            peer_0 = 1
            while peer_0 < num_nodes:            
                peer_0 *= 2
        
            peer_0 //= 2
            
            # this is the node under the top level node, so there is only one children node, and conduct the intra reduce in this node
            for channel in range(num_channel_per_tree): 
                intra_reduce(node_offset=peer_0, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel)

            #reduce to the top level node, the accepted gpu in the top node is last gpu
            for channel in range(num_channel_per_tree): 
                c1 = chunk(peer_0*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel)
                chunk(0*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel).reduce(c1, ch=channel)                 
            
            # conduct the intra reduce in the top level node to the last gpu
            for channel in range(num_channel_per_tree): 
                intra_reduce(node_offset=0, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel)

            
            
            # Broadcast tree - root is Rank 0
            # conduct the intra broadcast in the top level node from the last gpu
            for channel in range(num_channel_per_tree): 
                intra_broadcast_peer1(node_offset=0, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel) 
            
            for channel in range(num_channel_per_tree): 
                chunk(0*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel).copy(peer_0*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel, ch=channel) 
            
            for channel in range(num_channel_per_tree): 
                intra_broadcast_peer1(node_offset=peer_0, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel) 
            
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
                    
                    for channel in range(num_channel_per_tree): 
                        chunk(rank*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel).copy(peer_0*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel, ch=channel) 
                    
                    for channel in range(num_channel_per_tree): 
                        intra_broadcast_peer1(node_offset=peer_0, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel) 
                    

                    peer_1 = rank + low_bit
                    while peer_1 >= num_nodes:
                        peer_1 = rank + low_bit
                        low_bit //= 2

                    if peer_1 > rank:
                        for channel in range(num_channel_per_tree): 
                            chunk(rank*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel).copy(peer_1*num_local_gpus + combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+channel*num_chunks_per_channel, ch=channel)

                        for channel in range(num_channel_per_tree): 
                            intra_broadcast_peer1(node_offset=peer_1, num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel) 
                    
                step //= 2
  


        # tree1: channel1 3->2->1->0
        # tree1: channel3 0->1->2->3
        # tree1 have two different channels
        # Reduce tree - reducing onto Rank N-1
        # if the number of node is even, the second tree is a mirrored tree
        
        if (trees == 2) and (num_nodes % 2 == 0):
            for chunk_step in range(0, num_chunks_per_channel):


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

                         # peer_0 is one of the two children nodes, the first step is in the children node: reduce to the last gpu.
                        for channel in range(num_channel_per_tree): 
                            intra_reduce(node_offset=(num_nodes-1-peer_0), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)

                        # the children node peer_0 is ready, reduce to the last-1 gpu in the parent node rank.
                        for channel in range(num_channel_per_tree):  
                            c1 = chunk((num_nodes-1-peer_0)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
                            chunk((num_nodes-1-rank)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree)
                            
                        peer_1 = rank + low_bit
                        while peer_1 >= num_nodes:
                            peer_1 = rank + low_bit
                            low_bit //= 2

                        if peer_1 > rank:
                            # peer_1 is another children node, the first step is in the children node: reduce to the last gpu.
                            for channel in range(num_channel_per_tree): 
                                intra_reduce(node_offset=(num_nodes-1-peer_1), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)
                        
                            # the children node peer_1 is ready, reduce to the last gpu in the parent node rank.
                            for channel in range(num_channel_per_tree):  
                                c1 = chunk((num_nodes-1-peer_1)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
                                chunk((num_nodes-1-rank)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree) 
                    step *= 2
                    
                peer_0 = 1
                while peer_0 < num_nodes:            
                    peer_0 *= 2
            
                peer_0 //= 2

                # this is the node under the top level node, so there is only one children node, and conduct the intra reduce in this node
                for channel in range(num_channel_per_tree):
                    intra_reduce(node_offset=(num_nodes-1-peer_0), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)

                #reduce to the top level node, the accepted gpu in the top node is last gpu
                for channel in range(num_channel_per_tree):
                    c1 = chunk((num_nodes-1-peer_0)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
                    chunk((num_nodes-1-0)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree)     
     

                 # conduct the intra reduce in the top level node to the last gpu
                for channel in range(num_channel_per_tree):
                    intra_reduce(node_offset=(num_nodes-1-0), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)


                # Broadcast tree - root is Rank N-1
                # conduct the intra broadcast in the top level node from the last gpu
                for channel in range(num_channel_per_tree):
                    intra_broadcast_peer1(node_offset=(num_nodes-1-0), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 

                for channel in range(num_channel_per_tree):
                    chunk((num_nodes-1-0)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy((num_nodes-1-peer_0)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

                for channel in range(num_channel_per_tree):
                    intra_broadcast_peer1(node_offset=(num_nodes-1-peer_0), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
            
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

                        for channel in range(num_channel_per_tree):
                            chunk((num_nodes-1-rank)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy((num_nodes-1-peer_0)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

                        for channel in range(num_channel_per_tree):
                            intra_broadcast_peer1(node_offset=(num_nodes-1-peer_0), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
                    
                        peer_1 = rank + low_bit
                        while peer_1 >= num_nodes:
                            peer_1 = rank + low_bit
                            low_bit //= 2
                        
                        
                        for channel in range(num_channel_per_tree):
                            chunk((num_nodes-1-rank)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy((num_nodes-1-peer_1)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

                        for channel in range(num_channel_per_tree):
                            intra_broadcast_peer1(node_offset=(num_nodes-1-peer_1), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
                    
                    step //= 2
        
        
        # tree1: channel1 3->2->1->0
        # tree1: channel3 3->2->1->0
        # each tree has two same channel
        # Reduce tree - reducing onto Rank N-1
        # if the number of node is odd, the second tree is a shifted tree
        
        elif (trees == 2) and (num_nodes % 2 == 1):
            for chunk_step in range(0, num_chunks_per_channel):


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
                        
                        
                        # peer_0 is one of the two children nodes, the first step is in the children node: reduce to the last gpu.
                        for channel in range(num_channel_per_tree): 
                            intra_reduce(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)

                        # the children node peer_0 is ready, reduce to the last-1 gpu in the parent node rank.
                        for channel in range(num_channel_per_tree):  
                            c1 = chunk(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
                            chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree)
                            
                        peer_1 = rank + low_bit
                        while peer_1 >= num_nodes:
                            peer_1 = rank + low_bit
                            low_bit //= 2

                        if peer_1 > rank:
                            # peer_1 is another children node, the first step is in the children node: reduce to the last gpu.
                            for channel in range(num_channel_per_tree): 
                                intra_reduce(node_offset=((peer_1+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)
                        
                            # the children node peer_1 is ready, reduce to the last gpu in the parent node rank.
                            for channel in range(num_channel_per_tree):  
                                c1 = chunk(((peer_1+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
                                chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree) 
                    step *= 2
                    
                peer_0 = 1
                while peer_0 < num_nodes:            
                    peer_0 *= 2
            
                peer_0 //= 2

                # this is the node under the top level node, so there is only one children node, and conduct the intra reduce in this node
                for channel in range(num_channel_per_tree):
                    intra_reduce(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)

                #reduce to the top level node, the accepted gpu in the top node is last gpu
                for channel in range(num_channel_per_tree):
                    c1 = chunk(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
                    chunk(((0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree)     
     

                 # conduct the intra reduce in the top level node to the last gpu
                for channel in range(num_channel_per_tree):
                    intra_reduce(node_offset=((0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)


                # Broadcast tree - root is Rank N-1
                # conduct the intra broadcast in the top level node from the last gpu
                for channel in range(num_channel_per_tree):
                    intra_broadcast_peer1(node_offset=((0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 

                for channel in range(num_channel_per_tree):
                    chunk(((0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

                for channel in range(num_channel_per_tree):
                    intra_broadcast_peer1(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
            
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

                        for channel in range(num_channel_per_tree):
                            chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

                        for channel in range(num_channel_per_tree):
                            intra_broadcast_peer1(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
                    
                        peer_1 = rank + low_bit
                        while peer_1 >= num_nodes:
                            peer_1 = rank + low_bit
                            low_bit //= 2
                        
                        if peer_1 > rank:                        
                            for channel in range(num_channel_per_tree):
                                chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy(((peer_1+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.input, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

                            for channel in range(num_channel_per_tree):
                                intra_broadcast_peer1(node_offset=((peer_1+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
                    
                    step //= 2

                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, help='number of gpus per node')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--nchunks', type=int, help ='number of chunks')
parser.add_argument('--nchannel', type=int, help ='number of channels')
parser.add_argument('--instances', type=int, help ='number of instances')
parser.add_argument('--trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_binary_tree_hierarchical(args.num_nodes, args.num_gpus, args.nchunks, args.nchannel ,args.instances, args.protocol, args.trees)
