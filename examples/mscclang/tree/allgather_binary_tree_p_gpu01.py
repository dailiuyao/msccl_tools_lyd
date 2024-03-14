# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather 

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

combined_indices = []

chunk_offset_size_of_rank_index = None

num_chunks_per_channel = None
    
num_channel_per_tree = None 

num_local_gpus = None

gpu_size = None

def intra_gather_peer1(node_offset=0, tree_id=0, chunk_step=0):
    
    # global combined_indices 
    # global chunk_offset_size_of_rank_index
    # global num_chunks_per_channel
    # global num_channel_per_tree
    # global num_local_gpus
    # gpu_size
    
    rank_offset = node_offset * num_local_gpus
    for channel in range(num_channel_per_tree):
        gpu_index = combined_indices[channel]
        channel_step_total = int(channel + tree_id * num_channel_per_tree)
        for gpu_step in range(0, num_local_gpus-1):
            for index in range(gpu_step, num_local_gpus-1):
                
                chunk_offset_global = int(gpu_index[gpu_step+rank_offset]*chunk_offset_size_of_rank_index + channel_step_total*num_chunks_per_channel + chunk_step)
                other = chunk(gpu_index[int(index+rank_offset)], Buffer.output, chunk_offset_global)
                c1 = chunk(gpu_index[int(index+rank_offset+1)], Buffer.output, chunk_offset_global)
                print(f"c1 is {c1}")
                other.copy(c1, ch=int(channel_step_total))
        
def inter_gather(peer=0, rank=0, tree_id=0, chunk_step=0):
    # gather data from peer to rank
    
    # global combined_indices 
    # global chunk_offset_size_of_rank_index
    # global num_chunks_per_channel
    # global num_channel_per_tree
    # global num_local_gpus
    # gpu_size   
    
    peer_offset = peer * num_local_gpus 
    rank_offset = rank * num_local_gpus 
    
    for channel in range(num_channel_per_tree):
        gpu_index = combined_indices[channel]
        channel_step_total = channel + tree_id * num_channel_per_tree
        for gpu_step in range(0, num_local_gpus):
            peer_chunk_offset_global = gpu_index[gpu_step+peer_offset]*chunk_offset_size_of_rank_index + channel_step_total*num_chunks_per_channel + chunk_step

            other = chunk(gpu_index[num_local_gpus-1+peer_offset], Buffer.output, peer_chunk_offset_global)
            c1 = chunk(gpu_index[num_local_gpus-1+rank_offset], Buffer.output, peer_chunk_offset_global)
            other.copy(c1, ch=channel_step_total)
    
   
                
        
def intra_broadcast_peer1(node_offset=0, tree_id=0, chunk_step=0):    
    
    # global combined_indices 
    # global chunk_offset_size_of_rank_index
    # global num_chunks_per_channel
    # global num_channel_per_tree
    # global num_local_gpus
    # gpu_size
    
    rank_offset = node_offset * num_local_gpus
    for channel in range(num_channel_per_tree):
        gpu_index = combined_indices[channel]
        channel_step_total = channel + tree_id * num_channel_per_tree
        for chunk_gpu_step in range(0, gpu_size):    
            for index in range(0, num_local_gpus-1): 
                chunk_offset_global = gpu_index[chunk_gpu_step]*chunk_offset_size_of_rank_index + channel_step_total*num_chunks_per_channel + chunk_step
                c1 = chunk(gpu_index[num_local_gpus - 1 - index + rank_offset], Buffer.output, chunk_offset_global)
                c1.copy(gpu_index[num_local_gpus - 2 - index + rank_offset], Buffer.output, chunk_offset_global, ch=channel_step_total)

def inter_broadcast(peer=0, rank=0, tree_id=0, chunk_step=0):
    # broadcast data from rank to peer
    
    # global combined_indices 
    # global chunk_offset_size_of_rank_index
    # global num_chunks_per_channel
    # global num_channel_per_tree
    # global num_local_gpus
    # gpu_size   
    
    peer_offset = peer * num_local_gpus 
    rank_offset = rank * num_local_gpus 
    
    for channel in range(num_channel_per_tree):
        gpu_index = combined_indices[channel]
        channel_step_total = channel + tree_id * num_channel_per_tree
        for chunk_gpu_step in range(0, gpu_size):
            chunk_offset_global = gpu_index[chunk_gpu_step]*chunk_offset_size_of_rank_index + channel_step_total*num_chunks_per_channel + chunk_step

            other = chunk(gpu_index[num_local_gpus-1+rank_offset], Buffer.output, chunk_offset_global)
            c1 = chunk(gpu_index[num_local_gpus-1+peer_offset], Buffer.output, chunk_offset_global)
            other.copy(c1, ch=channel_step_total)
            

def allgather_binary_tree(num_nodes, num_gpus, num_chunks, num_channel, instances, protocol):   
    
    if (num_channel == 1):
        trees=1
    else:
        trees=2
    
    global chunk_offset_size_of_rank_index
    global num_chunks_per_channel
    global num_channel_per_tree
    global num_local_gpus
    global combined_indices 
    global gpu_size
    
    num_local_gpus = num_gpus
    size = num_local_gpus * num_nodes 
    gpu_size = size
    total_chunks = int(trees * num_chunks * num_channel * size)
    para_chunks = int(total_chunks/size)
    print(f"para_chunks is {para_chunks}, size is {size}")
    
    chunk_offset_size_of_rank_index = para_chunks

    num_chunks_per_channel = num_chunks
    
    num_channel_per_tree = num_channel

    topology = fully_connected(size)
    collective = AllGather(size, para_chunks, True)
    
    # tree0: channel0 3->2->1->0
    # tree0: channel2 0->1->2->3
    # each tree has one channel
    # gather tree - gathering onto Rank 0
    gpu_index0 = list(range(0, size, 1))
    # gpu_index1 = gpu_index0
    gpu_index1 = list(reversed(gpu_index0))
    combined_indices = [gpu_index0, gpu_index1] 

    with MSCCLProgram("allgather_binary_tree", topology, collective, instances, protocol=protocol):

        tree_id = 0
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
                    
                    # peer_0 is one of the two children nodes, the first step is in the children node: gather to the last gpu.
                    intra_gather_peer1(peer_0, tree_id, chunk_step)
                    
                    # the children node peer_0 is ready, gather to the last-1 gpu in the parent node rank. 
                    inter_gather(peer_0, rank, tree_id, chunk_step)
                    
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= num_nodes:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    
                    if peer_1 > rank:
                        # peer_1 is another children node, the first step is in the children node: gather to the last gpu.
                        intra_gather_peer1(peer_1, tree_id, chunk_step)
                        
                        # the children node peer_1 is ready, gather to the last gpu in the parent node rank.
                        inter_gather(peer_1, rank, tree_id, chunk_step)
                step *= 2
                
            peer_0 = 1
            while peer_0 < num_nodes:            
                peer_0 *= 2
        
            peer_0 //= 2
            
            # this is the node under the top level node, so there is only one children node, and conduct the intra gather in this node
            intra_gather_peer1(peer_0, tree_id, chunk_step)
            
            #gather to the top level node, the accepted gpu in the top node is last gpu
            inter_gather(peer_0, 0, tree_id, chunk_step)
            
            # conduct the intra gather in the top level node to the last gpu
            intra_gather_peer1(0, tree_id, chunk_step)
            
            
            
            # Broadcast tree - root is Rank 0
            # conduct the intra broadcast in the top level node from the last gpu
            intra_broadcast_peer1(0, tree_id, chunk_step)
            
            inter_broadcast(peer_0, 0, tree_id, chunk_step)
            
            intra_broadcast_peer1(peer_0, tree_id, chunk_step)
            
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
                    
                    inter_broadcast(peer_0, rank, tree_id, chunk_step)
                    
                    intra_broadcast_peer1(peer_0, tree_id, chunk_step)
                   

                    peer_1 = rank + low_bit
                    while peer_1 >= num_nodes:
                        peer_1 = rank + low_bit
                        low_bit //= 2

                    if peer_1 > rank:
                        inter_broadcast(peer_1, rank, tree_id, chunk_step)
                        
                        intra_broadcast_peer1(peer_1, tree_id, chunk_step)
                       
                step //= 2
  


        # tree1: channel1 3->2->1->0
        # tree1: channel3 0->1->2->3
        # tree1 have two different channels
        # gather tree - gathering onto Rank N-1
        # if the number of node is even, the second tree is a mirrored tree
        
        if (trees == 2) and (num_nodes % 2 == 0):
            tree_id = 1
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

                        # peer_0 is one of the two children nodes, the first step is in the children node: gather to the last gpu.
                        intra_gather_peer1(num_nodes-1-peer_0, tree_id, chunk_step)
                        
                        # the children node peer_0 is ready, gather to the last-1 gpu in the parent node rank.
                        inter_gather(num_nodes-1-peer_0, num_nodes-1-rank, tree_id, chunk_step)
                        
                        peer_1 = rank + low_bit
                        while peer_1 >= num_nodes:
                            peer_1 = rank + low_bit
                            low_bit //= 2

                        if peer_1 > rank:
                            # peer_1 is another children node, the first step is in the children node: gather to the last gpu.
                            intra_gather_peer1(num_nodes-1-peer_1, tree_id, chunk_step)
                            
                            # the children node peer_1 is ready, gather to the last gpu in the parent node rank.
                            inter_gather(num_nodes-1-peer_1, num_nodes-1-rank, tree_id, chunk_step)
                            
                    step *= 2
                    
                peer_0 = 1
                while peer_0 < num_nodes:            
                    peer_0 *= 2
            
                peer_0 //= 2

                # this is the node under the top level node, so there is only one children node, and conduct the intra gather in this node
                intra_gather_peer1(num_nodes-1-peer_0, tree_id, chunk_step)
                
                #gather to the top level node, the accepted gpu in the top node is last gpu
                inter_gather(num_nodes-1-peer_0, num_nodes-1-0, tree_id, chunk_step)

                 # conduct the intra gather in the top level node to the last gpu
                intra_gather_peer1(num_nodes-1-0, tree_id, chunk_step)
                
                # Broadcast tree - root is Rank N-1
                # conduct the intra broadcast in the top level node from the last gpu
                intra_broadcast_peer1(num_nodes-1-0, tree_id, chunk_step)
                
                inter_broadcast(num_nodes-1-peer_0, num_nodes-1-0, tree_id, chunk_step)
                
                intra_broadcast_peer1(num_nodes-1-peer_0, tree_id, chunk_step)
               
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

                        inter_broadcast(num_nodes-1-peer_0, num_nodes-1-rank, tree_id, chunk_step)
                        
                        intra_broadcast_peer1(num_nodes-1-peer_0, tree_id, chunk_step)
                        
                        peer_1 = rank + low_bit
                        while peer_1 >= num_nodes:
                            peer_1 = rank + low_bit
                            low_bit //= 2
                        
                        inter_broadcast(num_nodes-1-peer_1, num_nodes-1-rank, tree_id, chunk_step)
                        
                        intra_broadcast_peer1(num_nodes-1-peer_1, tree_id, chunk_step)
                        
                    step //= 2
        
        
        # # tree1: channel1 3->2->1->0
        # # tree1: channel3 3->2->1->0
        # # each tree has two same channel
        # # gather tree - gathering onto Rank N-1
        # # if the number of node is odd, the second tree is a shifted tree
        
        # elif (trees == 2) and (num_nodes % 2 == 1):
        #     tree_id = 1
        #     for chunk_step in range(0, num_chunks_per_channel):


        #         num_level_ori = math.log(num_nodes,2)
        #         num_level = math.ceil(num_level_ori) - 1
        #         step = 2
        #         while step <= 2**num_level:       
        #             for rank in range(step, num_nodes, step*2):
        #                 bit = 1
        #                 while bit < num_nodes:            
        #                     if bit & rank:
        #                         break
        #                     bit *= 2   
                        
        #                 low_bit = bit // 2
        #                 peer_0 = rank - low_bit
                        
                        
        #                 # peer_0 is one of the two children nodes, the first step is in the children node: reduce to the last gpu.
        #                 for channel in range(num_channel_per_tree): 
        #                     intra_reduce(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)

        #                 # the children node peer_0 is ready, reduce to the last-1 gpu in the parent node rank.
        #                 for channel in range(num_channel_per_tree):  
        #                     c1 = chunk(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
        #                     chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-2], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree)
                            
        #                 peer_1 = rank + low_bit
        #                 while peer_1 >= num_nodes:
        #                     peer_1 = rank + low_bit
        #                     low_bit //= 2

        #                 if peer_1 > rank:
        #                     # peer_1 is another children node, the first step is in the children node: reduce to the last gpu.
        #                     for channel in range(num_channel_per_tree): 
        #                         intra_reduce(node_offset=((peer_1+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)
                        
        #                     # the children node peer_1 is ready, reduce to the last gpu in the parent node rank.
        #                     for channel in range(num_channel_per_tree):  
        #                         c1 = chunk(((peer_1+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
        #                         chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree) 
        #             step *= 2
                    
        #         peer_0 = 1
        #         while peer_0 < num_nodes:            
        #             peer_0 *= 2
            
        #         peer_0 //= 2

        #         # this is the node under the top level node, so there is only one children node, and conduct the intra reduce in this node
        #         for channel in range(num_channel_per_tree):
        #             intra_reduce(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)

        #         #reduce to the top level node, the accepted gpu in the top node is last gpu
        #         for channel in range(num_channel_per_tree):
        #             c1 = chunk(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel)
        #             chunk(((0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).reduce(c1, ch=channel+num_channel_per_tree)     
     

        #          # conduct the intra reduce in the top level node to the last gpu
        #         for channel in range(num_channel_per_tree):
        #             intra_reduce(node_offset=((0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree)


        #         # Broadcast tree - root is Rank N-1
        #         # conduct the intra broadcast in the top level node from the last gpu
        #         for channel in range(num_channel_per_tree):
        #             intra_broadcast_peer1(node_offset=((0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 

        #         for channel in range(num_channel_per_tree):
        #             chunk(((0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

        #         for channel in range(num_channel_per_tree):
        #             intra_broadcast_peer1(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
            
        #         step = 2**num_level
        #         while step >= 2:       
        #             for rank in range(step, num_nodes, step*2):
        #                 bit = 1
        #                 while bit < num_nodes:            
        #                     if bit & rank:
        #                         break
        #                     bit *= 2   
                        
        #                 low_bit = bit // 2
        #                 peer_0 = rank - low_bit 

        #                 for channel in range(num_channel_per_tree):
        #                     chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-2], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy(((peer_0+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

        #                 for channel in range(num_channel_per_tree):
        #                     intra_broadcast_peer1(node_offset=((peer_0+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
                    
        #                 peer_1 = rank + low_bit
        #                 while peer_1 >= num_nodes:
        #                     peer_1 = rank + low_bit
        #                     low_bit //= 2
                        
        #                 if peer_1 > rank:                        
        #                     for channel in range(num_channel_per_tree):
        #                         chunk(((rank+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel).copy(((peer_1+1)%num_nodes)*num_local_gpus+combined_indices[channel][num_local_gpus-1], Buffer.output, chunk_step+(channel+num_channel_per_tree)*num_chunks_per_channel, ch=channel+num_channel_per_tree) 

        #                     for channel in range(num_channel_per_tree):
        #                         intra_broadcast_peer1(node_offset=((peer_1+1)%num_nodes), num_local_gpus=num_local_gpus, chunk_step=chunk_step, gpu_index=combined_indices[channel], num_chunks_per_channel=num_chunks_per_channel, ch_idx=channel+num_channel_per_tree) 
                    
        #             step //= 2

                
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, help='number of gpus per node')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--nchunk', type=int, help ='number of chunks')
parser.add_argument('--nchannel', type=int, help ='number of channels')
parser.add_argument('--instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allgather_binary_tree(args.num_nodes, args.num_gpus, args.nchunk, args.nchannel ,args.instances, args.protocol)
