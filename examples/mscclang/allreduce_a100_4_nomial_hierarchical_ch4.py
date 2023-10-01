# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# intra node allreduce using ring algorithm
def ring_reduce_scatter(size, rank_offset=0, local_chunk_size=1):
    for ch in range(0, size):
        index = ch * local_chunk_size
        for step in range(0, size-1):
            other = chunk(((step+1+ch) % size) +rank_offset, Buffer.input, index, local_chunk_size)
            c = chunk(((step+2+ch) % size)+rank_offset, Buffer.input, index, local_chunk_size)
            c.reduce(other, ch=index)

def ring_all_gather(size, rank_offset=0, local_chunk_size=1):
    for ch in range(0, size):
        index = ch  * local_chunk_size
        for step in range(0, size-1):
            c = chunk(((step+ch) % size) + rank_offset, Buffer.input, index, local_chunk_size)
            c.copy(((step+ch+1) % size) + rank_offset, Buffer.input, index, local_chunk_size, ch=index)
                        

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_4_nomial_tree(num_nodes, num_local_gpus, instances, protocol):
    num_gpus = num_nodes * num_local_gpus
    topology = fully_connected(num_gpus)
    collective = AllReduce(num_gpus, num_local_gpus, True)
    with MSCCLProgram("allreduce_4_nomial_tree", topology, collective, instances, protocol=protocol, interleaved_replication=False):
        
        # intra node reduce scatter
        for n in range(num_nodes):
            ring_reduce_scatter(size=num_local_gpus, rank_offset=n * num_local_gpus)
            
        cross_node_chunk_size = int(num_local_gpus/4)   
        
        # tree0
        # Reduce tree0 - reducing onto Rank 0        
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 1
        step_parent_parent = 4
        current_level = 1
        while current_level <= num_level:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child)*num_local_gpus+n, Buffer.input, n)
                        chunk(rank*num_local_gpus+n, Buffer.input, n).reduce(c1, ch=n)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2)*num_local_gpus+n, Buffer.input, n)
                        chunk(rank*num_local_gpus+n, Buffer.input, n).reduce(c1, ch=n)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3)*num_local_gpus+n, Buffer.input, n)
                        chunk(rank*num_local_gpus+n, Buffer.input, n).reduce(c1, ch=n)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1

             
        # Broadcast tree0 - root is Rank 0         
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 4**(num_level-1)
        step_parent_parent = 4**num_level
        current_level = num_level
        while current_level >= 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk(rank*num_local_gpus+n, Buffer.input, n).copy((rank + int(step_parent_child))*num_local_gpus+n, Buffer.input, n, ch=n)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk(rank*num_local_gpus+n, Buffer.input, n).copy((rank + int(step_parent_child)*2)*num_local_gpus+n, Buffer.input, n, ch=n)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk(rank*num_local_gpus+n, Buffer.input, n).copy((rank + int(step_parent_child)*3)*num_local_gpus+n, Buffer.input, n, ch=n)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
            
            
            
        # tree1
        # Reduce tree1 - reducing onto Rank 1
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 1
        step_parent_parent = 4
        current_level = 1
        intra_offset=1
        inter_offset=1
            
        if current_level == 1:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child-inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1
            
        while current_level <= num_level:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1
        
            
             
        # Broadcast tree1 - root is Rank 1
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 4**(num_level-1)
        step_parent_parent = 4**num_level
        current_level = num_level
        intra_offset=1
        inter_offset=1
        while current_level > 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child) + inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*2 + inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*3 + inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
        
        if current_level == 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child) - inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
            
            
            
        
        # tree2
        # Reduce tree2 - reducing onto Rank 2
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 1
        step_parent_parent = 4
        current_level = 1
        intra_offset=2
        inter_offset=1
        if current_level == 1:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2-inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1
            
        while current_level <= num_level:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2 + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3 + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1
             
        # Broadcast tree2 - root is Rank 2
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 4**(num_level-1)
        step_parent_parent = 4**num_level
        current_level = num_level
        intra_offset=2
        inter_offset=1
        while current_level > 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child) + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*2 + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*3 + inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
            
        if current_level == 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child))*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*2 - inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
            

        # tree3
        # Reduce tree3 - reducing onto Rank 3
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 1
        step_parent_parent = 4
        current_level = 1
        intra_offset=3
        inter_offset=1
        if current_level == 1:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3-inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1
        
        while current_level <= num_level:
            for rank in range (0, num_nodes, step_parent_parent):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*2 + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):                    
                        c1 = chunk((rank + step_parent_child*3 + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset)
                        chunk((rank + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).reduce(c1, ch=n+intra_offset)
            step_parent_child *= 4
            step_parent_parent *= 4
            current_level += 1
             
        # Broadcast tree3 - root is Rank 3
        num_level_ori = math.log(num_nodes,4)
        num_level = math.ceil(num_level_ori)
        step_parent_child = 4**(num_level-1)
        step_parent_parent = 4**num_level
        current_level = num_level
        intra_offset=3
        inter_offset=1
        while current_level > 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child) + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*2 + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*3 + inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
            
        if current_level == 1:
            for rank in range (0, num_nodes, int(step_parent_parent)):
                if rank + step_parent_child < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child))*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*2 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*2)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
                if rank + step_parent_child*3 < num_nodes:
                    for n in range(0, cross_node_chunk_size):
                        chunk((rank+inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset).copy((rank + int(step_parent_child)*3 - inter_offset*3)*num_local_gpus+n+intra_offset, Buffer.input, n+intra_offset, ch=n+intra_offset)
            step_parent_child /= 4
            step_parent_parent /= 4
            current_level -= 1
            
                      
        # intra node all gather
        for n in range(num_nodes):
            ring_all_gather(size=num_local_gpus, rank_offset=n * num_local_gpus, local_chunk_size=1) 

        

        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help='number of gpus per node')
parser.add_argument('num_nodes', type=int, help='number of nodes')
# parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_4_nomial_tree(args.num_nodes, args.num_gpus, args.instances, args.protocol)
