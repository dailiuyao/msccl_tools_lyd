# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_binary_tree(size, instances, trees, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, trees, True)
    with MSCCLProgram("allreduce_binary_tree", topology, collective, instances, protocol=protocol):
        # Reduce tree - reducing onto Rank 0
        num_level_ori = math.log(size,2)
        num_level = math.ceil(num_level_ori) - 1
        step = 2
        while step <= 2**num_level:       
            for rank in range(step, size, step*2):
                bit = 1
                while bit < size:            
                    if bit & rank:
                        break
                    bit *= 2   
                
                low_bit = bit // 2
                peer_0 = rank - low_bit
                c1 = chunk(peer_0, Buffer.input, 0)
                chunk(rank, Buffer.input, 0).reduce(c1, 0)
                
                peer_1 = rank + low_bit
                while peer_1 >= size:
                    peer_1 = rank + low_bit
                    low_bit //= 2
                c1 = chunk(peer_1, Buffer.input, 0)
                chunk(rank, Buffer.input, 0).reduce(c1, 0)
            step *= 2
            
        peer_0 = 1
        while peer_0 < size:            
            peer_0 *= 2
     
        peer_0 //= 2
        c1 = chunk(peer_0, Buffer.input, 0)
        chunk(0, Buffer.input, 0).reduce(c1, 0)         
                
         # Broadcast tree - root is Rank 0
        chunk(0, Buffer.input, 0).copy(peer_0, Buffer.input, 0) 
        
        step = 2**num_level
        while step >= 2:       
            for rank in range(step, size, step*2):
                bit = 1
                while bit < size:            
                    if bit & rank:
                        break
                    bit *= 2   
                
                low_bit = bit // 2
                peer_0 = rank - low_bit
                chunk(rank, Buffer.input, 0).copy(peer_0, Buffer.input, 0) 
                
                peer_1 = rank + low_bit
                while peer_1 >= size:
                    peer_1 = rank + low_bit
                    low_bit //= 2
                chunk(rank, Buffer.input, 0).copy(peer_1, Buffer.input, 0) 
            step //= 2

        # Mirrored version of the second tree for even ranks
        # Reduce tree - reducing onto Rank N-1
        if (trees == 2) and (size % 2 == 0):
            # Reduce tree - reducing onto Rank N-1
            num_level_ori = math.log(size,2)
            num_level = math.ceil(num_level_ori) - 1
            step = 2
            while step <= 2**num_level:       
                for rank in range(step, size, step*2):
                    bit = 1
                    while bit < size:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    c1 = chunk(size-1-peer_0, Buffer.input, 1)
                    chunk(size-1-rank, Buffer.input, 1).reduce(c1)
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= size:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    c1 = chunk(size-1-peer_1, Buffer.input, 1)
                    chunk(size-1-rank, Buffer.input, 1).reduce(c1)
                step *= 2
                
            peer_0 = 1
            while peer_0 < size:            
                peer_0 *= 2
        
            peer_0 //= 2
            c1 = chunk(size-1-peer_0, Buffer.input, 1)
            chunk(size-1-0, Buffer.input, 1).reduce(c1)         
                    
            # Broadcast tree - root is Rank N-1
            chunk(size-1-0, Buffer.input, 1).copy(size-1-peer_0, Buffer.input, 1) 
            
            step = 2**num_level
            while step >= 2:       
                for rank in range(step, size, step*2):
                    bit = 1
                    while bit < size:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    chunk(size-1-rank, Buffer.input, 1).copy(size-1-peer_0, Buffer.input, 1) 
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= size:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    chunk(size-1-rank, Buffer.input, 1).copy(size-1-peer_1, Buffer.input, 1) 
                step //= 2
        
        # shifted version of the second tree for odd ranks
        elif (trees == 2) and (size % 2 == 1):
            # Reduce tree - reducing onto Rank 1
            num_level_ori = math.log(size,2)
            num_level = math.ceil(num_level_ori) - 1
            step = 2
            while step <= 2**num_level:       
                for rank in range(step, size, step*2):
                    bit = 1
                    while bit < size:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    c1 = chunk((peer_0+1)%size, Buffer.input, 1)
                    chunk((rank+1)%size, Buffer.input, 1).reduce(c1)
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= size:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    c1 = chunk((peer_1+1)%size, Buffer.input, 1)
                    chunk((rank+1)%size, Buffer.input, 1).reduce(c1)
                step *= 2
                
            peer_0 = 1
            while peer_0 < size:            
                peer_0 *= 2
        
            peer_0 //= 2
            c1 = chunk((peer_0+1)%size, Buffer.input, 1)
            chunk((0+1)%size, Buffer.input, 1).reduce(c1)         
                    
            # Broadcast tree - root is Rank 1
            chunk((0+1)%size, Buffer.input, 1).copy((peer_0+1)%size, Buffer.input, 1) 
            
            step = 2**num_level
            while step >= 2:       
                for rank in range(step, size, step*2):
                    bit = 1
                    while bit < size:            
                        if bit & rank:
                            break
                        bit *= 2   
                    
                    low_bit = bit // 2
                    peer_0 = rank - low_bit
                    chunk((rank+1)%size, Buffer.input, 1).copy((peer_0+1)%size, Buffer.input, 1) 
                    
                    peer_1 = rank + low_bit
                    while peer_1 >= size:
                        peer_1 = rank + low_bit
                        low_bit //= 2
                    chunk((rank+1)%size, Buffer.input, 1).copy((peer_1+1)%size, Buffer.input, 1) 
                step //= 2 
        
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_binary_tree(args.num_gpus, args.instances, args.trees, args.protocol)