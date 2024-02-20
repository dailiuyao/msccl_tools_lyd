# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import math
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

# Binomial tree and mirrored binomial tree
# Mirrored trees adopted from: http://algo2.iti.kit.edu/documents/2tree.pdf
def allreduce_fibonacci_tree(size, instances, trees, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, trees, True)
    with MSCCLProgram("allreduce_fibonacci_tree", topology, collective, instances, protocol=protocol):
        # Reduce tree0 - reducing onto Rank 15
        # level 0
        c1 = chunk(0, Buffer.input, 0)
        chunk(11, Buffer.input, 0).reduce(c1, 0)
        c1 = chunk(1, Buffer.input, 0)
        chunk(11, Buffer.input, 0).reduce(c1, 0) 
        
        # level 1
        c1 = chunk(2, Buffer.input, 0)
        chunk(8, Buffer.input, 0).reduce(c1, 0)  
        c1 = chunk(3, Buffer.input, 0)
        chunk(9, Buffer.input, 0).reduce(c1, 0)     
        c1 = chunk(4, Buffer.input, 0)
        chunk(9, Buffer.input, 0).reduce(c1, 0)   
        c1 = chunk(5, Buffer.input, 0)
        chunk(10, Buffer.input, 0).reduce(c1, 0)
        c1 = chunk(6, Buffer.input, 0)
        chunk(10, Buffer.input, 0).reduce(c1, 0)
        c1 = chunk(7, Buffer.input, 0)
        chunk(13, Buffer.input, 0).reduce(c1, 0)   
        
        # level 1
        c1 = chunk(11, Buffer.input, 0)
        chunk(8, Buffer.input, 0).reduce(c1, 0)
        
        
        # level 2
        c1 = chunk(8, Buffer.input, 0)
        chunk(12, Buffer.input, 0).reduce(c1, 0) 
        c1 = chunk(9, Buffer.input, 0)
        chunk(12, Buffer.input, 0).reduce(c1, 0)
        c1 = chunk(10, Buffer.input, 0)
        chunk(13, Buffer.input, 0).reduce(c1, 0)
        
        # level 3
        c1 = chunk(12, Buffer.input, 0)
        chunk(14, Buffer.input, 0).reduce(c1, 0)
        c1 = chunk(13, Buffer.input, 0)
        chunk(14, Buffer.input, 0).reduce(c1, 0)
        
        # level 4
        c1 = chunk(14, Buffer.input, 0)
        chunk(15, Buffer.input, 0).reduce(c1, 0)
                
        # Broadcast tree0 - root is Rank 15
        # level 0
        chunk(15, Buffer.input, 0).copy(14, Buffer.input, 0) 
        
        # level 1
        chunk(14, Buffer.input, 0).copy(12, Buffer.input, 0) 
        chunk(14, Buffer.input, 0).copy(13, Buffer.input, 0)
        
        # level 2
        chunk(12, Buffer.input, 0).copy(8, Buffer.input, 0) 
        chunk(12, Buffer.input, 0).copy(9, Buffer.input, 0)   
        chunk(13, Buffer.input, 0).copy(10, Buffer.input, 0)  
        chunk(13, Buffer.input, 0).copy(7, Buffer.input, 0)  
        
        # level 3
        chunk(8, Buffer.input, 0).copy(11, Buffer.input, 0)  
        chunk(8, Buffer.input, 0).copy(2, Buffer.input, 0) 
        chunk(9, Buffer.input, 0).copy(3, Buffer.input, 0)   
        chunk(9, Buffer.input, 0).copy(4, Buffer.input, 0)  
        chunk(10, Buffer.input, 0).copy(5, Buffer.input, 0)  
        chunk(10, Buffer.input, 0).copy(6, Buffer.input, 0)  
        
        # level 4
        chunk(11, Buffer.input, 0).copy(0, Buffer.input, 0)  
        chunk(11, Buffer.input, 0).copy(1, Buffer.input, 0)  
        

        # Mirrored version of the second tree for even ranks
        if trees == 2:
            # Reduce tree1 - reducing onto Rank 0
            # level 0
            c1 = chunk(14, Buffer.input, 1)
            chunk(8, Buffer.input, 1).reduce(c1)
            c1 = chunk(15, Buffer.input, 1)
            chunk(8, Buffer.input, 1).reduce(c1) 
            c1 = chunk(9, Buffer.input, 1)
            chunk(4, Buffer.input, 1).reduce(c1)  
            c1 = chunk(10, Buffer.input, 1)
            chunk(5, Buffer.input, 1).reduce(c1)     
            c1 = chunk(11, Buffer.input, 1)
            chunk(5, Buffer.input, 1).reduce(c1)   
            c1 = chunk(12, Buffer.input, 1)
            chunk(6, Buffer.input, 1).reduce(c1)
            c1 = chunk(13, Buffer.input, 1)
            chunk(6, Buffer.input, 1).reduce(c1)
            c1 = chunk(7, Buffer.input, 1)
            chunk(3, Buffer.input, 1).reduce(c1)   
            
            # level 1
            c1 = chunk(8, Buffer.input, 1)
            chunk(4, Buffer.input, 1).reduce(c1)
            
            
            # level 2
            c1 = chunk(4, Buffer.input, 1)
            chunk(2, Buffer.input, 1).reduce(c1) 
            c1 = chunk(5, Buffer.input, 1)
            chunk(2, Buffer.input, 1).reduce(c1)
            c1 = chunk(6, Buffer.input, 1)
            chunk(3, Buffer.input, 1).reduce(c1)
            
            # level 3
            c1 = chunk(2, Buffer.input, 1)
            chunk(1, Buffer.input, 1).reduce(c1)
            c1 = chunk(3, Buffer.input, 1)
            chunk(1, Buffer.input, 1).reduce(c1)
            
            # level 4
            c1 = chunk(1, Buffer.input, 1)
            chunk(0, Buffer.input, 1).reduce(c1)
                    
            # Broadcast tree1 - root is Rank 0
            # level 0
            chunk(0, Buffer.input, 1).copy(1, Buffer.input, 1) 
            
            # level 1
            chunk(1, Buffer.input, 1).copy(2, Buffer.input, 1) 
            chunk(1, Buffer.input, 1).copy(3, Buffer.input, 1)
            
            # level 2
            chunk(2, Buffer.input, 1).copy(4, Buffer.input, 1) 
            chunk(2, Buffer.input, 1).copy(5, Buffer.input, 1)   
            chunk(3, Buffer.input, 1).copy(6, Buffer.input, 1)  
            chunk(3, Buffer.input, 1).copy(7, Buffer.input, 1)  
            
            # level 3
            chunk(4, Buffer.input, 1).copy(8, Buffer.input, 1)  
            chunk(4, Buffer.input, 1).copy(9, Buffer.input, 1) 
            chunk(5, Buffer.input, 1).copy(10, Buffer.input, 1)   
            chunk(5, Buffer.input, 1).copy(11, Buffer.input, 1)  
            chunk(6, Buffer.input, 1).copy(12, Buffer.input, 1)  
            chunk(6, Buffer.input, 1).copy(13, Buffer.input, 1)  
            
            # level 4
            chunk(8, Buffer.input, 1).copy(14, Buffer.input, 1)  
            chunk(8, Buffer.input, 1).copy(15, Buffer.input, 1)          
        
        
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('trees', type=int, choices=[1, 2], help ='number of trees')
parser.add_argument('instances', type=int, help ='number of instances')

parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce_fibonacci_tree(args.num_gpus, args.instances, args.trees, args.protocol)