# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce
import math
import numpy as np
        
def generate_gpu_indices(num_elements=64, num_gpus=4):

    # Generate gpu_indices
    gpu_indices = []
    
    if num_gpus == 4:
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
        gpu_indices.append([3, 2, 1, 0])
        gpu_indices.append([1, 0, 3, 2])
        gpu_indices.append(list(reversed([3, 2, 1, 0])))
        gpu_indices.append(list(reversed([1, 0, 3, 2])))
    elif num_gpus == 8:
        gpu_indices.append([7,6,5,4,3,2,1,0])  
        gpu_indices.append([0,7,6,5,4,3,2,1]) 
        gpu_indices.append([1,0,7,6,5,4,3,2]) 
        gpu_indices.append([2,1,0,7,6,5,4,3])
        gpu_indices.append([3,2,1,0,7,6,5,4])
        gpu_indices.append([4,3,2,1,0,7,6,5])
        gpu_indices.append([5,4,3,2,1,0,7,6])
        gpu_indices.append([6,5,4,3,2,1,0,7])
        gpu_indices.append(list(reversed([7,6,5,4,3,2,1,0])))
        gpu_indices.append(list(reversed([0,7,6,5,4,3,2,1])))
        gpu_indices.append(list(reversed([1,0,7,6,5,4,3,2])))
        gpu_indices.append(list(reversed([2,1,0,7,6,5,4,3])))
        gpu_indices.append(list(reversed([3,2,1,0,7,6,5,4])))
        gpu_indices.append(list(reversed([4,3,2,1,0,7,6,5])))
        gpu_indices.append(list(reversed([5,4,3,2,1,0,7,6])))
        gpu_indices.append(list(reversed([6,5,4,3,2,1,0,7])))
        gpu_indices.append([7,6,5,4,3,2,1,0])  
        gpu_indices.append([0,7,6,5,4,3,2,1]) 
        gpu_indices.append([1,0,7,6,5,4,3,2]) 
        gpu_indices.append([2,1,0,7,6,5,4,3])
        gpu_indices.append([3,2,1,0,7,6,5,4])
        gpu_indices.append([4,3,2,1,0,7,6,5])
        gpu_indices.append([5,4,3,2,1,0,7,6])
        gpu_indices.append([6,5,4,3,2,1,0,7])
        gpu_indices.append(list(reversed([7,6,5,4,3,2,1,0])))
        gpu_indices.append(list(reversed([0,7,6,5,4,3,2,1])))
        gpu_indices.append(list(reversed([1,0,7,6,5,4,3,2])))
        gpu_indices.append(list(reversed([2,1,0,7,6,5,4,3])))
        gpu_indices.append(list(reversed([3,2,1,0,7,6,5,4])))
        gpu_indices.append(list(reversed([4,3,2,1,0,7,6,5])))
        gpu_indices.append(list(reversed([5,4,3,2,1,0,7,6])))
        gpu_indices.append(list(reversed([6,5,4,3,2,1,0,7])))
    
    # # Number of groups after the initial one
    # num_groups = num_elements // num_gpus - 1  # Subtract 1 for the initial group already defined
    
    # # Generate the subsequent groups by adding 8 * group_number to each element of the previous group
    # NCCL_MAX_CHANNELS = 32  
    # for gpu_indices_id in range(NCCL_MAX_CHANNELS):
    #     for group in range(1, num_groups + 1):
    #         new_group = [(x + num_gpus) % num_elements for x in gpu_indices[gpu_indices_id][-num_gpus:]]  # Use modulo 64 to ensure numbers are within bounds
    #         gpu_indices[gpu_indices_id].extend(new_group)
            

    return gpu_indices


# reduce from 0 -> num_nodes-1
def intra_reduce(node_offset=0, num_gpus=4, gpu_indices=[0,0,0,0], chunk_id=0, ring_id=0):
    rank_offset = node_offset * num_gpus
    for index in range(0, num_gpus-1):
        other = chunk(int((gpu_indices[index])+rank_offset), Buffer.input, chunk_id)
        c1 = chunk(int((gpu_indices[index+1])+rank_offset), Buffer.input, chunk_id) 
        c1.reduce(other, ch=int(ring_id))
        
# broadcast from num_nodes-1 -> 0       
def intra_broadcast(node_offset=0, num_gpus=4, gpu_indices=[0,0,0,0], chunk_id=0, ring_id=0):    
    rank_offset = node_offset * num_gpus
    for index in range(0, num_gpus-1):
        c = chunk(int((gpu_indices[num_gpus - 1 - index])%num_gpus + rank_offset), Buffer.input, chunk_id)
        c.copy(int((gpu_indices[num_gpus - 2 - index])%num_gpus + rank_offset), Buffer.input, chunk_id, ch=ring_id)
        

def allreduce_ring(num_nodes, num_gpus, instances, nchunks, channels, protocol):
    size = num_gpus * num_nodes
    
    chunksperchannel = num_nodes
    topology = fully_connected(size)
    collective = AllReduce(size, nchunks * num_nodes * channels , True)
    
    with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances, protocol=protocol):

        gpu_indices = generate_gpu_indices(size, num_gpus)

        for chunk_id in range(nchunks):
            for ring_id in range(channels):
                
                chunk_offset = ring_id * num_nodes + chunk_id * num_nodes * channels
                
                # reduce from 0 -> num_nodes-1 for all nodes
                for node_offset in range(num_nodes):
                    for index in range(num_nodes):
                        intra_reduce(node_offset, num_gpus, gpu_indices[ring_id], index + chunk_offset, ring_id) 
                
                # allreduce inter-node
                for step in range(num_nodes - 1):
                    for index in range(num_nodes):
                        rank = ((index + step) % num_nodes)*num_gpus+gpu_indices[ring_id][num_gpus-1]
                        next_rank = ((index + step + 1) % num_nodes)*num_gpus+gpu_indices[ring_id][num_gpus-1]
                                                
                        c = chunk(int(next_rank), Buffer.input, index + chunk_offset)
                        channel_index = int(ring_id+channels)
                        c.reduce(chunk(rank, Buffer.input, index + chunk_offset), ch=channel_index)

                for step in range(-1, num_nodes - 2):
                    for index in range(num_nodes):
                        rank = ((index + step) % num_nodes)*num_gpus+gpu_indices[ring_id][num_gpus-1]
                        next_rank = ((index + step + 1) % num_nodes)*num_gpus+gpu_indices[ring_id][num_gpus-1]
                                                
                        channel_index = int(ring_id+2*channels)
                        chunk(rank, Buffer.input, index + chunk_offset).copy(next_rank, Buffer.input, index + chunk_offset, ch=channel_index)

                # reduce from 0 -> num_nodes-1 for all nodes
                for node_offset in range(num_nodes):
                    for index in range(num_nodes):
                        intra_broadcast(node_offset, num_gpus, gpu_indices[ring_id], index + chunk_offset, ring_id+3*channels) 
        
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, help ='number of gpus')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--nchannel', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('--nchunks', type=int, help='Number of chunks')
parser.add_argument('--instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()

allreduce_ring(args.num_nodes ,args.num_gpus, args.instances, args.nchunks, args.nchannel, args.protocol)
