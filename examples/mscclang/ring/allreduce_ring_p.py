# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce
import math
import numpy as np

# Ring all reduce for A100s
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
# def allreduce_ring(num_nodes, num_gpus, instances, channels, protocol):
#     size = num_gpus * num_nodes
#     # for each ring, the number of chunks = size 
#     # because we have 2 rings, the number of chunks = 2 * size
#     # if we have multiple channels, we need to divide the chunks into multiple channels, so we need chunksperchannel
    
#     #hardcode for 1 channel 
#     rings = 2
    
#     chunksperchannel = int((rings*size)/channels)
#     topology = fully_connected(size)
#     collective = AllReduce(size, rings*size, True)
#     with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances,
#          protocol=protocol):
        
#         # this hardcode just for 4gpus per node
#         gpu_index0 = list(range(0, size, 1))
#         gpu_index1 = list(reversed(gpu_index0))

#         # Reduce ring
#         for step in range(0, size-1):
#             for index in range(0, size):
#                 rank = gpu_index0[(index + step) % size]
#                 next_rank = gpu_index0[(index + step + 1) % size]
#                 c = chunk(next_rank, Buffer.input, index)
#                 c.reduce(chunk(rank, Buffer.input, index), ch=int(index/chunksperchannel))
        
#         if rings == 2:
#             for step in range(0, size-1):
#                 for index in range(0, size):
#                     rank = gpu_index1[(index + step) % size]
#                     next_rank = gpu_index1[(index + step + 1) % size]
#                     c = chunk(next_rank, Buffer.input, index+size)
#                     c.reduce(chunk(rank, Buffer.input, index+size), ch=int((index/chunksperchannel) + (channels/2)))
             
       
                
#         # Propagate ring
#         for step in range(-1, size-2):
#             for index in range(0, size):
#                 rank = gpu_index0[(index + step) % size]
#                 next_rank = gpu_index0[(index + step + 1) % size]
#                 c = chunk(rank, Buffer.input, index)
#                 c = c.copy(next_rank, Buffer.input, index, ch=int(index/chunksperchannel))
        
#         if rings == 2:
#             for step in range(-1, size-2):
#                 for index in range(0, size):
#                     rank = gpu_index1[(index + step) % size]
#                     next_rank = gpu_index1[(index + step + 1) % size]
#                     c = chunk(rank, Buffer.input, index+size)
#                     c = c.copy(next_rank, Buffer.input, index+size, ch=int((index/chunksperchannel) + (channels/2)))
                
        
               
#         XML()
#         Check()


def generate_gpu_indices(num_elements=64, num_gpus=4):

    # # Generate gpu_indices0
    # gpu_indices0 = [3, 2, 1, 0]
    
    # gpu_indices1 = [1, 0, 3, 2]
    
    gpu_indices = []
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
    
    # Number of groups after the initial one
    num_groups = num_elements // num_gpus - 1  # Subtract 1 for the initial group already defined
    
    # Generate the subsequent groups by adding 8 * group_number to each element of the previous group
    NCCL_MAX_CHANNELS = 32  
    for gpu_indices_id in range(NCCL_MAX_CHANNELS):
        for group in range(1, num_groups + 1):
            new_group = [(x + num_gpus) % num_elements for x in gpu_indices[gpu_indices_id][-num_gpus:]]  # Use modulo 64 to ensure numbers are within bounds
            gpu_indices[gpu_indices_id].extend(new_group)
            

    return gpu_indices


def allreduce_ring(num_nodes, num_gpus, instances, nchunks, channels, protocol):
    size = num_gpus * num_nodes
    
    chunksperchannel = size
    topology = fully_connected(size)
    collective = AllReduce(size, nchunks * size * channels , True)
    
    with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances, protocol=protocol):
        
        # gpu_indices = []
        # gpu_indices.append([7,6,5,4,3,2,1,0])  # gpu_index0
        # gpu_indices.append([0,7,6,5,4,3,2,1])  # gpu_index1

        gpu_indices = generate_gpu_indices(size, num_gpus)
                
        # # Print the first 16 elements of each list to verify the pattern
        # print("gpu_indices0:", gpu_indices0[:64])
        # print("gpu_indices1:", gpu_indices1[:64])



        # Using NumPy broadcasting to calculate ranks and next_ranks for both rings
        # steps = np.arange(size - 1)
        # indices = np.arange(size)
        # ranks_ring0 = (np.newaxis + steps) % size
        # next_ranks_ring0 = (np.newaxis + steps + 1) % size

        # ranks_ring1 = (np.newaxis + steps) % size
        # next_ranks_ring1 = (np.newaxis + steps + 1) % size

        for chunk_id in range(nchunks):
            for ring_id in range(channels):
                for step in range(size - 1):
                    for index in range(size):
                        rank = gpu_indices[ring_id][(index + step) % size]
                        next_rank = gpu_indices[ring_id][(index + step + 1) % size]
                        
                        offset = ring_id * size + chunk_id * size * channels
                        
                        c = chunk(int(next_rank), Buffer.input, index + offset)
                        channel_index = int(ring_id)
                        c.reduce(chunk(rank, Buffer.input, index + offset), ch=channel_index)

                for step in range(-1, size - 2):
                    for index in range(size):
                        rank = gpu_indices[ring_id][(index + step) % size]
                        next_rank = gpu_indices[ring_id][(index + step + 1) % size]
                        
                        offset = ring_id * size + chunk_id * size * channels

                        channel_index = int(ring_id)
                        chunk(rank, Buffer.input, index + offset).copy(next_rank, Buffer.input, index + offset, ch=channel_index)
        
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
