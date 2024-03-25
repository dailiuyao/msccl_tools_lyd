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

def generate_gpu_indices(num_elements=64):
    sets_needed = num_elements // 8

    # Generate gpu_indices0
    gpu_indices0 = [n for set_num in range(sets_needed) for n in range(set_num * 8 + 7, set_num * 8 - 1, -1)]
    
    # Generate gpu_indices1 with the specified starting pattern and then following the pattern of gpu_indices0
    gpu_indices1 = []
    for set_num in range(sets_needed):
        offset = set_num * 8
        if set_num == 0:  # For the first set, follow the specific starting pattern
            gpu_indices1.extend([1, 0, 7, 6, 5, 4, 3, 2])
        else:  # For subsequent sets, create the pattern based on the offset
            new_pattern = [((n + 1) % 8) + offset if ((n + 1) % 8) != 0 else offset for n in range(offset, offset + 8)]
            gpu_indices1.extend(new_pattern)
    
    return gpu_indices0, gpu_indices1


def allreduce_ring(num_nodes, num_gpus, instances, nchunks, channels, protocol):
    size = num_gpus * num_nodes
    rings = 2
    
    chunksperchannel = int(rings * size)
    topology = fully_connected(size)
    collective = AllReduce(size, rings * size * channels, True)
    
    with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances, protocol=protocol):
        
        # gpu_indices = []
        # gpu_indices.append([7,6,5,4,3,2,1,0])  # gpu_index0
        # gpu_indices.append([0,7,6,5,4,3,2,1])  # gpu_index1

        gpu_indices0, gpu_indices1 = generate_gpu_indices(64)

        # # Print the first 16 elements of each list to verify the pattern
        # print("gpu_indices0:", gpu_indices0[:64])
        # print("gpu_indices1:", gpu_indices1[:64])



        # Using NumPy broadcasting to calculate ranks and next_ranks for both rings
        steps = np.arange(size - 1)
        indices = np.arange(size)
        # ranks_ring0 = (np.newaxis + steps) % size
        # next_ranks_ring0 = (np.newaxis + steps + 1) % size

        # ranks_ring1 = (np.newaxis + steps) % size
        # next_ranks_ring1 = (np.newaxis + steps + 1) % size

        for ring in [0, 1]:
            for step in range(size - 1):
                for index in range(size):
                    if ring == 0:
                        rank = gpu_indices0[(index + step) % size]
                        next_rank = gpu_indices0[(index + step + 1) % size]
                        offset = 0
                    else:
                        rank = gpu_indices1[(index + step) % size]
                        next_rank = gpu_indices1[(index + step + 1) % size]
                        offset = size
                    
                    c = chunk(next_rank, Buffer.input, index + offset)
                    channel_index = int((index / chunksperchannel) + (channels / 2) * ring)
                    c.reduce(chunk(rank, Buffer.input, index + offset), ch=channel_index)

            for step in range(-1, size - 2):
                for index in range(size):
                    if ring == 0:
                        rank = gpu_indices0[(index + step) % size]
                        next_rank = gpu_indices0[(index + step + 1) % size]
                        offset = 0
                    else:
                        rank = gpu_indices1[(index + step) % size]
                        next_rank = gpu_indices1[(index + step + 1) % size]
                        offset = size
                    
                    channel_index = int((index / chunksperchannel) + (channels / 2) * ring)
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
