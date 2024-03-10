# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce
import math

# Ring all reduce for A100s
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
def allreduce_ring(num_nodes, num_gpus, instances, channels, protocol):
    size = num_gpus * num_nodes
    # for each ring, the number of chunks = size 
    # because we have 2 rings, the number of chunks = 2 * size
    # if we have multiple channels, we need to divide the chunks into multiple channels, so we need chunksperchannel 
    chunksperchannel = int((2*size)/channels)
    topology = fully_connected(size)
    collective = AllReduce(size, 2*size, True)
    with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        
        # this hardcode just for 4gpus per node
        gpu_index0 = [(n + 4*i) % (int(num_nodes)*4) for i in range(int(num_nodes)) for n in [3, 2, 5, 4]]
        gpu_index1 = [(n + 4*i) % (int(num_nodes)*4) for i in range(int(num_nodes)) for n in [3, 2, 1, 0]]

        # Reduce ring
        for step in range(0, size-1):
            for index in range(0, size):
                rank = gpu_index0[(index + step) % size]
                next_rank = gpu_index0[(index + step + 1) % size]
                c = chunk(next_rank, Buffer.input, index)
                c.reduce(chunk(rank, Buffer.input, index), ch=int(index/chunksperchannel), recvtb=int(index/chunksperchannel), sendtb=int(index/chunksperchannel))
        
        for step in range(0, size-1):
            for index in range(0, size):
                rank = gpu_index1[(index + step) % size]
                next_rank = gpu_index1[(index + step + 1) % size]
                c = chunk(next_rank, Buffer.input, index+size)
                c.reduce(chunk(rank, Buffer.input, index+size), ch=int((index/chunksperchannel) + (channels/2)), recvtb=int((index/chunksperchannel) + (channels/2)), sendtb=int((index/chunksperchannel) + (channels/2)))
             
       
                
        # Propagate ring
        for step in range(-1, size-2):
            for index in range(0, size):
                rank = gpu_index0[(index + step) % size]
                next_rank = gpu_index0[(index + step + 1) % size]
                c = chunk(rank, Buffer.input, index)
                c = c.copy(next_rank, Buffer.input, index, ch=int(index/chunksperchannel), recvtb=int(index/chunksperchannel), sendtb=int(index/chunksperchannel))
        
        for step in range(-1, size-2):
            for index in range(0, size):
                rank = gpu_index1[(index + step) % size]
                next_rank = gpu_index1[(index + step + 1) % size]
                c = chunk(rank, Buffer.input, index+size)
                c = c.copy(next_rank, Buffer.input, index+size, ch=int((index/chunksperchannel) + (channels/2)), recvtb=int((index/chunksperchannel) + (channels/2)), sendtb=int((index/chunksperchannel) + (channels/2)))
            
        
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, help ='number of gpus')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('--instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allreduce_ring(args.num_nodes ,args.num_gpus, args.instances, args.channels, args.protocol)
