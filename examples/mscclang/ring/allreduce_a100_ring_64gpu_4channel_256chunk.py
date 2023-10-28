# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce
import math

# ngpus=64 nchunks=256 
# Ring all reduce for A100s 
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
def allreduce_ring(size, instances, nchunks, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, nchunks, True)
    with MSCCLProgram(f"allreduce_ring_{nchunks}channelsperring", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        # Reduce ring
        gpu_index0 = []
        for i in range(64):
            gpu_index0.append(((i // 4) * 4 + [0, 3, 2, 5][i % 4])%64)
        
        gpu_index1 = []
        for i in range(64):
            gpu_index1.append(((i // 4) * 4 + [0, 7, 6, 5][i % 4])%64)

        for channel_same_link in range(0, 2):
            for step in range(0, size-1):
                for index in range(0, size):
                    rank = gpu_index0[(index + step) % size]
                    next_rank = gpu_index0[(index + step + 1) % size]
                    channel = math.floor((index)/64) + channel_same_link*2
                    c = chunk(next_rank, Buffer.input, index + channel_same_link*128)
                    c.reduce(chunk(rank, Buffer.input, index + channel_same_link*128), ch=channel, recvtb=channel, sendtb=channel)
            
            for step in range(0, size-1):
                for index in range(0, size):
                    rank = gpu_index1[(index + step) % size]
                    next_rank = gpu_index1[(index + step + 1) % size]
                    channel = math.floor((index + 64)/64) + channel_same_link*2
                    c = chunk(next_rank, Buffer.input, index + 64 + channel_same_link*128)
                    c.reduce(chunk(rank, Buffer.input, index + 64 + channel_same_link*128), ch=channel, recvtb=channel, sendtb=channel)
                    
                
        # Propagate ring
        for channel_same_link in range(0, 2):
            for step in range(-1, size-2):
                for index in range(0, size):
                    rank = gpu_index0[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + channel_same_link*128)
                    next_rank = gpu_index0[(index + step + 1) % size]
                    channel = math.floor((index)/64) + channel_same_link*2
                    c = c.copy(next_rank, Buffer.input, index + channel_same_link*128, ch=channel, recvtb=channel, sendtb=channel)
            
            for step in range(-1, size-2):
                for index in range(0, size):
                    rank = gpu_index1[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + 64 + channel_same_link*128)
                    next_rank = gpu_index1[(index + step + 1) % size]
                    channel = math.floor((index + 64)/64) + channel_same_link*2
                    c = c.copy(next_rank, Buffer.input, index + 64 + channel_same_link*128, ch=channel, recvtb=channel, sendtb=channel)
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('nchunks', type=int, help='Number of chunks to use for 1 instance of the ring')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allreduce_ring(args.num_gpus, args.instances, args.nchunks, args.protocol)
