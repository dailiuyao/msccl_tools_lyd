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
def allreduce_ring(size, instances, channels, protocol):
    topology = fully_connected(size)
    collective = AllReduce(size, channels, True)
    with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        # Reduce ring
        for channel_same_link in range(0, 4):
            for step in range(0, size-1):
                for index in range(0, size):
                    rank = (index + step) % size
                    next_rank = (index + step + 1) % size
                    channel = math.floor((index)/4) + channel_same_link*6
                    c = chunk(next_rank, Buffer.input, index + channel_same_link*24)
                    c.reduce(chunk(rank, Buffer.input, index + channel_same_link*24), ch=channel, recvtb=channel, sendtb=channel)
            
            for step in range(0, size-1):
                gpu_index = [0, 1, 3, 2]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 4)/4) + channel_same_link*6
                    c = chunk(next_rank, Buffer.input, index + 4 + channel_same_link*24)
                    c.reduce(chunk(rank, Buffer.input, index + 4 + channel_same_link*24), ch=channel, recvtb=channel, sendtb=channel)
                    
            for step in range(0, size-1):
                gpu_index = [0, 2, 3, 1]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 8)/4) + channel_same_link*6
                    c = chunk(next_rank, Buffer.input, index + 8 + channel_same_link*24)
                    c.reduce(chunk(rank, Buffer.input, index + 8 + channel_same_link*24), ch=channel, recvtb=channel, sendtb=channel)
                    
            for step in range(0, size-1):
                gpu_index = [0, 2, 1, 3]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 12)/4) + channel_same_link*6
                    c = chunk(next_rank, Buffer.input, index + 12 + channel_same_link*24)
                    c.reduce(chunk(rank, Buffer.input, index + 12 + channel_same_link*24), ch=channel, recvtb=channel, sendtb=channel)
                    
            for step in range(0, size-1):
                gpu_index = [0, 3, 1, 2]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 16)/4) + channel_same_link*6
                    c = chunk(next_rank, Buffer.input, index + 16 + channel_same_link*24)
                    c.reduce(chunk(rank, Buffer.input, index + 16 + channel_same_link*24), ch=channel, recvtb=channel, sendtb=channel)
            
            for step in range(0, size-1):
                gpu_index = [0, 3, 2, 1]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 20)/4) + channel_same_link*6
                    c = chunk(next_rank, Buffer.input, index + 20 + channel_same_link*24)
                    c.reduce(chunk(rank, Buffer.input, index + 20 + channel_same_link*24), ch=channel, recvtb=channel, sendtb=channel)
                
        # Propagate ring
        for channel_same_link in range(0, 4):
            for step in range(-1, size-2):
                for index in range(0, size):
                    rank = (index + step) % size
                    c = chunk(rank, Buffer.input, index + channel_same_link*24)
                    next_rank = (index + step + 1) % size
                    channel = math.floor((index)/4) + channel_same_link*6
                    c = c.copy(next_rank, Buffer.input, index + channel_same_link*24, ch=channel, recvtb=channel, sendtb=channel)
            
            for step in range(-1, size-2):
                gpu_index = [0, 1, 3, 2]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + 4 + channel_same_link*24)
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 4)/4) + channel_same_link*6
                    c = c.copy(next_rank, Buffer.input, index + 4 + channel_same_link*24, ch=channel, recvtb=channel, sendtb=channel)
                    
            for step in range(-1, size-2):
                gpu_index = [0, 2, 3, 1]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + 8 + channel_same_link*24)
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 8)/4) + channel_same_link*6
                    c = c.copy(next_rank, Buffer.input, index + 8 + channel_same_link*24, ch=channel, recvtb=channel, sendtb=channel)
                    
            for step in range(-1, size-2):
                gpu_index = [0, 2, 1, 3]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + 12 + channel_same_link*24)
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 12)/4) + channel_same_link*6
                    c = c.copy(next_rank, Buffer.input, index + 12 + channel_same_link*24, ch=channel, recvtb=channel, sendtb=channel)
            
            for step in range(-1, size-2):
                gpu_index = [0, 3, 1, 2]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + 16 + channel_same_link*24)
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 16)/4) + channel_same_link*6
                    c = c.copy(next_rank, Buffer.input, index + 16 + channel_same_link*24, ch=channel, recvtb=channel, sendtb=channel)
                    
            for step in range(-1, size-2):
                gpu_index = [0, 3, 2, 1]
                for index in range(0, size):
                    rank = gpu_index[(index + step) % size]
                    c = chunk(rank, Buffer.input, index + 20 + channel_same_link*24)
                    next_rank = gpu_index[(index + step + 1) % size]
                    channel = math.floor((index + 20)/4) + channel_same_link*6
                    c = c.copy(next_rank, Buffer.input, index + 20 + channel_same_link*24, ch=channel, recvtb=channel, sendtb=channel)
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allreduce_ring(args.num_gpus, args.instances, args.channels, args.protocol)
