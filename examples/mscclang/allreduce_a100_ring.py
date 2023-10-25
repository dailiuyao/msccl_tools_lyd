# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

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
        ### channel 0-3
        step = 0
        for index in range(0, size):
            rank = (index + step) % size
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = chunk(rank, Buffer.input, index)
            c.copy(next_rank, 'scratch', index, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(1, size-2):
            for index in range(0, size):
                rank = (index + step) % size
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = chunk(rank, Buffer.input, index)
                c.reduce(chunk(rank, 'scratch', index), ch=channel, recvtb=channel, sendtb=channel)
                c1 = chunk(rank, Buffer.input, index)
                c1.copy(next_rank, 'scratch', index, ch=channel, recvtb=channel, sendtb=channel)
        
        step = size-2
        for index in range(0, size):
            rank = (index + step) % size
            next_rank = (index + step + 1) % size
            channel = index%channels
            c = chunk(rank, Buffer.input, index)
            c.reduce(chunk(rank, 'scratch', index), ch=channel, recvtb=channel, sendtb=channel)
            c1 = chunk(rank, Buffer.input, index)
            c1.copy(next_rank, 'scratch', index, ch=channel, recvtb=channel, sendtb=channel)
            c2 = chunk(next_rank, Buffer.input, index) 
            c2.reduce(chunk(next_rank, 'scratch', index), ch=channel, recvtb=channel, sendtb=channel)
        
        # channel 4-7
        gpu_index = [0, 1, 3, 2]
        step = 0
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 4
            c = chunk(rank, Buffer.input, index + 4)
            c.copy(next_rank, 'scratch', index + 4, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(1, size-2):
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 4
                c = chunk(rank, Buffer.input, index + 4)
                c.reduce(chunk(rank, 'scratch', index + 4), ch=channel, recvtb=channel, sendtb=channel)
                c1 = chunk(rank, Buffer.input, index + 4)
                c1.copy(next_rank, 'scratch', index + 4, ch=channel, recvtb=channel, sendtb=channel)
        
        step = size-2
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 4
            c = chunk(rank, Buffer.input, index + 4)
            c.reduce(chunk(rank, 'scratch', index + 4), ch=channel, recvtb=channel, sendtb=channel)
            c1 = chunk(rank, Buffer.input, index + 4)
            c1.copy(next_rank, 'scratch', index + 4, ch=channel, recvtb=channel, sendtb=channel)
            c2 = chunk(next_rank, Buffer.input, index + 4) 
            c2.reduce(chunk(next_rank, 'scratch', index + 4), ch=channel, recvtb=channel, sendtb=channel)
        
        ### channel 8-11        
        gpu_index = [0, 2, 3, 1]
        step = 0
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 8
            c = chunk(rank, Buffer.input, index + 8)
            c.copy(next_rank, 'scratch', index + 8, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(1, size-2):
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 8
                c = chunk(rank, Buffer.input, index + 8)
                c.reduce(chunk(rank, 'scratch', index + 8), ch=channel, recvtb=channel, sendtb=channel)
                c1 = chunk(rank, Buffer.input, index + 8)
                c1.copy(next_rank, 'scratch', index + 8, ch=channel, recvtb=channel, sendtb=channel)
        
        step = size-2
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 8
            c = chunk(rank, Buffer.input, index + 8)
            c.reduce(chunk(rank, 'scratch', index + 8), ch=channel, recvtb=channel, sendtb=channel)
            c1 = chunk(rank, Buffer.input, index + 8)
            c1.copy(next_rank, 'scratch', index + 8, ch=channel, recvtb=channel, sendtb=channel)
            c2 = chunk(next_rank, Buffer.input, index + 8) 
            c2.reduce(chunk(next_rank, 'scratch', index + 8), ch=channel, recvtb=channel, sendtb=channel)
        
        
        # channel 12-15         
        gpu_index = [0, 2, 1, 3]
        step = 0
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 12
            c = chunk(rank, Buffer.input, index + 12)
            c.copy(next_rank, 'scratch', index + 12, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(1, size-2):
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 12
                c = chunk(rank, Buffer.input, index + 12)
                c.reduce(chunk(rank, 'scratch', index + 12), ch=channel, recvtb=channel, sendtb=channel)
                c1 = chunk(rank, Buffer.input, index + 12)
                c1.copy(next_rank, 'scratch', index + 12, ch=channel, recvtb=channel, sendtb=channel)
        
        step = size-2
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 12
            c = chunk(rank, Buffer.input, index + 12)
            c.reduce(chunk(rank, 'scratch', index + 12), ch=channel, recvtb=channel, sendtb=channel)
            c1 = chunk(rank, Buffer.input, index + 12)
            c1.copy(next_rank, 'scratch', index + 12, ch=channel, recvtb=channel, sendtb=channel)
            c2 = chunk(next_rank, Buffer.input, index + 12) 
            c2.reduce(chunk(next_rank, 'scratch', index + 12), ch=channel, recvtb=channel, sendtb=channel)
        
        # channel 16-19              
        gpu_index = [0, 3, 1, 2]
        step = 0
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 16
            c = chunk(rank, Buffer.input, index + 16)
            c.copy(next_rank, 'scratch', index + 16, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(1, size-2):
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 16
                c = chunk(rank, Buffer.input, index + 16)
                c.reduce(chunk(rank, 'scratch', index + 16), ch=channel, recvtb=channel, sendtb=channel)
                c1 = chunk(rank, Buffer.input, index + 16)
                c1.copy(next_rank, 'scratch', index + 16, ch=channel, recvtb=channel, sendtb=channel)
        
        step = size-2
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 16
            c = chunk(rank, Buffer.input, index + 16)
            c.reduce(chunk(rank, 'scratch', index + 16), ch=channel, recvtb=channel, sendtb=channel)
            c1 = chunk(rank, Buffer.input, index + 16)
            c1.copy(next_rank, 'scratch', index + 16, ch=channel, recvtb=channel, sendtb=channel)
            c2 = chunk(next_rank, Buffer.input, index + 16) 
            c2.reduce(chunk(next_rank, 'scratch', index + 16), ch=channel, recvtb=channel, sendtb=channel)
        
        # channel 20-23
        gpu_index = [0, 3, 2, 1]
        step = 0
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 20
            c = chunk(rank, Buffer.input, index + 20)
            c.copy(next_rank, 'scratch', index + 20, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(1, size-2):
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 20
                c = chunk(rank, Buffer.input, index + 20)
                c.reduce(chunk(rank, 'scratch', index + 20), ch=channel, recvtb=channel, sendtb=channel)
                c1 = chunk(rank, Buffer.input, index + 20)
                c1.copy(next_rank, 'scratch', index + 20, ch=channel, recvtb=channel, sendtb=channel)
        
        step = size-2
        for index in range(0, size):
            rank = gpu_index[(index + step) % size]
            next_rank = gpu_index[(index + step + 1) % size]
            channel = index%channels + 20
            c = chunk(rank, Buffer.input, index + 20)
            c.reduce(chunk(rank, 'scratch', index + 20), ch=channel, recvtb=channel, sendtb=channel)
            c1 = chunk(rank, Buffer.input, index + 20)
            c1.copy(next_rank, 'scratch', index + 20, ch=channel, recvtb=channel, sendtb=channel)
            c2 = chunk(next_rank, Buffer.input, index + 20) 
            c2.reduce(chunk(next_rank, 'scratch', index + 20), ch=channel, recvtb=channel, sendtb=channel)
                
        # Propagate ring
        for step in range(-1, size-2):
            for index in range(0, size):
                rank = (index + step) % size
                c = chunk(rank, Buffer.input, index)
                next_rank = (index + step + 1) % size
                channel = index%channels
                c = c.copy(next_rank, Buffer.input, index, ch=channel, recvtb=channel, sendtb=channel)
        
        for step in range(-1, size-2):
            gpu_index = [0, 1, 3, 2]
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                c = chunk(rank, Buffer.input, index + 4)
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 4
                c = c.copy(next_rank, Buffer.input, index + 4, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(-1, size-2):
            gpu_index = [0, 2, 3, 1]
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                c = chunk(rank, Buffer.input, index + 8)
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 8
                c = c.copy(next_rank, Buffer.input, index + 8, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(-1, size-2):
            gpu_index = [0, 2, 1, 3]
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                c = chunk(rank, Buffer.input, index + 12)
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 12
                c = c.copy(next_rank, Buffer.input, index + 12, ch=channel, recvtb=channel, sendtb=channel)
        
        for step in range(-1, size-2):
            gpu_index = [0, 3, 1, 2]
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                c = chunk(rank, Buffer.input, index + 16)
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 16
                c = c.copy(next_rank, Buffer.input, index + 16, ch=channel, recvtb=channel, sendtb=channel)
                
        for step in range(-1, size-2):
            gpu_index = [0, 3, 2, 1]
            for index in range(0, size):
                rank = gpu_index[(index + step) % size]
                c = chunk(rank, Buffer.input, index + 20)
                next_rank = gpu_index[(index + step + 1) % size]
                channel = index%channels + 20
                c = c.copy(next_rank, Buffer.input, index + 20, ch=channel, recvtb=channel, sendtb=channel)
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allreduce_ring(args.num_gpus, args.instances, args.channels, args.protocol)
