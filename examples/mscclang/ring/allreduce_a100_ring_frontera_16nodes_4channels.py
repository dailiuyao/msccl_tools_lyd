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
    channelsPerRing = int(channels/2)
    topology = fully_connected(size)
    collective = AllReduce(size, channels*size, True)
    with MSCCLProgram(f"allreduce_ring_{channels}channelsperring", topology, collective, instances,
         protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        
        gpu_index0 = [(n + 4*i) % (16*4) for i in range(16) for n in [0, 1, 2, 3]]
        gpu_index1 = [(n + 4*i) % (16*4) for i in range(16) for n in [3, 2, 1, 0]]
        # Reduce ring
        for step in range(0, size-1):
            for index in range(0, size):
                rank = gpu_index0[(index + step) % size]
                next_rank = gpu_index0[(index + step + 1) % size]
                for chid in range(0, channelsPerRing):
                    c = chunk(next_rank, Buffer.input, index+chid*size)
                    c.reduce(chunk(rank, Buffer.input, index+chid*size), ch=chid, recvtb=chid, sendtb=chid)
                

        for step in range(0, size-1):
            for index in range(0, size):
                rank = gpu_index1[(index + step) % size]
                next_rank = gpu_index1[(index + step + 1) % size]
                for chid in range(channelsPerRing, channels):
                    c = chunk(next_rank, Buffer.input, index+chid*size)
                    c.reduce(chunk(rank, Buffer.input, index+chid*size), ch=chid, recvtb=chid, sendtb=chid)
            
       
                
        # Propagate ring
        for step in range(-1, size-2):
            for index in range(0, size):
                rank = gpu_index0[(index + step) % size]
                next_rank = gpu_index0[(index + step + 1) % size]
                for chid in range(0, channelsPerRing):
                    c = chunk(rank, Buffer.input, index+chid*size)
                    c = c.copy(next_rank, Buffer.input, index+chid*size, ch=chid, recvtb=chid, sendtb=chid)
        

        for step in range(-1, size-2):
            for index in range(0, size):
                rank = gpu_index1[(index + step) % size]
                next_rank = gpu_index1[(index + step + 1) % size]
                for chid in range(channelsPerRing, channels):
                    c = chunk(rank, Buffer.input, index+chid*size)
                    c = c.copy(next_rank, Buffer.input, index+chid*size, ch=chid, recvtb=chid, sendtb=chid)
            
        
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allreduce_ring(args.num_gpus, args.instances, args.channels, args.protocol)
