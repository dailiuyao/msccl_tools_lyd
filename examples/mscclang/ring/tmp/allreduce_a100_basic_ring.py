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

        for index in range(0, size-1):
            rank = (index) % size
            next_rank = (index + 1) % size
            c = chunk(next_rank, Buffer.input, 0)
            c.reduce(chunk(rank, Buffer.input, 0), ch=0, recvtb=0, sendtb=0)
                

                
        # Propagate ring
        for index in range(0, size-1):
            rank = (index-1) % size
            c = chunk(rank, Buffer.input, 0)
            next_rank = (index) % size
            c = c.copy(next_rank, Buffer.input, 0, ch=0, recvtb=0, sendtb=0)
        
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allreduce_ring(args.num_gpus, args.instances, args.channels, args.protocol)
