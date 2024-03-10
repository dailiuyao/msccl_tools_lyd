# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather
import math

# Ring all reduce for A100s
# Vary channels from [1-8] to divide parts of the ring over multiple channels/tbs.
# channels=1 is standard ring, all chunks are assigned to the same tb/channel
# channels=8 devotes 1 tb/channel to handling 1 chunk of the data
def allgather_ring_pip(num_chunks:int, num_nodes:int, num_gpus:int, instances:int, channels:int, protocol:str):
    # for each ring, the number of original chunks for each gpu = num_chunks
    # the number of chunks to AllGather() is the original number of chunks for each gpu
    # total number of chunks = number_of_rings * number of channels * num_chunks * size
    size = num_gpus * num_nodes 
    number_of_rings = 2
    total_chunks = int(number_of_rings * num_chunks * channels * size)
    para_chunks = int(total_chunks/size)
    
    chunk_offset_of_index = para_chunks
    
    chunksperchannel = num_chunks
    
    channels_per_ring = channels
    
    topology = fully_connected(size)
    collective = AllGather(size, para_chunks, True)
    with MSCCLProgram(f"allgather_ring_{channels}channelsperring", topology, collective, instances,
         protocol=protocol):        
        # this hardcode just for 4gpus per node
        gpu_index0 = [(n + 4*i) % (int(num_nodes)*4) for i in range(int(num_nodes)) for n in [0, 1, 2, 3]]
        gpu_index1 = gpu_index0
                
        # Propagate ring
        ring_id = 0
        for chunk_step in range(0, chunksperchannel):
            for channel in range(0, channels_per_ring):
                channel_id_total = channel+ring_id*channels_per_ring
                chunk_offset_of_current_channel = channel_id_total*chunksperchannel
                for step in range(0, size-1):
                    for index in range(0, size):
                        rank = gpu_index0[(index + step) % size]
                        next_rank = gpu_index0[(index + step + 1) % size]
                        # print("rank, next_rank", rank, next_rank)
                        # print(index+channel_id_total*chunksperchannel)
                        # print("Before chunk:", rank, Buffer.output, index+channel_id_total*chunksperchannel)
                        # print("Before chunk:",rank, Buffer.output, index*chunk_offset_of_index + channel_id_total*chunksperchannel + chunk_step)
                        c = chunk(rank, Buffer.output, index*chunk_offset_of_index + channel_id_total*chunksperchannel + chunk_step)
                        # print("After chunk:", c)
                        c.copy(next_rank, Buffer.output, index*chunk_offset_of_index + channel_id_total*chunksperchannel + chunk_step, ch=channel_id_total)
        
        ring_id = 1
        for chunk_step in range(0, chunksperchannel):
            for channel in range(0, channels_per_ring):
                channel_id_total = channel+ring_id*channels_per_ring
                chunk_offset_of_current_channel = channel_id_total*chunksperchannel
                for step in range(0, size-1):
                    for index in range(size-1, -1, -1):
                        rank = gpu_index0[(index - step) % size]
                        next_rank = gpu_index0[(index - step - 1) % size]
                        # print("rank, next_rank", rank, next_rank)
                        # print(index+channel_id_total*chunksperchannel)
                        # print("Before chunk:", rank, Buffer.output, index+channel_id_total*chunksperchannel)
                        c = chunk(rank, Buffer.output, index*chunk_offset_of_index + channel_id_total*chunksperchannel + chunk_step)
                        # print("After chunk:", c)
                        c.copy(next_rank, Buffer.output, index*chunk_offset_of_index + channel_id_total*chunksperchannel + chunk_step, ch=channel_id_total)
        
        
               
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, help ='number of gpus')
parser.add_argument('--num_nodes', type=int, help='number of nodes')
parser.add_argument('--channels', type=int, help='Number of channels to use for 1 instance of the ring [1-8]')
parser.add_argument('--num_chunks', type=int, help='number of chunks')
parser.add_argument('--instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL128', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: LL128')
args = parser.parse_args()



allgather_ring_pip(args.num_chunks, args.num_nodes ,args.num_gpus, args.instances, args.channels, args.protocol)
