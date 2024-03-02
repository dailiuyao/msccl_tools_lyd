# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllReduce

def calculate_index_vector_halving_distance_doubling(group_index, count, size):
    # Calculate the number of groups based on the current count
    num_groups = size / (count*2)
    # Calculate the size of each group
    group_size = size // num_groups
    for rank in range(size):
        # Calculate the position within the group
        position_in_group = rank % group_size
        # Adjust the position based on the count to generate the pattern
        group_index[rank] += (position_in_group//count)*num_groups

def calculate_index_vector_doubling_distance_halving(group_index, count, size):
    # Calculate the number of groups based on the current count
    num_groups = size / (count*2)
    # Calculate the size of each group
    group_size = size // num_groups
    for rank in range(size):
        # Calculate the position within the group
        position_in_group = rank % group_size
        # Adjust the position based on the count to generate the pattern
        group_index[rank] -= (position_in_group//count)*num_groups

def reduce_scatter_vector_halving_distance_doubling(size, group_index):
    count = 1
    while count < size:
        calculate_index_vector_halving_distance_doubling(group_index, count, size)
        size_count = int((size // 2)/count)
        for rank in range(size):
            peer = rank ^ count
            index = int(group_index[peer])
            # print(index)
            c1 = chunk(rank, Buffer.input, index, size=size_count)
            chunk(peer, Buffer.output, index, size=size_count).reduce(c1, sendtb=peer, recvtb=rank, ch=0)
        count *= 2

def allgather_recursive_vector_doubling_distance_halving(size, group_index):
    count = size // 2
    size_count = int((size // 2) / count)
    while count >= 1:
        # print(group_index)
        size_count = int((size // 2) / count)
        for rank in range(size):
            peer = rank ^ count
            index = int(group_index[rank])
            chunk(rank, Buffer.output, index, size=size_count).copy(peer, Buffer.output, index, sendtb=peer, recvtb=rank, ch=0) 
        count //= 2
        if (count != 0):
            calculate_index_vector_doubling_distance_halving(group_index, count*2, size)

def allreduce(size, instances, protocol):
    topology = fully_connected(size)
    logical_chunk = size
    collective = AllReduce(size, logical_chunk, True)
    with MSCCLProgram("allreduce_recursive_doubling_halving", topology, collective, instances, protocol,
         interleaved_replication=False, threadblock_policy=ThreadblockPolicy.manual):
        group_index = [0] * size
        reduce_scatter_vector_halving_distance_doubling(size, group_index)
        allgather_recursive_vector_doubling_distance_halving(size, group_index)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, help ='number of gpus')
parser.add_argument('--instances', type=int, help='number of instances')
parser.add_argument('--protocol', type=str, default='LL', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()
allreduce(args.num_gpus, args.instances, args.protocol)
