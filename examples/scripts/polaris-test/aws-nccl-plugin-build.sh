#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N install-ofi-nccl
#PBS -o install-ofi-nccl.out
#PBS -e install-ofi-nccl.error

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

#install aws-ofi-nccl
pushd /home/yuke/ncclPG/aws-ofi-nccl-1.7.4-aws

rm -rf build

mkdir build

cd build

../configure --prefix=/home/yuke/ncclPG/aws-ofi-nccl-1.7.4-aws/build --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda --with-hwloc=/home/yuke/install/hwloc/      

make clean

make -j8 && make install