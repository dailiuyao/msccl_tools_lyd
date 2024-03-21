#!/bin/bash

source /home1/09168/ldai1/anaconda3/etc/profile.d/conda.sh

conda create -n msccl_tools python=3.7

conda activate msccl_tools

pip install --upgrade pip

cd /home1/09168/ldai1/ccl-build/msccl_tools_lyd

pip install -e .

# echo 'eval "$(register-python-argcomplete msccl)"' >> ~/.bashrc

# source ~/.bashrc