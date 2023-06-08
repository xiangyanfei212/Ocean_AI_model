#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fourcastnet
#export NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING=FALSE
#export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=eno1
#export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export MIOPEN_USER_DB_PATH=/tmp/xiongwei06_user
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/xiongwei06_chahe
export MIOPEN_FIND_MODE=5
export MIOPEN_SYSTEM_DB_PATH=/tmp/xiongwei06

export LD_LIBRARY_PATH=/public/home/acrzcyisbk/miniconda3/lib:$LD_LIBRARY_PATH

module purge 
module load mpi/hpcx/2.7.4-gcc-7.3.1
module load compiler/dtk/22.04.2

