#!/bin/bash 
#SBATCH --gres=gpu:4   # number of GPUs per node
#SBATCH -N 4	       # 请求节点的个数
#SBATCH --output=%j.log
#SBATCH --qos=gpugpu # 告诉 SLURM 调度器请求使用 GPU 资源

module load compilers/gcc/12.2.0
module load anaconda/2021.11
module load compilers/cuda/11.7
module load cudnn/8.4.0.27_cuda11.x
source activate torch2.0


export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo ${host[$k]}
done

export PYTHONUNBUFFERED=1

config_file='./config/AFNO.yaml'
config='afno_backbone'
run_num=$(date "+%Y%m%d-%H%M%S")

# python -m torch.distributed.launch 命令用于启动 PyTorch 分布式训练
# --nnodes 参数指定节点数，
# --nproc_per_node 参数指定每个节点的进程数
# --node_rank 参数指定当前节点的排名
# --master_addr 参数指定主节点的地址
# --master_port 参数指定主节点的端口号

# srun 命令用于在指定节点上启动一个任务。
# -N 参数指定节点数，
# -n 参数指定进程数，
# -c 参数指定每个进程使用的 CPU 核心数
# --gres=gpu:4 参数指定使用 4 个 GPU。

python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 --node_rank=0 --master_addr=${host[1]} --master_port=12321 train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num >> rank0_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[2]} python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 --node_rank=1 --master_addr=${host[1]} --master_port=12321 train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num >> rank1_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[3]} python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 --node_rank=2 --master_addr=${host[1]} --master_port=12321 train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num >> rank2_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[4]} python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 --node_rank=3 --master_addr=${host[1]} --master_port=12321 train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num >> rank3_${SLURM_JOB_ID}.log 2>&1 &
wait

