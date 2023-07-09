#!/bin/bash
#SBATCH --qos gpugpu
#SBATCH -N 2           # 请求节点的个数
#SBATCH --gres=gpu:4   # number of GPUs per node
#SBATCH -p vip_gpu_scx6115
#SBATCH --output=./logs/%j.log

module purge
module load compilers/cuda/11.7 
module load compilers/gcc/12.2.0
module load anaconda/2021.11
module load cudnn/8.4.0.27_cuda11.x

#export LD_LIBRARY_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/lib:$LIBRARY_PATH
#export CPATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/include:$CPATH
#export C_INCLUDE_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/include 

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml

# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TIMEOUT=23
# export NCCL_IB_RETRY_CNT=7
# export NCCL_DEBUG=INFO

source activate torch2.0
for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo ${host[$k]}
done

export PYTHONUNBUFFERED=1

# config_file='./config/AFNO.yaml'
# config='afno_backbone'
# run_num=$(date "+%Y%m%d-%H%M%S")

# config_file='./config/Model.yaml'
# config='Masked_AE_fusion'
# wandb_group='025_daily'
# run_num=$(date "+%Y%m%d-%H%M%S")

# config_file='./config/Model_2.yaml'
# config='Masked_AE_Ocean'
# wandb_group='025_daily_15_levels'
# run_num=$(date "+%Y%m%d-%H%M%S")

config_file='./config/Model_2.yaml'
config='afno'
wandb_group='025_daily_15_levels'
run_num=$(date "+%Y%m%d-%H%M%S")

# torchrun 命令用于启动 PyTorch 分布式训练
# --nnodes 参数指定节点数，
# --nproc_per_node 参数指定每个节点的进程数

# -N 参数指定节点数，
# -n 参数指定进程数，
# -c 参数指定每个进程使用的 CPU 核心数
# --gres=gpu:4 参数指定使用 4 个 GPU。


### GPUS
GPUS=4

### 脚本名称
RANK_SCRIPT="rank.sh"

### Job Path
JOB_PATH=`pwd`

### Job ID
JOB_ID="${SLURM_JOB_ID}"

### 获取节点主机名
for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo ${host[$k]}
done

### 设置主节点,将第一个节点主机名做为 master 地址.
MASTER_ADDR=${host[1]}

### Nodes
NODES="${#host[@]}"

### nodes gpus rank master_addr job_id
bash ${RANK_SCRIPT} ${NODES} ${GPUS} 0 ${MASTER_ADDR} ${JOB_ID} &

for((i=2;i<=${NODES};i++));
do
   node_host=${host[$i]}
   node_rank=${rank[$i]}
   echo "nodes:${NODES}, host:${node_host}, node_rank:${node_rank}, master_addr:${MASTER_ADDR}"
   srun -N 1 --gres=gpu:$GPUS -w $node_host bash ${RANK_SCRIPT} ${NODES} ${GPUS} $node_rank ${MASTER_ADDR} ${JOB_ID} &  
done
wait
