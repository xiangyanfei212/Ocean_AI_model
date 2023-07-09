#!/bin/bash 
#SBATCH --gres=gpu:4   # number of GPUs per node
#SBATCH --qos=gpugpu   
#SBATCH -N 4	       # 请求节点的个数
#SBATCH -p vip_gpu_scx6115
#SBATCH --output=./logs/%j.log
module purge
module load compilers/cuda/11.7 
module load compilers/gcc/12.2.0
module load anaconda/2021.11
module load cudnn/8.4.0.27_cuda11.x
export OMP_NUM_THREADS=1
#export LD_LIBRARY_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/lib:$LIBRARY_PATH
#export CPATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/include:$CPATH
#export C_INCLUDE_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/include 

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3


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

# yaml_config='./config/AFNO.yaml'
# config='afno_backbone'
# run_num=$(date "+%Y%m%d-%H%M%S")

# yaml_config='./config/Model.yaml'
# config='Masked_AE_fusion'
# wandb_group='025_daily'
# run_num=$(date "+%Y%m%d-%H%M%S")

wandb_group='025_daily_15_levels'
yaml_config='./config/Model_2.yaml'
config='Masked_AE_Ocean'
run_num=$(date "+%Y%m%d-%H%M%S")
batch_size=32
multi_steps_finetune=1
finetune_max_epochs=0 # valid when multi_steps_finetune>1

# yaml_config='./config/Model_2.yaml'
# config='afno'
# wandb_group='025_daily_15_levels'
# run_num=$(date "+%Y%m%d-%H%M%S")

############### multi steps finetune ##############
# wandb_group='025_daily_15_levels'
# config='Masked_AE_Ocean'
# run_num="20230628-131150"
# batch_size=16
# yaml_config=./exp_15_levels/${config}/${run_num}/config.yaml
# multi_steps_finetune=3 # 1: train with single step loss, 2: train with two steps loss, 3: train with three steps loss
# finetune_max_epochs=50

# torchrun 命令用于启动 PyTorch 分布式训练
# --nnodes 参数指定节点数，
# --nproc_per_node 参数指定每个节点的进程数

# -N 参数指定节点数，
# -n 参数指定进程数，
# -c 参数指定每个进程使用的 CPU 核心数
# --gres=gpu:4 参数指定使用 4 个 GPU。

torchrun \
       --nnodes=4 \
       --nproc_per_node=4 \
       --rdzv_id=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${host[1]}:29503" \
       train_025_multi_steps.py --enable_amp --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[2]} torchrun --nnodes=4  --nproc_per_node=4  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="${host[1]}:29503"  train_025_multi_steps.py --enable_amp --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank1_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[3]} torchrun --nnodes=4 --nproc_per_node=4  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="${host[1]}:29503" train_025_multi_steps.py --enable_amp --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank2_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[4]} torchrun --nnodes=4 --nproc_per_node=4  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="${host[1]}:29503" train_025_multi_steps.py --enable_amp --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank3_${SLURM_JOB_ID}.log 2>&1 &

wait

