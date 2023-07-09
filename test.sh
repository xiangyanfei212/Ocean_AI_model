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
source activate torch2.0
#export LD_LIBRARY_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/lib:$LIBRARY_PATH
#export CPATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/include:$CPATH
#export C_INCLUDE_PATH=/home/bingxing2/apps/nccl/2.14.3-1_cuda11.7/include 



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

### 启用IB通信

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml

for i in `scontrol show hostnames`
do
        let k=k+1
        host[$k]=$i
        echo ${host[$k]}
done


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

torchrun \
       --nnodes=4 \
       --node_rank=0 \ 
       --nproc_per_node=4 \
       --master_addr="${host[1]}" \
       --rdzv_id=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${host[1]}:29503" \
       train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num  --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}.log 2>&1 &

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -N 1 --gres=gpu:4 -w ${host[2]} torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=4 \
    --master_addr="${host[1]}" \
       --rdzv_id=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${host[1]}:29503" \
       train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num  --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}.log 2>&1 &

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -N 1 --gres=gpu:4 -w ${host[3]} torchrun \
    --nnodes=3 \
    --node_rank=2 \
    --nproc_per_node=4 \
    --master_addr="${host[1]}" \
    --rdzv_id=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${host[1]}:29503" \
       train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num  --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}.log 2>&1 &

    
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -N 1 --gres=gpu:4 -w ${host[4]} torchrun \
    --nnodes=4 \
    --node_rank=3 \
    --nproc_per_node=4 \
    --master_addr="${host[1]}" \
   --rdzv_backend=c10d \
       --rdzv_endpoint="${host[1]}:29503" \
       train_025_bingxing.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num  --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}.log 2>&1 &

wait
echo "over"




