#SBATCH --gres=gpu:4   # Number of GPUs per node
#SBATCH --qos=gpugpu   
#SBATCH -N 4           # Number of requested nodes
#SBATCH -p vip_gpu_scx6115  # Partition name
#SBATCH --output=./logs/%j.log  # Path to save the job output logs (%j is the job ID)

module purge
module load compilers/cuda/11.7 
module load compilers/gcc/12.2.0
module load anaconda/2021.11
module load cudnn/8.4.0.27_cuda11.x

export OMP_NUM_THREADS=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3

source activate torch2.0


for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  echo ${host[$k]}
done

wandb_group='025_daily_15_levels'
yaml_config='./config/config_backbone.yaml'
config='Masked_AE_Ocean' # Model type
batch_size=32
run_num=$(date "+%Y%m%d-%H%M%S") # Run ID based on the current timestamp
multi_steps_finetune=1
finetune_max_epochs=0

torchrun \
       --nnodes=4 \
       --nproc_per_node=4 \
       --rdzv_id=1 \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${host[1]}:29503" \
       train_backbone.py --enable_amp --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs --wandb_group=$wandb_group >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}_${run_num}.log 2>&1 &

wait
