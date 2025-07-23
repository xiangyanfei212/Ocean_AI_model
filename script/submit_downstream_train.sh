#!/bin/bash 
#SBATCH --gres=gpu:4   # Number of GPUs per node
#SBATCH --qos=gpugpu   # Quality of Service (QoS) for the job
#SBATCH -N 4           # Number of nodes requested
#SBATCH -p vip_gpu_scx6115  # Partition to use
#SBATCH --output=./logs/%j.log  # Path to save job logs (%j is replaced by the job ID)

# Load necessary modules
module purge
module load compilers/cuda/11.7
module load compilers/gcc/12.2.0
module load anaconda/2021.11
module load cudnn/8.4.0.27_cuda11.x

export OMP_NUM_THREADS=1
# NCCL settings for distributed training
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3

# Activate Python environment
source activate torch2.0

# Resolve the hostnames of allocated nodes
for i in `scontrol show hostnames`  # Get a list of all allocated nodes
do
  let k=k+1
  host[$k]=$i
  echo ${host[$k]}
done

export PYTHONUNBUFFERED=1  # Prevent Python from buffering output

# Define training configuration variables
yaml_config='./config/config_downstream.yaml'  # Path to the YAML configuration file
pretrained_dir='./exps/Masked_AE_Ocean/20230628-131150/2_steps_finetune'  # Directory for pretrained model weights
downstream_config='DownScalingNet'  # Configuration of the downstream model (e.g., DownScalingNet, BiochemicalNet, WaveNet)

# downstream options
freeze_backbone=1  # Whether to freeze the backbone model during training downstream model (1 = True, 0 = False)
use_mom_loss=1  # Whether to use momentum loss (1 = True, 0 = False)
add_noise=1  # Whether to add noise to the data (1 = True, 0 = False)
noise_mean=0  # Mean of the Gaussian noise
noise_std=0.2  # Standard deviation of the Gaussian noise

# Training options
downstream_max_epochs=100  # Maximum number of epochs for training
batch_size=32  # Training batch size
wandb_group='Downstream_'${downstream_config}'_noise'  # Weights & Biases (W&B) group name for experiment tracking
run_num=$(date "+%Y%m%d-%H%M%S")  # Generate a unique run ID based on the current timestamp

# Launch distributed training on the primary node (rank 0)
torchrun \
       --nnodes=4 \  # Number of nodes participating in training
       --nproc_per_node=4 \  # Number of processes per node (one per GPU)
       --rdzv_id=1 \  # Rendezvous ID for distributed communication
       --rdzv_backend=c10d \  # Backend for rendezvous and process group initialization
       --rdzv_endpoint="${host[1]}:29503" \  # Address and port of the rendezvous endpoint
       train_downstream.py --enable_amp --yaml_config=$yaml_config --pretrained_dir=$pretrained_dir --downstream_config=$downstream_config --freeze_backbone=$freeze_backbone --use_mom_loss=$use_mom_loss --add_noise=$add_noise --noise_mean=$noise_mean --noise_std=$noise_std --run_num=$run_num --batch_size=$batch_size --downstream_max_epochs=$downstream_max_epochs --wandb_group=$wandb_group >> ./logs/${wandb_group}_rank0_${SLURM_JOB_ID}.log 2>&1 &

# Launch training on other nodes
srun -N 1 --gres=gpu:4 -w ${host[2]} torchrun --nnodes=4 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="${host[1]}:29503" train_025_downstream.py --enable_amp --yaml_config=$yaml_config --pretrained_dir=$pretrained_dir --downstream_config=$downstream_config --run_num=$run_num --freeze_backbone=$freeze_backbone --use_mom_loss=$use_mom_loss --add_noise=$add_noise --noise_mean=$noise_mean --noise_std=$noise_std --batch_size=$batch_size --downstream_max_epochs=$downstream_max_epochs --wandb_group=$wandb_group >> ./logs/${wandb_group}_rank1_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[3]} torchrun --nnodes=4 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="${host[1]}:29503" train_025_downstream.py --enable_amp --yaml_config=$yaml_config --pretrained_dir=$pretrained_dir --downstream_config=$downstream_config --run_num=$run_num --freeze_backbone=$freeze_backbone --use_mom_loss=$use_mom_loss --add_noise=$add_noise --noise_mean=$noise_mean --noise_std=$noise_std --batch_size=$batch_size --downstream_max_epochs=$downstream_max_epochs --wandb_group=$wandb_group >> ./logs/${wandb_group}_rank2_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 -w ${host[4]} torchrun --nnodes=4 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="${host[1]}:29503" train_025_downstream.py --enable_amp --yaml_config=$yaml_config --pretrained_dir=$pretrained_dir --downstream_config=$downstream_config --run_num=$run_num --batch_size=$batch_size --freeze_backbone=$freeze_backbone --use_mom_loss=$use_mom_loss --add_noise=$add_noise --noise_mean=$noise_mean --noise_std=$noise_std --downstream_max_epochs=$downstream_max_epochs --wandb_group=$wandb_group >> ./logs/${wandb_group}_rank3_${SLURM_JOB_ID}.log 2>&1 &

# Wait for all background processes to complete
wait
