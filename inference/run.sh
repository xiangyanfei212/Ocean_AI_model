#!/bin/bash
#SBATCH --gres=gpu:4   # number of GPUs per node
#SBATCH -N 1	       # 请求节点的个数
#SBATCH -p vip_gpu_scx6115
#SBATCH --qos=gpugpu # 告诉 SLURM 调度器请求使用 GPU 资源
#SBATCH --output=%j.log
module purge
module load compilers/cuda/11.7 
module load compilers/gcc/12.2.0
module load anaconda/2021.11
module load cudnn/8.4.0.27_cuda11.x
source activate torch2.0

# yaml_config='../config/Model_2.yaml'
prediction_length=35
decorrelation_time=50
n_samples_per_year=300 

# config='afno_backbone'
# run_num='20230609-155507'

# config='Masked_AE_Ocean'
# run_num='20230610'

# config='Masked_AE_fusion'
# run_num='20230613-203822'

# config='Masked_AE_Ocean'
# run_num='20230610'

# exp_dir='../exp_15_levels'
# config='afno'
# run_num='20230628-131420'
# finetune_dir=''

exp_dir='../exp_15_levels'
config='Masked_AE_Ocean'
run_num='20230628-131150'
finetune_dir='2_steps_finetune'

# run_num='20230629-222717'
# finetune_dir=''
# run_num='20230701-220232'

# python inference_025.py --yaml_config=${yaml_config} --config=${config} --run_num=${run_num} --prediction_length=${prediction_length} --decorrelation_time=${decorrelation_time} --n_samples_per_year=${n_samples_per_year} 
python inference_025.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --decorrelation_time=${decorrelation_time} --n_samples_per_year=${n_samples_per_year} 



