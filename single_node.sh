#!/bin/bash
#SBATCH --output=./logs/%j.log          # 指定作业标准结果输出文件名称
export MIOPEN_DEBUG_DISABLE_FIND_DB=1   # 禁用 MIOpen 库在运行时自动查找并加载性能数据库
#export NCCL_SOCKET_IFNAME=eno1         # NCCL通信，指定网络接口
export NCCL_IB_HCA=mlx5_0:200Gbs        # NCCL 通信， 指定用于数据传输和接收的HCA（网络适配器）
export HSA_USERPTR_FOR_PAGED_MEM=0      #  
#export MIOPEN_DEBUG_CONV_WINOGRAD=0    # 
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0 
export MIOPEN_FIND_MODE=5 

# source /public/software/apps/DeepLearning/PyTorch/pytorch-env.sh
# config='Model' 
# config_path=./config/Model.yaml
config_path=./config/AFNO.yaml
config='afno_backbone' 
run_num=$(date "+%Y%m%d-%H%M%S")

comm_size=$OMPI_COMM_WORLD_SIZE
comm_rank=$OMPI_COMM_WORLD_RANK
local_rank=$OMPI_COMM_WORLD_LOCAL_RANK

# APP="python -u train.py --dist-url tcp://${1}:34567 --world-size=${comm_size} --comm_rank=${comm_rank} --local_rank=${local_rank} --enable_amp --yaml_config=$config_path --config=$config --run_num=$run_num"
APP="python -u 03_train_025.py --dist-url tcp://${1}:34567 --world-size=${comm_size} --comm_rank=${comm_rank} --local_rank=${local_rank} --enable_amp --yaml_config=$config_path --config=$config --run_num=$run_num"

# 多节点多卡的并行计算参数设置
case ${local_rank} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  NCCL_SOCKET_IFNAME=ib0  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP} 
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}  
  ;;
esac
