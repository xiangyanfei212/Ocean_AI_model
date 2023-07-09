### 加载环境
module purge
source /home/HPCBase/tools/module-5.2.0/init/profile.sh 
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/cuda/11.3.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate torch1.11
source /home/share/jincsuan/home/yeesuanAi26/env.sh
module load libs/netcdf/4.7.4_kgcc9.3.1_hmpi1.2.0
module load libs/hdf5/1.12.0_kgcc9.3.1_hmpi1.2.0

### 配置NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
#export NCCL_IB_HCA=mlx5_1:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

### 配置MASTER,RANK,NODES
OMPI_COMM_WORLD_SIZE=$1
NODES_FILE=$2
HOST=`hostname`
flock -x ${NODES_FILE} -c "echo ${HOST} >> ${NODES_FILE}"
MASTER_IP=`head -n 1 nodes_train`
HOST_RANK=`sed -n "/${HOST}/=" ${NODES_FILE}`
echo $OMPI_COMM_WORLD_SIZE
echo $HOST_RANK
let OMPI_COMM_WORLD_RANK=HOST_RANK-1
echo $OMPI_COMM_WORLD_RANK

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10081
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${seed}" ] && seed=42
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

#export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "seed" $seed

echo "start training"

config_path='./config/AFNO.yaml'
config='afno_backbone'
run_num=$(date "+%Y%m%d-%H%M%S")

#torchrun --nproc_per_node=${n_gpu} --nnodes=${OMPI_COMM_WORLD_SIZE} --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_IP} --master_port=${MASTER_PORT} train_025.py --config=$config --run_num=$run_num --yaml_config=$config_path --dist-url "tcp://${MASTER_IP}:${MASTER_PORT}"
#echo python -u train_025.py --config=$config --run_num=$run_num --yaml_config=$config_path --dist-url "tcp://${MASTER_IP}:${MASTER_PORT}"
#mpirun -n 4 python -u train_025.py --config=$config --run_num=$run_num --yaml_config=$config_path --dist-url "tcp://${MASTER_IP}:${MASTER_PORT}" --local_rank=${OMPI_COMM_WORLD_LOCAL_RANK}

torchrun --nproc_per_node=${n_gpu} --nnodes=${OMPI_COMM_WORLD_SIZE} --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_IP} --master_port=${MASTER_PORT} train_025_yeesuan.py --config=$config --run_num=$run_num --yaml_config=$config_path --dist-url "tcp://${MASTER_IP}:${MASTER_PORT}"
