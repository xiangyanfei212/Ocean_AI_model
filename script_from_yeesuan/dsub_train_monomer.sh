#!/bin/bash
#DSUB -n train_025 
#DSUB -A root.jincsuan
#DSUB -R 'cpu=128;gpu=4;mem=240000'
#DSUB -N 4
#DSUB -eo /home/share/jincsuan/home/yeesuanAi26/Ocean_AI_model/log/%J.%I.err
#DSUB -oo /home/share/jincsuan/home/yeesuanAi26/Ocean_AI_model/log/%J.%I.out

### 脚本名称
RANK_SCRIPT="train_monomer.sh"

### 计算目录
JOB_PATH="/home/share/jincsuan/home/yeesuanAi26/Ocean_AI_model"

### 获取节点数量、定义节点列表文件名
NNODES=4
NODES_FILE="nodes_train"

cd ${JOB_PATH}
/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${NODES_FILE}

if [[ -f "${NODES_FILE}" ]];then
  rm -rf ${NODES_FILE}
fi

