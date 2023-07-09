#!/bin/bash
#SBATCH -J train 
#SBATCH -p xahdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --exclusive
##SBATCH -x f11r4n03,f12r4n19,j12r4n05,f11r4n[05,15],f11r4n13,f12r4n09,j12r4n15,f11r4n02,f11r4n12
##SBATCH -x i02r1n14,e09r2n02,f11r1n12,f12r4n02,b13r2n[01-02,13-19],b13r3n[00-19],b13r4n[00-02],i02r2n13,e09r3n08,e11r2n19,e11r3n[00-14]
##SBATCH -w f12r4n[03-18]
#set -x

source `pwd`/env.sh

hostfile=/$SLURM_JOB_ID # 获取节点号
scontrol show hostnames $SLURM_JOB_NODELIST > ./hostfile/${hostfile} # 将运行当前作业的计算节点列表输出到指定的文件，$SLURM_JOB_NODELIST表示当前作业所分配的节点列表，
rm `pwd`/hostfile/hostfile-dl -f

for i in `cat ./hostfile/$hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID # 节点号
done
np=$(cat ./hostfile/$hostfile|sort|uniq |wc -l)  # 节点去重
np=$(($np*4))

nodename=$(cat ./hostfile/$hostfile |sed -n "1p") # 读取每行节点 第一个是主节点
echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`

chmod a+x ./single_node.sh
mpirun -np $np --allow-run-as-root --hostfile ./hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none ./single_node.sh  $dist_url -x ucx_rc_verbs_timeout=5000000.00us -x ucx_rc_verbs_timeout=260000.00us

