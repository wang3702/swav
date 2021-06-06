#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=swav_finetune
#SBATCH --time=72:00:00
#SBATCH --mem=150G
#SBATCH --partition=learnfair
#SBATCH --mail-user=wang3702@fb.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=end
#SBATCH --constraint="volta"

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH="imagenet"
EXPERIMENT_PATH="./swav_finetune"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u eval_lincls.py \
--data_path $DATASET_PATH \
--dist_url $dist_url \
--dump_path $EXPERIMENT_PATH \
--pretrained /checkpoint/wang3702/swav/swav_400ep_bs256_pretrain/checkpoint.pth.tar 
