#!/bin/bash

# This script is for launching training on a single node

DATE=$(date +"%m-%d")

export TRITON_CACHE_DIR='/tmp/triton_cache'

# Some pytorch settings
export OMP_NUM_THREADS=1

# Add this if using WandB
export WANDB_API_KEY='6087caaab6a2ecb90e54e11b142b0432e0d22f3a'

#conda activate ttt-video

NUM_GPUS=4

# For 9 seconds and onward, you should use a checkpoint and uncomment the override flag below
CHECKPOINT_WEIGHTS_DIR="/home/user/ttt-video-dit_v2/exp/06-01-ttt-video-3s-BS64-5000steps/checkpoint/"
#CHECKPOINT_WEIGHTS_DIR="/home/user/ttt-video-dit_v1/CogVideoX-2b-sat/CogVideoX-5b_converted/"
#CONFIG_FILE="./configs/train/ttt-mlp/3s-Copy1.toml"
CONFIG_FILE="./configs/train/ttt-mlp/30s.toml"

#EXP_NAME="${DATE}-ttt-video-3s-BS64-5000steps"
EXP_NAME="06-01-ttt-video-3s-BS64-5000steps"

torchrun --nproc_per_node=${NUM_GPUS} \
	--rdzv_backend c10d \
	--rdzv_endpoint="localhost:0" \
	--local-ranks-filter 0 \
	--role rank \
	--tee 3 \
	train.py \
	--wandb.disable \
	--job.config_file ${CONFIG_FILE} \
	--job.exp_name="${EXP_NAME}" \
	--training.global_batch_size=4\
	--parallelism.dp_replicate=1 \
	--parallelism.dp_sharding=4\
	--parallelism.tp_sharding=1\
    --checkpoint.resume \
    --checkpoint.init_state_dir=${CHECKPOINT_WEIGHTS_DIR} # uncomment this line to use a checkpoint
