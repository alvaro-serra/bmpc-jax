#!/bin/bash

#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --partition=gpu-medium
#SBATCH --constraint="A100.80G|A100.4g.40gb|A100.3g.40gb"
#SBATCH --array=0-8               # 7 envs × 3 seeds → indices 0-20.  Update if you add more.

# EXPERIMENT GRID
ENV_LIST=(h1hand-hurdle-v0
 h1hand-balance_simple-v0
 h1hand-stair-v0)   # Extend this with as many envs as you like
SEED_LIST=(0 1 2)                           # Exactly three seeds, change if needed
NUM_SEEDS=${#SEED_LIST[@]}

# DECODE ARRAY INDEX
ENV_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

ENV_ID=${ENV_LIST[$ENV_INDEX]}
SEED=${SEED_LIST[$SEED_INDEX]}
ENV_BACKEND=${BACKEND[$ENV_ID]}

# ENVIRONMENT SETUP
module load cuDNN/8.9.7.29-CUDA-12.3.2
#export MUJOCO_GL=egl

# LAUNCH TRAINING
python bmpc_jax/train.py \
       env.backend=humanoid \
       env.env_id=${ENV_ID} \
       wandb.project=${ENV_ID} \
       wandb.log_wandb=True \
       wandb.name=bmpc_repo_main \
       seed=${SEED} \
       >"${ENV_ID}_s${SEED}.out" 2>&1