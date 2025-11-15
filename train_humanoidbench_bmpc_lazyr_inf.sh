#!/bin/bash

#SBATCH --partition=gpu-a100-80g,gpu-mig-40g  # try MIG first, then A100-80G
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1                           # or: --gres=gpu:4g.40gb:1 for a MIG 40GB slice
#SBATCH --cpus-per-gpu=4
#SBATCH --array=0-20               # 7 envs × 3 seeds → indices 0-5.  Update if you add more.

# EXPERIMENT GRID
ENV_LIST=(h1hand-walk-v0 h1hand-reach-v0 h1hand-hurdle-v0 h1hand-crawl-v0 h1hand-maze-v0 h1hand-stand-v0
 h1hand-run-v0)
#  h1hand-sit_simple-v0 h1hand-sit_hard-v0 h1hand-balance_simple-v0 h1hand-balance_hard-v0
# h1hand-stair-v0 h1hand-slide-v0 h1hand-pole-v0)   # Extend this with as many envs as you like
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
       bmpc.reanalyze_interval=10000000.0
       wandb.project=${ENV_ID} \
       wandb.log_wandb=True \
       wandb.name=bmpc_repo_main_lazyr_inf \
       seed=${SEED} \
       >"${ENV_ID}_s${SEED}.out" 2>&1