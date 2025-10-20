#!/bin/bash
#SBATCH --job-name=gnan-muta
#SBATCH --array=0-2                     # 3 jobs: 0,1,2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# ----- Uncomment/adapt these if you want GPUs -----
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --constraint="gpu"
# --------------------------------------------------
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# --- Dropout choices mapped from the array index ---
DROPOUTS=(0.0 0.3 0.6)
DROPOUT="${DROPOUTS[$SLURM_ARRAY_TASK_ID]}"

# --- Tunables with sane defaults (override via --export=ALL,VAR=...) ---
DATA_NAME="${DATA_NAME:-mutagenicity}"
MODEL_NAME="${MODEL_NAME:-gnan}"
SEED="${SEED:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
WD="${WD:-0}"
LR="${LR:-0.01}"
N_LAYERS="${N_LAYERS:-3}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"

echo "[$(date)] Starting GNAN training"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK_ID=$SLURM_ARRAY_TASK_ID  DROPOUT=$DROPOUT"
echo "DATA_NAME=$DATA_NAME MODEL_NAME=$MODEL_NAME SEED=$SEED EPOCHS=$NUM_EPOCHS WD=$WD LR=$LR N_LAYERS=$N_LAYERS HIDDEN=$HIDDEN_CHANNELS"

# --- Environment / modules ---
# If your cluster has modules for these, load them here (adjust names to what Hendrix provides):
# module purge
# module load python/3.7   # if sticking to the very old torch/pyg listed by the repo
# module load cuda/10.0    # if using GPU and old torch
# module load pytorch/1.0.0
# module load pyg/1.0.0
# module load scipy/1.1.0 scikit-learn/0.20.0 pandas/0.23.4

# OR: if you maintain your own conda env, just activate it:
# source ~/.bashrc
# conda activate gnan-env

# Ensure logs dir exists
mkdir -p logs

# --- Run ---
# Prefer srun on clusters configured that way; otherwise `python` is fine.
srun python main.py \
  --seed="${SEED}" \
  --wd="${WD}" \
  --model_name="${MODEL_NAME}" \
  --data_name="${DATA_NAME}" \
  --dropout="${DROPOUT}" \
  --n_layers="${N_LAYERS}" \
  --hidden_channels="${HIDDEN_CHANNELS}" \
  --lr="${LR}" \
  --num_epochs="${NUM_EPOCHS}"

echo "[$(date)] Finished."
