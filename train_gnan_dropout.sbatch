#!/usr/bin/env bash
#SBATCH -J gnan_dropout
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH -o logs/gnan_dropout_%j_%a.out
#SBATCH -e logs/gnan_dropout_%j_%a.err
#SBATCH --chdir=/home/thf796/Graph-Neural-Additive-Networks---GNAN

set -euo pipefail
set -x  # echo commands as they run

# Map array index -> dropout values
DROPS=(0.0 0.3 0.6)
DROPOUT="${DROPS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Starting job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} with dropout=${DROPOUT}"
echo "Running on node: $(hostname -f)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-na}"

module purge
module load python/3.11.3
module load pytorch/2.2.2

# Activate your venv and make sure its site-packages win
source "$HOME/venvs/gnan_mod/bin/activate"
export PYTHONPATH="$HOME/venvs/gnan_mod/lib/python3.11/site-packages:${PYTHONPATH:-}"

# Quiet external logging & keep prints immediate
export WANDB_DISABLED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONFAULTHANDLER=1

mkdir -p logs

python - <<'PY'
import sys, torch, torch_geometric, platform
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("PyG:", torch_geometric.__version__)
print("Platform:", platform.platform())
PY

# Run training (line-buffered stdout/stderr)
stdbuf -oL -eL python -u main.py \
  --seed 0 \
  --wd 0 \
  --model_name gnan \
  --data_name mutagenicity \
  --dropout "${DROPOUT}" \
  --n_layers 3 \
  --hidden_channels 64 \
  --lr 0.001 \
  --num_epochs 1000 \
  --wandb_flag 0 \
  --early_stop 1 \
  --processed_data_dir processed_data

# Print a quick summary from the log lines emitted by main.py
echo "[SUMMARY] tail of metrics:"
grep -E 'Best Val Acc Model|Best Train Loss Model|^Epoch:|^Test Loss:' -n logs/gnan_dropout_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out || true

echo "[$(date)] Done job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} (dropout=${DROPOUT})"
