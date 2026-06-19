#!/bin/bash
#SBATCH --job-name=qg_pbpb_pp
#SBATCH --partition=mlgpu_short
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

# =============================================================================
# Marvin SLURM script for qg-pbpb-pp/main.py
# Models supported by main.py:
#   HF/PyTorch: bert roberta mamba
#   Keras/TF:   efn mefn aefn oefn
# =============================================================================

# Project directory on Marvin after unzipping
PROJECT_DIR="${PROJECT_DIR:-$HOME/qg-pbpb-pp}"

# Dataset path on Marvin
DATA_ROOT="${DATA_ROOT:-/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50}"

# Default experiment settings
MODELS="${MODELS:-bert}"
FOLD="${FOLD:-1}"
NUM_DATA="${NUM_DATA:-10000}"
MAX_PARTICLES="${MAX_PARTICLES:-128}"
EPOCHS="${EPOCHS:-5}"
NUM_FOLDS="${NUM_FOLDS:-3}"
FINAL_TEST_RATIO="${FINAL_TEST_RATIO:-0.25}"
SEED="${SEED:-42}"

# Conda environment name
CONDA_ENV="${CONDA_ENV:-qg-marvin}"

mkdir -p logs

echo "========================================================================"
echo "Job ID        : ${SLURM_JOB_ID:-unknown}"
echo "Job name      : ${SLURM_JOB_NAME:-unknown}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "Project dir   : ${PROJECT_DIR}"
echo "Data root     : ${DATA_ROOT}"
echo "Models        : ${MODELS}"
echo "Fold          : ${FOLD}"
echo "Num data      : ${NUM_DATA}"
echo "Max particles : ${MAX_PARTICLES}"
echo "Epochs        : ${EPOCHS}"
echo "========================================================================"

module purge
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

export TMPDIR="${TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

export HF_HOME="$TMPDIR/huggingface"
export TRANSFORMERS_CACHE="$TMPDIR/huggingface/transformers"
export XDG_CACHE_HOME="$TMPDIR/cache"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - <<'PY'
import sys
print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch check failed:", repr(exc))

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("TF GPUs:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("TensorFlow check failed:", repr(exc))
PY

python main.py \
  --models ${MODELS} \
  --data-root "${DATA_ROOT}" \
  --class-0 vac \
  --class-1 rec \
  --num-data "${NUM_DATA}" \
  --max-particles "${MAX_PARTICLES}" \
  --num-folds "${NUM_FOLDS}" \
  --final-test-ratio "${FINAL_TEST_RATIO}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --fold "${FOLD}"

echo "========================================================================"
echo "End time: $(date)"
echo "Job finished successfully."
echo "========================================================================"
