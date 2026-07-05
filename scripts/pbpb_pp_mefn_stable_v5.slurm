#!/bin/bash
#SBATCH --job-name=pbpb_mefn_stable
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=1-4%4
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128/logs/mefn_stable_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128/logs/mefn_stable_%A_%a.err

set -euo pipefail

MODEL=mefn
FOLD="$SLURM_ARRAY_TASK_ID"

MAX_FILES_PER_CLASS=-1
MAX_PARTICLES=128
NUM_FOLDS=4
FINAL_TEST_RATIO=0.20
SEED=42
EPOCHS=200

BASE_RESULTS_DIR="mefn_results"

echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID / Fold: $FOLD"
echo "Model: $MODEL"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "================================================================================"

module purge
module load Miniforge3

source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

ENV=/home/s6oraydi_hpc/.conda/envs/pbpb-pp_keras
conda activate "$ENV"

export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
hash -r

export PYTHONNOUSERSITE=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export MALLOC_ARENA_MAX=2

export NVIDIA_LIBS=$(find "$ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: - || true)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ENV/lib:${LD_LIBRARY_PATH:-}"

cd /home/s6oraydi_hpc/pbpb_pp/5/pbpb-pp

RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128

mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions,configs_runtime}
mkdir -p "$RUNROOT/runs/$BASE_RESULTS_DIR/fold_$FOLD"

ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/$BASE_RESULTS_DIR" "$BASE_RESULTS_DIR"

TASK_CONFIG="$RUNROOT/configs_runtime/${MODEL}_stable_fold_${FOLD}_${SLURM_JOB_ID}.json"

python - <<PY
import json
from pathlib import Path

model = "$MODEL"
fold = "$FOLD"
base_results_dir = "$BASE_RESULTS_DIR"

base = json.load(open("configs/v5_mefn_stable_siluish_lr3e4_200p128.json"))
base[model]["results_dir_name"] = f"{base_results_dir}/fold_{fold}_stable_lr3e4_swish"

out = Path("$TASK_CONFIG")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(base, indent=2))

print(f"Wrote task config: {out}")
print(json.dumps(base[model], indent=2))
PY

echo "================================================================================"
echo "Environment check"
echo "================================================================================"
echo "Python: $(which python)"
python -c "import sys; print('Executable:', sys.executable)"
nvidia-smi || true

python - <<'PYTEST'
import tensorflow as tf
import sklearn
import energyflow
import tf_keras
from main import get_keras_runner_and_model

runner, module = get_keras_runner_and_model("mefn")

print("TensorFlow:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
print("sklearn OK")
print("EnergyFlow:", energyflow.__version__)
print("tf_keras:", tf_keras.__version__)
print("MEFN import OK:", module.__name__)
PYTEST

echo "================================================================================"
echo "Checking shared split"
echo "================================================================================"

SPLIT_DIR="splits/marvin_parts_obsvs_vac_vs_rec_numdata_all_maxp_${MAX_PARTICLES}_allfiles"
SPLIT_FILE="$SPLIT_DIR/seed_${SEED}_folds_${NUM_FOLDS}_test_0.2.npz"

if [ -f "$SPLIT_FILE" ]; then
  ls -lh "$SPLIT_FILE"
else
  echo "No existing split found; main.py should create it in:"
  echo "$SPLIT_DIR"
fi

echo "================================================================================"
echo "Starting stable MEFN training: fold $FOLD"
echo "================================================================================"

python -u main.py \
  --models "$MODEL" \
  --data-root /lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50 \
  --class-0 vac \
  --class-1 rec \
  --num-data -1 \
  --max-files-per-class "$MAX_FILES_PER_CLASS" \
  --max-particles "$MAX_PARTICLES" \
  --num-folds "$NUM_FOLDS" \
  --final-test-ratio "$FINAL_TEST_RATIO" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --fold "$FOLD" \
  --optimized-config "$TASK_CONFIG"

echo "================================================================================"
echo "GPU state at end of job"
echo "================================================================================"
nvidia-smi || true

echo "End: $(date)"
