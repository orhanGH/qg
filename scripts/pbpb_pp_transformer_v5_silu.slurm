#!/bin/bash
#SBATCH --job-name=pbpb_te_silu
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=1-4%4
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128/logs/transformer_fold_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128/logs/transformer_fold_%A_%a.err

set -euo pipefail

MODEL=transformer_encoder
FOLD="$SLURM_ARRAY_TASK_ID"

MAX_FILES_PER_CLASS=-1
MAX_PARTICLES=128
NUM_FOLDS=4
FINAL_TEST_RATIO=0.20
SEED=42
EPOCHS=200

BASE_RESULTS_DIR="transformer_encoder_results"

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

ENV=/home/s6oraydi_hpc/.conda/envs/pbpb_pp_torch
conda activate "$ENV"

export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

cd /home/s6oraydi_hpc/pbpb_pp/5/pbpb-pp

RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128

mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions,configs_runtime}
mkdir -p "$RUNROOT/runs/$BASE_RESULTS_DIR/fold_$FOLD"

ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/$BASE_RESULTS_DIR" "$BASE_RESULTS_DIR"

TASK_CONFIG="$RUNROOT/configs_runtime/${MODEL}_fold_${FOLD}_${SLURM_JOB_ID}.json"

python - <<PY
import json
from pathlib import Path

model = "$MODEL"
fold = "$FOLD"
base_results_dir = "$BASE_RESULTS_DIR"

base = json.load(open("configs/v5_allmodels_silu_lr1e3_wd1e4_200p128.json"))
base[model]["results_dir_name"] = f"{base_results_dir}/fold_{fold}"

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
import torch
import transformers
import accelerate
import sklearn
from models.hf import transformer_encoder

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("sklearn OK")
print("Transformer default config:")
print(transformer_encoder.get_default_config())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
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
echo "Starting transformer training with silu/lr1e-3/wd1e-4"
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
