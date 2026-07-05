#!/bin/bash
#SBATCH --job-name=pbpb_aefn_F
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=1-8%4
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128/logs/aefnF_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs_v5_silu_lr1e3_wd1e4_200p128/logs/aefnF_%A_%a.err

set -euo pipefail

MAX_FILES_PER_CLASS=-1
MAX_PARTICLES=128
NUM_FOLDS=4
FINAL_TEST_RATIO=0.20
SEED=42
EPOCHS=200

# Use lowercase aliases because main.py lowercases requested model names.
AEFN_MODELS=(aefn_before_f aefn_before_phi_before_f)

TASK_ZERO=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_INDEX=$((TASK_ZERO / 4))
FOLD=$((TASK_ZERO % 4 + 1))
MODEL=${AEFN_MODELS[$MODEL_INDEX]}

case "$MODEL" in
  aefn_before_f) BASE_RESULTS_DIR="aefn_before_F_results" ;;
  aefn_before_phi_before_f) BASE_RESULTS_DIR="aefn_before_phi_before_F_results" ;;
  *) echo "Unknown AEFN model: $MODEL"; exit 1 ;;
esac

echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Fold: $FOLD"
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
mkdir -p "$RUNROOT"/runs/{aefn_before_F_results,aefn_before_phi_before_F_results}
mkdir -p "$RUNROOT/runs/$BASE_RESULTS_DIR/fold_$FOLD"

ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/aefn_before_F_results" aefn_before_F_results
ln -sfn "$RUNROOT/runs/aefn_before_phi_before_F_results" aefn_before_phi_before_F_results

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

echo "End: $(date)"
