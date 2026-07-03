#!/bin/bash
#SBATCH --job-name=pbpb_oefn_f34_200G
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --array=14-15%1
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/oefnGPU200p128_fold_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/oefnGPU200p128_fold_%A_%a.err
#SBATCH --mem=200G set -euo pipefail MAX_FILES_PER_CLASS=-1
MAX_PARTICLES=128
NUM_FOLDS=4
FINAL_TEST_RATIO=0.20
SEED=42
EPOCHS=200 KERAS_MODELS=(efn mefn aefn oefn) MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
FOLD=$((SLURM_ARRAY_TASK_ID % 4 + 1))
MODEL=${KERAS_MODELS[$MODEL_INDEX]} case "$MODEL" in efn) BASE_RESULTS_DIR="efn_results" ;; mefn) BASE_RESULTS_DIR="mefn_results" ;; aefn) BASE_RESULTS_DIR="aefn_results" ;; oefn) BASE_RESULTS_DIR="oefn_results" ;; *) echo "Unknown model: $MODEL"; exit 1 ;;
esac echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Fold: $FOLD"
echo "Host: $(hostname)"
echo "Start: $(date)" module purge
module load Miniforge3
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh ENV=/home/s6oraydi_hpc/.conda/envs/pbpb-pp_keras
conda activate "$ENV"
export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
hash -r export PYTHONNOUSERSITE=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export MALLOC_ARENA_MAX=2 export NVIDIA_LIBS=$(find "$ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: - || true)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ENV/lib:${LD_LIBRARY_PATH:-}" cd /home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions,configs_runtime}
mkdir -p "$RUNROOT"/runs/{efn_results,mefn_results,aefn_results,oefn_results}
mkdir -p "$RUNROOT/runs/$BASE_RESULTS_DIR/fold_$FOLD" ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/efn_results" efn_results
ln -sfn "$RUNROOT/runs/mefn_results" mefn_results
ln -sfn "$RUNROOT/runs/aefn_results" aefn_results
ln -sfn "$RUNROOT/runs/oefn_results" oefn_results TASK_CONFIG="$RUNROOT/configs_runtime/${MODEL}_fold_${FOLD}_${SLURM_JOB_ID}.json" python - <<PY
import json
from pathlib import Path model = "$MODEL"
fold = "$FOLD"
base_results_dir = "$BASE_RESULTS_DIR" base = json.load(open("configs/big50_parallel_base.json"))
base[model]["results_dir_name"] = f"{base_results_dir}/fold_{fold}" out = Path("$TASK_CONFIG")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(base, indent=2))
print(f"Wrote task config: {out}")
print(json.dumps(base[model], indent=2))
PY echo "Python: $(which python)"
python -c "import sys; print('Executable:', sys.executable)"
nvidia-smi || true python - <<'PYTEST'
import tensorflow as tf
import sklearn
import energyflow
import tf_keras print("TensorFlow:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
print("sklearn OK")
print("EnergyFlow:", energyflow.__version__)
print("tf_keras:", tf_keras.__version__)
PYTEST echo "Checking whether shared split already exists:" if [ "$MAX_FILES_PER_CLASS" -le 0 ]; then MAXFILES_TAG="allfiles"
else MAXFILES_TAG="maxfiles_${MAX_FILES_PER_CLASS}"
fi SPLIT_DIR="splits/marvin_parts_obsvs_vac_vs_rec_numdata_all_maxp_${MAX_PARTICLES}_${MAXFILES_TAG}" SPLIT_FILE_A="$SPLIT_DIR/seed_${SEED}_folds_${NUM_FOLDS}_test_${FINAL_TEST_RATIO}.npz"
SPLIT_FILE_B="$SPLIT_DIR/seed_${SEED}_folds_${NUM_FOLDS}_test_0.2.npz" if [ -f "$SPLIT_FILE_A" ]; then ls -lh "$SPLIT_FILE_A"
elif [ -f "$SPLIT_FILE_B" ]; then ls -lh "$SPLIT_FILE_B"
else echo "No existing split found yet; main.py should create it in:" echo "$SPLIT_DIR"
fi echo "Hard TensorFlow GPU check before training:"
python - <<'GPUCHECK'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("TF GPUs:", gpus)
if not gpus: sys.exit("ERROR: TensorFlow sees no GPU. Aborting to avoid CPU-only oEFN run.")
GPUCHECK
python -u main.py \ --models "$MODEL" \ --data-root /lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50 \ --class-0 vac \ --class-1 rec \ --num-data -1 \ --max-files-per-class "$MAX_FILES_PER_CLASS" \ --max-particles "$MAX_PARTICLES" \ --num-folds "$NUM_FOLDS" \ --final-test-ratio "$FINAL_TEST_RATIO" \ --seed "$SEED" \ --epochs "$EPOCHS" \ --fold "$FOLD" \ --optimized-config "$TASK_CONFIG" echo "================================================================================"
echo "GPU state at end of job:"
echo "================================================================================"
nvidia-smi || true echo "End: $(date)"
