#!/bin/bash
#SBATCH --job-name=pbpb_oefn200p128
#SBATCH --partition=mlgpu_medium
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/oefn200p128_fold_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/oefn200p128_fold_%A_%a.err set -euo pipefail MODEL="oefn"
FOLD=$((SLURM_ARRAY_TASK_ID + 1)) MAX_FILES_PER_CLASS=-1
MAX_PARTICLES=128
NUM_FOLDS=4
FINAL_TEST_RATIO=0.20
SEED=42
EPOCHS=200 RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs
REPO=/home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp mkdir -p "$RUNROOT/logs" "$RUNROOT/configs_runtime" "$RUNROOT/runs/oefn_results" cd "$REPO" ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/oefn_results" oefn_results TASK_CONFIG="$RUNROOT/configs_runtime/oefn_fold_${FOLD}_${SLURM_JOB_ID}.json" cat > "$TASK_CONFIG" <<JSON
{ "oefn": { "batch_size": 512, "use_early_stopping": true, "patience": 30, "early_stopping_threshold": 0.0001, "learning_rate": 0.0003, "weight_decay": 1e-05, "epochs": 200, "activation": "gelu", "results_dir_name": "oefn_results/fold_${FOLD}" }
}
JSON echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-NA}"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Fold: $FOLD"
echo "Host: $(hostname)"
echo "Task config: $TASK_CONFIG"
cat "$TASK_CONFIG" module purge
module load Miniforge3 source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/s6oraydi_hpc/.conda/envs/pbpb-pp_keras echo "Python: $(which python)"
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
PY nvidia-smi python -u main.py \ --models "$MODEL" \ --data-root /lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50 \ --class-0 vac \ --class-1 rec \ --num-data -1 \ --max-files-per-class "$MAX_FILES_PER_CLASS" \ --max-particles "$MAX_PARTICLES" \ --num-folds "$NUM_FOLDS" \ --final-test-ratio "$FINAL_TEST_RATIO" \ --seed "$SEED" \ --epochs "$EPOCHS" \ --fold "$FOLD" \ --optimized-config "$TASK_CONFIG" echo "Finished oEFN fold $FOLD"
nvidia-smi
