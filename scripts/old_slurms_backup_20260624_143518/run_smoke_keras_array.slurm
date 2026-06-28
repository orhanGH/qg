#!/bin/bash
#SBATCH --job-name=pbpb_keras_smoke
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/keras_smoke_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/keras_smoke_%A_%a.err

set -euo pipefail

MODELS=(efn mefn aefn oefn)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Host: $(hostname)"
echo "Start: $(date)"

module purge
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pbpb-pp_keras

export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"

export NVIDIA_LIBS=$(find "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: - || true)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp

mkdir -p /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/{logs,runs,splits,checkpoints,plots,predictions}
mkdir -p /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/{efn_results,mefn_results,aefn_results,oefn_results}

ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/splits splits
ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/efn_results efn_results
ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/mefn_results mefn_results
ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/aefn_results aefn_results
ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/oefn_results oefn_results

echo "Python: $(which python)"
nvidia-smi || true

python - <<'PYTEST'
import tensorflow as tf
import sklearn
import energyflow
import tf_keras

print("TensorFlow:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))
print("sklearn OK")
print("EnergyFlow:", energyflow.__version__)
print("tf_keras:", tf_keras.__version__)
PYTEST

python -u main.py \
  --models "$MODEL" \
  --data-root /lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50 \
  --class-0 vac \
  --class-1 rec \
  --num-data -1 \
  --max-files-per-class 4 \
  --max-particles 64 \
  --num-folds 2 \
  --final-test-ratio 0.25 \
  --seed 42 \
  --epochs 1 \
  --fold 1

echo "End: $(date)"
