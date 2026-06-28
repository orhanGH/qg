#!/bin/bash
#SBATCH --job-name=pbpb_keras_big50
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/keras_big50_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/keras_big50_%A_%a.err

set -euo pipefail

MODELS=(efn mefn aefn oefn)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Host: $(hostname)"
echo "Start: $(date)"

module purge
module load Miniforge3
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

ENV=/home/s6oraydi_hpc/.conda/envs/pbpb-pp_keras
conda activate "$ENV"
export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
hash -r

export PYTHONNOUSERSITE=1

# TensorFlow pip CUDA library paths
export NVIDIA_LIBS=$(find "$ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: - || true)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ENV/lib:${LD_LIBRARY_PATH:-}"

cd /home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp

RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs

mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions}
mkdir -p "$RUNROOT"/runs/{efn_results,mefn_results,aefn_results,oefn_results}

ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/efn_results" efn_results
ln -sfn "$RUNROOT/runs/mefn_results" mefn_results
ln -sfn "$RUNROOT/runs/aefn_results" aefn_results
ln -sfn "$RUNROOT/runs/oefn_results" oefn_results

echo "Python: $(which python)"
python -c "import sys; print('Executable:', sys.executable)"
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
  --max-files-per-class 80 \
  --max-particles 64 \
  --num-folds 4 \
  --final-test-ratio 0.20 \
  --seed 42 \
  --epochs 50 \
  --optimized-config configs/big50_samebatch_earlystop.json


echo "================================================================================"
echo "GPU state at end of job:"
echo "================================================================================"
nvidia-smi || true

echo "End: $(date)"
