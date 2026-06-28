#!/bin/bash
#SBATCH --job-name=pbpb_oefn50
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/oefn_big50_recover_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/oefn_big50_recover_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
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
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1
export MALLOC_ARENA_MAX=2

# TensorFlow pip CUDA library paths
export NVIDIA_LIBS=$(find "$ENV/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd: - || true)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ENV/lib:${LD_LIBRARY_PATH:-}"

cd /home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp

RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs

mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions}
mkdir -p "$RUNROOT"/runs/oefn_results

ln -sfn "$RUNROOT/splits" splits
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

echo "Checking split exists:"
ls -lh splits/marvin_parts_obsvs_vac_vs_rec_numdata_all_maxp_64_maxfiles_80/seed_42_folds_4_test_0.2.npz

python -u main.py \
  --models oefn \
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
  --optimized-config configs/oefn_big50_recover.json

echo "================================================================================"
echo "GPU state at end of job:"
echo "================================================================================"
nvidia-smi || true

echo "End: $(date)"
