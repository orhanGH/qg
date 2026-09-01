#!/bin/bash
#SBATCH --job-name=aefn_train_smoke
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/experiments/aefn_smoke/train_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/experiments/aefn_smoke/train_%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_DIR="$HOME/nlp-lab/pp_pbpb/implementation/qg-pbpb_vs_pp_git"
ENV="$HOME/.conda/envs/pbpb-pp_keras"
PYTHON="$ENV/bin/python"

# ----------------------------------------------------------------------
# Keep host CPU use minimal.
# ----------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

# TensorFlow / NVIDIA libraries installed in the conda environment.
export NVIDIA_LIBS=$(find "$ENV/lib/python3.11/site-packages/nvidia" \
    -type d -name lib | paste -sd: - || true)

export LD_LIBRARY_PATH="$NVIDIA_LIBS:$ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found: $PYTHON"
    exit 1
fi

mkdir -p \
    /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/experiments/aefn_smoke

echo "============================================================"
echo "AEFN END-TO-END GPU TRAINING SMOKE TEST"
echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Project      : $PROJECT_DIR"
echo "Python       : $PYTHON"
"$PYTHON" --version
echo "Start        : $(date)"
echo "============================================================"

nvidia-smi

echo
echo "Checking TensorFlow GPU..."
"$PYTHON" - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError("TensorFlow cannot see a GPU.")
PY

echo
echo "============================================================"
echo "Starting 7-model training smoke test"
echo "20,000 jets | 128 particles | 1 fold | 3 epochs"
echo "============================================================"

"$PYTHON" -u main.py \
    --models \
        aefn_before_phi \
        aefn_before_sum \
        aefn_after_sum \
        aefn_before_phi_before_sum \
        aefn_before_phi_after_sum \
        aefn_before_sum_after_sum \
        aefn_all_three \
    --num-data 20000 \
    --max-particles 128 \
    --num-folds 3 \
    --final-test-ratio 0.25 \
    --seed 42 \
    --epochs 3 \
    --fold 1

echo
echo "============================================================"
echo "ALL AEFN TRAINING SMOKE TESTS COMPLETED"
echo "End: $(date)"
echo "============================================================"
