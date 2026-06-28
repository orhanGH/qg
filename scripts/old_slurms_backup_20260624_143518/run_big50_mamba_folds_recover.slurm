#!/bin/bash
#SBATCH --job-name=pbpb_mamba50f
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=1-4%1
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/mamba_big50_fold_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/mamba_big50_fold_%A_%a.err

set -euo pipefail

FOLD="$SLURM_ARRAY_TASK_ID"

echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID / Fold: $FOLD"
echo "Host: $(hostname)"
echo "Start: $(date)"

module purge
module load Miniforge3
source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

ENV=/home/s6oraydi_hpc/.conda/envs/pbpb_pp_mamba
conda activate "$ENV"
export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
hash -r

export PYTHONNOUSERSITE=1

cd /home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp

RUNROOT=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs

mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions}
mkdir -p "$RUNROOT"/runs/mamba_hf_trainer_results

ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/mamba_hf_trainer_results" mamba_hf_trainer_results

echo "Python: $(which python)"
python -c "import sys; print('Executable:', sys.executable)"
nvidia-smi || true

python - <<'PYTEST'
import torch
import transformers
import accelerate
import sklearn
from models.hf import mamba
from runners.hf_runner import run_hf_experiment

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("sklearn OK")
print("Project Mamba import OK")
print("Mamba config:", mamba.get_default_config())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYTEST

echo "Checking split exists:"
ls -lh splits/marvin_parts_obsvs_vac_vs_rec_numdata_all_maxp_64_maxfiles_80/seed_42_folds_4_test_0.2.npz

python -u main.py \
  --models mamba \
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
  --fold "$FOLD" \
  --optimized-config configs/mamba_big50_recover.json

echo "================================================================================"
echo "GPU state at end of job:"
echo "================================================================================"
nvidia-smi || true

echo "End: $(date)"
