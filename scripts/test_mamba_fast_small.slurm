#!/bin/bash
#SBATCH --job-name=test_mamba_fast
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/test_mamba_fast_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/test_mamba_fast_%j.err

set -euo pipefail

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
mkdir -p "$RUNROOT"/{logs,runs,splits,checkpoints,plots,predictions,configs_runtime}
mkdir -p "$RUNROOT/runs/mamba_hf_trainer_results/test_fast"

ln -sfn "$RUNROOT/splits" splits
ln -sfn "$RUNROOT/runs/mamba_hf_trainer_results" mamba_hf_trainer_results

python - <<'PY'
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("selective_scan_fn:", selective_scan_fn is not None)
print("mamba_inner_fn:", mamba_inner_fn is not None)
print("causal_conv1d_fn:", causal_conv1d_fn is not None)
print("causal_conv1d_update:", causal_conv1d_update is not None)
PY

python -u main.py \
  --models mamba \
  --data-root /lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50 \
  --class-0 vac \
  --class-1 rec \
  --num-data -1 \
  --max-files-per-class 4 \
  --max-particles 64 \
  --num-folds 2 \
  --final-test-ratio 0.20 \
  --seed 42 \
  --epochs 2 \
  --fold 1 \
  --optimized-config configs/big50_parallel_base.json

echo "End: $(date)"
