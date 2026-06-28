#!/bin/bash
#SBATCH --job-name=pbpb_torch_smoke
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/torch_smoke_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/torch_smoke_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"

module purge
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pbpb_pp_torch

export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /home/s6oraydi_hpc/pbpb_pp/3/pbpb-pp

mkdir -p /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/{logs,runs,splits,checkpoints,plots,predictions}
mkdir -p /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/{efn_results,mefn_results,aefn_results,oefn_results,bert_hf_trainer_results,roberta_hf_trainer_results,mamba_hf_trainer_results}

ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/splits splits
ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/bert_hf_trainer_results bert_hf_trainer_results
ln -sfn /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/runs/roberta_hf_trainer_results roberta_hf_trainer_results

echo "Python: $(which python)"
nvidia-smi || true

python - <<'PYTEST'
import torch
import transformers
import sklearn

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("sklearn OK")

from models.hf import bert, roberta
from runners.hf_runner import run_hf_experiment

print("HF project imports OK")
print("BERT config:", bert.get_default_config())
print("RoBERTa config:", roberta.get_default_config())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYTEST

python -u main.py \
  --models bert roberta \
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
