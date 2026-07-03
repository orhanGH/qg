#!/bin/bash
#SBATCH --job-name=install_mamba_fast
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/install_mamba_fast_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs/install_mamba_fast_%j.err set -euo pipefail mkdir -p /lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/logs module purge
module load Miniforge3
module load CUDA/12.8 source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh ENV=/home/s6oraydi_hpc/.conda/envs/pbpb_pp_mamba
conda activate "$ENV"
export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
hash -r
export PYTHONNOUSERSITE=1 echo "Python: $(which python)"
python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda available', torch.cuda.is_available())" echo "CUDA compiler:"
which nvcc
nvcc --version nvidia-smi # Torch 2.11 requires setuptools < 82, so do NOT upgrade setuptools blindly.
python -m pip install --force-reinstall "setuptools<82"
python -m pip install --upgrade pip wheel packaging ninja python -m pip uninstall -y causal-conv1d mamba-ssm || true # First try causal-conv1d with CUDA compiler available.
python -m pip install --no-build-isolation "causal-conv1d>=1.4.0" # Then mamba-ssm.
python -m pip install --no-build-isolation "mamba-ssm[causal-conv1d]" python - <<'PY'
import importlib.util
import torch print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available()) for m in ["mamba_ssm", "causal_conv1d"]: print(m, "FOUND" if importlib.util.find_spec(m) else "MISSING") from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update print("selective_scan_fn:", selective_scan_fn is not None)
print("mamba_inner_fn:", mamba_inner_fn is not None)
print("causal_conv1d_fn:", causal_conv1d_fn is not None)
print("causal_conv1d_update:", causal_conv1d_update is not None)
PY
