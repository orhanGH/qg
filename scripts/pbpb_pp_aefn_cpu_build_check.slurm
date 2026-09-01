#!/bin/bash
#SBATCH --job-name=aefn_cpu_check
#SBATCH --partition=intelsr_devel
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/experiments/aefn_smoke/cpu_build_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/s6oraydi_hpc_runs/experiments/aefn_smoke/cpu_build_%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_DIR="$HOME/nlp-lab/pp_pbpb/implementation/qg-pbpb_vs_pp_git"
ENV="$HOME/.conda/envs/pbpb-pp_keras"

module purge
module load Miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Python      : $(which python)"
echo "Project     : $PROJECT_DIR"
echo "Start       : $(date)"
echo "GPU request : none"
echo "============================================================"

python -u - <<'PY'
import importlib

import numpy as np
import tensorflow as tf
from tf_keras import backend as K

MODELS = [
    "aefn_before_phi",
    "aefn_before_sum",
    "aefn_after_sum",
    "aefn_before_phi_before_sum",
    "aefn_before_phi_after_sum",
    "aefn_before_sum_after_sum",
    "aefn_all_three",
]

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

rng = np.random.default_rng(42)

# Two artificial padded jets with 128 constituent slots.
z = np.zeros((2, 128), dtype=np.float32)
z[:, :10] = rng.uniform(0.01, 1.0, size=(2, 10))
z[:, :10] /= z[:, :10].sum(axis=1, keepdims=True)

p = np.zeros((2, 128, 2), dtype=np.float32)
p[:, :10, :] = rng.normal(
    loc=0.0,
    scale=0.2,
    size=(2, 10, 2),
).astype(np.float32)

# Apply the same particle permutation to z and coordinates.
permutation = rng.permutation(128)
z_permuted = z[:, permutation]
p_permuted = p[:, permutation, :]

failures = []

for model_name in MODELS:
    print("\n" + "=" * 72)
    print("Checking:", model_name)
    print("=" * 72)

    try:
        module = importlib.import_module(f"models.efn.{model_name}")
        config = module.get_default_config()

        model = module.build_model(
            config=config,
            extra_info={"num_particles": 128},
        )

        output = model([z, p], training=False).numpy()
        output_permuted = model(
            [z_permuted, p_permuted],
            training=False,
        ).numpy()

        max_permutation_difference = float(
            np.max(np.abs(output - output_permuted))
        )

        print("Model name        :", config["model_name"])
        print("Parameters        :", f"{model.count_params():,}")
        print("Output shape      :", output.shape)
        print("Output finite     :", bool(np.isfinite(output).all()))
        print(
            "Permutation diff  :",
            f"{max_permutation_difference:.3e}",
        )

        if output.shape != (2, 2):
            raise ValueError(
                f"Expected output shape (2, 2), received {output.shape}"
            )

        if not np.isfinite(output).all():
            raise ValueError("Model produced non-finite outputs")

        if max_permutation_difference > 1e-5:
            raise ValueError(
                "Permutation-invariance check exceeded tolerance: "
                f"{max_permutation_difference}"
            )

        print("STATUS            : PASS")

    except Exception as exc:
        failures.append((model_name, repr(exc)))
        print("STATUS            : FAIL")
        print("ERROR             :", repr(exc))

    finally:
        K.clear_session()

print("\n" + "=" * 72)
print("FINAL SUMMARY")
print("=" * 72)

if failures:
    for model_name, error in failures:
        print(f"FAIL: {model_name}: {error}")

    raise SystemExit(
        f"{len(failures)} of {len(MODELS)} model checks failed"
    )

print(f"All {len(MODELS)} AEFN variants passed.")
PY

echo "============================================================"
echo "End: $(date)"
echo "CPU model-build check completed successfully."
echo "============================================================"
