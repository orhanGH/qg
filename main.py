from pathlib import Path
import argparse
import time

import numpy as np
from energyflow.datasets import qg_jets

from utils import (
    preprocess_jets,
    create_fixed_test_cv_splits,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert", "roberta", "mamba", "efn", "mefn", "oefn"],
        help="Models to run. Options: bert roberta mamba efn mefn oefn",
    )

    parser.add_argument(
        "--num-data",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--max-particles",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--final-test-ratio",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    return parser.parse_args()


def save_split_indices(project_root: Path, dev_idx, final_test_idx, folds, shared_config):
    """
    Saves the split indices used by all models.

    This is useful for proving that every model used exactly the same:
        - 75% development set
        - 25% fixed test set
        - CV train/validation folds
    """

    splits_dir = project_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_path = (
        splits_dir
        / f"qg_numdata_{shared_config['num_data']}"
        / f"seed_{shared_config['seed']}_folds_{shared_config['num_folds']}_test_{shared_config['final_test_ratio']}.npz"
    )

    split_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "dev_idx": dev_idx,
        "final_test_idx": final_test_idx,
    }

    for fold_info in folds:
        fold = fold_info["fold"]
        save_dict[f"fold_{fold}_train_idx"] = fold_info["train_idx"]
        save_dict[f"fold_{fold}_val_idx"] = fold_info["val_idx"]
        save_dict[f"fold_{fold}_test_idx"] = fold_info["test_idx"]

    np.savez(split_path, **save_dict)

    print(f"Saved split indices to: {split_path}")

def get_hf_runner_and_model(model_name: str):
    """
    Import Hugging Face / PyTorch code only when an HF model is requested.
    This allows the Keras environment to run EFN/MEFN/OEFN without torch.
    """
    from runners.hf_runner import run_hf_experiment

    if model_name == "bert":
        from models.hf import bert as model_module
    elif model_name == "roberta":
        from models.hf import roberta as model_module
    elif model_name == "mamba":
        from models.hf import mamba as model_module
    else:
        raise ValueError(f"Unknown HF model: {model_name}")

    return run_hf_experiment, model_module


def get_keras_runner_and_model(model_name: str):
    """
    Import TensorFlow/Keras/EnergyFlow code only when a Keras model is requested.
    This allows the PyTorch environment to run BERT/RoBERTa/Mamba without TensorFlow.
    """
    from runners.keras_runner import run_keras_experiment

    if model_name == "efn":
        from models.efn import efn as model_module
    elif model_name == "mefn":
        from models.efn import mefn as model_module
    elif model_name == "oefn":
        from models.efn import oefn as model_module
    else:
        raise ValueError(f"Unknown Keras model: {model_name}")

    return run_keras_experiment, model_module

def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent

    shared_config = {
        "seed": args.seed,

        "num_data": args.num_data,
        "max_particles": args.max_particles,

        "num_folds": args.num_folds,
        "final_test_ratio": args.final_test_ratio,

        # Shared defaults.
        # Individual model configs may override these if their get_default_config includes them.
        "batch_size": 512,
        "epochs": args.epochs,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "use_early_stopping": True,
        "patience": 30,
        "early_stopping_threshold": 1e-4,

    }

    print("=" * 80)
    print("Loading qg_jets dataset")
    print("=" * 80)

    X_raw, y = qg_jets.load(num_data=shared_config["num_data"])

    print(f"Raw X shape: {X_raw.shape}")
    print(f"Raw y shape: {y.shape}")

    print("=" * 80)
    print("Preprocessing once for all models")
    print("=" * 80)

    X = preprocess_jets(
        X_raw,
        max_particles=shared_config["max_particles"],
        sort_by_pt=True,
    )

    y = y.astype(np.int64)

    print(f"Processed X shape: {X.shape}")
    print(f"Processed y shape: {y.shape}")

    print("=" * 80)
    print("Creating one shared 75/25 split and shared CV folds")
    print("=" * 80)

    dev_idx, final_test_idx, folds = create_fixed_test_cv_splits(
        y=y,
        num_folds=shared_config["num_folds"],
        final_test_ratio=shared_config["final_test_ratio"],
        seed=shared_config["seed"],
    )

    print(f"Development samples: {len(dev_idx)}")
    print(f"Fixed test samples : {len(final_test_idx)}")

    for fold_info in folds:
        print(
            f"Fold {fold_info['fold']}: "
            f"train={len(fold_info['train_idx'])}, "
            f"val={len(fold_info['val_idx'])}, "
            f"test={len(fold_info['test_idx'])}"
        )

    save_split_indices(
        project_root=project_root,
        dev_idx=dev_idx,
        final_test_idx=final_test_idx,
        folds=folds,
        shared_config=shared_config,
    )

    hf_model_names = {"bert", "roberta", "mamba"}
    keras_model_names = {"efn", "mefn", "oefn"}

    requested_models = [name.lower() for name in args.models]

    for model_name in requested_models:
        start_time = time.perf_counter()

        print("\n" + "-" * 80)
        print(f"Starting model: {model_name.upper()}")
        print("-" * 80)

        if model_name in hf_model_names:
            run_hf_experiment, model_module = get_hf_runner_and_model(model_name)
            model_config = model_module.get_default_config()

            run_hf_experiment(
                X=X,
                y=y,
                folds=folds,
                shared_config=shared_config,
                model_config=model_config,
                build_model_fn=model_module.build_model,
                get_model_summary_fields_fn=model_module.get_model_summary_fields,
            )

        elif model_name in keras_model_names:
            run_keras_experiment, model_module = get_keras_runner_and_model(model_name)
            model_config = model_module.get_default_config()

            run_keras_experiment(
                X=X,
                y=y,
                folds=folds,
                shared_config=shared_config,
                model_config=model_config,
                build_model_fn=model_module.build_model,
                prepare_fold_inputs_fn=model_module.prepare_fold_inputs,
                get_model_summary_fields_fn=model_module.get_model_summary_fields,
            )

        else:
            print(f"[WARN] Unknown model name: {model_name}")
            continue

        elapsed = time.perf_counter() - start_time

        print("-" * 80)
        print(f"Finished {model_name.upper()} in {elapsed / 60:.2f} min")
        print("-" * 80)


if __name__ == "__main__":
    main()
