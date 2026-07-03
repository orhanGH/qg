# =============================================================================
# main.py
# =============================================================================

from pathlib import Path
import argparse
import time
import json
import numpy as np

from utils import (
    get_or_create_file_level_test_cv_splits,
    load_marvin_parts_dataset,
    load_marvin_parts_and_obsvs_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["transformer_encoder", "mamba", "efn", "mefn", "aefn", "oefn"],
        help="Models to run. Options: transformer_encoder mamba efn mefn aefn oefn.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50",
        help="Path to Marvin ptmin50 dataset root.",
    )

    parser.add_argument(
        "--class-0",
        type=str,
        default="vac",
        choices=["rec", "vac"],
        help="Directory name used as label 0.",
    )

    parser.add_argument(
        "--class-1",
        type=str,
        default="rec",
        choices=["rec", "vac"],
        help="Directory name used as label 1.",
    )

    parser.add_argument(
        "--num-data",
        type=int,
        default=10000,
        help="Total number of jets to load. Use -1 to load all available jets.",
    )

    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Optional debug limit for number of .npz files per class.",
    )

    parser.add_argument(
        "--max-particles",
        type=int,
        default=128,
        help="Maximum number of constituents per jet after truncation/padding.",
    )

    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--final-test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only one fold, 1-based. If not set, run all folds.",
    )

    parser.add_argument(
        "--optimized-config",
        type=str,
        default=None,
        help="Path to JSON file containing optimized hyperparameters per model.",
    )

    return parser.parse_args()


def load_optimized_configs(path):
    if path is None:
        return {}

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Optimized config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        optimized_configs = json.load(f)

    print("=" * 80)
    print(f"Loaded optimized config from: {config_path}")
    print("Available optimized configs:", list(optimized_configs.keys()))
    print("=" * 80)

    return optimized_configs


def save_split_indices(project_root, dev_idx, final_test_idx, folds, shared_config):
    splits_dir = project_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    num_data_name = "all" if shared_config["num_data"] <= 0 else str(shared_config["num_data"])

    split_path = (
        splits_dir
        / (
            f"marvin_parts_obsvs_{shared_config['class_0']}_vs_{shared_config['class_1']}"
            f"_numdata_{num_data_name}_maxp_{shared_config['max_particles']}"
        )
        / (
            f"seed_{shared_config['seed']}"
            f"_folds_{shared_config['num_folds']}"
            f"_test_{shared_config['final_test_ratio']}.npz"
        )
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
        save_dict[f"fold_{fold}_train_files"] = fold_info["train_files"]
        save_dict[f"fold_{fold}_val_files"] = fold_info["val_files"]
        save_dict[f"fold_{fold}_test_files"] = fold_info["test_files"]

    np.savez(split_path, **save_dict)
    print(f"Saved split indices to: {split_path}")


def get_hf_runner_and_model(model_name):
    from runners.hf_runner import run_hf_experiment

    if model_name == "transformer_encoder":
        from models.hf import transformer_encoder as model_module
    elif model_name == "mamba":
        from models.hf import mamba as model_module
    else:
        raise ValueError(f"Unknown HF model: {model_name}")

    return run_hf_experiment, model_module


def get_keras_runner_and_model(model_name):
    from runners.keras_runner import run_keras_experiment

    if model_name == "efn":
        from models.efn import efn as model_module
    elif model_name == "mefn":
        from models.efn import mefn as model_module
    elif model_name == "aefn":
        from models.efn import aefn as model_module
    elif model_name == "oefn":
        from models.efn import oefn as model_module
    elif model_name == "aefn_before_phi":
        from models.efn import aefn_before_phi as model_module
#######################################################
    elif model_name == "aefn_after_phi":
        from models.efn import aefn_after_phi as model_module

    elif model_name == "aefn_before_f":
        from models.efn import aefn_before_f as model_module

    elif model_name == "aefn_all":
        from models.efn import aefn_all as model_module
######################################################
    else:
        raise ValueError(f"Unknown Keras model: {model_name}")

    return run_keras_experiment, model_module


def apply_config_overrides(
    model_name,
    model_config,
    shared_config,
    optimized_configs,
    args,
):
    if model_name in optimized_configs:
        print(f"Applying optimized config for {model_name}:")
        print(json.dumps(optimized_configs[model_name], indent=2))
        model_config.update(optimized_configs[model_name])

    model_config["num_data"] = shared_config["num_data"]
    model_config["max_particles"] = shared_config["max_particles"]
    model_config["num_folds"] = shared_config["num_folds"]
    model_config["final_test_ratio"] = shared_config["final_test_ratio"]
    model_config["seed"] = shared_config["seed"]
    model_config["epochs"] = args.epochs
    model_config["learning_rate"] = shared_config["learning_rate"]
    model_config["weight_decay"] = shared_config["weight_decay"]

    print(f"Final config for {model_name}:")
    print(json.dumps(model_config, indent=2))
    print("=" * 80)

    return model_config


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    shared_config = {
        "seed": args.seed,
        "num_data": args.num_data,
        "max_particles": args.max_particles,
        "num_folds": args.num_folds,
        "final_test_ratio": args.final_test_ratio,
        "class_0": args.class_0,
        "class_1": args.class_1,
        "data_root": args.data_root,
        "max_files_per_class": args.max_files_per_class,
        "batch_size": 512,
        "epochs": args.epochs,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
        "use_early_stopping": True,
        "patience": 30,
        "early_stopping_threshold": 1e-4,
    }

    optimized_configs = load_optimized_configs(args.optimized_config)

    hf_model_names = {"transformer_encoder", "mamba"}
    keras_model_names = {
    "efn",
    "mefn",
    "aefn",
    "oefn",
    "aefn_before_phi",
    "aefn_after_phi",
    "aefn_before_f",
    "aefn_all",
    }
    requested_models = [name.lower() for name in args.models]
    need_obsvs = "oefn" in requested_models

    print("=" * 80)
    if need_obsvs:
        print("Loading Marvin parts + observable dataset")
    else:
        print("Loading Marvin parts dataset only; observables skipped because oEFN is not requested")
    print("=" * 80)
    print(f"Data root           : {args.data_root}")
    print(f"Label 0             : {args.class_0}")
    print(f"Label 1             : {args.class_1}")
    print(f"num_data            : {args.num_data} (-1 means all)")
    print(f"max_particles       : {args.max_particles}")
    print(f"max_files_per_class : {args.max_files_per_class}")

    max_jets = None if args.num_data <= 0 else args.num_data

    if need_obsvs:
        (
            X_parts,
            X_obsvs,
            y,
            file_ids,
            file_labels,
            file_paths,
            obsvs_paths,
        ) = load_marvin_parts_and_obsvs_dataset(
            data_root=Path(args.data_root),
            class_0=args.class_0,
            class_1=args.class_1,
            max_particles=args.max_particles,
            max_jets=max_jets,
            max_files_per_class=args.max_files_per_class,
            sort_by_pt=True,
            seed=args.seed,
            return_file_paths=True,
        )
    else:
        (
            X_parts,
            y,
            file_ids,
            file_labels,
            file_paths,
        ) = load_marvin_parts_dataset(
            data_root=Path(args.data_root),
            class_0=args.class_0,
            class_1=args.class_1,
            max_particles=args.max_particles,
            max_jets=max_jets,
            max_files_per_class=args.max_files_per_class,
            sort_by_pt=True,
            seed=args.seed,
            return_file_paths=True,
        )
        X_obsvs = None
        obsvs_paths = [""] * len(file_paths)

    y = y.astype(np.int64)

    print("=" * 80)
    print("Loaded and preprocessed Marvin dataset")
    print("=" * 80)
    print(f"X_parts shape       : {X_parts.shape}")
    if X_obsvs is None:
        print("X_obsvs             : skipped")
    else:
        print(f"X_obsvs shape       : {X_obsvs.shape}")
    print(f"y shape             : {y.shape}")
    print(f"file_ids shape      : {file_ids.shape}")
    print(f"num files loaded    : {len(file_labels)}")
    print(f"class counts        : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"X_parts dtype       : {X_parts.dtype}")
    if X_obsvs is not None:
        print(f"X_obsvs dtype       : {X_obsvs.dtype}")
    print("Parts convention:")
    print("  X_parts[..., 0] = z = pt_i / sum_j pt_j")
    print("  X_parts[..., 1] = delta_eta")
    print("  X_parts[..., 2] = delta_phi")
    if X_obsvs is not None:
        print("Observables convention:")
        print("  X_obsvs[:, 0:60]    = nsubs")
        print("  X_obsvs[:, 60:83]   = eecs")
        print("  X_obsvs[:, 83:3472] = efps")
    print("=" * 80)

    print("Creating one shared FILE-LEVEL split and shared CV folds")
    print("=" * 80)

    dev_idx, final_test_idx, folds = get_or_create_file_level_test_cv_splits(
        project_root=project_root,
        y=y,
        file_ids=file_ids,
        file_labels=file_labels,
        shared_config=shared_config,
        file_paths=file_paths,
    )

    print(f"Development samples: {len(dev_idx)}")
    print(f"Fixed test samples : {len(final_test_idx)}")

    for fold_info in folds:
        print(
            f"Fold {fold_info['fold']}: "
            f"train={len(fold_info['train_idx'])}, "
            f"val={len(fold_info['val_idx'])}, "
            f"test={len(fold_info['test_idx'])}, "
            f"train_files={len(fold_info['train_files'])}, "
            f"val_files={len(fold_info['val_files'])}, "
            f"test_files={len(fold_info['test_files'])}"
        )

    save_split_indices(
        project_root=project_root,
        dev_idx=dev_idx,
        final_test_idx=final_test_idx,
        folds=folds,
        shared_config=shared_config,
    )

    file_list_path = project_root / "splits" / "last_marvin_file_list.txt"
    file_list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_list_path, "w", encoding="utf-8") as f:
        for i, path in enumerate(file_paths):
            obs_path = obsvs_paths[i] if i < len(obsvs_paths) else ""
            f.write(f"{i}\t{file_labels[i]}\tparts={path}\tobsvs={obs_path}\n")

    print(f"Saved loaded file list to: {file_list_path}")

    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]

        if len(folds) != 1:
            raise ValueError(f"Invalid fold {args.fold}. Must be between 1 and {args.num_folds}.")

        print("=" * 80)
        print(f"Running only fold {args.fold}")
        print("=" * 80)

    for model_name in requested_models:
        start_time = time.perf_counter()

        print("\n" + "-" * 80)
        print(f"Starting model: {model_name.upper()}")
        print("-" * 80)

        if model_name in hf_model_names:
            run_hf_experiment, model_module = get_hf_runner_and_model(model_name)

            model_config = model_module.get_default_config()
            model_config = apply_config_overrides(
                model_name=model_name,
                model_config=model_config,
                shared_config=shared_config,
                optimized_configs=optimized_configs,
                args=args,
            )

            run_hf_experiment(
                X=X_parts,
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
            model_config = apply_config_overrides(
                model_name=model_name,
                model_config=model_config,
                shared_config=shared_config,
                optimized_configs=optimized_configs,
                args=args,
            )

            if model_name == "oefn" and X_obsvs is None:
                raise RuntimeError("oEFN requires observables, but X_obsvs was not loaded.")

            X_model = (
                {"parts": X_parts, "obsvs": X_obsvs}
                if model_name == "oefn"
                else X_parts
            )

            run_keras_experiment(
                X=X_model,
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
