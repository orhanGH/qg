from pathlib import Path
import argparse
import gc
import json

import numpy as np
import optuna

from sklearn.metrics import roc_auc_score

from tf_keras.callbacks import EarlyStopping
from tf_keras import backend as K

from utils import (
    get_or_create_file_level_test_cv_splits,
    load_marvin_parts_dataset,
    load_marvin_parts_and_obsvs_dataset,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    "/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"
)

HPO_ROOT = Path(
    "/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/"
    "s6oraydi_hpc_runs/hpo"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["efn", "mefn", "oefn"],
    )

    parser.add_argument("--n-trials", type=int, default=15)

    # For a fast smoke:
    # --folds 1
    # Production:
    # --folds 4
    parser.add_argument("--folds", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_module(model_name):
    if model_name == "efn":
        from models.efn import efn
        return efn

    if model_name == "mefn":
        from models.efn import mefn
        return mefn

    if model_name == "oefn":
        from models.efn import oefn
        return oefn

    raise ValueError(model_name)


def suggest_config(trial, model_name, module, epochs):
    config = module.get_default_config()

    # ------------------------------------------------------------------
    # Fixed comparison settings
    # ------------------------------------------------------------------
    config["max_particles"] = 128
    config["num_folds"] = 4
    config["final_test_ratio"] = 0.2
    config["seed"] = 42

    config["batch_size"] = 512
    config["epochs"] = epochs
    config["patience"] = 30
    config["use_early_stopping"] = True
    config["early_stopping_threshold"] = 1e-4

    config["activation"] = "silu"

    # ------------------------------------------------------------------
    # Common categorical architecture choices
    # ------------------------------------------------------------------
    phi_choice = trial.suggest_categorical(
        "Phi_arch",
        ["small", "baseline", "wide"],
    )

    phi_archs = {
        "small": (64, 64, 64),
        "baseline": (100, 100, 128),
        "wide": (128, 128, 128),
    }

    f_choice = trial.suggest_categorical(
        "F_arch",
        ["small", "baseline", "wide"],
    )

    f_archs = {
        "small": (64, 64),
        "baseline": (100, 100, 100),
        "wide": (128, 128),
    }

    config["Phi_sizes"] = phi_archs[phi_choice]
    config["F_sizes"] = f_archs[f_choice]

    config["learning_rate"] = trial.suggest_categorical(
        "learning_rate",
        [3e-5, 1e-4, 3e-4, 1e-3],
    )

    # ------------------------------------------------------------------
    # EFN
    # ------------------------------------------------------------------
    if model_name == "efn":
        config["latent_dropout"] = trial.suggest_categorical(
            "latent_dropout",
            [0.0, 0.1, 0.2],
        )

        config["F_dropouts"] = trial.suggest_categorical(
            "F_dropout",
            [0.0, 0.1, 0.2],
        )

    # ------------------------------------------------------------------
    # MEFN
    # ------------------------------------------------------------------
    elif model_name == "mefn":
        config["latent_dim"] = trial.suggest_categorical(
            "latent_dim",
            [8, 16, 24],
        )

        config["moment_order"] = trial.suggest_categorical(
            "moment_order",
            [2, 3],
        )

        config["weight_decay"] = trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-5, 1e-4],
        )

        # Keep dropout fixed to avoid too many search dimensions.
        config["phi_dropout"] = 0.1
        config["moment_dropout"] = 0.1
        config["F_dropout"] = 0.1

    # ------------------------------------------------------------------
    # oEFN
    # ------------------------------------------------------------------
    elif model_name == "oefn":
        config["n_pca_components"] = trial.suggest_categorical(
            "n_pca_components",
            [8, 13, 24, 32],
        )

        config["obs_latent_dim"] = trial.suggest_categorical(
            "obs_latent_dim",
            [32, 64, 128],
        )

        config["obs_dropout"] = trial.suggest_categorical(
            "obs_dropout",
            [0.0, 0.1, 0.2],
        )

        # Keep particle-branch dropout fixed.
        config["phi_dropout"] = 0.1
        config["latent_dropout"] = 0.1
        config["F_dropout"] = 0.1

        # These define the actual observable setup.
        config["remove_eecs"] = True
        config["nsubs_dim"] = 60
        config["eecs_dim"] = 23

    return config


def main():
    args = parse_args()

    model_name = args.model
    module = get_module(model_name)

    set_seed(args.seed)

    shared_config = {
        "seed": 42,
        "num_data": -1,
        "max_particles": 128,
        "num_folds": 4,
        "final_test_ratio": 0.2,
        "class_0": "vac",
        "class_1": "rec",
        "data_root": str(DATA_ROOT),
        "max_files_per_class": None,
    }

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if model_name == "oefn":
        (
            X_parts,
            X_obsvs,
            y,
            file_ids,
            file_labels,
            file_paths,
            _,
        ) = load_marvin_parts_and_obsvs_dataset(
            data_root=DATA_ROOT,
            class_0="vac",
            class_1="rec",
            max_particles=128,
            max_jets=None,
            max_files_per_class=None,
            sort_by_pt=True,
            seed=42,
            return_file_paths=True,
        )

        X_model = {
            "parts": X_parts,
            "obsvs": X_obsvs,
        }

    else:
        (
            X_parts,
            y,
            file_ids,
            file_labels,
            file_paths,
        ) = load_marvin_parts_dataset(
            data_root=DATA_ROOT,
            class_0="vac",
            class_1="rec",
            max_particles=128,
            max_jets=None,
            max_files_per_class=None,
            sort_by_pt=True,
            seed=42,
            return_file_paths=True,
        )

        X_model = X_parts

    y = y.astype(np.int64)

    # ------------------------------------------------------------------
    # Reuse exact baseline split
    # ------------------------------------------------------------------
    _, _, folds = get_or_create_file_level_test_cv_splits(
        project_root=PROJECT_ROOT,
        y=y,
        file_ids=file_ids,
        file_labels=file_labels,
        shared_config=shared_config,
        file_paths=file_paths,
    )

    folds = folds[:args.folds]

    print(f"Using {len(folds)} CV fold(s) for HPO.")

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def objective(trial):
        config = suggest_config(
            trial,
            model_name,
            module,
            args.epochs,
        )

        fold_aucs = []

        for fold_info in folds:
            fold = fold_info["fold"]

            K.clear_session()
            gc.collect()

            set_seed(42 + fold)

            train_idx = fold_info["train_idx"]
            val_idx = fold_info["val_idx"]

            # IMPORTANT:
            # pass val indices as the temporary third argument too,
            # so the held-out final test is NEVER used by HPO.
            train_inputs, val_inputs, _, extra_info = (
                module.prepare_fold_inputs(
                    X_model,
                    train_idx,
                    val_idx,
                    val_idx,
                    config,
                    Path("/tmp"),
                    {},
                )
            )

            model = module.build_model(
                config,
                extra_info,
            )

            y_train = np.eye(2, dtype=np.float32)[y[train_idx]]
            y_val = np.eye(2, dtype=np.float32)[y[val_idx]]

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=30,
                min_delta=1e-4,
                restore_best_weights=True,
            )

            model.fit(
                train_inputs,
                y_train,
                validation_data=(val_inputs, y_val),
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                callbacks=[early_stop],
                verbose=2,
            )

            probs = model.predict(
                val_inputs,
                batch_size=config["batch_size"],
                verbose=0,
            )

            auc = roc_auc_score(
                y[val_idx],
                probs[:, 1],
            )

            fold_aucs.append(float(auc))

            trial.report(
                float(np.mean(fold_aucs)),
                step=fold,
            )

            print(
                f"Trial {trial.number} | "
                f"fold {fold} | "
                f"val AUC = {auc:.6f}"
            )

        return float(np.mean(fold_aucs))

    # ------------------------------------------------------------------
    # Optuna
    # ------------------------------------------------------------------
    output_dir = HPO_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = (
        f"sqlite:///{output_dir / 'study.db'}"
    )

    study = optuna.create_study(
        study_name=f"pbpb_{model_name}_hpo",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
    )

    result = {
        "model": model_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "num_trials": len(study.trials),
    }

    with open(
        output_dir / "best_params.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()