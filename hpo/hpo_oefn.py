from pathlib import Path
import argparse
import gc
import json

import numpy as np
import optuna

from sklearn.metrics import roc_auc_score

from tf_keras.callbacks import Callback, EarlyStopping
from tf_keras import backend as K

from utils import (
    get_or_create_file_level_test_cv_splits,
    load_marvin_parts_and_obsvs_dataset,
    set_seed,
)

from models.efn import oefn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    "/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"
)

HPO_ROOT = Path(
    "/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/"
    "s6oraydi_hpc_runs/hpo_reduced/oefn"
)


class OptunaPruningCallback(Callback):

    def __init__(self, trial, fold):
        super().__init__()
        self.trial = trial
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        val_loss = logs.get(
            "val_loss"
        )

        if val_loss is None:
            return

        score = -float(
            val_loss
        )

        step = (
            (self.fold - 1) * 1000
            + epoch
        )

        self.trial.report(
            score,
            step=step,
        )

        if self.trial.should_prune():

            raise optuna.TrialPruned(
                f"Trial {self.trial.number} "
                f"pruned at fold {self.fold}, "
                f"epoch {epoch + 1}"
            )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    return parser.parse_args()


def suggest_config(
    trial,
    epochs,
    patience,
):

    config = oefn.get_default_config()

    config["max_particles"] = 128
    config["num_folds"] = 4
    config["final_test_ratio"] = 0.2
    config["seed"] = 42

    config["epochs"] = epochs
    config["patience"] = patience

    config["use_early_stopping"] = True

    config[
        "early_stopping_threshold"
    ] = 1e-4

    config["batch_size"] = 512

    config["activation"] = (
        trial.suggest_categorical(
            "activation",
            [
                "gelu",
                "silu",
            ],
        )
    )

    config["learning_rate"] = (
        trial.suggest_categorical(
            "learning_rate",
            [
                1e-4,
                3e-4,
                1e-3,
            ],
        )
    )

    phi_choice = (
        trial.suggest_categorical(
            "Phi_arch",
            [
                "baseline",
                "wide",
            ],
        )
    )

    phi_archs = {

        "baseline": (
            100,
            100,
            128,
        ),

        "wide": (
            128,
            128,
            128,
        ),
    }

    config["Phi_sizes"] = (
        phi_archs[
            phi_choice
        ]
    )

    f_choice = (
        trial.suggest_categorical(
            "F_arch",
            [
                "baseline",
                "wide",
            ],
        )
    )

    f_archs = {

        "baseline": (
            100,
            100,
            100,
        ),

        "wide": (
            128,
            128,
        ),
    }

    config["F_sizes"] = (
        f_archs[
            f_choice
        ]
    )

    config["n_pca_components"] = (
        trial.suggest_categorical(
            "n_pca_components",
            [
                13,
                32,
            ],
        )
    )

    config["obs_latent_dim"] = (
        trial.suggest_categorical(
            "obs_latent_dim",
            [
                64,
                128,
            ],
        )
    )

    config["obs_dropout"] = (
        trial.suggest_categorical(
            "obs_dropout",
            [
                0.0,
                0.1,
                0.2,
            ],
        )
    )

    config["phi_dropout"] = (
        trial.suggest_categorical(
            "phi_dropout",
            [
                0.0,
                0.1,
                0.2,
            ],
        )
    )

    config["latent_dropout"] = (
        trial.suggest_categorical(
            "latent_dropout",
            [
                0.0,
                0.1,
                0.2,
            ],
        )
    )

    config["F_dropout"] = (
        trial.suggest_categorical(
            "F_dropout",
            [
                0.0,
                0.1,
                0.2,
            ],
        )
    )

    config["remove_eecs"] = True

    config["nsubs_dim"] = 60

    config["eecs_dim"] = 23

    return config


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    shared_config = {

        "seed": 42,
        "num_data": -1,
        "max_particles": 128,
        "num_folds": 4,
        "final_test_ratio": 0.2,

        "class_0": "vac",
        "class_1": "rec",

        "data_root": str(
            DATA_ROOT
        ),

        "max_files_per_class": None,
    }

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

    y = y.astype(
        np.int64
    )

    _, _, folds = (
        get_or_create_file_level_test_cv_splits(

            project_root=PROJECT_ROOT,

            y=y,

            file_ids=file_ids,

            file_labels=file_labels,

            shared_config=shared_config,

            file_paths=file_paths,
        )
    )

    folds = folds[
        :args.folds
    ]

    HPO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    def objective(trial):

        config = suggest_config(
            trial,
            args.epochs,
            args.patience,
        )

        fold_aucs = []

        for fold_info in folds:

            fold = (
                fold_info["fold"]
            )

            K.clear_session()
            gc.collect()

            set_seed(
                args.seed
                + fold
            )

            train_idx = (
                fold_info[
                    "train_idx"
                ]
            )

            val_idx = (
                fold_info[
                    "val_idx"
                ]
            )

            (
                train_inputs,
                val_inputs,
                _,
                extra_info,
            ) = oefn.prepare_fold_inputs(

                X_model,

                train_idx,

                val_idx,

                val_idx,

                config,

                Path("/tmp"),

                {},
            )

            model = oefn.build_model(
                config,
                extra_info,
            )

            y_train = np.eye(
                2,
                dtype=np.float32,
            )[y[train_idx]]

            y_val = np.eye(
                2,
                dtype=np.float32,
            )[y[val_idx]]

            callbacks = [

                EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                    verbose=1,
                ),

                OptunaPruningCallback(
                    trial,
                    fold,
                ),
            ]

            model.fit(

                train_inputs,

                y_train,

                validation_data=(
                    val_inputs,
                    y_val,
                ),

                epochs=config[
                    "epochs"
                ],

                batch_size=config[
                    "batch_size"
                ],

                callbacks=callbacks,

                verbose=2,
            )

            probs = model.predict(

                val_inputs,

                batch_size=config[
                    "batch_size"
                ],

                verbose=0,
            )

            auc = roc_auc_score(

                y[val_idx],

                probs[:, 1],
            )

            fold_aucs.append(
                float(auc)
            )

            print(
                f"Trial {trial.number} | "
                f"Fold {fold} | "
                f"AUC={auc:.6f}"
            )

        final_auc = float(
            np.mean(
                fold_aucs
            )
        )

        trial.set_user_attr(
            "fold_aucs",
            fold_aucs,
        )

        trial.set_user_attr(
            "auc_std",
            float(
                np.std(
                    fold_aucs
                )
            ),
        )

        return final_auc


    storage = (
        f"sqlite:///"
        f"{HPO_ROOT / 'study.db'}"
    )

    study = optuna.create_study(

        study_name=
            "pbpb_oefn_reduced_hpo",

        direction="maximize",

        storage=storage,

        load_if_exists=True,

        sampler=
            optuna.samplers.TPESampler(
                seed=args.seed,
                multivariate=True,
            ),

        pruner=
            optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1,
            ),
    )

    study.optimize(

        objective,

        n_trials=args.n_trials,

        gc_after_trial=True,
    )

    result = {

        "model": "oefn",

        "best_value":
            study.best_value,

        "best_params":
            study.best_params,

        "best_trial_number":
            study.best_trial.number,

        "best_trial_fold_aucs":
            study.best_trial
            .user_attrs
            .get("fold_aucs"),

        "best_trial_auc_std":
            study.best_trial
            .user_attrs
            .get("auc_std"),

        "num_trials":
            len(study.trials),
    }

    with open(
        HPO_ROOT / "best_params.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    study.trials_dataframe().to_csv(
        HPO_ROOT / "all_trials.csv",
        index=False,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
