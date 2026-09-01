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
    "s6oraydi_hpc_runs/hpo_large_search"
)


# ============================================================
# Optuna pruning callback for Keras
# ============================================================

class OptunaPruningCallback(Callback):

    def __init__(self, trial, fold):
        super().__init__()

        self.trial = trial
        self.fold = fold

    def on_epoch_end(
        self,
        epoch,
        logs=None,
    ):

        logs = logs or {}

        val_loss = logs.get(
            "val_loss"
        )

        if val_loss is None:
            return

        # We maximize the reported value.
        # Therefore report negative validation loss.
        score = -float(
            val_loss
        )

        # Unique step across all folds.
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
                f"epoch {epoch + 1}, "
                f"val_loss={val_loss:.6f}"
            )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "efn",
            "mefn",
            "oefn",
        ],
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
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


# ============================================================
# Model import
# ============================================================

def get_module(
    model_name,
):

    if model_name == "efn":

        from models.efn import efn

        return efn

    if model_name == "mefn":

        from models.efn import mefn

        return mefn

    if model_name == "oefn":

        from models.efn import oefn

        return oefn

    raise ValueError(
        model_name
    )


# ============================================================
# Hyperparameter search space
# ============================================================

def suggest_config(
    trial,
    model_name,
    module,
    epochs,
    patience,
):

    config = (
        module
        .get_default_config()
    )


    # ========================================================
    # Fixed experiment settings
    # ========================================================

    config["max_particles"] = 128

    config["num_folds"] = 4

    config["final_test_ratio"] = 0.2

    config["seed"] = 42

    config["epochs"] = epochs

    config["use_early_stopping"] = True

    config["patience"] = patience

    config[
        "early_stopping_threshold"
    ] = 1e-4


    # ========================================================
    # Shared activation search
    # ========================================================

    config["activation"] = (
        trial.suggest_categorical(
            "activation",
            [
                "relu",
                "gelu",
                "silu",
            ],
        )
    )


    # ========================================================
    # Batch size
    # ========================================================

    config["batch_size"] = (
        trial.suggest_categorical(
            "batch_size",
            [
                256,
                512,
                1024,
            ],
        )
    )


    # ========================================================
    # Learning rate
    # ========================================================

    config["learning_rate"] = (
        trial.suggest_categorical(
            "learning_rate",
            [
                3e-5,
                1e-4,
                3e-4,
                1e-3,
            ],
        )
    )


    # ========================================================
    # Phi architecture
    # ========================================================

    phi_choice = (
        trial.suggest_categorical(
            "Phi_arch",
            [
                "small",
                "baseline",
                "wide",
                "deep",
            ],
        )
    )

    phi_archs = {

        "small":
            (
                64,
                64,
                64,
            ),

        "baseline":
            (
                100,
                100,
                128,
            ),

        "wide":
            (
                128,
                128,
                128,
            ),

        "deep":
            (
                128,
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


    # ========================================================
    # F architecture
    # ========================================================

    f_choice = (
        trial.suggest_categorical(
            "F_arch",
            [
                "small",
                "baseline",
                "wide",
                "deep",
            ],
        )
    )

    f_archs = {

        "small":
            (
                64,
                64,
            ),

        "baseline":
            (
                100,
                100,
                100,
            ),

        "wide":
            (
                128,
                128,
            ),

        "deep":
            (
                128,
                128,
                128,
                128,
            ),
    }

    config["F_sizes"] = (
        f_archs[
            f_choice
        ]
    )


    # ========================================================
    # EFN
    # ========================================================

    if model_name == "efn":

        config[
            "latent_dropout"
        ] = trial.suggest_categorical(
            "latent_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "F_dropouts"
        ] = trial.suggest_categorical(
            "F_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )


    # ========================================================
    # MEFN
    # ========================================================

    elif model_name == "mefn":

        config[
            "latent_dim"
        ] = trial.suggest_categorical(
            "latent_dim",
            [
                8,
                16,
                24,
                32,
            ],
        )

        config[
            "moment_order"
        ] = trial.suggest_categorical(
            "moment_order",
            [
                2,
                3,
            ],
        )

        config[
            "phi_dropout"
        ] = trial.suggest_categorical(
            "phi_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "moment_dropout"
        ] = trial.suggest_categorical(
            "moment_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "F_dropout"
        ] = trial.suggest_categorical(
            "F_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "weight_decay"
        ] = trial.suggest_categorical(
            "weight_decay",
            [
                0.0,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
            ],
        )

        config[
            "clipnorm"
        ] = trial.suggest_categorical(
            "clipnorm",
            [
                0.5,
                1.0,
                2.0,
            ],
        )


    # ========================================================
    # oEFN
    # ========================================================

    elif model_name == "oefn":

        config[
            "n_pca_components"
        ] = trial.suggest_categorical(
            "n_pca_components",
            [
                8,
                13,
                24,
                32,
                64,
            ],
        )

        config[
            "obs_latent_dim"
        ] = trial.suggest_categorical(
            "obs_latent_dim",
            [
                32,
                64,
                128,
                256,
            ],
        )

        config[
            "obs_dropout"
        ] = trial.suggest_categorical(
            "obs_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "phi_dropout"
        ] = trial.suggest_categorical(
            "phi_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "latent_dropout"
        ] = trial.suggest_categorical(
            "latent_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "F_dropout"
        ] = trial.suggest_categorical(
            "F_dropout",
            [
                0.0,
                0.05,
                0.10,
                0.20,
                0.30,
            ],
        )

        config[
            "remove_eecs"
        ] = True

        config[
            "nsubs_dim"
        ] = 60

        config[
            "eecs_dim"
        ] = 23


    return config


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    model_name = (
        args.model
    )

    module = (
        get_module(
            model_name
        )
    )

    set_seed(
        args.seed
    )


    # ========================================================
    # Shared experimental config
    # ========================================================

    shared_config = {

        "seed":
            42,

        "num_data":
            -1,

        "max_particles":
            128,

        "num_folds":
            4,

        "final_test_ratio":
            0.2,

        "class_0":
            "vac",

        "class_1":
            "rec",

        "data_root":
            str(
                DATA_ROOT
            ),

        "max_files_per_class":
            None,
    }


    # ========================================================
    # Load data
    # ========================================================

    if model_name == "oefn":

        (
            X_parts,
            X_obsvs,
            y,
            file_ids,
            file_labels,
            file_paths,
            _,
        ) = (
            load_marvin_parts_and_obsvs_dataset(

                data_root=
                    DATA_ROOT,

                class_0=
                    "vac",

                class_1=
                    "rec",

                max_particles=
                    128,

                max_jets=
                    None,

                max_files_per_class=
                    None,

                sort_by_pt=
                    True,

                seed=
                    42,

                return_file_paths=
                    True,
            )
        )

        X_model = {

            "parts":
                X_parts,

            "obsvs":
                X_obsvs,
        }

    else:

        (
            X_parts,
            y,
            file_ids,
            file_labels,
            file_paths,
        ) = (
            load_marvin_parts_dataset(

                data_root=
                    DATA_ROOT,

                class_0=
                    "vac",

                class_1=
                    "rec",

                max_particles=
                    128,

                max_jets=
                    None,

                max_files_per_class=
                    None,

                sort_by_pt=
                    True,

                seed=
                    42,

                return_file_paths=
                    True,
            )
        )

        X_model = (
            X_parts
        )


    y = y.astype(
        np.int64
    )


    # ========================================================
    # Reuse same file-level CV split
    # ========================================================

    _, _, folds = (
        get_or_create_file_level_test_cv_splits(

            project_root=
                PROJECT_ROOT,

            y=
                y,

            file_ids=
                file_ids,

            file_labels=
                file_labels,

            shared_config=
                shared_config,

            file_paths=
                file_paths,
        )
    )

    folds = folds[
        :args.folds
    ]

    print(
        f"Model: {model_name}"
    )

    print(
        f"Using {len(folds)} CV folds."
    )

    print(
        f"Trials requested: {args.n_trials}"
    )

    print(
        f"Max epochs per fold: {args.epochs}"
    )


    # ========================================================
    # Objective
    # ========================================================

    def objective(
        trial
    ):

        config = (
            suggest_config(
                trial,
                model_name,
                module,
                args.epochs,
                args.patience,
            )
        )

        print(
            "\n"
            "================================================"
        )

        print(
            f"Starting trial {trial.number}"
        )

        print(
            json.dumps(
                {
                    key:
                        (
                            list(value)
                            if isinstance(
                                value,
                                tuple,
                            )
                            else value
                        )
                    for key, value
                    in config.items()
                },
                indent=2,
                default=str,
            )
        )

        print(
            "================================================"
        )

        fold_aucs = []


        # ====================================================
        # CV folds
        # ====================================================

        for fold_info in folds:

            fold = (
                fold_info[
                    "fold"
                ]
            )


            # Clean previous model from memory
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


            # =================================================
            # Prepare model inputs
            # =================================================

            train_inputs, \
            val_inputs, \
            _, \
            extra_info = (
                module.prepare_fold_inputs(

                    X_model,

                    train_idx,

                    val_idx,

                    # Important:
                    # validation is reused here.
                    # Final test is not touched during HPO.
                    val_idx,

                    config,

                    Path("/tmp"),

                    {},
                )
            )


            # =================================================
            # Build model
            # =================================================

            model = (
                module.build_model(
                    config,
                    extra_info,
                )
            )


            y_train = (
                np.eye(
                    2,
                    dtype=np.float32,
                )[
                    y[
                        train_idx
                    ]
                ]
            )

            y_val = (
                np.eye(
                    2,
                    dtype=np.float32,
                )[
                    y[
                        val_idx
                    ]
                ]
            )


            # =================================================
            # Callbacks
            # =================================================

            early_stop = (
                EarlyStopping(

                    monitor=
                        "val_loss",

                    patience=
                        args.patience,

                    min_delta=
                        1e-4,

                    restore_best_weights=
                        True,

                    verbose=
                        1,
                )
            )


            pruning_callback = (
                OptunaPruningCallback(
                    trial,
                    fold,
                )
            )


            # =================================================
            # Train
            # =================================================

            model.fit(

                train_inputs,

                y_train,

                validation_data=(
                    val_inputs,
                    y_val,
                ),

                epochs=
                    config[
                        "epochs"
                    ],

                batch_size=
                    config[
                        "batch_size"
                    ],

                callbacks=[
                    early_stop,
                    pruning_callback,
                ],

                verbose=
                    2,
            )


            # =================================================
            # Validation prediction
            # =================================================

            probs = (
                model.predict(

                    val_inputs,

                    batch_size=
                        config[
                            "batch_size"
                        ],

                    verbose=
                        0,
                )
            )


            auc = (
                roc_auc_score(

                    y[
                        val_idx
                    ],

                    probs[
                        :,
                        1,
                    ],
                )
            )


            fold_aucs.append(
                float(
                    auc
                )
            )


            mean_auc = float(
                np.mean(
                    fold_aucs
                )
            )


            print(
                f"Trial {trial.number} | "
                f"Fold {fold} | "
                f"AUC={auc:.6f} | "
                f"Running mean={mean_auc:.6f}"
            )


            # =================================================
            # Additional pruning after complete fold
            # =================================================

            fold_step = (
                10000
                + fold
            )

            trial.report(
                mean_auc,
                step=fold_step,
            )

            if (
                trial.should_prune()
            ):

                raise (
                    optuna
                    .TrialPruned(
                        f"Trial "
                        f"{trial.number} "
                        f"pruned after "
                        f"fold {fold}"
                    )
                )


        # ====================================================
        # Final objective:
        # mean validation AUC across folds
        # ====================================================

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


    # ========================================================
    # Output directory
    # ========================================================

    output_dir = (
        HPO_ROOT
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Persistent Optuna storage
    # ========================================================

    storage = (
        f"sqlite:///"
        f"{output_dir / 'study.db'}"
    )


    # ========================================================
    # Optuna study
    # ========================================================

    study = (
        optuna.create_study(

            study_name=
                f"pbpb_"
                f"{model_name}_"
                f"large_hpo",

            direction=
                "maximize",

            storage=
                storage,

            load_if_exists=
                True,

            sampler=
                optuna
                .samplers
                .TPESampler(

                    seed=
                        args.seed,

                    multivariate=
                        True,
                ),

            pruner=
                optuna
                .pruners
                .MedianPruner(

                    # First 5 trials run
                    # without pruning.
                    n_startup_trials=
                        5,

                    # Give each new trial
                    # a few epochs before
                    # comparing it.
                    n_warmup_steps=
                        3,

                    interval_steps=
                        1,
                ),
        )
    )


    print(
        "\nOptuna study:"
    )

    print(
        study.study_name
    )

    print(
        "Existing trials:",
        len(
            study.trials
        ),
    )


    # ========================================================
    # Optimize
    # ========================================================

    study.optimize(

        objective,

        n_trials=
            args.n_trials,

        gc_after_trial=
            True,

        show_progress_bar=
            False,
    )


    # ========================================================
    # Study statistics
    # ========================================================

    completed = [
        t
        for t in study.trials
        if (
            t.state
            ==
            optuna
            .trial
            .TrialState
            .COMPLETE
        )
    ]

    pruned = [
        t
        for t in study.trials
        if (
            t.state
            ==
            optuna
            .trial
            .TrialState
            .PRUNED
        )
    ]

    failed = [
        t
        for t in study.trials
        if (
            t.state
            ==
            optuna
            .trial
            .TrialState
            .FAIL
        )
    ]


    # ========================================================
    # Best result
    # ========================================================

    result = {

        "model":
            model_name,

        "best_value":
            study.best_value,

        "best_params":
            study.best_params,

        "best_trial_number":
            study.best_trial.number,

        "best_trial_fold_aucs":
            study.best_trial.user_attrs.get(
                "fold_aucs"
            ),

        "best_trial_auc_std":
            study.best_trial.user_attrs.get(
                "auc_std"
            ),

        "num_trials_total":
            len(
                study.trials
            ),

        "num_complete":
            len(
                completed
            ),

        "num_pruned":
            len(
                pruned
            ),

        "num_failed":
            len(
                failed
            ),
    }


    # ========================================================
    # Save JSON
    # ========================================================

    with open(
        output_dir
        / "best_params.json",

        "w",

        encoding=
            "utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )


    # ========================================================
    # Save all trials as CSV
    # ========================================================

    trials_df = (
        study
        .trials_dataframe()
    )

    trials_df.to_csv(
        output_dir
        / "all_trials.csv",

        index=
            False,
    )


    # ========================================================
    # Print result
    # ========================================================

    print(
        "\n"
        "================================================"
    )

    print(
        "HPO FINISHED"
    )

    print(
        "================================================"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":

    main()
