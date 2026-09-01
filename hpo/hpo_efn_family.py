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
    "s6oraydi_hpc_runs/hpo_reduced"
)


# ============================================================
# Keras -> Optuna pruning
# ============================================================

class OptunaPruningCallback(Callback):

    def __init__(self, trial, fold):
        super().__init__()
        self.trial = trial
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        val_loss = logs.get("val_loss")

        if val_loss is None:
            return

        # Study maximizes the reported value.
        # Therefore use negative validation loss.
        score = -float(val_loss)

        # Unique step for every epoch/fold.
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
# CLI
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


# ============================================================
# Model module
# ============================================================

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

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# Reduced HPO search space
# ============================================================

def suggest_config(
    trial,
    model_name,
    module,
    epochs,
    patience,
):

    config = module.get_default_config()


    # ========================================================
    # Fixed experiment settings
    # ========================================================

    config["max_particles"] = 128

    config["num_folds"] = 4

    config["final_test_ratio"] = 0.2

    config["seed"] = 42

    config["epochs"] = epochs

    config["patience"] = patience

    config["use_early_stopping"] = True

    config["early_stopping_threshold"] = 1e-4


    # ========================================================
    # Shared: activation
    #
    # Reduced from:
    # relu / gelu / silu
    #
    # to:
    # gelu / silu
    # ========================================================

    config["activation"] = trial.suggest_categorical(
        "activation",
        [
            "gelu",
            "silu",
        ],
    )


    # ========================================================
    # Shared: fixed batch size
    #
    # Do NOT search 256 / 512 / 1024.
    # This keeps the comparison fair and saves time.
    # ========================================================

    config["batch_size"] = 512


    # ========================================================
    # Shared: learning rate
    # ========================================================

    config["learning_rate"] = trial.suggest_categorical(
        "learning_rate",
        [
            1e-4,
            3e-4,
            1e-3,
        ],
    )


    # ========================================================
    # Shared: Phi architecture
    #
    # Only baseline and wide.
    # ========================================================

    phi_choice = trial.suggest_categorical(
        "Phi_arch",
        [
            "baseline",
            "wide",
        ],
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


    # ========================================================
    # Shared: F architecture
    # ========================================================

    f_choice = trial.suggest_categorical(
        "F_arch",
        [
            "baseline",
            "wide",
        ],
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


    # ========================================================
    # EFN-specific
    # ========================================================

    if model_name == "efn":

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

        # EFN uses the plural key F_dropouts.
        config["F_dropouts"] = (
            trial.suggest_categorical(
                "F_dropout",
                [
                    0.0,
                    0.1,
                    0.2,
                ],
            )
        )


    # ========================================================
    # MEFN-specific
    # ========================================================

    elif model_name == "mefn":

        config["latent_dim"] = (
            trial.suggest_categorical(
                "latent_dim",
                [
                    16,
                    24,
                ],
            )
        )

        config["moment_order"] = (
            trial.suggest_categorical(
                "moment_order",
                [
                    2,
                    3,
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

        config["moment_dropout"] = (
            trial.suggest_categorical(
                "moment_dropout",
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

        config["weight_decay"] = (
            trial.suggest_categorical(
                "weight_decay",
                [
                    0.0,
                    1e-4,
                ],
            )
        )

        # Keep fixed.
        config["clipnorm"] = 1.0


    # ========================================================
    # oEFN-specific
    # ========================================================

    elif model_name == "oefn":

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

        # Fixed observable definition.
        config["remove_eecs"] = True

        config["nsubs_dim"] = 60

        config["eecs_dim"] = 23


    return config


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    model_name = args.model

    module = get_module(
        model_name
    )

    set_seed(
        args.seed
    )


    # ========================================================
    # Common data/split setup
    # ========================================================

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


    y = y.astype(
        np.int64
    )


    # ========================================================
    # Reuse exact file-level split
    # ========================================================

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


    print(
        "========================================"
    )

    print(
        f"MODEL: {model_name}"
    )

    print(
        f"TRIALS: {args.n_trials}"
    )

    print(
        f"FOLDS: {len(folds)}"
    )

    print(
        f"MAX EPOCHS: {args.epochs}"
    )

    print(
        f"PATIENCE: {args.patience}"
    )

    print(
        "========================================"
    )


    # ========================================================
    # Objective
    # ========================================================

    def objective(trial):

        config = suggest_config(

            trial=trial,

            model_name=model_name,

            module=module,

            epochs=args.epochs,

            patience=args.patience,
        )


        print(
            "\n========================================"
        )

        print(
            f"STARTING TRIAL {trial.number}"
        )

        print(
            "========================================"
        )


        printable_config = {

            key: (
                list(value)
                if isinstance(
                    value,
                    tuple,
                )
                else value
            )

            for key, value
            in config.items()
        }


        print(
            json.dumps(
                printable_config,
                indent=2,
                default=str,
            )
        )


        fold_aucs = []


        # ====================================================
        # Cross validation
        # ====================================================

        for fold_info in folds:

            fold = (
                fold_info[
                    "fold"
                ]
            )


            print(
                "\n----------------------------------------"
            )

            print(
                f"TRIAL {trial.number} | FOLD {fold}"
            )

            print(
                "----------------------------------------"
            )


            # Clear old Keras model.
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
            # Prepare data
            #
            # IMPORTANT:
            # val_idx is deliberately used as the third
            # temporary argument as well.
            #
            # Final test data is NOT used during HPO.
            # =================================================

            (
                train_inputs,
                val_inputs,
                _,
                extra_info,
            ) = module.prepare_fold_inputs(

                X_model,

                train_idx,

                val_idx,

                val_idx,

                config,

                Path("/tmp"),

                {},
            )


            # =================================================
            # Model
            # =================================================

            model = module.build_model(

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


            # =================================================
            # Early stopping
            # =================================================

            early_stop = EarlyStopping(

                monitor="val_loss",

                patience=args.patience,

                min_delta=1e-4,

                restore_best_weights=True,

                verbose=1,
            )


            # =================================================
            # Median pruning
            # =================================================

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

                epochs=config[
                    "epochs"
                ],

                batch_size=config[
                    "batch_size"
                ],

                callbacks=[
                    early_stop,
                    pruning_callback,
                ],

                verbose=2,
            )


            # =================================================
            # Validation predictions
            # =================================================

            probs = model.predict(

                val_inputs,

                batch_size=config[
                    "batch_size"
                ],

                verbose=0,
            )


            auc = roc_auc_score(

                y[
                    val_idx
                ],

                probs[
                    :,
                    1
                ],
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
                f"Mean={mean_auc:.6f}"
            )


            # =================================================
            # Pruning after full fold
            # =================================================

            fold_step = (
                10000
                + fold
            )


            trial.report(

                mean_auc,

                step=fold_step,
            )


            if trial.should_prune():

                raise optuna.TrialPruned(

                    f"Trial {trial.number} "
                    f"pruned after fold "
                    f"{fold}"
                )


            # Memory cleanup
            del model

            del train_inputs

            del val_inputs

            gc.collect()


        # ====================================================
        # Final trial score
        # ====================================================

        final_auc = float(
            np.mean(
                fold_aucs
            )
        )


        auc_std = float(
            np.std(
                fold_aucs
            )
        )


        trial.set_user_attr(
            "fold_aucs",
            fold_aucs,
        )


        trial.set_user_attr(
            "auc_std",
            auc_std,
        )


        print(
            f"\nTrial {trial.number} finished"
        )

        print(
            f"Mean AUC: {final_auc:.6f}"
        )

        print(
            f"Std AUC: {auc_std:.6f}"
        )


        return final_auc


    # ========================================================
    # Output
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
    # Persistent database
    # ========================================================

    storage = (

        f"sqlite:///"

        f"{output_dir / 'study.db'}"
    )


    # ========================================================
    # Optuna
    # ========================================================

    study = optuna.create_study(

        study_name=(
            f"pbpb_"
            f"{model_name}_"
            f"reduced_hpo"
        ),

        direction="maximize",

        storage=storage,

        load_if_exists=True,


        sampler=optuna.samplers.TPESampler(

            seed=args.seed,

            multivariate=True,
        ),


        pruner=optuna.pruners.MedianPruner(

            # First 5 trials establish baseline.
            n_startup_trials=5,

            # Do not prune immediately.
            n_warmup_steps=3,

            interval_steps=1,
        ),
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
    # Optimization
    # ========================================================

    study.optimize(

        objective,

        n_trials=args.n_trials,

        gc_after_trial=True,

        show_progress_bar=False,
    )


    # ========================================================
    # Statistics
    # ========================================================

    completed = [

        t

        for t in study.trials

        if (
            t.state
            ==
            optuna.trial.TrialState.COMPLETE
        )
    ]


    pruned = [

        t

        for t in study.trials

        if (
            t.state
            ==
            optuna.trial.TrialState.PRUNED
        )
    ]


    failed = [

        t

        for t in study.trials

        if (
            t.state
            ==
            optuna.trial.TrialState.FAIL
        )
    ]


    # ========================================================
    # Final result
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
            study.best_trial
            .user_attrs
            .get(
                "fold_aucs"
            ),

        "best_trial_auc_std":
            study.best_trial
            .user_attrs
            .get(
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
    # Save best params
    # ========================================================

    with open(

        output_dir
        / "best_params.json",

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            result,

            f,

            indent=2,
        )


    # ========================================================
    # Save all trials
    # ========================================================

    trials_df = (
        study
        .trials_dataframe()
    )


    trials_df.to_csv(

        output_dir
        / "all_trials.csv",

        index=False,
    )


    # ========================================================
    # Print final result
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "HPO FINISHED"
    )

    print(
        "========================================"
    )


    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":

    main()
