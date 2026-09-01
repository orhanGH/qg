from pathlib import Path
import argparse
import gc
import json

import numpy as np
import optuna
import torch

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

from utils import (
    get_or_create_file_level_test_cv_splits,
    load_marvin_parts_dataset,
    set_seed,
)

from models.hf import mamba
from runners.hf_runner import JetDataset, compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    "/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"
)

HPO_ROOT = Path(
    "/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/"
    "s6oraydi_hpc_runs/hpo_reduced/mamba"
)


# ============================================================
# Optuna pruning callback
# ============================================================

class OptunaPruningCallback(TrainerCallback):

    def __init__(self, trial, fold):
        self.trial = trial
        self.fold = fold

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs,
    ):

        metrics = metrics or {}

        auc = metrics.get(
            "eval_roc_auc"
        )

        if auc is None:
            return control

        epoch = int(
            state.epoch or 0
        )

        # Unique step for every fold / epoch.
        step = (
            (self.fold - 1) * 1000
            + epoch
        )

        self.trial.report(
            float(auc),
            step=step,
        )

        if self.trial.should_prune():

            raise optuna.TrialPruned(
                f"Trial {self.trial.number} "
                f"pruned at fold {self.fold}, "
                f"epoch {epoch}, "
                f"AUC={auc:.6f}"
            )

        return control


# ============================================================
# Arguments
# ============================================================

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


# ============================================================
# Reduced search space
# ============================================================

def suggest_config(
    trial,
    epochs,
    patience,
):

    config = (
        mamba
        .get_default_config()
    )

    config.update(
        {
            "model_name":
                "mamba",

            "max_particles":
                128,

            # -------------------------
            # Architecture
            # -------------------------

            "hidden_dim":
                trial.suggest_categorical(
                    "hidden_dim",
                    [
                        64,
                        128,
                    ],
                ),

            "num_layers":
                trial.suggest_categorical(
                    "num_layers",
                    [
                        2,
                        4,
                    ],
                ),

            "state_size":
                trial.suggest_categorical(
                    "state_size",
                    [
                        16,
                        32,
                    ],
                ),

            # Keep these fixed
            # to reduce runtime/search space.
            "conv_kernel":
                4,

            "expand":
                2,

            # -------------------------
            # Regularization
            # -------------------------

            "dropout":
                trial.suggest_categorical(
                    "dropout",
                    [
                        0.0,
                        0.1,
                        0.2,
                    ],
                ),

            # -------------------------
            # Activation
            # -------------------------

            "activation":
                trial.suggest_categorical(
                    "activation",
                    [
                        "gelu",
                        "silu",
                    ],
                ),

            # -------------------------
            # Optimization
            # -------------------------

            "learning_rate":
                trial.suggest_categorical(
                    "learning_rate",
                    [
                        3e-5,
                        1e-4,
                        3e-4,
                    ],
                ),

            "weight_decay":
                trial.suggest_categorical(
                    "weight_decay",
                    [
                        0.0,
                        1e-5,
                        1e-4,
                    ],
                ),

            # Fixed for fair comparison.
            "batch_size":
                512,

            "epochs":
                epochs,

            "use_early_stopping":
                True,

            "patience":
                patience,

            "early_stopping_threshold":
                1e-4,
        }
    )

    return config


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(
        args.seed
    )

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
            str(DATA_ROOT),

        "max_files_per_class":
            None,
    }


    # ========================================================
    # Load data
    # ========================================================

    (
        X,
        y,
        file_ids,
        file_labels,
        file_paths,
    ) = load_marvin_parts_dataset(

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


    y = y.astype(
        np.int64
    )


    # ========================================================
    # Same CV split as other models
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


    HPO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Objective
    # ========================================================

    def objective(trial):

        config = suggest_config(
            trial,
            args.epochs,
            args.patience,
        )

        fold_aucs = []


        for fold_info in folds:

            fold = (
                fold_info[
                    "fold"
                ]
            )


            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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


            train_dataset = (
                JetDataset(
                    X[train_idx],
                    y[train_idx],
                )
            )


            val_dataset = (
                JetDataset(
                    X[val_idx],
                    y[val_idx],
                )
            )


            model = (
                mamba
                .build_model(
                    config
                )
            )


            trial_dir = (
                HPO_ROOT
                / f"trial_{trial.number}"
                / f"fold_{fold}"
            )


            training_args = TrainingArguments(

                output_dir=
                    str(
                        trial_dir
                    ),

                num_train_epochs=
                    config[
                        "epochs"
                    ],

                per_device_train_batch_size=
                    config[
                        "batch_size"
                    ],

                per_device_eval_batch_size=
                    config[
                        "batch_size"
                    ],

                learning_rate=
                    config[
                        "learning_rate"
                    ],

                weight_decay=
                    config[
                        "weight_decay"
                    ],

                lr_scheduler_type=
                    "cosine",

                warmup_ratio=
                    0.05,

                eval_strategy=
                    "epoch",

                save_strategy=
                    "epoch",

                logging_strategy=
                    "epoch",

                load_best_model_at_end=
                    True,

                metric_for_best_model=
                    "roc_auc",

                greater_is_better=
                    True,

                save_total_limit=
                    1,

                report_to=
                    "none",

                seed=
                    args.seed
                    + fold,

                data_seed=
                    args.seed
                    + fold,

                remove_unused_columns=
                    False,
            )


            trainer = Trainer(

                model=
                    model,

                args=
                    training_args,

                train_dataset=
                    train_dataset,

                eval_dataset=
                    val_dataset,

                compute_metrics=
                    compute_metrics,

                callbacks=[

                    EarlyStoppingCallback(

                        early_stopping_patience=
                            args.patience,

                        early_stopping_threshold=
                            1e-4,
                    ),

                    OptunaPruningCallback(
                        trial,
                        fold,
                    ),
                ],
            )


            # =================================================
            # Train
            # =================================================

            trainer.train()


            # =================================================
            # Validation
            # =================================================

            metrics = trainer.evaluate(
                eval_dataset=
                    val_dataset
            )


            auc = float(
                metrics[
                    "eval_roc_auc"
                ]
            )


            fold_aucs.append(
                auc
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
                step=
                    fold_step,
            )


            if trial.should_prune():

                raise optuna.TrialPruned(
                    f"Trial {trial.number} "
                    f"pruned after fold "
                    f"{fold}"
                )


            # =================================================
            # Cleanup
            # =================================================

            del trainer
            del model
            del train_dataset
            del val_dataset

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        # ====================================================
        # Trial result
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


        return final_auc


    # ========================================================
    # Study storage
    # ========================================================

    storage = (
        f"sqlite:///"
        f"{HPO_ROOT / 'study.db'}"
    )


    study = optuna.create_study(

        study_name=
            "pbpb_mamba_reduced_hpo",

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

                n_startup_trials=
                    5,

                n_warmup_steps=
                    3,

                interval_steps=
                    1,
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
    # Result
    # ========================================================

    result = {

        "model":
            "mamba",

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

        "num_trials":
            len(
                study.trials
            ),
    }


    # ========================================================
    # JSON
    # ========================================================

    with open(

        HPO_ROOT
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
    # CSV
    # ========================================================

    study.trials_dataframe().to_csv(

        HPO_ROOT
        / "all_trials.csv",

        index=
            False,
    )


    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
