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
)

from utils import (
    get_or_create_file_level_test_cv_splits,
    load_marvin_parts_dataset,
    set_seed,
)

from models.hf import transformer_encoder
from runners.hf_runner import JetDataset, compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    "/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"
)

HPO_ROOT = Path(
    "/lustre/scratch/data/s6oraydi_hpc-pbpb_pp/"
    "s6oraydi_hpc_runs/hpo/transformer_encoder"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def suggest_config(trial, epochs):
    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [64, 128, 256],
    )

    config = transformer_encoder.get_default_config()

    config.update(
        {
            "model_name": "transformer_encoder",
            "max_particles": 128,

            "hidden_dim": hidden_dim,

            "num_layers": trial.suggest_categorical(
                "num_layers",
                [1, 2, 3, 4],
            ),

            "num_heads": trial.suggest_categorical(
                "num_heads",
                [2, 4, 8],
            ),

            "dropout": trial.suggest_categorical(
                "dropout",
                [0.0, 0.1, 0.2],
            ),

            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [1e-5, 3e-5, 1e-4, 3e-4],
            ),

            "weight_decay": trial.suggest_categorical(
                "weight_decay",
                [0.0, 1e-5, 1e-4],
            ),

            "activation": "silu",

            "batch_size": 512,
            "epochs": epochs,

            "use_early_stopping": True,
            "patience": 30,
            "early_stopping_threshold": 1e-4,
        }
    )

    return config


def main():
    args = parse_args()

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

    (
        X,
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

    y = y.astype(np.int64)

    _, _, folds = get_or_create_file_level_test_cv_splits(
        project_root=PROJECT_ROOT,
        y=y,
        file_ids=file_ids,
        file_labels=file_labels,
        shared_config=shared_config,
        file_paths=file_paths,
    )

    folds = folds[:args.folds]

    HPO_ROOT.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        config = suggest_config(
            trial,
            args.epochs,
        )

        fold_aucs = []

        for fold_info in folds:
            fold = fold_info["fold"]

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            set_seed(42 + fold)

            train_idx = fold_info["train_idx"]
            val_idx = fold_info["val_idx"]

            train_dataset = JetDataset(
                X[train_idx],
                y[train_idx],
            )

            val_dataset = JetDataset(
                X[val_idx],
                y[val_idx],
            )

            model = transformer_encoder.build_model(
                config
            )

            trial_dir = (
                HPO_ROOT
                / f"trial_{trial.number}"
                / f"fold_{fold}"
            )

            training_args = TrainingArguments(
                output_dir=str(trial_dir),
                num_train_epochs=config["epochs"],

                per_device_train_batch_size=config[
                    "batch_size"
                ],
                per_device_eval_batch_size=config[
                    "batch_size"
                ],

                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],

                lr_scheduler_type="cosine",
                warmup_ratio=0.05,

                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",

                load_best_model_at_end=True,
                metric_for_best_model="roc_auc",
                greater_is_better=True,

                save_total_limit=1,
                report_to="none",

                seed=42 + fold,
                data_seed=42 + fold,

                remove_unused_columns=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=30,
                        early_stopping_threshold=1e-4,
                    )
                ],
            )

            trainer.train()

            metrics = trainer.evaluate(
                eval_dataset=val_dataset,
            )

            auc = float(metrics["eval_roc_auc"])

            fold_aucs.append(auc)

            trial.report(
                float(np.mean(fold_aucs)),
                step=fold,
            )

            print(
                f"Trial {trial.number} | "
                f"fold {fold} | "
                f"val AUC={auc:.6f}"
            )

            del trainer
            del model
            del train_dataset
            del val_dataset

        return float(np.mean(fold_aucs))

    storage = (
        f"sqlite:///{HPO_ROOT / 'study.db'}"
    )

    study = optuna.create_study(
        study_name="pbpb_transformer_hpo",
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
        "model": "transformer_encoder",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "num_trials": len(study.trials),
    }

    with open(
        HPO_ROOT / "best_params.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()