import inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from transformers import Trainer, TrainingArguments

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils import (
    set_seed,
    save_dict_csv,
    save_list_of_dicts_csv,
    append_dict_csv,
)


class JetDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "particles": self.X[idx],
            "labels": self.y[idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
    }


def make_training_args(fold_dir: Path, config: dict) -> TrainingArguments:
    kwargs = {
        "output_dir": str(fold_dir / "checkpoints"),
        "num_train_epochs": config["epochs"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "roc_auc",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": config["seed"],
        "data_seed": config["seed"],
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    allowed_args = set(signature.parameters.keys())

    if "eval_strategy" in allowed_args:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in allowed_args:
        kwargs["evaluation_strategy"] = "epoch"

    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in allowed_args
    }

    return TrainingArguments(**kwargs)


def run_one_hf_fold(
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict,
    output_dir: Path,
    build_model_fn,
) -> dict:
    print("=" * 80)
    print(f"{config['model_name'].upper()} Fold {fold}/{config['num_folds']}")
    print("=" * 80)

    fold_seed = config["seed"] + fold
    set_seed(fold_seed)

    train_dataset = JetDataset(X[train_idx], y[train_idx])
    val_dataset = JetDataset(X[val_idx], y[val_idx])
    test_dataset = JetDataset(X[test_idx], y[test_idx])

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_fn(config)

    num_parameters = sum(p.numel() for p in model.parameters())
    num_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Train samples        : {len(train_dataset)}")
    print(f"Validation samples   : {len(val_dataset)}")
    print(f"Fixed test samples   : {len(test_dataset)}")
    print(f"Parameters           : {num_parameters:,}")
    print(f"Trainable parameters : {num_trainable_parameters:,}")

    training_args = make_training_args(
        fold_dir=fold_dir,
        config={**config, "seed": fold_seed},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("CUDA available:", torch.cuda.is_available())
    print("Trainer device:", trainer.args.device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    trainer.train()

    save_list_of_dicts_csv(
        trainer.state.log_history,
        fold_dir / "trainer_log_history.csv",
    )

    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )

    prediction_output = trainer.predict(test_dataset)

    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = prediction_output.label_ids
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prediction_rows = []
    for i in range(len(y_true)):
        prediction_rows.append(
            {
                "index": i,
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred[i]),
                "prob_class_0": float(probs[i, 0]),
                "prob_class_1": float(probs[i, 1]),
            }
        )

    save_list_of_dicts_csv(
        prediction_rows,
        fold_dir / "test_predictions.csv",
    )

    trainer.save_model(str(fold_dir / "best_model"))

    row = {
        "fold": fold,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "num_parameters": num_parameters,
        "num_trainable_parameters": num_trainable_parameters,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    row.update(test_metrics)

    save_list_of_dicts_csv(
        [row],
        fold_dir / "test_metrics.csv",
    )

    print("Fold test results:")
    for key, value in row.items():
        print(f"{key}: {value}")

    return row

def build_cv_summary(fold_rows: list[dict]) -> list[dict]:
    """
    Build mean/std summary over folds.

    This is Option A:
        each fold model is evaluated separately on the fixed test set,
        then we report mean ± std over folds.
    """

    summary_rows = []

    metric_names = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_loss",
        "test_runtime",
        "test_samples_per_second",
        "test_steps_per_second",
    ]

    for metric_name in metric_names:
        values = [
            row[metric_name]
            for row in fold_rows
            if metric_name in row and row[metric_name] is not None
        ]

        if len(values) == 0:
            continue

        values = np.asarray(values, dtype=float)

        summary_rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )

    return summary_rows


def build_run_summary_row(
    run_id: str,
    config: dict,
    summary_rows: list[dict],
    output_dir: Path,
    fold_rows: list[dict],
    model_summary_fields: dict,
) -> dict:
    """
    Build one flat row for runs_summary.csv.
    """

    row = {
        "run_id": run_id,
        "run_path": str(output_dir),
        "model_name": config["model_name"],

        # Dataset / preprocessing
        "num_data": config["num_data"],
        "max_particles": config["max_particles"],

        # Training
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],

        # Fixed holdout + CV
        "num_folds": config["num_folds"],
        "final_test_ratio": config["final_test_ratio"],
        "seed": config["seed"],
    }

    # Add model-specific fields:
    # BERT/RoBERTa: hidden_dim, num_layers, num_heads, dropout
    # Mamba: hidden_dim, num_layers, state_size, conv_kernel, expand, dropout
    row.update(model_summary_fields)

    if fold_rows:
        row["num_parameters"] = fold_rows[0].get("num_parameters")
        row["num_trainable_parameters"] = fold_rows[0].get("num_trainable_parameters")
    else:
        row["num_parameters"] = None
        row["num_trainable_parameters"] = None

    for metric_row in summary_rows:
        metric = metric_row["metric"]
        row[f"{metric}_mean"] = metric_row["mean"]
        row[f"{metric}_std"] = metric_row["std"]

    return row


def run_hf_experiment(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[dict],
    shared_config: dict,
    model_config: dict,
    build_model_fn,
    get_model_summary_fields_fn,
):
    """
    Shared Hugging Face experiment runner.

    Important:
        This does NOT create new splits.
        It receives the already-created folds from main.py.

    Therefore BERT/RoBERTa/Mamba all use:
        same X
        same y
        same train_idx per fold
        same val_idx per fold
        same fixed test_idx per fold
    """

    config = {}
    config.update(model_config)
    config.update(shared_config)
    set_seed(config["seed"])

    # qg/runners/hf_runner.py -> parents[1] is qg/
    project_root = Path(__file__).resolve().parents[1]

    base_output_dir = project_root / config["results_dir_name"]
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

    output_dir = base_output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config["run_id"] = run_id

    save_dict_csv(
        config,
        output_dir / "config.csv",
    )

    fold_rows = []

    for fold_info in folds:
        fold_row = run_one_hf_fold(
            fold=fold_info["fold"],
            X=X,
            y=y,
            train_idx=fold_info["train_idx"],
            val_idx=fold_info["val_idx"],
            test_idx=fold_info["test_idx"],
            config=config,
            output_dir=output_dir,
            build_model_fn=build_model_fn,
        )

        fold_rows.append(fold_row)

    save_list_of_dicts_csv(
        fold_rows,
        output_dir / "fold_results.csv",
    )

    summary_rows = build_cv_summary(fold_rows)

    print("\nCross-validation summary:")
    for row in summary_rows:
        print(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}")

    save_list_of_dicts_csv(
        summary_rows,
        output_dir / "cv_summary.csv",
    )

    run_summary_row = build_run_summary_row(
        run_id=run_id,
        config=config,
        summary_rows=summary_rows,
        output_dir=output_dir,
        fold_rows=fold_rows,
        model_summary_fields=get_model_summary_fields_fn(config),
    )

    save_list_of_dicts_csv(
        [run_summary_row],
        base_output_dir / "latest_run_summary.csv",
    )

    append_dict_csv(
        run_summary_row,
        base_output_dir / "runs_summary.csv",
    )

    print(f"\nSaved this run to: {output_dir}")
    print(f"Saved latest summary to: {base_output_dir / 'latest_run_summary.csv'}")
    print(f"Updated all-runs summary: {base_output_dir / 'runs_summary.csv'}")