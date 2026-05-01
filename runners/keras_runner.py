import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from tf_keras.callbacks import EarlyStopping

from utils import (
    set_seed,
    save_dict_csv,
    save_list_of_dicts_csv,
    append_dict_csv,
)


def one_hot_labels(y: np.ndarray, num_classes: int = 2) -> np.ndarray:
    y = y.astype(int)
    return np.eye(num_classes, dtype=np.float32)[y]


def normalize_binary_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)

    if probs.ndim == 1:
        prob_1 = probs
        prob_0 = 1.0 - prob_1
        return np.stack([prob_0, prob_1], axis=1)

    if probs.ndim == 2 and probs.shape[1] == 1:
        prob_1 = probs[:, 0]
        prob_0 = 1.0 - prob_1
        return np.stack([prob_0, prob_1], axis=1)

    return probs


def categorical_crossentropy_from_probs(y_true, probs, eps: float = 1e-8) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))


def compute_metrics_from_probs(prefix: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    probs = normalize_binary_probs(probs)

    y_true = y_true.astype(int)
    y_pred = np.argmax(probs, axis=1)

    return {
        f"{prefix}_loss": categorical_crossentropy_from_probs(y_true, probs),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, probs[:, 1]),
    }


def count_parameters(model) -> int:
    if hasattr(model, "count_params"):
        return int(model.count_params())

    if hasattr(model, "model") and hasattr(model.model, "count_params"):
        return int(model.model.count_params())

    return -1


def save_model_safely(model, path: Path) -> None:
    try:
        model.save(str(path))
        return
    except Exception:
        pass

    try:
        if hasattr(model, "model"):
            model.model.save(str(path))
            return
    except Exception:
        pass

    print(f"[WARN] Could not save model to {path}")


def history_to_rows(history, train_runtime: float, num_train: int, batch_size: int):
    rows = []
    history_dict = history.history

    losses = history_dict.get("loss", [])
    num_epochs_ran = len(losses)

    for epoch_idx in range(num_epochs_ran):
        row = {
            "epoch": epoch_idx + 1,
        }

        if "loss" in history_dict:
            row["loss"] = history_dict["loss"][epoch_idx]

        if "val_loss" in history_dict:
            row["eval_loss"] = history_dict["val_loss"][epoch_idx]

        if "accuracy" in history_dict:
            row["accuracy"] = history_dict["accuracy"][epoch_idx]

        if "val_accuracy" in history_dict:
            row["eval_accuracy"] = history_dict["val_accuracy"][epoch_idx]

        rows.append(row)

    if num_epochs_ran > 0:
        num_steps = np.ceil(num_train / batch_size) * num_epochs_ran

        rows.append(
            {
                "train_runtime": float(train_runtime),
                "train_samples_per_second": float(num_train * num_epochs_ran / train_runtime),
                "train_steps_per_second": float(num_steps / train_runtime),
                "train_loss": float(losses[-1]) if losses else None,
                "epoch": float(num_epochs_ran),
            }
        )

    return rows


def run_one_keras_fold(
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict,
    output_dir: Path,
    build_model_fn,
    prepare_fold_inputs_fn,
    context: dict,
) -> dict:
    print("=" * 80)
    print(f"{config['model_name'].upper()} Fold {fold}/{config['num_folds']}")
    print("=" * 80)

    fold_seed = config["seed"] + fold
    set_seed(fold_seed)

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_inputs, val_inputs, test_inputs, extra_info = prepare_fold_inputs_fn(
        X,
        train_idx,
        val_idx,
        test_idx,
        config,
        fold_dir,
        context,
    )

    y_onehot = one_hot_labels(y, num_classes=2)

    y_train = y_onehot[train_idx]
    y_val = y_onehot[val_idx]
    y_test = y_onehot[test_idx]

    model = build_model_fn(config, extra_info)

    num_parameters = count_parameters(model)

    print(f"Train samples      : {len(train_idx)}")
    print(f"Validation samples : {len(val_idx)}")
    print(f"Fixed test samples : {len(test_idx)}")
    print(f"Parameters         : {num_parameters:,}")

    callbacks = []

    if config.get("use_early_stopping", True):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config["patience"],
                restore_best_weights=True,
                verbose=1,
            )
        )

    start_time = time.perf_counter()

    history = model.fit(
        train_inputs,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(val_inputs, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    train_runtime = time.perf_counter() - start_time

    history_rows = history_to_rows(
        history=history,
        train_runtime=train_runtime,
        num_train=len(train_idx),
        batch_size=config["batch_size"],
    )

    save_list_of_dicts_csv(
        history_rows,
        fold_dir / "trainer_log_history.csv",
    )

    # Validation metrics after training.
    val_probs = model.predict(
        val_inputs,
        batch_size=config["batch_size"],
        verbose=0,
    )
    val_probs = normalize_binary_probs(val_probs)

    y_val_true = y[val_idx].astype(int)
    val_metrics = compute_metrics_from_probs(
        prefix="eval",
        y_true=y_val_true,
        probs=val_probs,
    )

    # Fixed holdout test metrics.
    test_start_time = time.perf_counter()

    test_probs = model.predict(
        test_inputs,
        batch_size=config["batch_size"],
        verbose=0,
    )

    test_runtime = time.perf_counter() - test_start_time

    test_probs = normalize_binary_probs(test_probs)

    y_test_true = y[test_idx].astype(int)
    y_test_pred = np.argmax(test_probs, axis=1)

    test_metrics = compute_metrics_from_probs(
        prefix="test",
        y_true=y_test_true,
        probs=test_probs,
    )

    test_metrics["test_runtime"] = float(test_runtime)
    test_metrics["test_samples_per_second"] = float(len(test_idx) / test_runtime)

    cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prediction_rows = []

    for i in range(len(y_test_true)):
        prediction_rows.append(
            {
                "index": i,
                "true_label": int(y_test_true[i]),
                "pred_label": int(y_test_pred[i]),
                "prob_class_0": float(test_probs[i, 0]),
                "prob_class_1": float(test_probs[i, 1]),
            }
        )

    save_list_of_dicts_csv(
        prediction_rows,
        fold_dir / "test_predictions.csv",
    )

    save_list_of_dicts_csv(
        [extra_info],
        fold_dir / "fold_extra_info.csv",
    )

    save_model_safely(
        model,
        fold_dir / "best_model.keras",
    )

    row = {
        "fold": fold,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "num_parameters": num_parameters,
        "num_trainable_parameters": num_parameters,
        "train_runtime": float(train_runtime),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    row.update(val_metrics)
    row.update(test_metrics)

    save_list_of_dicts_csv(
        [row],
        fold_dir / "test_metrics.csv",
    )

    print("Fold results:")
    for key, value in row.items():
        print(f"{key}: {value}")

    return row


def build_cv_summary(fold_rows: list[dict]) -> list[dict]:
    summary_rows = []

    metric_names = [
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_roc_auc",
        "eval_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_loss",
        "test_runtime",
        "test_samples_per_second",
        "train_runtime",
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
    row = {
        "run_id": run_id,
        "run_path": str(output_dir),
        "model_name": config["model_name"],

        "num_data": config["num_data"],
        "max_particles": config["max_particles"],

        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "patience": config["patience"],
        "learning_rate": config["learning_rate"],

        "num_folds": config["num_folds"],
        "final_test_ratio": config["final_test_ratio"],
        "seed": config["seed"],
    }

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


def run_keras_experiment(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[dict],
    shared_config: dict,
    model_config: dict,
    build_model_fn,
    prepare_fold_inputs_fn,
    get_model_summary_fields_fn,
):
    config = {}
    config.update(model_config)
    config.update(shared_config)

    set_seed(config["seed"])

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
    context = {}

    for fold_info in folds:
        fold_row = run_one_keras_fold(
            fold=fold_info["fold"],
            X=X,
            y=y,
            train_idx=fold_info["train_idx"],
            val_idx=fold_info["val_idx"],
            test_idx=fold_info["test_idx"],
            config=config,
            output_dir=output_dir,
            build_model_fn=build_model_fn,
            prepare_fold_inputs_fn=prepare_fold_inputs_fn,
            context=context,
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