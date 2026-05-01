import csv
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import random
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_fixed_test_cv_splits(
    y: np.ndarray,
    num_folds: int,
    final_test_ratio: float,
    seed: int,
):
    all_idx = np.arange(len(y))

    dev_idx, final_test_idx = train_test_split(
        all_idx,
        test_size=final_test_ratio,
        random_state=seed,
        stratify=y,
    )

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )

    y_dev = y[dev_idx]

    folds = []

    for fold, (train_local_idx, val_local_idx) in enumerate(
        skf.split(np.zeros(len(dev_idx)), y_dev),
        start=1,
    ):
        train_idx = dev_idx[train_local_idx]
        val_idx = dev_idx[val_local_idx]
        test_idx = final_test_idx

        folds.append(
            {
                "fold": fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
        )

    return dev_idx, final_test_idx, folds


def preprocess_jets(
    X: np.ndarray,
    max_particles: int = 60,
    sort_by_pt: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Convert raw qg_jets into particle tokens.

    Input:
        X.shape = (num_jets, num_particles, num_features)

    Raw features:
        X[..., 0] = pT
        X[..., 1] = rapidity/y
        X[..., 2] = phi
        X[..., 3] = PID, ignored here

    Output:
        X_proc.shape = (num_jets, max_particles, 3)

    Processed features:
        X_proc[..., 0] = z = pT_i / total_pT
        X_proc[..., 1] = centered_y
        X_proc[..., 2] = centered_phi
    """

    X = X[:, :, :3].astype(np.float32).copy()

    if sort_by_pt:
        sort_idx = np.argsort(-X[:, :, 0], axis=1)
        X = np.take_along_axis(X, sort_idx[:, :, None], axis=1)

    num_jets, num_particles, feature_dim = X.shape

    if num_particles > max_particles:
        X = X[:, :max_particles, :]
    elif num_particles < max_particles:
        padded = np.zeros(
            (num_jets, max_particles, feature_dim),
            dtype=np.float32,
        )
        padded[:, :num_particles, :] = X
        X = padded

    for jet in X:
        mask = jet[:, 0] > 0

        if not np.any(mask):
            continue

        total_pt = jet[:, 0].sum()

        if total_pt <= eps:
            continue

        yphi_center = np.average(
            jet[mask, 1:3],
            weights=jet[mask, 0],
            axis=0,
        )

        jet[mask, 1:3] -= yphi_center
        jet[mask, 0] /= total_pt
        jet[~mask, :] = 0.0

    return X.astype(np.float32)


def save_dict_csv(data: dict, path: Path):
    """
    Save one dictionary as a two-column CSV:
        key,value
    """

    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])

        for key, value in data.items():
            writer.writerow([key, value])


def save_list_of_dicts_csv(rows: list[dict], path: Path):
    """
    Save a list of dictionaries as a CSV table.
    Overwrites the file.
    """

    path.parent.mkdir(exist_ok=True, parents=True)

    if len(rows) == 0:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_dict_csv(row: dict, path: Path):
    """
    Append one dictionary row to a CSV file.

    If the CSV already exists and the new row has new columns,
    the whole file is rewritten with the union of all columns.
    """

    path.parent.mkdir(exist_ok=True, parents=True)

    existing_rows = []

    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    all_rows = existing_rows + [row]

    fieldnames = []

    for current_row in all_rows:
        for key in current_row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
