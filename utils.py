
# =============================================================================
# utils.py
# =============================================================================

import csv
from pathlib import Path
import random

import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split, StratifiedKFold


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ModuleNotFoundError:
        pass


def create_fixed_test_cv_splits(
    y,
    num_folds,
    final_test_ratio,
    seed,
):
    """
    Old jet-level split, kept for compatibility with the original qg_jets code.
    Do not use this for the final Marvin comparison.
    Use create_file_level_test_cv_splits instead.
    """

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


def create_file_level_test_cv_splits(
    y,
    file_ids,
    file_labels,
    num_folds,
    final_test_ratio,
    seed,
):
    """
    Marvin split function.

    Splits at file level, not jet level.
    All jets from the same .npz file stay together.
    """

    y = np.asarray(y)
    file_ids = np.asarray(file_ids)
    file_labels = np.asarray(file_labels)

    unique_files = np.arange(len(file_labels))

    if len(unique_files) != len(np.unique(file_ids)):
        raise ValueError(
            "file_labels length must match the number of unique file_ids. "
            f"Got len(file_labels)={len(file_labels)}, "
            f"unique file_ids={len(np.unique(file_ids))}."
        )

    labels, counts = np.unique(file_labels, return_counts=True)

    if len(labels) != 2:
        raise ValueError(f"Expected exactly two file labels, got labels={labels}.")

    if np.min(counts) < 2:
        raise ValueError(
            "Need at least 2 files per class for a stratified final test split. "
            f"File counts per class: {dict(zip(labels, counts))}"
        )

    dev_files, final_test_files = train_test_split(
        unique_files,
        test_size=final_test_ratio,
        random_state=seed,
        stratify=file_labels,
    )

    dev_file_labels = file_labels[dev_files]

    dev_labels, dev_counts = np.unique(dev_file_labels, return_counts=True)

    if np.min(dev_counts) < num_folds:
        raise ValueError(
            "Not enough development files per class for the requested number of folds. "
            f"Development file counts per class: {dict(zip(dev_labels, dev_counts))}, "
            f"num_folds={num_folds}. "
            "Increase --max-files-per-class or reduce --num-folds."
        )

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=seed,
    )

    final_test_idx = np.where(np.isin(file_ids, final_test_files))[0]

    folds = []

    for fold, (train_local_idx, val_local_idx) in enumerate(
        skf.split(np.zeros(len(dev_files)), dev_file_labels),
        start=1,
    ):
        train_files = dev_files[train_local_idx]
        val_files = dev_files[val_local_idx]

        train_idx = np.where(np.isin(file_ids, train_files))[0]
        val_idx = np.where(np.isin(file_ids, val_files))[0]
        test_idx = final_test_idx

        assert len(set(train_files) & set(val_files)) == 0
        assert len(set(train_files) & set(final_test_files)) == 0
        assert len(set(val_files) & set(final_test_files)) == 0

        folds.append(
            {
                "fold": fold,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "train_files": train_files,
                "val_files": val_files,
                "test_files": final_test_files,
            }
        )

    dev_idx = np.where(np.isin(file_ids, dev_files))[0]

    for fid in unique_files:
        jet_idx = np.where(file_ids == fid)[0]

        if len(jet_idx) == 0:
            continue

        jet_labels = np.unique(y[jet_idx])

        if len(jet_labels) != 1 or jet_labels[0] != file_labels[fid]:
            raise ValueError(
                f"Inconsistent labels for file_id={fid}: "
                f"jet_labels={jet_labels}, file_label={file_labels[fid]}"
            )

    return dev_idx, final_test_idx, folds


def delta_phi(phi, center):
    """
    Wrapped delta phi in [-pi, pi].
    """

    return (phi - center + np.pi) % (2.0 * np.pi) - np.pi


def weighted_phi_center(phi, weights, eps=1e-8):
    """
    pT-weighted circular mean for phi.
    This avoids problems at the -pi / pi boundary.
    """

    sin_mean = np.sum(weights * np.sin(phi))
    cos_mean = np.sum(weights * np.cos(phi))

    if np.abs(sin_mean) < eps and np.abs(cos_mean) < eps:
        return 0.0

    return float(np.arctan2(sin_mean, cos_mean))


def process_one_marvin_jet(
    pt,
    eta,
    phi,
    max_particles,
    sort_by_pt=True,
    eps=1e-8,
):
    """
    Convert one variable-length Marvin jet into a fixed-size token array.

    Output:
        out.shape = (max_particles, 3)

    Feature convention:
        out[:, 0] = z = pt_i / sum_j pt_j
        out[:, 1] = delta_eta = eta_i - eta_jet_center
        out[:, 2] = delta_phi = wrapped(phi_i - phi_jet_center)

    Padded rows are exactly zero.
    """

    pt = np.asarray(pt, dtype=np.float32)
    eta = np.asarray(eta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)

    out = np.zeros((max_particles, 3), dtype=np.float32)

    n = len(pt)

    if n == 0:
        return out

    valid = pt > 0

    if not np.any(valid):
        return out

    pt = pt[valid]
    eta = eta[valid]
    phi = phi[valid]

    if sort_by_pt:
        order = np.argsort(-pt)
        pt = pt[order]
        eta = eta[order]
        phi = phi[order]

    if len(pt) > max_particles:
        pt = pt[:max_particles]
        eta = eta[:max_particles]
        phi = phi[:max_particles]

    total_pt = float(np.sum(pt))

    if total_pt <= eps:
        return out

    z = pt / total_pt

    eta_center = float(np.sum(pt * eta) / total_pt)
    phi_center = weighted_phi_center(phi, pt, eps=eps)

    deta = eta - eta_center
    dphi = delta_phi(phi, phi_center)

    n_keep = len(pt)

    out[:n_keep, 0] = z.astype(np.float32)
    out[:n_keep, 1] = deta.astype(np.float32)
    out[:n_keep, 2] = dphi.astype(np.float32)

    return out


def load_marvin_parts_file(
    path,
    max_particles,
    label,
    file_id,
    sort_by_pt=True,
):
    """
    Load one Marvin parts .npz file.

    Expected arrays:
        pt, eta, phi, mass, offsets, w

    Only pt/eta/phi are used for this first pipeline.
    mass is ignored because the current models expect 3 features:
        z, delta_eta, delta_phi
    """

    path = Path(path)

    with np.load(path) as f:
        pt = f["pt"]
        eta = f["eta"]
        phi = f["phi"]
        offsets = f["offsets"]
        w = f["w"] if "w" in f.files else None

        n_jets = len(offsets) - 1

        X_file = np.zeros((n_jets, max_particles, 3), dtype=np.float32)
        counts_file = np.diff(offsets).astype(np.int32)

        for i in range(n_jets):
            start = int(offsets[i])
            end = int(offsets[i + 1])

            X_file[i] = process_one_marvin_jet(
                pt=pt[start:end],
                eta=eta[start:end],
                phi=phi[start:end],
                max_particles=max_particles,
                sort_by_pt=sort_by_pt,
            )

    y_file = np.full(n_jets, label, dtype=np.int64)
    file_ids_file = np.full(n_jets, file_id, dtype=np.int64)

    if w is not None:
        w_file = np.asarray(w, dtype=np.float32)
    else:
        w_file = None

    return X_file, y_file, file_ids_file, w_file, counts_file


def _balanced_keep_indices(y, max_jets, seed=42):
    """
    Select up to max_jets with approximately equal numbers per class.
    This avoids accidentally keeping only one class when debugging.
    """

    if max_jets is None or max_jets <= 0 or max_jets >= len(y):
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    if len(classes) != 2:
        raise ValueError(f"Expected two classes for balanced selection, got {classes}.")

    per_class = max_jets // 2
    remainder = max_jets - per_class * 2

    keep = []

    for class_pos, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        n_take = per_class + (1 if class_pos < remainder else 0)
        n_take = min(n_take, len(idx))

        keep.append(idx[:n_take])

    keep_idx = np.concatenate(keep)
    keep_idx.sort()

    return keep_idx


def load_marvin_parts_dataset(
    data_root,
    class_0="vac",
    class_1="rec",
    max_particles=128,
    max_jets=None,
    max_files_per_class=None,
    sort_by_pt=True,
    return_file_paths=False,
):
    """
    Load Marvin constituent-level dataset from:

        data_root/class_name/parts/*.npz

    Output:
        X.shape = (N_jets, max_particles, 3)

    Feature convention:
        X[..., 0] = z = pt_i / sum_j pt_j
        X[..., 1] = delta_eta
        X[..., 2] = delta_phi

    Labels:
        class_0 -> 0
        class_1 -> 1
    """

    data_root = Path(data_root)

    if class_0 == class_1:
        raise ValueError("class_0 and class_1 must be different.")

    all_X = []
    all_y = []
    all_file_ids = []
    all_counts = []

    file_labels = []
    file_paths = []

    classes = [
        (class_0, 0),
        (class_1, 1),
    ]

    file_id = 0

    for class_name, label in classes:
        parts_dir = data_root / class_name / "parts"

        if not parts_dir.exists():
            raise FileNotFoundError(f"Parts directory not found: {parts_dir}")

        files = sorted(parts_dir.glob("*.npz"))

        if len(files) == 0:
            raise FileNotFoundError(f"No .npz files found in: {parts_dir}")

        if max_files_per_class is not None:
            files = files[:max_files_per_class]

        print(f"Class {class_name!r} -> label {label}: {len(files)} files")

        for local_i, path in enumerate(files, start=1):
            print(f"[{class_name} {local_i}/{len(files)}] Loading {path.name}")

            X_file, y_file, ids_file, w_file, counts_file = load_marvin_parts_file(
                path=path,
                max_particles=max_particles,
                label=label,
                file_id=file_id,
                sort_by_pt=sort_by_pt,
            )

            all_X.append(X_file)
            all_y.append(y_file)
            all_file_ids.append(ids_file)
            all_counts.append(counts_file)

            file_labels.append(label)
            file_paths.append(str(path))
            file_id += 1

            print(
                f"    jets={len(y_file)}, "
                f"constituents mean={counts_file.mean():.2f}, "
                f"median={np.median(counts_file):.1f}, "
                f"max={counts_file.max()}"
            )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    file_ids = np.concatenate(all_file_ids, axis=0)
    counts = np.concatenate(all_counts, axis=0)
    file_labels = np.asarray(file_labels, dtype=np.int64)

    if max_jets is not None and max_jets > 0 and max_jets < len(y):
        keep_idx = _balanced_keep_indices(y, max_jets=max_jets, seed=42)

        X = X[keep_idx]
        y = y[keep_idx]
        file_ids = file_ids[keep_idx]
        counts = counts[keep_idx]

        old_unique = np.unique(file_ids)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(old_unique)}

        new_file_ids = np.asarray(
            [old_to_new[fid] for fid in file_ids],
            dtype=np.int64,
        )

        new_file_labels = np.asarray(
            [file_labels[old_id] for old_id in old_unique],
            dtype=np.int64,
        )

        new_file_paths = [file_paths[old_id] for old_id in old_unique]

        file_ids = new_file_ids
        file_labels = new_file_labels
        file_paths = new_file_paths

    print("=" * 80)
    print("Marvin parts summary")
    print("=" * 80)
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")
    print(f"num files     : {len(file_labels)}")
    print(f"class counts  : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"counts min    : {counts.min()}")
    print(f"counts mean   : {counts.mean():.2f}")
    print(f"counts median : {np.median(counts):.1f}")
    print(f"counts max    : {counts.max()}")
    print("=" * 80)

    if return_file_paths:
        return X, y, file_ids, file_labels, file_paths

    return X, y, file_ids, file_labels


def preprocess_jets(
    X,
    max_particles=60,
    sort_by_pt=True,
    eps=1e-8,
):
    """
    Old qg_jets preprocessing, kept for compatibility.

    Input:
        X.shape = (num_jets, num_particles, num_features)

    Raw features:
        X[..., 0] = pT
        X[..., 1] = rapidity/y
        X[..., 2] = phi
        X[..., 3] = PID, ignored here

    Output:
        X_proc.shape = (num_jets, max_particles, 3)
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


def compute_binary_classification_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).reshape(-1)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    except ValueError:
        metrics.update({"tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan})

    return metrics


def save_dict_csv(data, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])

        for key, value in data.items():
            writer.writerow([key, value])


def save_list_of_dicts_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    if len(rows) == 0:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_dict_csv(row, path):
    path = Path(path)
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
