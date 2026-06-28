# =============================================================================
# utils.py
# =============================================================================

import csv
import json
from pathlib import Path
import random

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold


# =============================================================================
# Reproducibility
# =============================================================================

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


# =============================================================================
# CSV helpers used by runners
# =============================================================================

def _to_serializable_value(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)

    return value


def save_dict_csv(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {k: _to_serializable_value(v) for k, v in row.items()}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_list_of_dicts_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)

    if len(rows) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    clean_rows = []

    for row in rows:
        clean_rows.append({k: _to_serializable_value(row.get(k, "")) for k in fieldnames})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def append_dict_csv(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {k: _to_serializable_value(v) for k, v in row.items()}

    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            old_rows = list(reader)

        fieldnames = list(old_fieldnames)

        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

        old_rows.append(row)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for old_row in old_rows:
                writer.writerow({k: old_row.get(k, "") for k in fieldnames})

    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


# =============================================================================
# Split utilities
# =============================================================================

def create_fixed_test_cv_splits(
    y,
    num_folds,
    final_test_ratio,
    seed,
):
    """
    Old jet-level split, kept for compatibility.
    For Marvin comparison, prefer create_file_level_test_cv_splits.
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

    unique_file_ids = np.unique(file_ids)

    if len(file_labels) != len(unique_file_ids):
        raise ValueError(
            "file_labels length must match the number of unique file_ids. "
            f"Got len(file_labels)={len(file_labels)}, "
            f"unique file_ids={len(unique_file_ids)}."
        )

    unique_files = np.arange(len(file_labels))

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
            raise ValueError(f"No jets found for file_id={fid}. This should not happen.")

        jet_labels = np.unique(y[jet_idx])

        if len(jet_labels) != 1 or jet_labels[0] != file_labels[fid]:
            raise ValueError(
                f"Inconsistent labels for file_id={fid}: "
                f"jet_labels={jet_labels}, file_label={file_labels[fid]}"
            )

    return dev_idx, final_test_idx, folds


def get_split_path(project_root, shared_config):
    project_root = Path(project_root)

    num_data_name = "all" if shared_config["num_data"] <= 0 else str(shared_config["num_data"])

    max_files_per_class = shared_config.get("max_files_per_class", None)
    use_all_files = max_files_per_class is None or int(max_files_per_class) <= 0
    max_files_name = "allfiles" if use_all_files else f"maxfiles_{max_files_per_class}"

    split_dir = (
        project_root
        / "splits"
        / (
            f"marvin_parts_obsvs_{shared_config['class_0']}_vs_{shared_config['class_1']}"
            f"_numdata_{num_data_name}"
            f"_maxp_{shared_config['max_particles']}"
            f"_{max_files_name}"
        )
    )

    split_file = (
        f"seed_{shared_config['seed']}"
        f"_folds_{shared_config['num_folds']}"
        f"_test_{shared_config['final_test_ratio']}.npz"
    )

    return split_dir / split_file


def save_split_indices(
    project_root,
    dev_idx,
    final_test_idx,
    folds,
    shared_config,
    file_paths=None,
    file_labels=None,
):
    split_path = get_split_path(
        project_root=project_root,
        shared_config=shared_config,
    )

    split_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "dev_idx": np.asarray(dev_idx),
        "final_test_idx": np.asarray(final_test_idx),
    }

    for fold_info in folds:
        fold = fold_info["fold"]

        save_dict[f"fold_{fold}_train_idx"] = np.asarray(fold_info["train_idx"])
        save_dict[f"fold_{fold}_val_idx"] = np.asarray(fold_info["val_idx"])
        save_dict[f"fold_{fold}_test_idx"] = np.asarray(fold_info["test_idx"])

        if "train_files" in fold_info:
            save_dict[f"fold_{fold}_train_files"] = np.asarray(fold_info["train_files"])

        if "val_files" in fold_info:
            save_dict[f"fold_{fold}_val_files"] = np.asarray(fold_info["val_files"])

        if "test_files" in fold_info:
            save_dict[f"fold_{fold}_test_files"] = np.asarray(fold_info["test_files"])

    metadata = {
        "seed": shared_config["seed"],
        "num_data": shared_config["num_data"],
        "max_particles": shared_config["max_particles"],
        "num_folds": shared_config["num_folds"],
        "final_test_ratio": shared_config["final_test_ratio"],
        "class_0": shared_config["class_0"],
        "class_1": shared_config["class_1"],
        "data_root": str(shared_config.get("data_root", "")),
        "max_files_per_class": shared_config.get("max_files_per_class", None),
    }

    save_dict["metadata_json"] = np.asarray(json.dumps(metadata, indent=2))

    if file_paths is not None:
        save_dict["file_paths"] = np.asarray(file_paths, dtype=str)

    if file_labels is not None:
        save_dict["file_labels"] = np.asarray(file_labels)

    np.savez(split_path, **save_dict)

    print("=" * 80)
    print(f"Saved split indices to: {split_path}")
    print("=" * 80)

    return split_path


def load_split_indices(project_root, shared_config):
    split_path = get_split_path(
        project_root=project_root,
        shared_config=shared_config,
    )

    if not split_path.exists():
        return None

    print("=" * 80)
    print(f"Using existing split file: {split_path}")
    print("=" * 80)

    with np.load(split_path, allow_pickle=False) as data:
        dev_idx = data["dev_idx"]
        final_test_idx = data["final_test_idx"]

        folds = []

        for fold in range(1, shared_config["num_folds"] + 1):
            train_key = f"fold_{fold}_train_idx"
            val_key = f"fold_{fold}_val_idx"
            test_key = f"fold_{fold}_test_idx"

            if train_key not in data or val_key not in data or test_key not in data:
                raise KeyError(
                    f"Split file is missing keys for fold {fold}. "
                    f"Expected {train_key}, {val_key}, {test_key} in {split_path}."
                )

            fold_info = {
                "fold": fold,
                "train_idx": data[train_key],
                "val_idx": data[val_key],
                "test_idx": data[test_key],
            }

            train_files_key = f"fold_{fold}_train_files"
            val_files_key = f"fold_{fold}_val_files"
            test_files_key = f"fold_{fold}_test_files"

            if train_files_key in data:
                fold_info["train_files"] = data[train_files_key]

            if val_files_key in data:
                fold_info["val_files"] = data[val_files_key]

            if test_files_key in data:
                fold_info["test_files"] = data[test_files_key]

            folds.append(fold_info)

    return dev_idx, final_test_idx, folds


def validate_loaded_splits(
    y,
    file_ids,
    file_labels,
    dev_idx,
    final_test_idx,
    folds,
):
    n = len(y)

    all_indices = [dev_idx, final_test_idx]

    for fold_info in folds:
        all_indices.append(fold_info["train_idx"])
        all_indices.append(fold_info["val_idx"])
        all_indices.append(fold_info["test_idx"])

    for idx in all_indices:
        if len(idx) == 0:
            raise ValueError("Loaded split contains an empty index array.")

        if np.min(idx) < 0 or np.max(idx) >= n:
            raise ValueError(
                "Loaded split indices do not match the loaded dataset size. "
                f"Dataset has n={n}, but split has min={np.min(idx)}, max={np.max(idx)}."
            )

    final_test_set = set(np.asarray(final_test_idx).tolist())

    for fold_info in folds:
        train_idx = np.asarray(fold_info["train_idx"])
        val_idx = np.asarray(fold_info["val_idx"])
        test_idx = np.asarray(fold_info["test_idx"])

        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        if len(train_set & val_set) > 0:
            raise ValueError(f"Fold {fold_info['fold']} has train/val leakage.")

        if len(train_set & test_set) > 0:
            raise ValueError(f"Fold {fold_info['fold']} has train/test leakage.")

        if len(val_set & test_set) > 0:
            raise ValueError(f"Fold {fold_info['fold']} has val/test leakage.")

        if test_set != final_test_set:
            raise ValueError(f"Fold {fold_info['fold']} test_idx differs from final_test_idx.")

        if "train_files" in fold_info and "val_files" in fold_info and "test_files" in fold_info:
            train_files = set(np.asarray(fold_info["train_files"]).tolist())
            val_files = set(np.asarray(fold_info["val_files"]).tolist())
            test_files = set(np.asarray(fold_info["test_files"]).tolist())

            if len(train_files & val_files) > 0:
                raise ValueError(f"Fold {fold_info['fold']} has train/val file leakage.")

            if len(train_files & test_files) > 0:
                raise ValueError(f"Fold {fold_info['fold']} has train/test file leakage.")

            if len(val_files & test_files) > 0:
                raise ValueError(f"Fold {fold_info['fold']} has val/test file leakage.")

    unique_files = np.arange(len(file_labels))

    for fid in unique_files:
        jet_idx = np.where(file_ids == fid)[0]

        if len(jet_idx) == 0:
            raise ValueError(f"No jets found for file_id={fid}. This should not happen.")

        jet_labels = np.unique(y[jet_idx])

        if len(jet_labels) != 1 or jet_labels[0] != file_labels[fid]:
            raise ValueError(
                f"Inconsistent labels for file_id={fid}: "
                f"jet_labels={jet_labels}, file_label={file_labels[fid]}"
            )

    print("=" * 80)
    print("Loaded split sanity checks passed.")
    print("=" * 80)


def get_or_create_file_level_test_cv_splits(
    project_root,
    y,
    file_ids,
    file_labels,
    shared_config,
    file_paths=None,
    force_recreate=False,
):
    if not force_recreate:
        loaded = load_split_indices(
            project_root=project_root,
            shared_config=shared_config,
        )

        if loaded is not None:
            dev_idx, final_test_idx, folds = loaded

            validate_loaded_splits(
                y=y,
                file_ids=file_ids,
                file_labels=file_labels,
                dev_idx=dev_idx,
                final_test_idx=final_test_idx,
                folds=folds,
            )

            return dev_idx, final_test_idx, folds

    print("=" * 80)
    print("Creating new shared FILE-LEVEL split and shared CV folds")
    print("=" * 80)

    dev_idx, final_test_idx, folds = create_file_level_test_cv_splits(
        y=y,
        file_ids=file_ids,
        file_labels=file_labels,
        num_folds=shared_config["num_folds"],
        final_test_ratio=shared_config["final_test_ratio"],
        seed=shared_config["seed"],
    )

    save_split_indices(
        project_root=project_root,
        dev_idx=dev_idx,
        final_test_idx=final_test_idx,
        folds=folds,
        shared_config=shared_config,
        file_paths=file_paths,
        file_labels=file_labels,
    )

    return dev_idx, final_test_idx, folds


# =============================================================================
# Marvin preprocessing and loading
# =============================================================================

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
    Convert one variable-length Marvin jet into fixed-size token array.

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

    if len(pt) == 0:
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


def _list_marvin_npz_files(data_root, class_name, subdir):
    root = Path(data_root) / class_name / subdir

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    files = sorted(root.glob("*.npz"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {root}")

    return files


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

    Only pt/eta/phi are used for sequence models.
    Output X has 3 features:
        z, delta_eta, delta_phi
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as f:
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


def load_marvin_obsvs_file(path, label, file_id):
    """
    Load one Marvin observable .npz file.

    Expected arrays:
        x          (N_jets, 3472)
        w          (N_jets,)
        nsubs      (N_jets, 60)
        eecs       (N_jets, 23)
        efps       (N_jets, 3389)

    We use x directly because:
        x = concat(nsubs, eecs, efps)
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as f:
        if "x" not in f.files:
            raise KeyError(f"{path} does not contain key 'x'. Available keys: {f.files}")

        X_file = np.asarray(f["x"], dtype=np.float32)
        w_file = np.asarray(f["w"], dtype=np.float32) if "w" in f.files else None

        if X_file.ndim != 2:
            raise ValueError(f"Expected obsvs/x to be 2D, got shape {X_file.shape} in {path}")

        n_jets = X_file.shape[0]

        if "nsubs" in f.files and "eecs" in f.files and "efps" in f.files:
            expected_dim = f["nsubs"].shape[1] + f["eecs"].shape[1] + f["efps"].shape[1]

            if X_file.shape[1] != expected_dim:
                raise ValueError(
                    f"Observable dimension mismatch in {path}: "
                    f"x has {X_file.shape[1]} features, but "
                    f"nsubs+eecs+efps gives {expected_dim}."
                )

    y_file = np.full(n_jets, label, dtype=np.int64)
    file_ids_file = np.full(n_jets, file_id, dtype=np.int64)

    return X_file, y_file, file_ids_file, w_file


def _select_files_for_max_jets(file_njets, file_labels, max_jets, seed=42):
    """
    Select complete files, not individual jets.

    This keeps the file-level CV split valid after applying --num-data.
    The selected number of jets can be slightly larger than max_jets because
    files are indivisible.
    """
    if max_jets is None:
        return np.arange(len(file_njets), dtype=np.int64)

    rng = np.random.default_rng(seed)

    file_njets = np.asarray(file_njets)
    file_labels = np.asarray(file_labels)

    classes = np.unique(file_labels)

    if len(classes) != 2:
        raise ValueError(f"Expected exactly two classes, got {classes}")

    target_per_class = max_jets // 2
    selected_files = []

    for cls in classes:
        cls_files = np.where(file_labels == cls)[0]
        rng.shuffle(cls_files)

        total = 0
        cls_selected = []

        for fid in cls_files:
            cls_selected.append(fid)
            total += int(file_njets[fid])

            if total >= target_per_class:
                break

        selected_files.extend(cls_selected)

    selected_files = np.asarray(sorted(selected_files), dtype=np.int64)

    return selected_files


def _filter_to_selected_files(
    X_parts,
    X_obsvs,
    y,
    file_ids,
    file_labels,
    file_paths,
    obsvs_paths,
    selected_files,
):
    """
    Keep only selected complete files and remap file_ids to 0..N_selected-1.
    """
    selected_files = np.asarray(selected_files, dtype=np.int64)

    keep_mask = np.isin(file_ids, selected_files)

    X_parts = X_parts[keep_mask]
    X_obsvs = X_obsvs[keep_mask]
    y = y[keep_mask]
    old_file_ids = file_ids[keep_mask]

    old_to_new = {old: new for new, old in enumerate(selected_files.tolist())}
    new_file_ids = np.asarray([old_to_new[int(fid)] for fid in old_file_ids], dtype=np.int64)

    new_file_labels = file_labels[selected_files]
    new_file_paths = [file_paths[int(fid)] for fid in selected_files]
    new_obsvs_paths = [obsvs_paths[int(fid)] for fid in selected_files]

    return (
        X_parts,
        X_obsvs,
        y,
        new_file_ids,
        new_file_labels,
        new_file_paths,
        new_obsvs_paths,
    )


def load_marvin_parts_dataset(
    data_root,
    class_0="vac",
    class_1="rec",
    max_particles=128,
    max_jets=None,
    max_files_per_class=None,
    sort_by_pt=True,
    seed=42,
    return_file_paths=False,
):
    """
    Load Marvin parts dataset from:

        data_root/class_name/parts/*.npz

    Returns:
        X_parts     shape (N_jets, max_particles, 3)
        y           shape (N_jets,)
        file_ids    shape (N_jets,)
        file_labels shape (N_files,)
        file_paths  list[str], optional
    """
    data_root = Path(data_root)

    X_all = []
    y_all = []
    file_ids_all = []
    file_labels = []
    file_paths = []
    file_njets = []

    file_id = 0

    for label, class_name in [(0, class_0), (1, class_1)]:
        files = _list_marvin_npz_files(data_root, class_name, "parts")

        if max_files_per_class is not None and int(max_files_per_class) > 0:
            files = files[:max_files_per_class]

        for path in files:
            X_file, y_file, file_ids_file, _, _ = load_marvin_parts_file(
                path=path,
                max_particles=max_particles,
                label=label,
                file_id=file_id,
                sort_by_pt=sort_by_pt,
            )

            X_all.append(X_file)
            y_all.append(y_file)
            file_ids_all.append(file_ids_file)
            file_labels.append(label)
            file_paths.append(str(path))
            file_njets.append(X_file.shape[0])

            file_id += 1

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    file_ids = np.concatenate(file_ids_all, axis=0)
    file_labels = np.asarray(file_labels, dtype=np.int64)
    file_njets = np.asarray(file_njets, dtype=np.int64)

    if max_jets is not None:
        selected_files = _select_files_for_max_jets(
            file_njets=file_njets,
            file_labels=file_labels,
            max_jets=max_jets,
            seed=seed,
        )

        keep_mask = np.isin(file_ids, selected_files)
        X = X[keep_mask]
        y = y[keep_mask]
        old_file_ids = file_ids[keep_mask]

        old_to_new = {old: new for new, old in enumerate(selected_files.tolist())}
        file_ids = np.asarray([old_to_new[int(fid)] for fid in old_file_ids], dtype=np.int64)

        file_labels = file_labels[selected_files]
        file_paths = [file_paths[int(fid)] for fid in selected_files]

    if return_file_paths:
        return X, y, file_ids, file_labels, file_paths

    return X, y, file_ids, file_labels


def load_marvin_obsvs_dataset(
    data_root,
    class_0="vac",
    class_1="rec",
    max_jets=None,
    max_files_per_class=None,
    seed=42,
    return_file_paths=False,
):
    """
    Load Marvin observable dataset from:

        data_root/class_name/obsvs/*.npz

    Returns:
        X_obsvs     shape (N_jets, 3472)
        y           shape (N_jets,)
        file_ids    shape (N_jets,)
        file_labels shape (N_files,)
        file_paths  list[str], optional
    """
    data_root = Path(data_root)

    X_all = []
    y_all = []
    file_ids_all = []
    file_labels = []
    file_paths = []
    file_njets = []

    file_id = 0

    for label, class_name in [(0, class_0), (1, class_1)]:
        files = _list_marvin_npz_files(data_root, class_name, "obsvs")

        if max_files_per_class is not None and int(max_files_per_class) > 0:
            files = files[:max_files_per_class]

        for path in files:
            X_file, y_file, file_ids_file, _ = load_marvin_obsvs_file(
                path=path,
                label=label,
                file_id=file_id,
            )

            X_all.append(X_file)
            y_all.append(y_file)
            file_ids_all.append(file_ids_file)
            file_labels.append(label)
            file_paths.append(str(path))
            file_njets.append(X_file.shape[0])

            file_id += 1

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    file_ids = np.concatenate(file_ids_all, axis=0)
    file_labels = np.asarray(file_labels, dtype=np.int64)
    file_njets = np.asarray(file_njets, dtype=np.int64)

    if max_jets is not None:
        selected_files = _select_files_for_max_jets(
            file_njets=file_njets,
            file_labels=file_labels,
            max_jets=max_jets,
            seed=seed,
        )

        keep_mask = np.isin(file_ids, selected_files)
        X = X[keep_mask]
        y = y[keep_mask]
        old_file_ids = file_ids[keep_mask]

        old_to_new = {old: new for new, old in enumerate(selected_files.tolist())}
        file_ids = np.asarray([old_to_new[int(fid)] for fid in old_file_ids], dtype=np.int64)

        file_labels = file_labels[selected_files]
        file_paths = [file_paths[int(fid)] for fid in selected_files]

    if return_file_paths:
        return X, y, file_ids, file_labels, file_paths

    return X, y, file_ids, file_labels


def load_marvin_parts_and_obsvs_dataset(
    data_root,
    class_0="vac",
    class_1="rec",
    max_particles=128,
    max_jets=None,
    max_files_per_class=None,
    sort_by_pt=True,
    seed=42,
    return_file_paths=True,
):
    """
    Load both Marvin parts and observable data.

    parts:
        data_root/class_name/parts/*.npz

    obsvs:
        data_root/class_name/obsvs/*.npz

    Files are matched by filename, e.g.
        parts/dijet_rec_0002.npz
        obsvs/dijet_rec_0002.npz

    Returns:
        X_parts     shape (N_jets, max_particles, 3)
        X_obsvs     shape (N_jets, 3472)
        y           shape (N_jets,)
        file_ids    shape (N_jets,)
        file_labels shape (N_files,)
        file_paths  list[str]
        obsvs_paths list[str]
    """
    data_root = Path(data_root)

    X_parts_all = []
    X_obsvs_all = []
    y_all = []
    file_ids_all = []
    file_labels = []
    file_paths = []
    obsvs_paths = []
    file_njets = []

    file_id = 0

    for label, class_name in [(0, class_0), (1, class_1)]:
        parts_files = _list_marvin_npz_files(data_root, class_name, "parts")
        obsvs_files = _list_marvin_npz_files(data_root, class_name, "obsvs")

        obsvs_by_name = {p.name: p for p in obsvs_files}

        if max_files_per_class is not None and int(max_files_per_class) > 0:
            parts_files = parts_files[:max_files_per_class]

        for parts_path in parts_files:
            if parts_path.name not in obsvs_by_name:
                raise FileNotFoundError(
                    f"No matching observable file for parts file {parts_path}. "
                    f"Expected: {data_root / class_name / 'obsvs' / parts_path.name}"
                )

            obsvs_path = obsvs_by_name[parts_path.name]

            X_parts_file, y_file, file_ids_file, _, _ = load_marvin_parts_file(
                path=parts_path,
                max_particles=max_particles,
                label=label,
                file_id=file_id,
                sort_by_pt=sort_by_pt,
            )

            X_obsvs_file, y_obs_file, file_ids_obs_file, _ = load_marvin_obsvs_file(
                path=obsvs_path,
                label=label,
                file_id=file_id,
            )

            if X_parts_file.shape[0] != X_obsvs_file.shape[0]:
                raise ValueError(
                    f"Jet-count mismatch for file pair:\n"
                    f"  parts: {parts_path} has {X_parts_file.shape[0]} jets\n"
                    f"  obsvs: {obsvs_path} has {X_obsvs_file.shape[0]} jets"
                )

            if not np.array_equal(y_file, y_obs_file):
                raise ValueError(f"Label mismatch between {parts_path} and {obsvs_path}")

            if not np.array_equal(file_ids_file, file_ids_obs_file):
                raise ValueError(f"file_id mismatch between {parts_path} and {obsvs_path}")

            X_parts_all.append(X_parts_file)
            X_obsvs_all.append(X_obsvs_file)
            y_all.append(y_file)
            file_ids_all.append(file_ids_file)

            file_labels.append(label)
            file_paths.append(str(parts_path))
            obsvs_paths.append(str(obsvs_path))
            file_njets.append(X_parts_file.shape[0])

            file_id += 1

    X_parts = np.concatenate(X_parts_all, axis=0)
    X_obsvs = np.concatenate(X_obsvs_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    file_ids = np.concatenate(file_ids_all, axis=0)

    file_labels = np.asarray(file_labels, dtype=np.int64)
    file_njets = np.asarray(file_njets, dtype=np.int64)

    if max_jets is not None:
        selected_files = _select_files_for_max_jets(
            file_njets=file_njets,
            file_labels=file_labels,
            max_jets=max_jets,
            seed=seed,
        )

        (
            X_parts,
            X_obsvs,
            y,
            file_ids,
            file_labels,
            file_paths,
            obsvs_paths,
        ) = _filter_to_selected_files(
            X_parts=X_parts,
            X_obsvs=X_obsvs,
            y=y,
            file_ids=file_ids,
            file_labels=file_labels,
            file_paths=file_paths,
            obsvs_paths=obsvs_paths,
            selected_files=selected_files,
        )

    if return_file_paths:
        return X_parts, X_obsvs, y, file_ids, file_labels, file_paths, obsvs_paths

    return X_parts, X_obsvs, y, file_ids, file_labels
