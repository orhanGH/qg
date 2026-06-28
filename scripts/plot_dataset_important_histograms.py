#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_key(npz, candidates):
    keys = set(npz.files)
    for c in candidates:
        if c in keys:
            return c
    return None


def wrap_phi(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def weighted_phi(phi, weight):
    s = np.sum(weight * np.sin(phi))
    c = np.sum(weight * np.cos(phi))
    return np.arctan2(s, c)


def safe_hist_values(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return x


def compare_hist(data, key, labels, out_path, title, xlabel, bins=80, logy=False, density=True, xlim=None):
    plt.figure(figsize=(8, 5))

    any_data = False
    for label in labels:
        values = safe_hist_values(data[label].get(key, []))
        if len(values) == 0:
            continue
        any_data = True
        plt.hist(values, bins=bins, density=density, histtype="step", linewidth=1.7, label=label)

    if not any_data:
        plt.close()
        print(f"[skip] no data for {key}")
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    if logy:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[ok] {out_path}")


def bar_plot(names, values, out_path, title, ylabel):
    plt.figure(figsize=(7, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[ok] {out_path}")


def maybe_downsample(values, max_size, rng):
    values = np.asarray(values)
    if len(values) <= max_size:
        return values
    idx = rng.choice(len(values), size=max_size, replace=False)
    return values[idx]


def list_part_files(data_root, sample, max_files_per_class):
    part_dir = Path(data_root) / sample / "parts"
    files = sorted(part_dir.glob("*.npz"))

    if not files:
        raise FileNotFoundError(f"No .npz files found in {part_dir}")

    if max_files_per_class is not None and max_files_per_class > 0:
        files = files[:max_files_per_class]

    return files


def process_sample(
    data_root,
    sample,
    max_files_per_class,
    max_particles,
    max_jets_per_class,
    max_jets_per_file,
    max_constituents_hist,
    seed,
):
    rng = np.random.default_rng(seed)

    files = list_part_files(data_root, sample, max_files_per_class)
    print(f"\n=== Processing {sample} ===")
    print(f"Files: {len(files)}")

    out = {
        "jets_per_file": [],
        "multiplicity": [],
        "jet_pt": [],
        "jet_eta": [],
        "jet_phi": [],
        "jet_mass": [],
        "leading_z": [],
        "second_z": [],
        "third_z": [],
        "sum_kept_z": [],
        "lost_pt_fraction": [],
        "padding_fraction": [],
        "truncated": [],
        "constituent_deltaR": [],
        "jet_width_girth": [],
        "pt_dispersion": [],
    }

    total_jets_used = 0
    missing_mass = False

    for file_idx, path in enumerate(files, start=1):
        print(f"[{sample}] file {file_idx}/{len(files)}: {path.name}")

        with np.load(path) as f:
            pt_key = find_key(f, ["pt", "pts", "pT", "particle_pt"])
            eta_key = find_key(f, ["eta", "etas", "particle_eta"])
            phi_key = find_key(f, ["phi", "phis", "particle_phi"])
            mass_key = find_key(f, ["mass", "m", "masses", "particle_mass"])
            offsets_key = find_key(f, ["offsets", "offset", "jet_offsets"])

            if pt_key is None or eta_key is None or phi_key is None or offsets_key is None:
                print("Available keys:", f.files)
                raise KeyError(
                    f"Missing required keys in {path}. Need pt/eta/phi/offsets-like arrays."
                )

            pt = np.asarray(f[pt_key], dtype=np.float64)
            eta = np.asarray(f[eta_key], dtype=np.float64)
            phi = np.asarray(f[phi_key], dtype=np.float64)
            offsets = np.asarray(f[offsets_key], dtype=np.int64)

            mass = None
            if mass_key is not None:
                mass = np.asarray(f[mass_key], dtype=np.float64)
            else:
                missing_mass = True

        n_jets_file = len(offsets) - 1
        out["jets_per_file"].append(n_jets_file)

        jet_indices = np.arange(n_jets_file)

        if max_jets_per_file is not None and max_jets_per_file > 0 and len(jet_indices) > max_jets_per_file:
            jet_indices = rng.choice(jet_indices, size=max_jets_per_file, replace=False)
            jet_indices = np.sort(jet_indices)

        if max_jets_per_class is not None and max_jets_per_class > 0:
            remaining = max_jets_per_class - total_jets_used
            if remaining <= 0:
                break
            if len(jet_indices) > remaining:
                jet_indices = jet_indices[:remaining]

        for j in jet_indices:
            start = offsets[j]
            end = offsets[j + 1]

            p = pt[start:end]
            e = eta[start:end]
            ph = phi[start:end]

            valid = np.isfinite(p) & np.isfinite(e) & np.isfinite(ph) & (p > 0)
            p = p[valid]
            e = e[valid]
            ph = ph[valid]

            if len(p) == 0:
                continue

            total_pt = np.sum(p)
            if total_pt <= 0 or not np.isfinite(total_pt):
                continue

            order = np.argsort(p)[::-1]
            p_sorted = p[order]
            e_sorted = e[order]
            ph_sorted = ph[order]

            kept = min(len(p_sorted), max_particles)

            p_kept = p_sorted[:kept]
            e_kept = e_sorted[:kept]
            ph_kept = ph_sorted[:kept]

            z_kept = p_kept / total_pt

            jet_eta = np.sum(p * e) / total_pt
            jet_phi = weighted_phi(ph, p)

            d_eta = e_kept - jet_eta
            d_phi = wrap_phi(ph_kept - jet_phi)
            dR = np.sqrt(d_eta**2 + d_phi**2)

            z_all_sorted = p_sorted / total_pt

            out["multiplicity"].append(len(p))
            out["jet_pt"].append(total_pt)
            out["jet_eta"].append(jet_eta)
            out["jet_phi"].append(jet_phi)

            out["leading_z"].append(z_all_sorted[0])
            out["second_z"].append(z_all_sorted[1] if len(z_all_sorted) > 1 else 0.0)
            out["third_z"].append(z_all_sorted[2] if len(z_all_sorted) > 2 else 0.0)

            sum_kept_z = np.sum(z_kept)
            out["sum_kept_z"].append(sum_kept_z)
            out["lost_pt_fraction"].append(max(0.0, 1.0 - sum_kept_z))
            out["padding_fraction"].append((max_particles - kept) / max_particles)
            out["truncated"].append(1 if len(p_sorted) > max_particles else 0)

            out["constituent_deltaR"].extend(dR.tolist())
            out["jet_width_girth"].append(np.sum(z_kept * dR))
            out["pt_dispersion"].append(np.sqrt(np.sum(z_kept**2)))

            if mass is not None:
                m_raw = mass[start:end][valid]
                m_sorted = m_raw[order]
                m_kept = m_sorted[:kept]

                px = p_kept * np.cos(ph_kept)
                py = p_kept * np.sin(ph_kept)
                pz = p_kept * np.sinh(e_kept)
                energy = np.sqrt(np.maximum(m_kept**2 + p_kept**2 * np.cosh(e_kept)**2, 0.0))

                E = np.sum(energy)
                PX = np.sum(px)
                PY = np.sum(py)
                PZ = np.sum(pz)
                m2 = E**2 - PX**2 - PY**2 - PZ**2
                out["jet_mass"].append(np.sqrt(max(m2, 0.0)))

            total_jets_used += 1

        # Keep constituent dR memory bounded
        if len(out["constituent_deltaR"]) > max_constituents_hist:
            out["constituent_deltaR"] = maybe_downsample(
                out["constituent_deltaR"], max_constituents_hist, rng
            ).tolist()

    if missing_mass:
        print(f"[info] No mass key found for some/all {sample} files; jet_mass plot may be skipped.")

    print(f"[{sample}] jets used: {total_jets_used}")

    # Final downsample for constituent dR
    out["constituent_deltaR"] = maybe_downsample(
        out["constituent_deltaR"], max_constituents_hist, rng
    ).tolist()

    return out


def write_summary_csv(data, labels, out_path):
    rows = []
    for label in labels:
        for key, values in data[label].items():
            arr = safe_hist_values(values)
            if len(arr) == 0:
                continue
            rows.append({
                "sample": label,
                "quantity": key,
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "p05": float(np.percentile(arr, 5)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(np.max(arr)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[ok] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-0", default="vac")
    parser.add_argument("--class-1", default="rec")
    parser.add_argument("--max-files-per-class", type=int, default=10)
    parser.add_argument("--max-jets-per-class", type=int, default=100000)
    parser.add_argument("--max-jets-per-file", type=int, default=0)
    parser.add_argument("--max-particles", type=int, default=64)
    parser.add_argument("--max-constituents-hist", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [args.class_0, args.class_1]

    data = {}
    data[args.class_0] = process_sample(
        args.data_root,
        args.class_0,
        args.max_files_per_class,
        args.max_particles,
        args.max_jets_per_class,
        args.max_jets_per_file if args.max_jets_per_file > 0 else None,
        args.max_constituents_hist,
        args.seed,
    )
    data[args.class_1] = process_sample(
        args.data_root,
        args.class_1,
        args.max_files_per_class,
        args.max_particles,
        args.max_jets_per_class,
        args.max_jets_per_file if args.max_jets_per_file > 0 else None,
        args.max_constituents_hist,
        args.seed + 1,
    )

    write_summary_csv(data, labels, out_dir / "dataset_histogram_summary.csv")

    compare_hist(
        data, "jets_per_file", labels,
        out_dir / "01_jets_per_file_rec_vs_vac.png",
        "Jets per file",
        "Number of jets in file",
        bins=40,
        density=False,
    )

    compare_hist(
        data, "jet_pt", labels,
        out_dir / "02_jet_pt_rec_vs_vac.png",
        "Jet pT",
        "Jet pT",
        bins=80,
        logy=True,
    )

    compare_hist(
        data, "jet_eta", labels,
        out_dir / "03_jet_eta_rec_vs_vac.png",
        "Jet eta",
        "pT-weighted jet eta",
        bins=80,
    )

    compare_hist(
        data, "jet_phi", labels,
        out_dir / "04_jet_phi_rec_vs_vac.png",
        "Jet phi",
        "pT-weighted jet phi",
        bins=80,
    )

    compare_hist(
        data, "jet_mass", labels,
        out_dir / "05_jet_mass_rec_vs_vac.png",
        "Jet mass",
        "Approximate jet mass from constituents",
        bins=80,
        logy=True,
    )

    compare_hist(
        data, "multiplicity", labels,
        out_dir / "06_constituent_multiplicity_rec_vs_vac.png",
        "Constituent multiplicity",
        "Number of constituents per jet",
        bins=80,
        logy=True,
    )

    compare_hist(
        data, "leading_z", labels,
        out_dir / "07_leading_particle_z_rec_vs_vac.png",
        "Leading particle z",
        "Leading constituent z = pT_i / sum pT",
        bins=80,
    )

    compare_hist(
        data, "sum_kept_z", labels,
        out_dir / "08_sum_kept_z_rec_vs_vac.png",
        f"Fraction of jet pT kept by top {args.max_particles} particles",
        "sum of kept z",
        bins=80,
        xlim=(0.0, 1.05),
    )

    compare_hist(
        data, "lost_pt_fraction", labels,
        out_dir / "09_lost_pt_fraction_rec_vs_vac.png",
        f"Lost pT fraction after keeping top {args.max_particles} particles",
        "1 - sum kept z",
        bins=80,
        logy=True,
        xlim=(0.0, 1.0),
    )

    compare_hist(
        data, "padding_fraction", labels,
        out_dir / "10_padding_fraction_rec_vs_vac.png",
        "Padding fraction",
        f"Fraction of padded rows in {args.max_particles}-particle input",
        bins=65,
        logy=True,
        xlim=(0.0, 1.0),
    )

    compare_hist(
        data, "constituent_deltaR", labels,
        out_dir / "11_constituent_deltaR_rec_vs_vac.png",
        "Constituent radial distance from jet axis",
        "Delta R",
        bins=100,
        logy=True,
    )

    compare_hist(
        data, "jet_width_girth", labels,
        out_dir / "12_jet_width_girth_rec_vs_vac.png",
        "Jet width / girth",
        "sum z_i * DeltaR_i",
        bins=80,
        logy=True,
    )

    compare_hist(
        data, "pt_dispersion", labels,
        out_dir / "13_pt_dispersion_rec_vs_vac.png",
        "pT dispersion",
        "sqrt(sum z_i^2)",
        bins=80,
    )

    truncated_rates = []
    for label in labels:
        arr = np.asarray(data[label]["truncated"], dtype=float)
        truncated_rates.append(float(np.mean(arr)) if len(arr) else 0.0)

    bar_plot(
        labels,
        truncated_rates,
        out_dir / "14_truncated_fraction_rec_vs_vac.png",
        f"Fraction of jets with more than {args.max_particles} particles",
        "Truncated fraction",
    )

    print("\nDone.")
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
