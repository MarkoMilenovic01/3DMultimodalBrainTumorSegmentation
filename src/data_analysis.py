from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ==========================================================
# 1) File Discovery
# ==========================================================

def find_one(case_dir: Path, suffix: str) -> Optional[Path]:
    matches = sorted(case_dir.glob(f"*{suffix}"))
    return matches[0] if len(matches) == 1 else None


def discover_case_files(case_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "T1": find_one(case_dir, "_t1.nii"),
        "T1CE": find_one(case_dir, "_t1ce.nii"),
        "T2": find_one(case_dir, "_t2.nii"),
        "FLAIR": find_one(case_dir, "_flair.nii"),
        "SEG": find_one(case_dir, "_seg.nii"),
    }


def is_case_complete(files: Dict[str, Optional[Path]]) -> bool:
    return all(files.values())


def list_case_directories(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.rglob("BraTS20_Training_*") if p.is_dir()])


# ==========================================================
# 2) Loading & Preprocessing
# ==========================================================

def load_nifti(path: Path, dtype=np.float32) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=dtype)


def brain_mask(image_4ch: np.ndarray) -> np.ndarray:
    return np.any(image_4ch != 0, axis=0)


def zscore(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = volume.copy().astype(np.float32)
    vox = v[mask]
    if vox.size < 100:
        return v
    mean, std = vox.mean(), vox.std()
    if std > 1e-6:
        v[mask] = (v[mask] - mean) / std
    else:
        v[mask] -= mean
    v[~mask] = 0
    return v


# ==========================================================
# 3) Plot Helpers
# ==========================================================

def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_hist(data, bins, title, xlabel, ylabel, path):
    fig = plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(fig, path)


def save_overlay(base, seg, title, path):
    fig = plt.figure()
    plt.imshow(base, cmap="gray")
    plt.imshow(seg, alpha=0.35)
    plt.title(title)
    plt.axis("off")
    save_fig(fig, path)


# ==========================================================
# 4) Dataset-Level Analysis
# ==========================================================

def dataset_analysis(dataset_root: Path, out_dir: Path) -> List[Path]:

    case_dirs = list_case_directories(dataset_root)
    complete_cases = []
    label_counts: Dict[int, int] = {}
    tumor_fractions = []

    for case_dir in case_dirs:
        files = discover_case_files(case_dir)
        if not is_case_complete(files):
            continue

        complete_cases.append(case_dir)

        seg = load_nifti(files["SEG"]).astype(np.int16)

        labels, counts = np.unique(seg, return_counts=True)
        for l, c in zip(labels, counts):
            label_counts[int(l)] = label_counts.get(int(l), 0) + int(c)

        tumor_fractions.append((seg > 0).sum() / seg.size)

    tf = np.array(tumor_fractions)

    with open(out_dir / "dataset_summary.txt", "w") as f:
        f.write("DATASET ANALYSIS\n")
        f.write("================\n")
        f.write(f"Total cases: {len(case_dirs)}\n")
        f.write(f"Complete cases: {len(complete_cases)}\n\n")
        for l in sorted(label_counts):
            f.write(f"Label {l}: {label_counts[l]}\n")
        f.write("\nTumor fraction:\n")
        f.write(f"Min:  {tf.min():.6f}\n")
        f.write(f"Mean: {tf.mean():.6f}\n")
        f.write(f"Max:  {tf.max():.6f}\n")

    save_hist(tf, 50,
              "Tumor Fraction Distribution",
              "Tumor voxels / total voxels",
              "Cases",
              out_dir / "tumor_fraction_histogram.png")

    return complete_cases


# ==========================================================
# 5) Random Case – Full Research Inspection
# ==========================================================

def random_case_analysis(case_dirs: List[Path], out_dir: Path, seed=42):

    random.seed(seed)
    case_dir = random.choice(case_dirs)
    files = discover_case_files(case_dir)

    print("Selected case:", case_dir.name)

    t1 = load_nifti(files["T1"])
    t1ce = load_nifti(files["T1CE"])
    t2 = load_nifti(files["T2"])
    flair = load_nifti(files["FLAIR"])
    seg = load_nifti(files["SEG"]).astype(np.int16)

    image = np.stack([t1, t1ce, t2, flair])
    mask = brain_mask(image)

    H, W, D = seg.shape
    slices = sorted({D//2, max(0, D//2-20), min(D-1, D//2+20)})

    # --------------------------------------------------
    # 1) Orthogonal Views
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(flair[:, :, D//2], cmap="gray")
    axes[0].set_title("Axial")

    axes[1].imshow(flair[:, W//2, :], cmap="gray")
    axes[1].set_title("Coronal")

    axes[2].imshow(flair[H//2, :, :], cmap="gray")
    axes[2].set_title("Sagittal")

    for ax in axes:
        ax.axis("off")

    save_fig(fig, out_dir / f"{case_dir.name}_orthogonal.png")

    # --------------------------------------------------
    # 2) Slice-wise Modality Comparison
    # --------------------------------------------------
    for s in slices:

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        axes[0].imshow(t1[:, :, s], cmap="gray")
        axes[0].set_title("T1")

        axes[1].imshow(t1ce[:, :, s], cmap="gray")
        axes[1].set_title("T1CE")

        axes[2].imshow(t2[:, :, s], cmap="gray")
        axes[2].set_title("T2")

        axes[3].imshow(flair[:, :, s], cmap="gray")
        axes[3].set_title("FLAIR")

        axes[4].imshow(flair[:, :, s], cmap="gray")
        axes[4].imshow(seg[:, :, s], alpha=0.35)
        axes[4].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        save_fig(fig, out_dir / f"{case_dir.name}_modalities_slice_{s}.png")

    # --------------------------------------------------
    # 3) Tumor-only Heatmap
    # --------------------------------------------------
    tumor_mask = seg > 0
    masked = flair * tumor_mask

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(masked[:, :, D//2], cmap="hot")
    plt.title("Tumor Only (FLAIR)")
    plt.axis("off")
    save_fig(fig, out_dir / f"{case_dir.name}_tumor_only.png")

    # --------------------------------------------------
    # 4) Difference Maps
    # --------------------------------------------------
    diff1 = t1ce[:, :, D//2] - t1[:, :, D//2]
    diff2 = t2[:, :, D//2] - flair[:, :, D//2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(diff1, cmap="bwr")
    axes[0].set_title("T1CE - T1")

    axes[1].imshow(diff2, cmap="bwr")
    axes[1].set_title("T2 - FLAIR")

    for ax in axes:
        ax.axis("off")

    save_fig(fig, out_dir / f"{case_dir.name}_difference_maps.png")

    # --------------------------------------------------
    # 5) Z-score Histograms
    # --------------------------------------------------
    modality_names = ["T1", "T1CE", "T2", "FLAIR"]

    for i, name in enumerate(modality_names):
        before = image[i][mask]
        after = zscore(image[i], mask)[mask]

        fig = plt.figure()
        plt.hist(before, bins=120, alpha=0.5, density=True, label="Before")
        plt.hist(after, bins=120, alpha=0.5, density=True, label="After")
        plt.legend()
        plt.title(f"{name} Before vs After Z-score")
        save_fig(fig, out_dir / f"{case_dir.name}_{name}_zscore.png")

    # --------------------------------------------------
    # 6) Per-label Intensity Distribution
    # --------------------------------------------------
    labels = [0, 1, 2, 4]
    for lbl in labels:
        region = flair[seg == lbl]
        if region.size < 50:
            continue

        fig = plt.figure()
        plt.hist(region, bins=100)
        plt.title(f"FLAIR Intensity - Label {lbl}")
        save_fig(fig, out_dir / f"{case_dir.name}_label_{lbl}_intensity.png")


# ==========================================================
# 6) Run Everything
# ==========================================================

def run_everything(dataset_root: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)

    print("Running FULL research analysis...")

    complete_cases = dataset_analysis(dataset_root, out_root)
    random_case_analysis(complete_cases, out_root)

    print("Analysis complete.")


def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python data_analysis.py <dataset_root> <output_folder>")
        return

    run_everything(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
