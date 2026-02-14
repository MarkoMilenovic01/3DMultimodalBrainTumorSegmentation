"""
===============================================================================
BraTS 2020 Preprocessing Pipeline
===============================================================================

High-Level Overview
-------------------
This file prepares the raw BraTS MRI dataset for deep learning training.

For each patient case:
    1) Load 4 MRI modalities + segmentation mask
    2) Normalize MRI intensities
    3) Remap segmentation labels to consecutive indices
    4) Crop around the brain (remove empty background)
    5) Resize to fixed shape (128×128×128)
    6) Filter out cases with too little tumor
    7) Save as fast-loading .npy files
    8) Create train/validation split

Final output:
    - images/{case_id}.npy   → shape (4, H, W, D)
    - masks/{case_id}.npy    → shape (H, W, D)
    - meta.csv               → tumor statistics
    - train.txt / val.txt    → dataset split
===============================================================================
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


MODALITIES = ["t1", "t1ce", "t2", "flair"]
ALL_FILES = MODALITIES + ["seg"]


# =============================================================================
# I/O
# =============================================================================

def load_nifti(path: Path) -> np.ndarray:
    """
    Load a NIfTI file and return a float32 numpy array.
    Used for both MRI volumes and segmentation masks.
    """
    return nib.load(str(path)).get_fdata().astype(np.float32)


def find_case_files(case_dir: Path) -> dict[str, Path] | None:
    """
    Locate required files inside a BraTS case folder.

    Case requirement:
        - t1, t1ce, t2, flair
        - seg

    Case outcome:
        - If all exist → return dictionary of paths
        - If any missing → return None (case skipped)
    """
    files = {}
    for name in ALL_FILES:
        matches = list(case_dir.glob(f"*_{name}.nii*"))
        if not matches:
            return None  # Case skipped
        files[name] = matches[0]
    return files


# =============================================================================
# PREPROCESSING
# =============================================================================

def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Normalize MRI intensities to [0,1].

    Case:
        - If no brain voxels → return zeros
        - Otherwise → z-score normalize, clip, rescale
    """
    brain = volume > 0
    if brain.sum() == 0:
        return np.zeros_like(volume)

    mean = volume[brain].mean()
    std = volume[brain].std() + 1e-8

    z = (volume - mean) / std
    z = np.clip(z, -5, 5)

    return ((z + 5) / 10).astype(np.float32)


def remap_brats_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert BraTS labels from {0,1,2,4} → {0,1,2,3}.

    Case:
        - Label 4 (enhancing tumor) becomes 3.
    """
    mask = mask.astype(np.int16)
    mask[mask == 4] = 3
    return mask


def compute_brain_bbox(img4: np.ndarray, margin: int = 10):
    """
    Compute bounding box around non-zero voxels.

    Case:
        - If brain found → tight bounding box + margin
        - If no brain found → return full volume
    """
    brain = np.any(img4 > 0, axis=0)
    coords = np.where(brain)

    H, W, D = img4.shape[1:]

    if coords[0].size == 0:
        return (0, H, 0, W, 0, D)

    h0, w0, d0 = (int(c.min()) for c in coords)
    h1, w1, d1 = (int(c.max()) + 1 for c in coords)

    h0 = max(0, h0 - margin); h1 = min(H, h1 + margin)
    w0 = max(0, w0 - margin); w1 = min(W, w1 + margin)
    d0 = max(0, d0 - margin); d1 = min(D, d1 + margin)

    return (h0, h1, w0, w1, d0, d1)


def crop_to_bbox(img4: np.ndarray, mask: np.ndarray, bbox):
    """
    Crop both image and mask using the same bounding box.
    Ensures spatial alignment remains correct.
    """
    h0, h1, w0, w1, d0, d1 = bbox
    return (
        img4[:, h0:h1, w0:w1, d0:d1],
        mask[h0:h1, w0:w1, d0:d1],
    )


def center_crop_or_pad(x: np.ndarray, target=(128, 128, 128)):
    """
    Force volume to target size.

    Cases:
        - If dimension > target → center crop
        - If dimension < target → center pad
        - If equal → unchanged
    """
    for axis, t in zip(range(-3, 0), target):
        s = x.shape[axis]

        if s > t:
            start = (s - t) // 2
            x = np.take(x, range(start, start + t), axis=axis)

        elif s < t:
            pad = t - s
            before = pad // 2
            after = pad - before

            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (before, after)
            x = np.pad(x, pad_width)

    return x


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def preprocess_brats(
    raw_dir: str,
    out_dir: str,
    target_shape=(128, 128, 128),
    min_tumor_fraction=0.001,
    val_split=0.2,
    seed=42,
):
    """
    Execute full preprocessing pipeline.

    Case handling:
        - Missing files → case skipped
        - Tumor fraction too small → case dropped
        - Valid case → processed and saved
    """

    raw = Path(raw_dir)
    out = Path(out_dir)

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(raw.glob("BraTS20_Training_*"))

    kept_ids = []
    meta_rows = []

    for case_dir in case_dirs:
        case_id = case_dir.name

        files = find_case_files(case_dir)
        if files is None:
            print(f"[SKIP] {case_id}")
            continue

        vols = {m: normalize_mri(load_nifti(files[m])) for m in MODALITIES}
        mask = remap_brats_mask(load_nifti(files["seg"]))

        img4 = np.stack([vols["t1"], vols["t1ce"], vols["t2"], vols["flair"]], axis=0)

        bbox = compute_brain_bbox(img4)
        img4, mask = crop_to_bbox(img4, mask, bbox)

        img4 = center_crop_or_pad(img4, target_shape)
        mask = center_crop_or_pad(mask, target_shape)

        tumor_fraction = float((mask > 0).sum()) / mask.size

        if tumor_fraction < min_tumor_fraction:
            print(f"[DROP] {case_id}")
            continue

        np.save(out / "images" / f"{case_id}.npy", img4)
        np.save(out / "masks" / f"{case_id}.npy", mask)

        kept_ids.append(case_id)
        meta_rows.append([case_id, tumor_fraction])

        print(f"[OK] {case_id}")

    with open(out / "meta.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "tumor_fraction"])
        writer.writerows(meta_rows)

    train_ids, val_ids = train_test_split(
        kept_ids, test_size=val_split, random_state=seed
    )

    (out / "train.txt").write_text("\n".join(train_ids))
    (out / "val.txt").write_text("\n".join(val_ids))

    print(f"Processed {len(kept_ids)} cases.")



# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Script entry point.

    This block runs only when the file is executed directly:
        python preprocess.py

    Modify the paths and parameters below if needed.
    """

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    RAW_DATA_DIR = "data/raw/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    OUTPUT_DIR = "data/processed"

    TARGET_SHAPE = (128, 128, 128)
    VALIDATION_SPLIT = 0.2
    MIN_TUMOR_FRACTION = 0.001
    RANDOM_SEED = 42
    
    # -------------------------------------------------------------------------
    # Run preprocessing
    # -------------------------------------------------------------------------
    preprocess_brats(
        raw_dir=RAW_DATA_DIR,
        out_dir=OUTPUT_DIR,
        target_shape=TARGET_SHAPE,
        min_tumor_fraction=MIN_TUMOR_FRACTION,
        val_split=VALIDATION_SPLIT,
        seed=RANDOM_SEED,
    )

    print("\nPreprocessing completed successfully.")
