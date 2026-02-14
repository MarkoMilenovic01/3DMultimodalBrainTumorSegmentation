import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv

# -------------------------
# Helpers
# -------------------------

def load_nii(path: str) -> np.ndarray:
    """Loads NIfTI and returns float32 array."""
    return nib.load(path).get_fdata().astype(np.float32)

def normalize_zscore_nonzero(x: np.ndarray, eps=1e-8, clip=5.0) -> np.ndarray:
    """
    Z-score normalize using only non-zero voxels (brain region),
    then clip to [-clip, clip], then rescale to [0, 1] (optional).
    """
    mask = x > 0
    if mask.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    mu = x[mask].mean()
    std = x[mask].std()
    x_norm = (x - mu) / (std + eps)
    x_norm = np.clip(x_norm, -clip, clip)

    # optional: map to [0,1] (helps some models)
    x_norm = (x_norm + clip) / (2 * clip)
    return x_norm.astype(np.float32)

def remap_mask(y: np.ndarray) -> np.ndarray:
    """BraTS: {0,1,2,4} -> {0,1,2,3} by mapping 4->3"""
    y = y.astype(np.int16)
    y[y == 4] = 3
    return y

def get_nonzero_bbox(vol4: np.ndarray, margin=2):
    """
    vol4 shape: (C,H,W,D). Compute bbox on union of non-zero across channels.
    """
    union = np.any(vol4 > 0, axis=0)  # (H,W,D)
    coords = np.array(np.where(union))
    if coords.size == 0:
        # fallback: full
        H, W, D = vol4.shape[1:]
        return (0, H, 0, W, 0, D)

    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1) + 1

    # NOTE: coords are (H,W,D) indexing? Actually np.where on (H,W,D) -> (h_idx, w_idx, d_idx)
    hmin, wmin, dmin = coords.min(axis=1)
    hmax, wmax, dmax = coords.max(axis=1) + 1

    hmin = max(0, hmin - margin); wmin = max(0, wmin - margin); dmin = max(0, dmin - margin)
    H, W, D = vol4.shape[1:]
    hmax = min(H, hmax + margin); wmax = min(W, wmax + margin); dmax = min(D, dmax + margin)

    return (hmin, hmax, wmin, wmax, dmin, dmax)

def crop_bbox(vol4: np.ndarray, mask: np.ndarray, bbox):
    hmin, hmax, wmin, wmax, dmin, dmax = bbox
    return vol4[:, hmin:hmax, wmin:wmax, dmin:dmax], mask[hmin:hmax, wmin:wmax, dmin:dmax]

def center_crop_or_pad_3d(x: np.ndarray, target=(128,128,128), pad_value=0):
    """
    x shape: (H,W,D) or (C,H,W,D)
    Center crop if too big, pad if too small.
    """
    is_4d = (x.ndim == 4)
    if is_4d:
        C, H, W, D = x.shape
    else:
        H, W, D = x.shape
        C = None

    tH, tW, tD = target

    def crop_pad_1dim(arr, dim_size, target_size, axis):
        # crop
        if dim_size > target_size:
            start = (dim_size - target_size) // 2
            end = start + target_size
            slicer = [slice(None)] * arr.ndim
            slicer[axis] = slice(start, end)
            arr = arr[tuple(slicer)]
        # pad
        dim_size = arr.shape[axis]
        if dim_size < target_size:
            pad_before = (target_size - dim_size) // 2
            pad_after = target_size - dim_size - pad_before
            pad_width = [(0,0)] * arr.ndim
            pad_width[axis] = (pad_before, pad_after)
            arr = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
        return arr

    if is_4d:
        x = crop_pad_1dim(x, x.shape[1], tH, axis=1)
        x = crop_pad_1dim(x, x.shape[2], tW, axis=2)
        x = crop_pad_1dim(x, x.shape[3], tD, axis=3)
    else:
        x = crop_pad_1dim(x, x.shape[0], tH, axis=0)
        x = crop_pad_1dim(x, x.shape[1], tW, axis=1)
        x = crop_pad_1dim(x, x.shape[2], tD, axis=2)

    return x

def tumor_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).sum()) / float(mask.size)

# -------------------------
# Main preprocessing
# -------------------------

def preprocess_brats(
    raw_root: str,
    out_root: str,
    target_shape=(128,128,128),
    min_tumor_fraction=0.001,   # 0.1% as a starting point
    val_ratio=0.2,
    seed=42
):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    img_out = out_root / "images"
    msk_out = out_root / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in raw_root.glob("BraTS20_Training_*") if p.is_dir()])
    kept_cases = []
    rows = []

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Find modality files (handles .nii or .nii.gz)
        def f(suffix):
            hits = list(case_dir.glob(f"*_{suffix}.nii")) + list(case_dir.glob(f"*_{suffix}.nii.gz"))
            return str(hits[0]) if hits else None

        flair = f("flair")
        t1    = f("t1")
        t1ce  = f("t1ce")
        t2    = f("t2")
        seg   = f("seg")

        if not all([flair, t1, t1ce, t2, seg]):
            print(f"[SKIP] Missing files in {case_id}")
            continue

        # Load
        v_flair = load_nii(flair)
        v_t1    = load_nii(t1)
        v_t1ce  = load_nii(t1ce)
        v_t2    = load_nii(t2)
        y       = load_nii(seg).astype(np.int16)

        # Normalize each modality
        v_flair = normalize_zscore_nonzero(v_flair)
        v_t1    = normalize_zscore_nonzero(v_t1)
        v_t1ce  = normalize_zscore_nonzero(v_t1ce)
        v_t2    = normalize_zscore_nonzero(v_t2)

        # Stack: (C,H,W,D)
        x = np.stack([v_t1, v_t1ce, v_t2, v_flair], axis=0).astype(np.float32)

        # Remap mask 4->3
        y = remap_mask(y)

        # Crop to non-zero bbox
        bbox = get_nonzero_bbox(x, margin=2)
        x, y = crop_bbox(x, y, bbox)

        # Center crop/pad to target
        x = center_crop_or_pad_3d(x, target=target_shape, pad_value=0)
        y = center_crop_or_pad_3d(y, target=target_shape, pad_value=0)

        # Drop if too little tumor
        tf = tumor_fraction(y)
        if tf < min_tumor_fraction:
            # You can choose to keep them later for "negative" learning,
            # but for now you asked to drop low-labeled volumes.
            print(f"[DROP] {case_id} tumor_fraction={tf:.6f}")
            continue

        # Save
        np.save(img_out / f"{case_id}.npy", x)
        np.save(msk_out / f"{case_id}.npy", y)

        kept_cases.append(case_id)
        rows.append([case_id, tf])

        print(f"[OK] {case_id} saved | tumor_fraction={tf:.6f}")

    # Save meta
    with open(out_root / "meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "tumor_fraction"])
        w.writerows(rows)

    # Split
    train_ids, val_ids = train_test_split(kept_cases, test_size=val_ratio, random_state=seed, shuffle=True)

    (out_root / "train.txt").write_text("\n".join(train_ids))
    (out_root / "val.txt").write_text("\n".join(val_ids))

    print(f"\nDone. Kept {len(kept_cases)} cases.")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

if __name__ == "__main__":
    preprocess_brats(
        raw_root="data/raw/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        out_root="data/processed",
        target_shape=(128,128,128),
        min_tumor_fraction=0.001
    )
