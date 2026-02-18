"""
preprocess.py — BraTS 2020 Preprocessing Pipeline
Steps:
  1. Load NIfTI volumes + sanity checks
  2. Intensity normalization (z-score per modality, brain-only)
  3. Foreground crop + pad to fixed size (240x240x155 → crop → pad to 192x192x144)
  4. Label remapping (0,1,2,4 → 0,1,2,3)
  5. Save preprocessed volumes as .npy for fast loading
"""

import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Config ───────────────────────────
MODALITIES   = ["flair", "t1", "t1ce", "t2"]
LABEL_REMAP  = {0: 0, 1: 1, 2: 2, 4: 3}   # WT/TC/ET → 0,1,2,3
TARGET_SHAPE = (192, 192, 144)              # after crop+pad
RAW_ROOT     = Path("data/raw/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
OUT_ROOT     = Path("data/preprocessed")
STATS_FILE   = OUT_ROOT / "dataset_stats.json"


# ─────────────────────────── I/O helpers ───────────────────────────

def get_case_paths(case_dir: Path) -> dict:
    """Return dict {modality: path, 'seg': path} for a BraTS case folder."""
    case_id = case_dir.name
    paths = {}
    for mod in MODALITIES:
        p = case_dir / f"{case_id}_{mod}.nii"
        if not p.exists():
            p = case_dir / f"{case_id}_{mod}.nii.gz"
        paths[mod] = p
    seg = case_dir / f"{case_id}_seg.nii"
    if not seg.exists():
        seg = case_dir / f"{case_id}_seg.nii.gz"
    paths["seg"] = seg
    return paths


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


# ─────────────────────────── Sanity Checks ───────────────────────────

def sanity_check(paths: dict, case_id: str) -> bool:
    """Check all files exist, shapes match, seg has valid labels."""
    ok = True

    # 1. All files exist
    for key, p in paths.items():
        if not p.exists():
            log.error(f"[{case_id}] Missing file: {p}")
            ok = False

    if not ok:
        return False

    # 2. Load and check shapes
    shapes = {}
    for key, p in paths.items():
        vol = load_volume(p)
        shapes[key] = vol.shape

    ref_shape = shapes[MODALITIES[0]]
    for key, sh in shapes.items():
        if sh != ref_shape:
            log.error(f"[{case_id}] Shape mismatch: {key} {sh} vs {MODALITIES[0]} {ref_shape}")
            ok = False

    # 3. Check seg labels
    seg = load_volume(paths["seg"])
    unique_labels = np.unique(seg).astype(int).tolist()
    valid_labels  = {0, 1, 2, 4}
    unexpected    = set(unique_labels) - valid_labels
    if unexpected:
        log.warning(f"[{case_id}] Unexpected seg labels: {unexpected}")

    # 4. Check for NaNs / Infs
    for key in MODALITIES:
        vol = load_volume(paths[key])
        if np.isnan(vol).any() or np.isinf(vol).any():
            log.warning(f"[{case_id}] NaN/Inf found in {key}")

    return ok


# ─────────────────────────── Normalization ───────────────────────────

def zscore_normalize(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalization using only brain voxels (mask > 0)."""
    brain_voxels = vol[mask > 0]
    if brain_voxels.size == 0:
        return vol
    mu  = brain_voxels.mean()
    std = brain_voxels.std()
    if std < 1e-8:
        std = 1.0
    normalized = (vol - mu) / std
    # Zero out background
    normalized[mask == 0] = 0.0
    return normalized.astype(np.float32)


# ─────────────────────────── Crop & Pad ───────────────────────────

def compute_brain_bbox(mask: np.ndarray, margin: int = 5) -> tuple:
    """Compute bounding box of non-zero mask with margin."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(mask.shape))
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])


def crop_to_bbox(vol: np.ndarray, bbox: tuple) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bbox
    return vol[x0:x1, y0:y1, z0:z1]


def pad_to_shape(vol: np.ndarray, target: tuple, pad_val: float = 0.0) -> np.ndarray:
    """Center-pad volume to target shape."""
    result = np.full(target, pad_val, dtype=vol.dtype)
    src_shape = vol.shape
    offsets = [(t - s) // 2 for t, s in zip(target, src_shape)]
    # Clamp if vol is already larger than target on some axis
    slices_dst = tuple(slice(o, o + s) for o, s in zip(offsets, src_shape))
    slices_src = tuple(slice(0, s) for s in src_shape)
    result[slices_dst] = vol[slices_src]
    return result


# ─────────────────────────── Label Remap ───────────────────────────

def remap_labels(seg: np.ndarray, remap: dict = LABEL_REMAP) -> np.ndarray:
    """Remap BraTS labels: {0,1,2,4} → {0,1,2,3}."""
    out = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in remap.items():
        out[seg == src] = dst
    return out


# ─────────────────────────── Main Pipeline ───────────────────────────

def preprocess_case(case_dir: Path, out_dir: Path, margin: int = 5) -> dict | None:
    """
    Full preprocessing for one case.
    Returns stats dict or None on failure.
    """
    case_id = case_dir.name
    paths   = get_case_paths(case_dir)

    # --- Sanity check ---
    if not sanity_check(paths, case_id):
        log.error(f"[{case_id}] Skipping due to sanity check failure.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.npz"
    if out_path.exists():
        log.info(f"[{case_id}] Already preprocessed, skipping.")
        return {"case_id": case_id, "status": "skipped"}

    # --- Load all modalities ---
    volumes = {mod: load_volume(paths[mod]) for mod in MODALITIES}
    seg_raw = load_volume(paths["seg"]).astype(np.float32)

    # --- Brain mask (union of all modalities > 0) ---
    brain_mask = np.zeros_like(volumes[MODALITIES[0]], dtype=bool)
    for mod in MODALITIES:
        brain_mask |= (volumes[mod] > 0)

    # --- Normalize each modality ---
    norm_volumes = {mod: zscore_normalize(volumes[mod], brain_mask) for mod in MODALITIES}

    # --- Crop ---
    bbox = compute_brain_bbox(brain_mask, margin=margin)
    cropped_vols = {mod: crop_to_bbox(norm_volumes[mod], bbox) for mod in MODALITIES}
    cropped_seg  = crop_to_bbox(seg_raw, bbox)
    cropped_shape = cropped_vols[MODALITIES[0]].shape

    # --- Pad ---
    padded_vols = {mod: pad_to_shape(cropped_vols[mod], TARGET_SHAPE, 0.0) for mod in MODALITIES}
    padded_seg  = pad_to_shape(cropped_seg, TARGET_SHAPE, 0.0)

    # --- Label remap ---
    remapped_seg = remap_labels(padded_seg)

    # --- Stack modalities: (4, H, W, D) ---
    image = np.stack([padded_vols[mod] for mod in MODALITIES], axis=0)   # (4, 192, 192, 144)
    label = remapped_seg[np.newaxis]                                       # (1, 192, 192, 144)

    # --- Save ---
    np.savez_compressed(out_path, image=image, label=label)

    stats = {
        "case_id"       : case_id,
        "status"        : "ok",
        "raw_shape"     : list(volumes[MODALITIES[0]].shape),
        "cropped_shape" : list(cropped_shape),
        "padded_shape"  : list(TARGET_SHAPE),
        "unique_labels" : np.unique(remapped_seg).tolist(),
        "bbox"          : list(bbox),
    }
    log.info(f"[{case_id}] Done. raw→crop {list(volumes[MODALITIES[0]].shape)}→{list(cropped_shape)}")
    return stats


def run_preprocessing(raw_root: Path = RAW_ROOT, out_root: Path = OUT_ROOT):
    case_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir() and "BraTS20_Training" in d.name])
    log.info(f"Found {len(case_dirs)} cases in {raw_root}")

    all_stats = []
    for case_dir in tqdm(case_dirs, desc="Preprocessing"):
        stats = preprocess_case(case_dir, out_root / "train")
        if stats:
            all_stats.append(stats)

    # Save dataset stats
    out_root.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(all_stats, f, indent=2)
    log.info(f"Stats saved to {STATS_FILE}")

    ok_count   = sum(1 for s in all_stats if s.get("status") == "ok")
    skip_count = sum(1 for s in all_stats if s.get("status") == "skipped")
    log.info(f"Preprocessing complete: {ok_count} processed, {skip_count} skipped.")


if __name__ == "__main__":
    run_preprocessing()