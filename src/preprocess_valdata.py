from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib


def load_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def normalize_zscore_nonzero(x: np.ndarray, eps=1e-8, clip=5.0) -> np.ndarray:
    mask = x > 0
    if mask.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)
    mu = x[mask].mean()
    std = x[mask].std()
    x = (x - mu) / (std + eps)
    x = np.clip(x, -clip, clip)
    x = (x + clip) / (2 * clip)  # -> [0,1]
    return x.astype(np.float32)


def get_nonzero_bbox(vol4: np.ndarray, margin=2):
    union = np.any(vol4 > 0, axis=0)  # (H,W,D)
    coords = np.array(np.where(union))
    if coords.size == 0:
        H, W, D = vol4.shape[1:]
        return (0, H, 0, W, 0, D)

    hmin, wmin, dmin = coords.min(axis=1)
    hmax, wmax, dmax = coords.max(axis=1) + 1

    hmin = max(0, hmin - margin); wmin = max(0, wmin - margin); dmin = max(0, dmin - margin)
    H, W, D = vol4.shape[1:]
    hmax = min(H, hmax + margin); wmax = min(W, wmax + margin); dmax = min(D, dmax + margin)

    return (hmin, hmax, wmin, wmax, dmin, dmax)


def crop_bbox(vol4: np.ndarray, bbox):
    hmin, hmax, wmin, wmax, dmin, dmax = bbox
    return vol4[:, hmin:hmax, wmin:wmax, dmin:dmax]


def center_crop_or_pad_3d(x: np.ndarray, target=(128, 128, 128), pad_value=0):
    is_4d = (x.ndim == 4)
    if is_4d:
        C, H, W, D = x.shape
    else:
        H, W, D = x.shape

    tH, tW, tD = target

    def crop_pad(arr, axis, t):
        s = arr.shape[axis]
        if s > t:
            start = (s - t) // 2
            end = start + t
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(start, end)
            arr = arr[tuple(sl)]
        s = arr.shape[axis]
        if s < t:
            before = (t - s) // 2
            after = t - s - before
            pad = [(0, 0)] * arr.ndim
            pad[axis] = (before, after)
            arr = np.pad(arr, pad, mode="constant", constant_values=pad_value)
        return arr

    if is_4d:
        x = crop_pad(x, 1, tH)
        x = crop_pad(x, 2, tW)
        x = crop_pad(x, 3, tD)
    else:
        x = crop_pad(x, 0, tH)
        x = crop_pad(x, 1, tW)
        x = crop_pad(x, 2, tD)

    return x


def preprocess_valdata(
    raw_root: str,
    out_root: str = "data/processed_val",
    target_shape=(128, 128, 128),
):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_img = out_root / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in raw_root.glob("BraTS20_Validation_*") if p.is_dir()])
    print("Found cases:", len(case_dirs))

    for case_dir in case_dirs:
        case_id = case_dir.name

        def f(suffix):
            hits = list(case_dir.glob(f"*_{suffix}.nii")) + list(case_dir.glob(f"*_{suffix}.nii.gz"))
            return str(hits[0]) if hits else None

        flair = f("flair")
        t1    = f("t1")
        t1ce  = f("t1ce")
        t2    = f("t2")

        if not all([flair, t1, t1ce, t2]):
            print("[SKIP] missing modality in", case_id)
            continue

        v_flair = normalize_zscore_nonzero(load_nii(flair))
        v_t1    = normalize_zscore_nonzero(load_nii(t1))
        v_t1ce  = normalize_zscore_nonzero(load_nii(t1ce))
        v_t2    = normalize_zscore_nonzero(load_nii(t2))

        # channel order matches earlier: [T1, T1CE, T2, FLAIR]
        x = np.stack([v_t1, v_t1ce, v_t2, v_flair], axis=0).astype(np.float32)

        bbox = get_nonzero_bbox(x, margin=2)
        x = crop_bbox(x, bbox)
        x = center_crop_or_pad_3d(x, target=target_shape, pad_value=0)

        np.save(out_img / f"{case_id}.npy", x)
        print("[OK]", case_id, "->", x.shape)

    print("Saved to:", out_img)


if __name__ == "__main__":
    preprocess_valdata(
        raw_root="data/raw/archive/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData",
        out_root="data/processed_val",
        target_shape=(128, 128, 128),
    )
