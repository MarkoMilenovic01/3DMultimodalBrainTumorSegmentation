"""
augment.py — BraTS 2020 Data Augmentation
RTX 4060-friendly: all ops on CPU via numpy/scipy, fast + lightweight.

Spatial:
  - Random flips (per-axis)
  - Small rotations (±15°)
  - Small scaling (0.9–1.1)

Intensity:
  - Brightness shift
  - Contrast scaling
  - Gamma correction
  - Gaussian noise

Applied to (4, H, W, D) image + (1, H, W, D) label.
"""

import numpy as np
from scipy.ndimage import rotate, zoom


# ─────────────────────────── Spatial Augmentations ───────────────────────────

def random_flip(image: np.ndarray, label: np.ndarray,
                p_per_axis: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Random flips along spatial axes (axes 1,2,3 for C,H,W,D layout)."""
    for axis in [1, 2, 3]:
        if np.random.rand() < p_per_axis:
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotation(image: np.ndarray, label: np.ndarray,
                    max_angle: float = 15.0,
                    p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Random rotation in one randomly chosen plane (xy, xz, or yz).
    Uses nearest-neighbour for label to avoid interpolation artifacts.
    """
    if np.random.rand() >= p:
        return image, label

    angle = np.random.uniform(-max_angle, max_angle)
    # Randomly pick which plane to rotate in (axes pairs within H,W,D → indices 1,2,3)
    plane = np.random.choice([(1, 2), (1, 3), (2, 3)])

    rotated_mods = []
    for c in range(image.shape[0]):
        rot = rotate(image[c], angle, axes=plane, reshape=False,
                     order=1, mode="constant", cval=0.0)
        rotated_mods.append(rot)
    image = np.stack(rotated_mods, axis=0)

    label_rot = rotate(label[0].astype(np.float32), angle, axes=plane,
                       reshape=False, order=0, mode="constant", cval=0.0)
    label = label_rot.astype(np.uint8)[np.newaxis]

    return image, label


def random_scale(image: np.ndarray, label: np.ndarray,
                 scale_range: tuple = (0.9, 1.1),
                 p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Random isotropic scaling then center-crop/pad back to original shape.
    """
    if np.random.rand() >= p:
        return image, label

    factor = np.random.uniform(*scale_range)
    orig_shape = image.shape[1:]  # (H, W, D)

    def _scale_vol(vol, order):
        scaled = zoom(vol, factor, order=order, mode="constant", cval=0.0)
        return _recrop_pad(scaled, orig_shape, 0.0)

    scaled_mods = [_scale_vol(image[c], order=1) for c in range(image.shape[0])]
    image = np.stack(scaled_mods, axis=0)

    label_scaled = _scale_vol(label[0].astype(np.float32), order=0).astype(np.uint8)
    label = label_scaled[np.newaxis]

    return image, label


def _recrop_pad(vol: np.ndarray, target: tuple, pad_val: float = 0.0) -> np.ndarray:
    """Center-crop then pad to match target shape exactly."""
    result = vol
    # Crop if too large
    slices = []
    for s, t in zip(result.shape, target):
        start = max(0, (s - t) // 2)
        slices.append(slice(start, start + min(s, t)))
    result = result[tuple(slices)]

    # Pad if too small
    pad_widths = []
    for s, t in zip(result.shape, target):
        total = max(0, t - s)
        pad_widths.append((total // 2, total - total // 2))
    result = np.pad(result, pad_widths, mode="constant", constant_values=pad_val)
    return result


# ─────────────────────────── Intensity Augmentations ───────────────────────────

def random_brightness(image: np.ndarray,
                      shift_range: float = 0.1,
                      p: float = 0.5) -> np.ndarray:
    """Add a small random brightness shift per modality."""
    if np.random.rand() >= p:
        return image
    shifts = np.random.uniform(-shift_range, shift_range, size=(image.shape[0], 1, 1, 1))
    return image + shifts.astype(np.float32)


def random_contrast(image: np.ndarray,
                    contrast_range: tuple = (0.85, 1.15),
                    p: float = 0.5) -> np.ndarray:
    """Scale intensities around per-modality mean."""
    if np.random.rand() >= p:
        return image
    factors = np.random.uniform(*contrast_range, size=(image.shape[0], 1, 1, 1)).astype(np.float32)
    means   = image.mean(axis=(1, 2, 3), keepdims=True)
    return (image - means) * factors + means


def random_gamma(image: np.ndarray,
                 gamma_range: tuple = (0.7, 1.5),
                 p: float = 0.3) -> np.ndarray:
    """
    Gamma correction per modality.
    Maps each modality to [0,1], applies gamma, maps back.
    Only applied to brain voxels (> min of brain region).
    """
    if np.random.rand() >= p:
        return image
    out = image.copy()
    for c in range(image.shape[0]):
        vol = out[c]
        mn, mx = vol.min(), vol.max()
        if mx - mn < 1e-6:
            continue
        gamma     = np.random.uniform(*gamma_range)
        vol_norm  = (vol - mn) / (mx - mn)
        vol_gamma = np.power(np.clip(vol_norm, 0, 1), gamma)
        out[c]    = vol_gamma * (mx - mn) + mn
    return out


def random_gaussian_noise(image: np.ndarray,
                          noise_std_range: tuple = (0.0, 0.05),
                          p: float = 0.3) -> np.ndarray:
    """Add small Gaussian noise."""
    if np.random.rand() >= p:
        return image
    std   = np.random.uniform(*noise_std_range)
    noise = np.random.normal(0, std, size=image.shape).astype(np.float32)
    return image + noise


# ─────────────────────────── Patch Sampling ───────────────────────────

PATCH_SIZE = (96, 96, 96)

def extract_patch(image: np.ndarray, label: np.ndarray,
                  patch_size: tuple = PATCH_SIZE,
                  fg_bias: float = 0.33) -> tuple[np.ndarray, np.ndarray]:
    """
    Biased patch sampling:
      - With prob `fg_bias`: center patch on a random foreground voxel (label > 0)
      - Otherwise: uniform random patch
    Returns (image_patch, label_patch) of shape (C, *patch_size) and (1, *patch_size).
    """
    ph, pw, pd = patch_size
    H, W, D    = image.shape[1:]

    if np.random.rand() < fg_bias:
        fg_coords = np.argwhere(label[0] > 0)
        if fg_coords.shape[0] > 0:
            idx    = np.random.randint(0, fg_coords.shape[0])
            center = fg_coords[idx]
        else:
            center = np.array([H // 2, W // 2, D // 2])
        # Clamp center so patch stays in bounds
        cx = int(np.clip(center[0], ph // 2, H - ph // 2 - 1))
        cy = int(np.clip(center[1], pw // 2, W - pw // 2 - 1))
        cz = int(np.clip(center[2], pd // 2, D - pd // 2 - 1))
        x0, y0, z0 = cx - ph // 2, cy - pw // 2, cz - pd // 2
    else:
        x0 = np.random.randint(0, max(1, H - ph + 1))
        y0 = np.random.randint(0, max(1, W - pw + 1))
        z0 = np.random.randint(0, max(1, D - pd + 1))

    # Clamp start
    x0 = int(np.clip(x0, 0, H - ph))
    y0 = int(np.clip(y0, 0, W - pw))
    z0 = int(np.clip(z0, 0, D - pd))

    img_patch = image[:, x0:x0+ph, y0:y0+pw, z0:z0+pd]
    lbl_patch = label[:, x0:x0+ph, y0:y0+pw, z0:z0+pd]
    return img_patch.copy(), lbl_patch.copy()


# ─────────────────────────── Composed Transform ───────────────────────────

def augment(image: np.ndarray, label: np.ndarray,
            patch_size: tuple = PATCH_SIZE,
            training: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Full augmentation pipeline applied to a full preprocessed volume.
    Returns a (C, pH, pW, pD) image patch and (1, pH, pW, pD) label patch.

    Args:
        image: (4, H, W, D) float32 preprocessed image
        label: (1, H, W, D) uint8 label
        patch_size: output patch size
        training: if False, skips augmentation (extract center patch only)
    """
    if training:
        # --- Spatial ---
        image, label = random_flip(image, label)
        image, label = random_rotation(image, label, max_angle=15.0, p=0.4)
        image, label = random_scale(image, label, scale_range=(0.9, 1.1), p=0.3)

        # --- Patch sampling (biased) ---
        image, label = extract_patch(image, label, patch_size, fg_bias=0.33)

        # --- Intensity (applied after patch crop for efficiency) ---
        image = random_brightness(image, shift_range=0.1,  p=0.5)
        image = random_contrast(image,  contrast_range=(0.85, 1.15), p=0.5)
        image = random_gamma(image,     gamma_range=(0.7, 1.5),       p=0.3)
        image = random_gaussian_noise(image, noise_std_range=(0.0, 0.05), p=0.3)
    else:
        # Validation: deterministic center patch
        H, W, D = image.shape[1:]
        ph, pw, pd = patch_size
        x0 = (H - ph) // 2
        y0 = (W - pw) // 2
        z0 = (D - pd) // 2
        image = image[:, x0:x0+ph, y0:y0+pw, z0:z0+pd].copy()
        label = label[:, x0:x0+ph, y0:y0+pw, z0:z0+pd].copy()

    return image, label


# ─────────────────────────── Quick Test ───────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    dummy_img = np.random.randn(4, 192, 192, 144).astype(np.float32)
    dummy_lbl = np.zeros((1, 192, 192, 144), dtype=np.uint8)
    dummy_lbl[0, 80:120, 80:120, 60:90] = 1  # fake tumor

    img_p, lbl_p = augment(dummy_img, dummy_lbl, training=True)
    print(f"Image patch: {img_p.shape}, dtype={img_p.dtype}")
    print(f"Label patch: {lbl_p.shape}, dtype={lbl_p.dtype}, unique={np.unique(lbl_p)}")
    print("augment.py OK")