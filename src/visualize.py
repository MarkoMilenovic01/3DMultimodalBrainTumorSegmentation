"""
visualize.py — BraTS 2020 Visualization Utilities

Functions:
  - plot_modalities_slice     : side-by-side 4 modalities + seg for one axial slice
  - plot_label_overlay        : overlay tumor mask on FLAIR
  - plot_augmentation_compare : before/after augmentation
  - plot_class_distribution   : bar chart of label class volumes
  - plot_intensity_histograms : per-modality intensity distributions
  - plot_patch_sample         : visualize a sampled patch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional

# Label index → name/color
LABEL_INFO = {
    0: ("Background",     "#000000"),
    1: ("NCR/NET",        "#FF0000"),  # Necrotic core
    2: ("Edema",          "#00FF00"),  # Peritumoral edema
    3: ("Enhancing",      "#FFFF00"),  # Enhancing tumor
}
MODALITY_NAMES = ["FLAIR", "T1", "T1CE", "T2"]


def _pick_slice(vol: np.ndarray, axis: int = 2, idx: Optional[int] = None) -> np.ndarray:
    """Extract a 2D slice along an axis."""
    if idx is None:
        idx = vol.shape[axis] // 2
    if axis == 0:   return vol[idx]
    elif axis == 1: return vol[:, idx]
    else:           return vol[:, :, idx]


def _seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """Convert 2D label map → RGB for display."""
    from matplotlib.colors import hex2color
    rgb = np.zeros((*seg.shape, 3), dtype=np.float32)
    for label_id, (_, color) in LABEL_INFO.items():
        mask = seg == label_id
        c = hex2color(color)
        for ch in range(3):
            rgb[..., ch][mask] = c[ch]
    return rgb


# ─────────────────────────── Core Plots ───────────────────────────

def plot_modalities_slice(
    image: np.ndarray,   # (4, H, W, D)
    label: np.ndarray,   # (1, H, W, D) or (H, W, D)
    slice_idx: Optional[int] = None,
    axis: int = 2,
    save_path: Optional[str] = None,
    title: str = "BraTS Modalities",
):
    """Plot all 4 modalities + segmentation for one slice."""
    if label.ndim == 4:
        label = label[0]

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (mod_name, ax) in enumerate(zip(MODALITY_NAMES, axes[:4])):
        sl = _pick_slice(image[i], axis, slice_idx)
        vmin, vmax = np.percentile(sl[sl != 0], [2, 98]) if sl.any() else (0, 1)
        ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(mod_name, fontsize=10)
        ax.axis("off")

    # Segmentation grayscale
    seg_sl = _pick_slice(label, axis, slice_idx)
    axes[4].imshow(seg_sl.T, cmap="nipy_spectral", origin="lower", vmin=0, vmax=3)
    axes[4].set_title("Segmentation", fontsize=10)
    axes[4].axis("off")

    # Overlay: FLAIR + colored seg
    flair_sl = _pick_slice(image[0], axis, slice_idx)
    flair_norm = np.clip((flair_sl - flair_sl.min()) / (flair_sl.max() - flair_sl.min() + 1e-8), 0, 1)
    flair_rgb  = np.stack([flair_norm]*3, axis=-1)
    seg_rgb    = _seg_to_rgb(seg_sl)
    tumor_mask = (seg_sl > 0)[..., np.newaxis]
    overlay    = flair_rgb * 0.55 + seg_rgb * 0.45
    overlay[~tumor_mask[..., 0]] = flair_rgb[~tumor_mask[..., 0]]
    axes[5].imshow(overlay.transpose(1, 0, 2), origin="lower")
    axes[5].set_title("Overlay", fontsize=10)
    axes[5].axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=n) for _, (n, c) in LABEL_INFO.items() if _ > 0]
    axes[5].legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_augmentation_compare(
    image_orig: np.ndarray, label_orig: np.ndarray,
    image_aug:  np.ndarray, label_aug:  np.ndarray,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """2-row comparison: original vs augmented (FLAIR + seg overlay)."""
    def _make_overlay(img, lbl):
        lbl = lbl[0] if lbl.ndim == 4 else lbl
        sl  = _pick_slice(img[0], 2, slice_idx)
        ls  = _pick_slice(lbl,    2, slice_idx)
        norm = np.clip((sl - sl.min()) / (sl.max() - sl.min() + 1e-8), 0, 1)
        rgb  = np.stack([norm]*3, axis=-1)
        seg_rgb = _seg_to_rgb(ls)
        mask = (ls > 0)[..., np.newaxis]
        out  = rgb * 0.55 + seg_rgb * 0.45
        out[~mask[..., 0]] = rgb[~mask[..., 0]]
        return out.transpose(1, 0, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(_make_overlay(image_orig, label_orig), origin="lower")
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(_make_overlay(image_aug, label_aug), origin="lower")
    axes[1].set_title("Augmented", fontsize=12)
    axes[1].axis("off")

    plt.suptitle("FLAIR + Segmentation Overlay", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(
    label: np.ndarray,  # (1,H,W,D) or (H,W,D)
    case_id: str = "",
    save_path: Optional[str] = None,
):
    """Bar chart showing voxel count per class."""
    lbl = label[0] if label.ndim == 4 else label
    counts = {lid: int((lbl == lid).sum()) for lid in LABEL_INFO}
    names  = [LABEL_INFO[lid][0] for lid in LABEL_INFO]
    values = [counts[lid] for lid in LABEL_INFO]
    colors = [LABEL_INFO[lid][1] for lid in LABEL_INFO]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Voxel Count")
    ax.set_title(f"Class Distribution{' — ' + case_id if case_id else ''}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_intensity_histograms(
    image: np.ndarray,  # (4, H, W, D)
    label: np.ndarray,  # (1, H, W, D)
    case_id: str = "",
    save_path: Optional[str] = None,
):
    """Per-modality intensity histogram, separating brain vs. background."""
    brain_mask = (image > 0).any(axis=0)  # (H, W, D)
    lbl = label[0] if label.ndim == 4 else label

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Intensity Distributions{' — ' + case_id if case_id else ''}", fontsize=13)

    for i, (ax, mod) in enumerate(zip(axes.flatten(), MODALITY_NAMES)):
        vol = image[i]
        brain_vals = vol[brain_mask]
        tumor_vals = vol[lbl > 0]

        ax.hist(brain_vals, bins=100, color="steelblue", alpha=0.7, label="Brain",
                density=True, histtype="stepfilled")
        if tumor_vals.size > 0:
            ax.hist(tumor_vals, bins=100, color="crimson", alpha=0.7, label="Tumor",
                    density=True, histtype="stepfilled")
        ax.set_title(mod)
        ax.set_xlabel("Intensity (z-scored)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_patch_sample(
    image: np.ndarray,  # (4, 96, 96, 96)
    label: np.ndarray,  # (1, 96, 96, 96)
    save_path: Optional[str] = None,
):
    """Visualize center slice of a 96^3 patch (all modalities + overlay)."""
    plot_modalities_slice(image, label, save_path=save_path, title="Sampled 96³ Patch")


def plot_three_planes(
    image: np.ndarray,  # (4, H, W, D)
    label: np.ndarray,  # (1, H, W, D)
    modality_idx: int = 0,
    save_path: Optional[str] = None,
):
    """Plot axial, coronal, sagittal views for one modality + segmentation."""
    lbl = label[0] if label.ndim == 4 else label
    vol = image[modality_idx]
    mod_name = MODALITY_NAMES[modality_idx]

    axes_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
    axis_ids    = [2, 1, 0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"{mod_name} — Three Planes", fontsize=13, fontweight="bold")

    for col, (ax_lbl, ax_id) in enumerate(zip(axes_labels, axis_ids)):
        sl  = _pick_slice(vol, ax_id)
        ls  = _pick_slice(lbl, ax_id)
        vmin, vmax = np.percentile(sl[sl != 0], [2, 98]) if sl.any() else (0, 1)

        axes[0, col].imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(ax_lbl, fontsize=10)
        axes[0, col].axis("off")

        # Overlay
        norm = np.clip((sl - sl.min()) / (sl.max() - sl.min() + 1e-8), 0, 1)
        rgb  = np.stack([norm]*3, axis=-1)
        seg_rgb = _seg_to_rgb(ls)
        mask = (ls > 0)[..., np.newaxis]
        overlay = rgb * 0.55 + seg_rgb * 0.45
        overlay[~mask[..., 0]] = rgb[~mask[..., 0]]
        axes[1, col].imshow(overlay.transpose(1, 0, 2), origin="lower")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Modality", fontsize=9)
    axes[1, 0].set_ylabel("Overlay", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────── Quick Test ───────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    data_dir = Path("data/preprocessed/train")
    files    = sorted(data_dir.glob("*.npz"))

    if not files:
        print("No preprocessed data. Generating dummy data for demo...")
        image = np.random.randn(4, 192, 192, 144).astype(np.float32)
        label = np.zeros((1, 192, 192, 144), dtype=np.uint8)
        label[0, 80:110, 80:110, 70:90] = 1
        label[0, 90:105, 90:105, 72:88] = 3
        label[0, 75:115, 75:115, 68:92] = 2
        case_id = "demo"
    else:
        d = np.load(files[0])
        image, label = d["image"], d["label"]
        case_id = files[0].stem

    print(f"Visualizing: {case_id}")
    plot_modalities_slice(image, label, title=f"Case: {case_id}")
    plot_class_distribution(label, case_id=case_id)
    plot_intensity_histograms(image, label, case_id=case_id)
    plot_three_planes(image, label)
    print("visualize.py OK")