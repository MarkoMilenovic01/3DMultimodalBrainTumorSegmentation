"""
===============================================================================
BraTS 2020 Data Visualization
===============================================================================

High-Level Overview
-------------------
This file visualizes preprocessed BraTS data to verify preprocessing quality.

Visualization Features:
    - Display all 4 MRI modalities (T1, T1ce, T2, FLAIR)
    - Show segmentation mask with color-coded tumor regions
    - View any slice (axial/coronal/sagittal)
    - Overlay segmentation on MRI for context
    - Interactive case browsing

Usage Examples:
    # Visualize a specific case
    visualize_case("data/processed", "BraTS20_Training_001")
    
    # Visualize random cases from train set
    visualize_random_cases("data/processed", split="train", n=3)
    
    # Save visualizations to files
    visualize_case("data/processed", "BraTS20_Training_001", save_path="viz.png")

===============================================================================
"""

from pathlib import Path
from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CORE VISUALIZATION
# =============================================================================

def visualize_case(
    data_root: str | Path,
    case_id: str,
    slice_idx: Optional[int] = None,
    plane: Literal["axial", "coronal", "sagittal"] = "axial",
    save_path: Optional[str] = None,
    figsize: tuple = (20, 4),
):
    """
    Visualize one BraTS case: 4 modalities + segmentation mask.
    
    This creates a 2-row figure:
        Row 1: T1, T1ce, T2, FLAIR (grayscale MRI slices)
        Row 2: Mask overlays on each modality (colored tumor regions)
    
    Args:
        data_root: Path to processed data directory
        case_id: Case identifier (e.g., "BraTS20_Training_001")
        slice_idx: Which slice to show (None = middle slice)
        plane: Viewing plane - "axial" (top-down), "coronal" (front), "sagittal" (side)
        save_path: If provided, save figure to this path instead of showing
        figsize: Figure size in inches (width, height)
    
    Slice Planes Explained:
        - axial: Horizontal slices (looking down at head) - DEFAULT
        - coronal: Vertical slices (looking at face)
        - sagittal: Vertical slices (looking from side)
    """
    
    root = Path(data_root)
    
    # -------------------------
    # Load data files
    # -------------------------
    img_path = root / "images" / f"{case_id}.npy"
    mask_path = root / "masks" / f"{case_id}.npy"
    
    if not img_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Case {case_id} not found in {data_root}")
    
    # Load arrays
    # img: (4, H, W, D) - 4 MRI modalities
    # mask: (H, W, D) - segmentation labels {0, 1, 2, 3}
    img = np.load(img_path)    # shape: (4, 128, 128, 128)
    mask = np.load(mask_path)  # shape: (128, 128, 128)
    
    # -------------------------
    # Select viewing plane and slice
    # -------------------------
    # Extract 2D slice from 3D volume based on viewing plane
    # mask shape: (H, W, D)
    if plane == "axial":
        # Top-down view (most common for brain imaging)
        axis = 2  # Slice along depth (D) dimension
        default_slice = mask.shape[2] // 2  # Middle slice
    elif plane == "coronal":
        # Front view
        axis = 0  # Slice along height (H) dimension
        default_slice = mask.shape[0] // 2
    elif plane == "sagittal":
        # Side view
        axis = 1  # Slice along width (W) dimension
        default_slice = mask.shape[1] // 2
    else:
        raise ValueError(f"Invalid plane: {plane}. Use 'axial', 'coronal', or 'sagittal'")
    
    # Use provided slice or default to middle
    if slice_idx is None:
        slice_idx = default_slice
    
    # -------------------------
    # FIX 2: Validate slice bounds
    # -------------------------
    max_idx = mask.shape[axis] - 1
    if not (0 <= slice_idx <= max_idx):
        raise ValueError(
            f"slice_idx must be in [0, {max_idx}] for plane={plane}, "
            f"but got {slice_idx}"
        )
    
    # -------------------------
    # Extract 2D slices for each modality
    # -------------------------
    # np.take extracts the specified slice along the given axis
    slices = [np.take(img[i], slice_idx, axis=axis) for i in range(4)]
    mask_slice = np.take(mask, slice_idx, axis=axis)
    
    # -------------------------
    # Create figure with 2 rows
    # -------------------------
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(
        f"{case_id} - {plane.capitalize()} Plane - Slice {slice_idx}",
        fontsize=16,
        fontweight="bold"
    )
    
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    
    # -------------------------
    # Row 1: Display MRI modalities (grayscale)
    # -------------------------
    for i, (ax, name, slice_2d) in enumerate(zip(axes[0], modality_names, slices)):
        ax.imshow(slice_2d, cmap="gray", origin="lower")
        ax.set_title(f"{name}", fontsize=14, fontweight="bold")
        ax.axis("off")
    
    # -------------------------
    # Row 2: Display segmentation overlays (colored masks on MRI)
    # -------------------------
    for i, (ax, name, slice_2d) in enumerate(zip(axes[1], modality_names, slices)):
        # Show MRI as background
        ax.imshow(slice_2d, cmap="gray", origin="lower")
        
        # -------------------------
        # FIX 3: True transparency for background
        # -------------------------
        # Overlay segmentation mask with transparency
        # Only overlay where tumor exists (mask > 0)
        mask_colored = create_colored_mask(mask_slice)
        alpha = (mask_slice > 0).astype(np.float32) * 0.5  # 50% transparent for tumor only
        ax.imshow(mask_colored, alpha=alpha, origin="lower")
        
        ax.set_title(f"{name} + Mask", fontsize=14)
        ax.axis("off")
    
    # -------------------------
    # Add legend explaining colors
    # -------------------------
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                   markersize=10, label='Background'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Necrotic Core (1)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='Edema (2)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Enhancing Tumor (3)'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        fontsize=11,
        frameon=True
    )
    
    plt.tight_layout()
    
    # -------------------------
    # Save or show
    # -------------------------
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_colored_mask(mask_slice: np.ndarray) -> np.ndarray:
    """
    Convert integer mask to RGB colored mask for visualization.
    
    Color Mapping:
        0 (background) → transparent (handled by alpha in imshow)
        1 (necrotic core) → red
        2 (edema) → green
        3 (enhancing tumor) → blue
    
    Args:
        mask_slice: 2D array of shape (H, W) with labels {0, 1, 2, 3}
    
    Returns:
        RGB array of shape (H, W, 3) with values in [0, 1]
    """
    H, W = mask_slice.shape
    colored = np.zeros((H, W, 3), dtype=np.float32)
    
    # Red channel: necrotic core (label 1)
    colored[mask_slice == 1, 0] = 1.0  # Full red
    
    # Green channel: edema (label 2)
    colored[mask_slice == 2, 1] = 1.0  # Full green
    
    # Blue channel: enhancing tumor (label 3)
    colored[mask_slice == 3, 2] = 1.0  # Full blue
    
    return colored


# =============================================================================
# BATCH VISUALIZATION
# =============================================================================

def visualize_random_cases(
    data_root: str | Path,
    split: Literal["train", "val"] = "train",
    n: int = 3,
    plane: Literal["axial", "coronal", "sagittal"] = "axial",
    seed: Optional[int] = None,  # FIX 5: Add reproducibility option
):
    """
    Visualize multiple random cases from a dataset split.
    
    Useful for:
        - Verifying preprocessing worked correctly
        - Exploring dataset diversity
        - Quality checking augmentation
    
    Args:
        data_root: Path to processed data directory
        split: Which split to sample from ("train" or "val")
        n: Number of cases to visualize
        plane: Viewing plane for all cases
        seed: Random seed for reproducible sampling (optional)
    """
    root = Path(data_root)
    
    # -------------------------
    # FIX 5: Set random seed if provided
    # -------------------------
    if seed is not None:
        np.random.seed(seed)
    
    # -------------------------
    # Load case IDs from split file
    # -------------------------
    split_file = root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    case_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    
    if len(case_ids) == 0:
        raise ValueError(f"No cases found in {split}.txt")
    
    # -------------------------
    # Sample random cases
    # -------------------------
    n = min(n, len(case_ids))  # Don't sample more than available
    sampled = np.random.choice(case_ids, size=n, replace=False)
    
    print(f"\nVisualizing {n} random cases from {split} split:")
    print(f"Plane: {plane}")
    if seed is not None:
        print(f"Random seed: {seed}")
    print("-" * 60)
    
    # -------------------------
    # Visualize each case
    # -------------------------
    for i, case_id in enumerate(sampled, 1):
        print(f"[{i}/{n}] {case_id}")
        visualize_case(data_root, case_id, plane=plane)


def visualize_case_comparison(
    data_root: str | Path,
    case_id: str,
    slice_indices: Optional[tuple[int, int, int]] = None,
    save_path: Optional[str] = None,
):
    """
    Show the same anatomical region in all 3 viewing planes (axial, coronal, sagittal).
    
    This creates a 3-column comparison to understand 3D tumor structure.
    
    Args:
        data_root: Path to processed data directory
        case_id: Case to visualize
        slice_indices: Tuple of (axial_slice, coronal_slice, sagittal_slice).
                      If None, uses middle slice for each plane.
        save_path: Optional path to save figure
    """
    root = Path(data_root)
    
    # -------------------------
    # Load data
    # -------------------------
    img = np.load(root / "images" / f"{case_id}.npy")
    mask = np.load(root / "masks" / f"{case_id}.npy")
    
    # -------------------------
    # FIX 4: Use appropriate middle slice for each plane
    # -------------------------
    if slice_indices is None:
        # Each plane gets its own middle slice
        axial_slice = mask.shape[2] // 2    # Middle of D (depth)
        coronal_slice = mask.shape[0] // 2  # Middle of H (height)
        sagittal_slice = mask.shape[1] // 2 # Middle of W (width)
    else:
        axial_slice, coronal_slice, sagittal_slice = slice_indices
    
    # -------------------------
    # Extract slices from each plane
    # -------------------------
    # Use T1ce (index 1) as representative modality for display
    
    # Axial: top-down (slice along D)
    axial_img = img[1, :, :, axial_slice]
    axial_mask = mask[:, :, axial_slice]
    
    # Coronal: front view (slice along H)
    coronal_img = img[1, coronal_slice, :, :]
    coronal_mask = mask[coronal_slice, :, :]
    
    # Sagittal: side view (slice along W)
    sagittal_img = img[1, :, sagittal_slice, :]
    sagittal_mask = mask[:, sagittal_slice, :]
    
    # -------------------------
    # Create comparison figure
    # -------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"{case_id} - Multi-Plane View\n"
        f"Slices: Axial={axial_slice}, Coronal={coronal_slice}, Sagittal={sagittal_slice}",
        fontsize=16,
        fontweight="bold"
    )
    
    planes = ["Axial (Top-Down)", "Coronal (Front)", "Sagittal (Side)"]
    images = [axial_img, coronal_img, sagittal_img]
    masks = [axial_mask, coronal_mask, sagittal_mask]
    
    for i, (plane_name, img_slice, mask_slice) in enumerate(zip(planes, images, masks)):
        # Row 1: MRI only
        axes[0, i].imshow(img_slice, cmap="gray", origin="lower")
        axes[0, i].set_title(f"{plane_name}\nT1ce", fontsize=12, fontweight="bold")
        axes[0, i].axis("off")
        
        # Row 2: MRI + mask overlay (with true transparency)
        axes[1, i].imshow(img_slice, cmap="gray", origin="lower")
        mask_colored = create_colored_mask(mask_slice)
        alpha = (mask_slice > 0).astype(np.float32) * 0.5
        axes[1, i].imshow(mask_colored, alpha=alpha, origin="lower")
        axes[1, i].set_title(f"{plane_name}\nWith Segmentation", fontsize=12)
        axes[1, i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# STATISTICS AND ANALYSIS
# =============================================================================

def print_case_statistics(data_root: str | Path, case_id: str):
    """
    Print detailed statistics about a case.
    
    Information Displayed:
        - Data shapes
        - Intensity ranges for each modality
        - Label distribution (voxel counts per class)
        - Tumor volume percentage
    """
    root = Path(data_root)
    
    img = np.load(root / "images" / f"{case_id}.npy")
    mask = np.load(root / "masks" / f"{case_id}.npy")
    
    print(f"\n{'='*60}")
    print(f"Case: {case_id}")
    print(f"{'='*60}")
    
    # -------------------------
    # Shape information
    # -------------------------
    print(f"\nShape Information:")
    print(f"  Image: {img.shape} (channels, height, width, depth)")
    print(f"  Mask:  {mask.shape} (height, width, depth)")
    
    # -------------------------
    # Intensity statistics per modality
    # -------------------------
    modalities = ["T1", "T1ce", "T2", "FLAIR"]
    print(f"\nIntensity Ranges (after normalization):")
    for i, name in enumerate(modalities):
        brain = img[i] > 0  # Only brain voxels
        if brain.sum() > 0:
            print(f"  {name:6s}: [{img[i][brain].min():.3f}, {img[i][brain].max():.3f}]")
    
    # -------------------------
    # Label distribution
    # -------------------------
    print(f"\nSegmentation Label Distribution:")
    total_voxels = mask.size
    
    for label in range(4):
        count = (mask == label).sum()
        percentage = 100 * count / total_voxels
        
        label_names = {
            0: "Background",
            1: "Necrotic Core",
            2: "Edema",
            3: "Enhancing Tumor"
        }
        
        print(f"  Label {label} ({label_names[label]:16s}): "
              f"{count:7d} voxels ({percentage:5.2f}%)")
    
    # -------------------------
    # Tumor metrics
    # -------------------------
    tumor_voxels = (mask > 0).sum()
    tumor_percentage = 100 * tumor_voxels / total_voxels
    
    print(f"\nTumor Metrics:")
    print(f"  Total tumor voxels: {tumor_voxels}")
    print(f"  Tumor percentage:   {tumor_percentage:.2f}%")
    print(f"{'='*60}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Demo script showing various visualization options.
    
    Uncomment the examples you want to run.
    """
    
    DATA_ROOT = "data/processed"
    
    # -------------------------
    # Example 1: Visualize a specific case
    # -------------------------
    print("Example 1: Single case visualization")
    visualize_case(
        DATA_ROOT,
        "BraTS20_Training_005",  # Change to your case ID
        plane="axial"
    )
    
    # -------------------------
    # Example 2: Multi-plane comparison
    # -------------------------
    print("Example 2: Multi-plane comparison")
    visualize_case_comparison(
        DATA_ROOT,
        "BraTS20_Training_001"
    )
    
    # -------------------------
    # Example 3: Random cases from train set (reproducible)
    # -------------------------
    print("Example 3: Random training cases")
    visualize_random_cases(
        DATA_ROOT,
        split="train",
        n=3,
        plane="axial",
    )
    
    # -------------------------
    # Example 4: Print detailed statistics
    # -------------------------
    print_case_statistics(DATA_ROOT, "BraTS20_Training_001")
    
    # -------------------------
    # Example 5: Save visualization to file
    # -------------------------
    visualize_case(
        DATA_ROOT,
        "BraTS20_Training_001",
        save_path="visualization.png"
    )


if __name__ == "__main__":
    main()