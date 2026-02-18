"""
===============================================================================
BraTS 2020 PyTorch Dataset Loader
===============================================================================

High-Level Overview
-------------------
This file provides a PyTorch Dataset class for loading preprocessed BraTS data.

Key Features:
    - Loads .npy files created by the preprocessing pipeline
    - Supports train/validation splits
    - Optional data augmentation (random flips)
    - Returns PyTorch tensors ready for model training
    - Includes validation checks for data integrity

Usage Example:
    dataset = BratsNpy3D(root="data/processed", split="train", augment=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for images, masks in loader:
        # images: (batch, 4, 128, 128, 128)
        # masks:  (batch, 128, 128, 128)
        pass

===============================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# UTILITIES
# =============================================================================

def read_ids(txt_path: str | Path) -> List[str]:
    """
    Read case IDs from a text file (one ID per line).
    
    Used to load train.txt or val.txt which contain the dataset split.
    Empty lines are automatically filtered out.
    """
    txt_path = Path(txt_path)
    ids = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
    return ids


# =============================================================================
# DATASET CLASS
# =============================================================================

class BratsNpy3D(Dataset):
    """
    PyTorch Dataset for preprocessed BraTS brain tumor MRI data.
    
    Data Format:
        - Images: (4, H, W, D) float32 array with 4 MRI modalities
        - Masks:  (H, W, D) int64 array with labels {0, 1, 2, 3}
    
    Returns:
        x: torch.FloatTensor of shape (4, H, W, D) - MRI volumes
        y: torch.LongTensor of shape (H, W, D) - segmentation labels
        meta: dict with case info (optional, if return_meta=True)
    
    Arguments:
        root: Path to processed data directory
        split: "train" or "val" - determines which split to load
        return_meta: If True, also return metadata dict with case info
        augment: If True, apply random flips during training
        expect_channels: Validate that images have this many channels (e.g., 4)
    """

    def __init__(
        self,
        root: str | Path = "data/processed",
        split: str = "train",
        return_meta: bool = False,
        augment: bool = False,
        expect_channels: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.return_meta = return_meta
        self.augment = augment
        self.expect_channels = expect_channels

        # -------------------------
        # Load case IDs from split file
        # -------------------------
        ids_path = self.root / f"{split}.txt"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing split file: {ids_path}")
        
        self.ids = read_ids(ids_path)

        # -------------------------
        # Verify data directories exist
        # -------------------------
        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"

        if not self.img_dir.exists() or not self.msk_dir.exists():
            raise FileNotFoundError(
                f"Expected directories to exist:\n"
                f"  - {self.img_dir}\n"
                f"  - {self.msk_dir}"
            )

    def __len__(self) -> int:
        """Return the number of cases in this split."""
        return len(self.ids)

    # =========================================================================
    # DATA AUGMENTATION
    # =========================================================================

    def _augment(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random 3D flips for data augmentation.
        
        Augmentation Strategy:
            - Random flip along H axis (up/down)
            - Random flip along W axis (left/right)
            - Random flip along D axis (front/back)
            - Each flip has 50% probability
        
        This is "label-safe" meaning:
            - Flips preserve anatomical validity
            - Labels remain correctly aligned with images
            - No interpolation artifacts
        
        Args:
            x: Image array of shape (C, H, W, D)
            y: Mask array of shape (H, W, D)
        
        Returns:
            Augmented (x, y) pair with same shapes
        """
        # Flip along height axis (H)
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :, :].copy()  # Reverse H dimension, keep channels
            y = y[::-1, :, :].copy()      # Reverse H dimension
        
        # Flip along width axis (W)
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1, :].copy()  # Reverse W dimension
            y = y[:, ::-1, :].copy()
        
        # Flip along depth axis (D)
        if np.random.rand() < 0.5:
            x = x[:, :, :, ::-1].copy()  # Reverse D dimension
            y = y[:, :, ::-1].copy()
        
        return x, y

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def __getitem__(self, idx: int):
        """
        Load and return a single case.
        
        Pipeline:
            1. Get case ID from index
            2. Load image and mask .npy files
            3. Validate shapes and channels
            4. Apply augmentation (if enabled and training)
            5. Convert to PyTorch tensors
            6. Optionally return metadata
        
        Args:
            idx: Index into the dataset (0 to len-1)
        
        Returns:
            If return_meta=False: (x_tensor, y_tensor)
            If return_meta=True:  (x_tensor, y_tensor, meta_dict)
        """
        # -------------------------
        # Step 1: Get case ID
        # -------------------------
        case_id = self.ids[idx]

        # -------------------------
        # Step 2: Construct file paths
        # -------------------------
        x_path = self.img_dir / f"{case_id}.npy"
        y_path = self.msk_dir / f"{case_id}.npy"

        # -------------------------
        # Step 3: Load numpy arrays
        # -------------------------
        x = np.load(x_path).astype(np.float32)  # Image: (C, H, W, D)
        y = np.load(y_path).astype(np.int64)    # Mask:  (H, W, D)

        # -------------------------
        # Step 4: Validate dimensions
        # -------------------------
        if x.ndim != 4:
            raise ValueError(
                f"{case_id}: Image should be 4D (C,H,W,D), got shape {x.shape}"
            )
        
        if y.ndim != 3:
            raise ValueError(
                f"{case_id}: Mask should be 3D (H,W,D), got shape {y.shape}"
            )

        # Optional: Validate number of channels
        if self.expect_channels is not None and x.shape[0] != self.expect_channels:
            raise ValueError(
                f"{case_id}: Expected {self.expect_channels} channels, "
                f"got {x.shape[0]}"
            )

        # -------------------------
        # Step 5: Apply augmentation (train only)
        # -------------------------
        if self.augment and self.split == "train":
            x, y = self._augment(x, y)

        # -------------------------
        # Step 6: Convert to PyTorch tensors
        # -------------------------
        x_tensor = torch.from_numpy(x)        # float32 → FloatTensor
        y_tensor = torch.from_numpy(y).long() # int64 → LongTensor (required for CrossEntropyLoss)

        # -------------------------
        # Step 7: Return data (with or without metadata)
        # -------------------------
        if self.return_meta:
            meta: Dict[str, Any] = {
                "id": case_id,                      # Case identifier
                "x_path": str(x_path),              # Path to image file
                "y_path": str(y_path),              # Path to mask file
                "shape": tuple(x.shape),            # Image shape (C,H,W,D)
                "labels": torch.unique(y_tensor).tolist(),  # Unique labels present
            }
            return x_tensor, y_tensor, meta
        
        return x_tensor, y_tensor


# =============================================================================
# DEMO / TESTING
# =============================================================================

def main():
    """
    Quick test to verify the dataset loads correctly.
    
    This demonstrates:
        - Creating a dataset
        - Creating a DataLoader
        - Loading one batch
        - Inspecting shapes and metadata
    """
    
    # -------------------------
    # Create dataset
    # -------------------------
    dataset = BratsNpy3D(
        root="data/processed",
        split="train",
        augment=True,          # Enable augmentation
        return_meta=True,      # Get metadata for debugging
        expect_channels=4      # Validate 4 MRI modalities
    )
    
    print(f"Dataset size: {len(dataset)} cases")
    
    # -------------------------
    # Create DataLoader
    # -------------------------
    # Note: batch_size=1 for 3D data due to memory constraints
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,        # Parallel data loading
        pin_memory=True       # Faster GPU transfer
    )
    
    # -------------------------
    # Load one batch
    # -------------------------
    x, y, meta = next(iter(loader))
    
    # -------------------------
    # Print information
    # -------------------------
    print("\n--- Batch Information ---")
    print(f"Images (x): {x.shape} | dtype: {x.dtype}")
    print(f"  Shape breakdown: (batch, channels, height, width, depth)")
    print(f"  Expected: (1, 4, 128, 128, 128)")
    
    print(f"\nMasks (y): {y.shape} | dtype: {y.dtype}")
    print(f"  Shape breakdown: (batch, height, width, depth)")
    print(f"  Expected: (1, 128, 128, 128)")
    
    print(f"\nUnique labels in mask: {torch.unique(y).tolist()}")
    print(f"  0 = background, 1 = necrotic core, 2 = edema, 3 = enhancing tumor")
    
    print(f"\nMetadata:")
    print(f"  Case ID: {meta['id'][0]}")
    print(f"  Shape: {meta['shape'][0]}")
    
    print("\n✓ Dataset loading successful!")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()