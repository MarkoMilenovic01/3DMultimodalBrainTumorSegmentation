"""
dataset.py — BraTS 2020 PyTorch Dataset
Loads preprocessed .npz files and applies augmentation on-the-fly.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
from augment import augment, PATCH_SIZE

PREPROCESSED_ROOT = Path("data/preprocessed/train")


# ─────────────────────────── Dataset ───────────────────────────

class BraTSDataset(Dataset):
    """
    Loads preprocessed BraTS2020 .npz files.
    Each .npz contains:
        image: (4, 192, 192, 144) float32
        label: (1, 192, 192, 144) uint8

    During training, augmentation + biased patch sampling is applied per-call.
    During validation, a deterministic center patch is returned.
    """

    def __init__(
        self,
        data_dir: Path = PREPROCESSED_ROOT,
        patch_size: tuple = PATCH_SIZE,
        training: bool = True,
        case_ids: Optional[list] = None,
    ):
        self.data_dir   = Path(data_dir)
        self.patch_size = patch_size
        self.training   = training

        all_files = sorted(self.data_dir.glob("*.npz"))
        if case_ids is not None:
            all_files = [f for f in all_files if f.stem in case_ids]

        self.files = all_files
        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found in {self.data_dir}. "
                "Run preprocess.py first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data    = np.load(self.files[idx])
        image   = data["image"]                   # (4, H, W, D) float32
        label   = data["label"].astype(np.uint8)  # (1, H, W, D) uint8
        case_id = self.files[idx].stem

        image, label = augment(image, label,
                               patch_size=self.patch_size,
                               training=self.training)

        return {
            "image"  : torch.from_numpy(image).float(),    # (4, 96, 96, 96)
            "label"  : torch.from_numpy(label).long(),     # (1, 96, 96, 96)
            "case_id": case_id,
        }


# ─────────────────────────── Train/Val Split ───────────────────────────

def get_train_val_split(
    data_dir: Path = PREPROCESSED_ROOT,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Return (train_ids, val_ids) lists of case stems."""
    all_ids = sorted([f.stem for f in Path(data_dir).glob("*.npz")])
    rng     = np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n_val   = max(1, int(len(all_ids) * val_ratio))
    val_ids = all_ids[:n_val]
    trn_ids = all_ids[n_val:]
    return trn_ids, val_ids


def get_dataloaders(
    data_dir: Path = PREPROCESSED_ROOT,
    patch_size: tuple = PATCH_SIZE,
    batch_size: int = 2,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).
    batch_size=2 is safe for RTX 4060 8GB with 96^3 patches.
    """
    train_ids, val_ids = get_train_val_split(data_dir, val_ratio, seed)

    train_ds = BraTSDataset(data_dir, patch_size, training=True,  case_ids=train_ids)
    val_ds   = BraTSDataset(data_dir, patch_size, training=False, case_ids=val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


# ─────────────────────────── Quick Test ───────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path("data/preprocessed/train")
    if not data_dir.exists():
        print("No preprocessed data found. Run preprocess.py first.")
    else:
        train_ids, val_ids = get_train_val_split(data_dir)
        print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

        ds = BraTSDataset(data_dir, training=True)
        sample = ds[0]
        print(f"Image shape : {sample['image'].shape}")
        print(f"Label shape : {sample['label'].shape}")
        print(f"Label unique: {sample['label'].unique()}")
        print(f"Case ID     : {sample['case_id']}")
        print("dataset.py OK")