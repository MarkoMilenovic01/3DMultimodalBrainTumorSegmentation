from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def read_ids(txt_path: str | Path) -> List[str]:
    txt_path = Path(txt_path)
    ids = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
    return ids


class BratsNpy3D(Dataset):
    """
    Loads preprocessed BraTS volumes saved as .npy:
      - image: (C, H, W, D) float32
      - mask:  (H, W, D) int16/int64 labels in {0,1,2,3}

    Returns:
      x: torch.FloatTensor (C,H,W,D)
      y: torch.LongTensor  (H,W,D)
      meta: dict (optional)
    """

    def __init__(
        self,
        root: str | Path = "data/processed",
        split: str = "train",                 # "train" or "val"
        return_meta: bool = False,
        augment: bool = False,
        expect_channels: Optional[int] = None # set to 4 or 3 if you want strict checking
    ):
        self.root = Path(root)
        self.split = split
        self.return_meta = return_meta
        self.augment = augment
        self.expect_channels = expect_channels

        ids_path = self.root / f"{split}.txt"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing split file: {ids_path}")

        self.ids = read_ids(ids_path)

        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"

        if not self.img_dir.exists() or not self.msk_dir.exists():
            raise FileNotFoundError(f"Expected {self.img_dir} and {self.msk_dir} to exist")

    def __len__(self) -> int:
        return len(self.ids)

    # ---------- basic 3D aug (safe, simple) ----------
    def _augment(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # x: (C,H,W,D), y: (H,W,D)
        # Random flips along axes (H/W/D). Keep it simple and label-safe.
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :, :].copy()
            y = y[::-1, :, :].copy()
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1, :].copy()
            y = y[:, ::-1, :].copy()
        if np.random.rand() < 0.5:
            x = x[:, :, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        return x, y

    def __getitem__(self, idx: int):
        case_id = self.ids[idx]

        x_path = self.img_dir / f"{case_id}.npy"
        y_path = self.msk_dir / f"{case_id}.npy"

        x = np.load(x_path).astype(np.float32)    # (C,H,W,D)
        y = np.load(y_path).astype(np.int64)      # (H,W,D)

        if x.ndim != 4:
            raise ValueError(f"{case_id}: expected x.ndim=4, got {x.shape}")
        if y.ndim != 3:
            raise ValueError(f"{case_id}: expected y.ndim=3, got {y.shape}")

        if self.expect_channels is not None and x.shape[0] != self.expect_channels:
            raise ValueError(f"{case_id}: expected C={self.expect_channels}, got {x.shape[0]}")

        # optional: augment only for train
        if self.augment and self.split == "train":
            x, y = self._augment(x, y)

        # to torch
        x_t = torch.from_numpy(x)                 # float32
        y_t = torch.from_numpy(y).long()          # int64 for CE loss

        if self.return_meta:
            meta: Dict[str, Any] = {
                "id": case_id,
                "x_path": str(x_path),
                "y_path": str(y_path),
                "shape": tuple(x.shape),
                "labels": torch.unique(y_t).tolist(),
            }
            return x_t, y_t, meta

        return x_t, y_t


def main():
    ds = BratsNpy3D(root="data/processed", split="train", augment=True, return_meta=True, expect_channels=4)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    x, y, meta = next(iter(dl))
    print("x:", x.shape, x.dtype)  # (B,C,H,W,D)
    print("y:", y.shape, y.dtype)  # (B,H,W,D)
    print("labels:", torch.unique(y).tolist())
    print("meta:", meta["id"][0], meta["shape"][0])

if __name__ == "__main__":
    main()