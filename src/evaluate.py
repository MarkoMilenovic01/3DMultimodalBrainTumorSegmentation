from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import BratsNpy3D
from src.model import UNet3D
from src.losses import dice_ce_loss


def move_to_conv3d_format(x: torch.Tensor, y: torch.Tensor):
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B,C,H,W,D)->(B,C,D,H,W)
    y = y.permute(0, 3, 1, 2).contiguous()      # (B,H,W,D)->(B,D,H,W)
    return x, y


@torch.no_grad()
def dice_score_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dices.append((2 * inter + eps) / (denom + eps))
    return torch.stack(dices).mean().item()


def main():
    # --- config ---
    root = "data/processed"
    ckpt_path = "checkpoints/unet3d_epoch20.pt"  # <-- change to your checkpoint
    in_channels = 4
    num_classes = 4
    base = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # --- data ---
    ds = BratsNpy3D(root=root, split="val", augment=False, expect_channels=in_channels)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # --- model ---
    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- evaluation ---
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y = move_to_conv3d_format(x, y)

            logits = model(x)
            loss = dice_ce_loss(logits, y, num_classes=num_classes, dice_weight=1.0, ce_weight=1.0, ignore_bg=True)

            pred = torch.argmax(logits, dim=1)  # (B,D,H,W)
            dice = dice_score_multiclass(pred, y, num_classes=num_classes)

            total_loss += loss.item()
            total_dice += dice

    total_loss /= max(1, len(dl))
    total_dice /= max(1, len(dl))

    print(f"VAL | loss={total_loss:.4f} | dice={total_dice:.4f}")


if __name__ == "__main__":
    main()
