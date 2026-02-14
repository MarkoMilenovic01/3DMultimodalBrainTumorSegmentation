from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import BratsNpy3D
from src.model import UNet3D


def to_conv3d(x: torch.Tensor, y: torch.Tensor | None = None):
    # x: (B,C,H,W,D) -> (B,C,D,H,W)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    if y is not None:
        # y: (B,H,W,D) -> (B,D,H,W)
        y = y.permute(0, 3, 1, 2).contiguous()
    return x, y


def overlay_mask(ax, base_img: np.ndarray, mask: np.ndarray, title: str):
    """
    base_img: (H,W) float
    mask:     (H,W) int with labels 0..3
    """
    ax.imshow(base_img, cmap="gray")
    ax.imshow(mask, alpha=0.35, vmin=0, vmax=3)  # color overlay
    ax.set_title(title)
    ax.axis("off")


@torch.no_grad()
def main():
    # ---- config ----
    root = "data/processed"
    ckpt_path = "checkpoints/unet3d_epoch20.pt"  # <-- change to your best checkpoint
    out_dir = Path("viz_val")
    out_dir.mkdir(parents=True, exist_ok=True)

    in_channels = 4
    num_classes = 4
    base = 16
    num_cases = 8  # how many cases to save

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- data ----
    ds = BratsNpy3D(root=root, split="val", augment=False, return_meta=True, expect_channels=in_channels)

    # ---- model ----
    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- loop ----
    for i in range(min(num_cases, len(ds))):
        x, y, meta = ds[i]  # x: (C,H,W,D), y:(H,W,D)
        case_id = meta["id"]

        xb = x.unsqueeze(0).to(device)  # (1,C,H,W,D)
        yb = y.unsqueeze(0).to(device)  # (1,H,W,D)
        xb, yb = to_conv3d(xb, yb)      # xb:(1,C,D,H,W), yb:(1,D,H,W)

        logits = model(xb)
        pred = torch.argmax(logits, dim=1)  # (1,D,H,W)

        # choose a slice with tumor (from GT), otherwise middle slice
        y_np = yb.squeeze(0).cpu().numpy()      # (D,H,W)
        pred_np = pred.squeeze(0).cpu().numpy() # (D,H,W)

        tumor_slices = np.where((y_np > 0).reshape(y_np.shape[0], -1).sum(axis=1) > 0)[0]
        if len(tumor_slices) > 0:
            z = int(tumor_slices[len(tumor_slices) // 2])
        else:
            z = y_np.shape[0] // 2

        # take one modality to display nicely (FLAIR usually best for edema)
        # your saved channel order (from earlier script): [T1, T1CE, T2, FLAIR]
        x_np = x.numpy()  # (C,H,W,D)
        flair_slice = x_np[3, :, :, z]  # (H,W)
        gt_slice = y.numpy()[:, :, z]   # (H,W) from original y (H,W,D)
        pred_slice = pred_np[z, :, :]   # (H,W) because pred is (D,H,W)

        # plot 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(flair_slice, cmap="gray")
        axes[0].set_title(f"{case_id} | FLAIR | z={z}")
        axes[0].axis("off")

        overlay_mask(axes[1], flair_slice, gt_slice, "GT mask overlay")
        overlay_mask(axes[2], flair_slice, pred_slice, "Pred mask overlay")

        fig.tight_layout()
        save_path = out_dir / f"{case_id}_z{z}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print("Saved:", save_path)


if __name__ == "__main__":
    main()
