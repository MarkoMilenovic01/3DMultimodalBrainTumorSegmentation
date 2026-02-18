from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from src.model import UNet3D


def to_conv3d(x: torch.Tensor):
    return x.permute(0, 1, 4, 2, 3).contiguous()  # (B,C,H,W,D)->(B,C,D,H,W)


@torch.no_grad()
def main():
    ckpt_path = "checkpoints/unet3d_epoch20.pt"  # change to your checkpoint
    in_channels = 4
    num_classes = 4
    base = 16

    img_dir = Path("data/processed_val/images")
    out_dir = Path("predictions_val")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = sorted(img_dir.glob("*.npy"))
    print("Found", len(files), "validation cases")

    for f in files:
        case_id = f.stem
        x = np.load(f).astype(np.float32)                 # (C,H,W,D)
        x = torch.from_numpy(x).unsqueeze(0).to(device)   # (1,C,H,W,D)
        x = to_conv3d(x)

        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0)     # (D,H,W)

        np.save(out_dir / f"{case_id}_pred.npy", pred.cpu().numpy().astype(np.uint8))
        print("[OK]", case_id)

    print("Saved predictions to:", out_dir)


if __name__ == "__main__":
    main()
