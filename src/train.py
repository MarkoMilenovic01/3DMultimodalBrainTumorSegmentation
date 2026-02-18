from __future__ import annotations
import os
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import BratsNpy3D          # <- your Dataset class name
from src.model import UNet3D                # <- make sure your model.py exposes UNet3D
from src.losses import dice_ce_loss         # <- your loss function


def move_to_conv3d_format(x: torch.Tensor, y: torch.Tensor):
    # x: (B,C,H,W,D) -> (B,C,D,H,W)
    x = x.permute(0, 1, 4, 2, 3).contiguous()
    # y: (B,H,W,D) -> (B,D,H,W)
    y = y.permute(0, 3, 1, 2).contiguous()
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


def plot_history(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    # Dice plot
    plt.figure()
    plt.plot(history["epoch"], history["val_dice"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dice_curve.png", dpi=150)
    plt.close()


def main():
    # ---- config ----
    root = "data/processed"
    in_channels = 4
    num_classes = 4
    base = 16
    batch_size = 1
    lr = 2e-4
    weight_decay = 1e-4
    epochs = 20
    num_workers = 2

    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print("device:", device, "| amp:", use_amp)

    # ---- data ----
    train_ds = BratsNpy3D(root=root, split="train", augment=True, expect_channels=in_channels)
    val_ds   = BratsNpy3D(root=root, split="val", augment=False, expect_channels=in_channels)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ---- model ----
    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- logging ----
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": []}
    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_dice"])

    # ---- train loop ----
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y = move_to_conv3d_format(x, y)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = dice_ce_loss(logits, y, num_classes=num_classes, dice_weight=1.0, ce_weight=1.0, ignore_bg=True)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item()

        train_loss /= max(1, len(train_dl))

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x, y = move_to_conv3d_format(x, y)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = dice_ce_loss(logits, y, num_classes=num_classes, dice_weight=1.0, ce_weight=1.0, ignore_bg=True)

                val_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                val_dice += dice_score_multiclass(pred, y, num_classes=num_classes)

        val_loss /= max(1, len(val_dl))
        val_dice /= max(1, len(val_dl))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_dice])

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict()},
            out_dir / f"unet3d_epoch{epoch:02d}.pt"
        )

    plot_history(history, out_dir)
    print("Saved:", csv_path, "and plots in", out_dir)


if __name__ == "__main__":
    main()
