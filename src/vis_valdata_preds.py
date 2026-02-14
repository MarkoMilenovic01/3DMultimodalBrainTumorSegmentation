from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def overlay_mask(ax, base_img: np.ndarray, mask: np.ndarray, title: str):
    ax.imshow(base_img, cmap="gray")
    ax.imshow(mask, alpha=0.35, vmin=0, vmax=3)
    ax.set_title(title)
    ax.axis("off")


def pick_best_slice(pred: np.ndarray) -> int:
    """
    pred: (D,H,W)
    pick slice with most non-zero predicted pixels; fallback to middle slice.
    """
    per_slice = (pred > 0).reshape(pred.shape[0], -1).sum(axis=1)
    if per_slice.max() == 0:
        return pred.shape[0] // 2
    return int(per_slice.argmax())


def main():
    img_dir = Path("data/processed_val/images")   # (C,H,W,D) .npy
    pred_dir = Path("predictions_val")            # (D,H,W) .npy
    out_dir = Path("viz_valdata")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print("Missing:", img_dir)
        return
    if not pred_dir.exists():
        print("Missing:", pred_dir)
        return

    files = sorted(img_dir.glob("*.npy"))
    print("Found images:", len(files))

    saved = 0
    for f in files[:50]:  # visualize first 50, adjust as you like
        case_id = f.stem
        pred_path = pred_dir / f"{case_id}_pred.npy"

        if not pred_path.exists():
            # You haven't predicted this case yet
            continue

        x = np.load(f).astype(np.float32)              # (C,H,W,D)
        pred = np.load(pred_path).astype(np.uint8)     # (D,H,W)

        # sanity
        if x.ndim != 4:
            print("Bad x shape:", case_id, x.shape)
            continue
        if pred.ndim != 3:
            print("Bad pred shape:", case_id, pred.shape)
            continue

        z = pick_best_slice(pred)  # slice with most predicted tumor

        # your channel order (from preprocessing): [T1, T1CE, T2, FLAIR]
        flair = x[3, :, :, z]          # (H,W)
        pred_slice = pred[z, :, :]     # (H,W)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(flair, cmap="gray")
        axes[0].set_title(f"{case_id} | FLAIR | z={z}")
        axes[0].axis("off")

        overlay_mask(axes[1], flair, pred_slice, "Pred mask overlay")

        fig.tight_layout()
        save_path = out_dir / f"{case_id}_z{z}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print("Saved:", save_path)
        saved += 1

    if saved == 0:
        print("\nSaved 0 images.")
        print("Most likely reasons:")
        print("1) predictions_val/ is empty (you didn't run predict_valdata)")
        print("2) file names don't match between processed_val and predictions_val")
        print("Check sample names in both folders.")
    else:
        print(f"\nDone. Saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
