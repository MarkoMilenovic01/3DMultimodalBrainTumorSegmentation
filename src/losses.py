from __future__ import annotations
import torch
import torch.nn.functional as F


def dice_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
    ignore_bg: bool = True
) -> torch.Tensor:
    """
    logits:  (B, C, D, H, W)
    targets: (B, D, H, W) long in [0..C-1]
    """
    probs = F.softmax(logits, dim=1)  # (B,C,D,H,W)
    onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    if ignore_bg and num_classes > 1:
        probs = probs[:, 1:]
        onehot = onehot[:, 1:]

    dims = (0, 2, 3, 4)
    inter = (probs * onehot).sum(dims)
    denom = (probs + onehot).sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def dice_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    ignore_bg: bool = True
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets)
    dl = dice_loss_multiclass(logits, targets, num_classes=num_classes, ignore_bg=ignore_bg)
    return dice_weight * dl + ce_weight * ce
