from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = (targets > 0.5).to(dtype=probs.dtype)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, dice_smooth: float = 1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth=dice_smooth)
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, (targets > 0.5).to(dtype=logits.dtype))
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
