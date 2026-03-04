from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class Metrics:
    dice: float
    iou: float
    acc: float

def _binarize_from_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > thr).to(dtype=torch.float32)

@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> Metrics:
    pred = _binarize_from_logits(logits, thr=thr)
    tgt = (targets > 0.5).to(dtype=torch.float32)

    pred_f = pred.view(pred.size(0), -1)
    tgt_f = tgt.view(tgt.size(0), -1)

    inter = (pred_f * tgt_f).sum(dim=1)
    union = pred_f.sum(dim=1) + tgt_f.sum(dim=1) - inter

    dice = (2 * inter + eps) / (pred_f.sum(dim=1) + tgt_f.sum(dim=1) + eps)
    iou = (inter + eps) / (union + eps)
    acc = (pred_f.eq(tgt_f)).to(torch.float32).mean(dim=1)

    return Metrics(dice=float(dice.mean().item()), iou=float(iou.mean().item()), acc=float(acc.mean().item()))
