from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def save_overlay(
    image: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    out_path: str | os.PathLike,
    title: Optional[str] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    gt = (mask_gt > 0.5).astype(np.float32)
    pr = (mask_pred > 0.5).astype(np.float32)

    fig = plt.figure(figsize=(10, 3))
    if title:
        fig.suptitle(title)

    ax1 = plt.subplot(1, 3, 1); ax1.imshow(img); ax1.axis("off"); ax1.set_title("Image")
    ax2 = plt.subplot(1, 3, 2); ax2.imshow(gt, cmap="gray"); ax2.axis("off"); ax2.set_title("GT")
    ax3 = plt.subplot(1, 3, 3); ax3.imshow(pr, cmap="gray"); ax3.axis("off"); ax3.set_title("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
