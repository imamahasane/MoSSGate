from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import AugmentConfig, build_train_transforms, build_val_transforms

@dataclass
class ISICDataConfig:
    root: str
    image_dir: str = "images"
    mask_dir: str = "masks"
    train_split: str = "splits/train.txt"
    val_split: str = "splits/val.txt"
    test_split: str = "splits/test.txt"
    img_size: int = 352

def _read_ids(split_file: Path) -> List[str]:
    ids = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids

def _imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _imread_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    # binarize to {0,1}
    m = (m > 127).astype(np.uint8) * 255
    return m

class ISICDataset(Dataset):

    def __init__(self, cfg: ISICDataConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.image_dir = self.root / cfg.image_dir
        self.mask_dir = self.root / cfg.mask_dir

        split_map = {"train": cfg.train_split, "val": cfg.val_split, "test": cfg.test_split}
        if split not in split_map:
            raise ValueError(f"Unknown split={split}")
        self.ids = _read_ids(self.root / split_map[split])

        aug_cfg = AugmentConfig(img_size=cfg.img_size)
        self.tf = build_train_transforms(aug_cfg) if split == "train" else build_val_transforms(aug_cfg)

    def __len__(self) -> int:
        return len(self.ids)

    def _resolve_image_path(self, _id: str) -> Path:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = self.image_dir / f"{_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find image for id={_id} in {self.image_dir}")

    def _resolve_mask_path(self, _id: str) -> Path:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = self.mask_dir / f"{_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find mask for id={_id} in {self.mask_dir}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        _id = self.ids[idx]
        img = _imread_rgb(self._resolve_image_path(_id))
        mask = _imread_mask(self._resolve_mask_path(_id))

        out = self.tf(image=img, mask=mask)
        x = out["image"]  
        y = out["mask"]   
        if y.ndim == 2:
            y = y[None, ...]
        y = (y > 127).to(dtype=torch.float32) if hasattr(y, "to") else (y > 127).astype(np.float32)
        if not hasattr(y, "to"):
            y = torch.from_numpy(y).float()
        return x, y, _id
