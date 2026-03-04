from __future__ import annotations
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2

@dataclass
class AugmentConfig:
    img_size: int = 352

def build_train_transforms(cfg: AugmentConfig):
    return A.Compose(
        [
            A.Resize(cfg.img_size, cfg.img_size, interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-20, 20),
                translate_percent=(0.0, 0.05),
                shear=(-5, 5),
                p=0.5,
            ),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
            A.Normalize(),  
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )

def build_val_transforms(cfg: AugmentConfig):
    return A.Compose(
        [
            A.Resize(cfg.img_size, cfg.img_size, interpolation=1),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )
